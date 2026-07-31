"""
Deterministically sample bounded text slices from HuggingFaceTB/smollm-corpus.

The script uses the public Hugging Face datasets-server row API, so it does not
need ``datasets`` or a multi-gigabyte Parquet download. It records the dataset
revision, selected row indices, per-document hashes, token estimates, and
licensing notes beside the exported UTF-8 text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dataset import clean_text, text_fingerprint


DATASET_ID = "HuggingFaceTB/smollm-corpus"
DATASET_API = f"https://huggingface.co/api/datasets/{DATASET_ID}"
ROWS_API = "https://datasets-server.huggingface.co/rows"
DEFAULT_SEED = "aurora-smollm-v1"
DEFAULT_PAGE_SIZE = 100
DEFAULT_FINEWEB_TOKENS = 7_500_000
DEFAULT_COSMOPEDIA_TOKENS = 2_000_000
USER_AGENT = "AuroraLM-corpus-sampler/1.0"


class SamplerError(RuntimeError):
    """Raised when a bounded corpus sample cannot be produced safely."""


@dataclass(frozen=True)
class SourceSpec:
    config: str
    output_name: str
    token_budget: int
    minimum_score: int | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_json(url: str, *, attempts: int = 12, timeout: int = 60) -> Any:
    delay = 1.0
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == attempts:
                raise SamplerError(f"HTTP {exc.code} while fetching {url}") from exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = max(delay, float(retry_after))
                except (TypeError, ValueError):
                    delay = max(delay, 30.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == attempts:
                raise SamplerError(f"Could not fetch JSON from {url}") from exc
        time.sleep(delay)
        delay = min(delay * 2.0, 8.0)
    raise AssertionError("unreachable")


def current_dataset_revision() -> dict[str, Any]:
    payload = _fetch_json(DATASET_API)
    revision = payload.get("sha")
    if not isinstance(revision, str) or not revision:
        raise SamplerError("Hugging Face dataset metadata omitted the revision")
    return {
        "dataset_id": payload.get("id", DATASET_ID),
        "revision_sha": revision,
        "last_modified": payload.get("lastModified"),
    }


def fetch_rows(config: str, offset: int, length: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": config,
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    payload = _fetch_json(f"{ROWS_API}?{query}")
    rows = payload.get("rows")
    total = payload.get("num_rows_total")
    if not isinstance(rows, list) or not isinstance(total, int):
        raise SamplerError(f"Unexpected datasets-server response for {config}")
    return {"rows": rows, "num_rows_total": total}


def deterministic_page_indices(
    *,
    total_rows: int,
    page_size: int,
    seed: str,
) -> list[int]:
    """Return every page index once in a deterministic pseudorandom order."""
    if total_rows < 1:
        raise ValueError("total_rows must be positive")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    page_count = (total_rows + page_size - 1) // page_size
    pages = list(range(page_count))
    random.Random(seed).shuffle(pages)
    return pages


def _row_text_and_tokens(
    spec: SourceSpec,
    row_wrapper: Mapping[str, Any],
) -> tuple[str, int, dict[str, Any]] | None:
    row = row_wrapper.get("row")
    if not isinstance(row, Mapping):
        raise SamplerError(f"{spec.config} returned a row without an object payload")

    text = clean_text(row.get("text"))
    if not text:
        return None

    safe_metadata: dict[str, Any] = {}
    if spec.config == "fineweb-edu-dedup":
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SamplerError("FineWeb-Edu row omitted metadata")
        score = metadata.get("int_score")
        if spec.minimum_score is not None:
            try:
                if int(score) < spec.minimum_score:
                    return None
            except (TypeError, ValueError) as exc:
                raise SamplerError("FineWeb-Edu row has invalid int_score") from exc
        if metadata.get("language") not in (None, "en"):
            return None
        raw_tokens = metadata.get("token_count")
        safe_metadata = {
            "source_id": row.get("id"),
            "int_score": score,
            "language_score": metadata.get("language_score"),
        }
    elif spec.config == "cosmopedia-v2":
        raw_tokens = row.get("token_length")
        safe_metadata = {
            "audience": row.get("audience"),
            "format": row.get("format"),
            "seed_data": row.get("seed_data"),
        }
    else:
        raise SamplerError(f"Unsupported SmolLM config {spec.config!r}")

    try:
        token_count = int(raw_tokens)
    except (TypeError, ValueError) as exc:
        raise SamplerError(f"{spec.config} row has no valid token count") from exc
    if token_count < 1:
        return None

    return text, token_count, safe_metadata


def sample_source(
    spec: SourceSpec,
    *,
    seed: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    fetcher: Callable[[str, int, int], Mapping[str, Any]] = fetch_rows,
    progress: Callable[[str], None] | None = print,
    request_delay: float = 0.0,
) -> dict[str, Any]:
    """Sample one source to its token budget using deterministic source pages."""
    if spec.token_budget < 1:
        raise ValueError("token budget must be positive")

    first = fetcher(spec.config, 0, 1)
    total_rows = first.get("num_rows_total")
    if not isinstance(total_rows, int) or total_rows < 1:
        raise SamplerError(f"{spec.config} did not report a valid row count")

    page_indices = deterministic_page_indices(
        total_rows=total_rows,
        page_size=page_size,
        seed=f"{seed}:{spec.config}",
    )
    selected_texts: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    token_total = 0
    pages_fetched = 0
    rows_examined = 0
    filtered_rows = 0
    duplicate_rows = 0

    for page_index in page_indices:
        offset = page_index * page_size
        length = min(page_size, total_rows - offset)
        payload = fetcher(spec.config, offset, length)
        pages_fetched += 1
        if request_delay > 0.0:
            time.sleep(request_delay)

        for row_wrapper in payload["rows"]:
            rows_examined += 1
            result = _row_text_and_tokens(spec, row_wrapper)
            if result is None:
                filtered_rows += 1
                continue
            text, token_count, metadata = result
            dedup_sha = text_fingerprint(text)
            if dedup_sha in seen_text:
                duplicate_rows += 1
                continue
            seen_text.add(dedup_sha)
            text_sha = _sha256_bytes(text.encode("utf-8"))

            row_index = row_wrapper.get("row_idx")
            if not isinstance(row_index, int):
                raise SamplerError(f"{spec.config} row omitted row_idx")
            selected_texts.append(text)
            selected_rows.append(
                {
                    "row_index": row_index,
                    "text_sha256": text_sha,
                    "token_count": token_count,
                    "metadata": metadata,
                }
            )
            token_total += token_count
            if token_total >= spec.token_budget:
                break

        if progress and (pages_fetched == 1 or pages_fetched % 25 == 0):
            progress(
                f"{spec.config}: {token_total:,}/{spec.token_budget:,} "
                f"tokens from {len(selected_texts):,} documents "
                f"({pages_fetched} pages)"
            )
        if token_total >= spec.token_budget:
            break
    else:
        raise SamplerError(
            f"{spec.config} exhausted all rows before reaching its token budget"
        )

    return {
        "config": spec.config,
        "output_name": spec.output_name,
        "token_budget": spec.token_budget,
        "token_count": token_total,
        "document_count": len(selected_texts),
        "pages_fetched": pages_fetched,
        "rows_examined": rows_examined,
        "filtered_rows": filtered_rows,
        "duplicate_rows": duplicate_rows,
        "minimum_score": spec.minimum_score,
        "texts": selected_texts,
        "selected_rows": selected_rows,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    revision: Mapping[str, Any],
    seed: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_paths = [output_dir / sample["output_name"] for sample in samples]
    manifest_path = output_dir / "smollm_sample_manifest.json"
    existing = [path for path in [*output_paths, manifest_path] if path.exists()]
    if existing and not overwrite:
        raise SamplerError(
            "Refusing to overwrite existing corpus files: "
            + ", ".join(path.name for path in existing)
        )

    source_manifests = []
    for sample, path in zip(samples, output_paths):
        text_payload = (
            "\n\n".join(sample["texts"]) + "\n"
        ).encode("utf-8")
        _atomic_write(path, text_payload)
        source_manifests.append(
            {
                key: value
                for key, value in sample.items()
                if key != "texts"
            }
            | {
                "file": path.name,
                "bytes": len(text_payload),
                "sha256": _sha256_bytes(text_payload),
            }
        )

    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dict(revision),
        "seed": seed,
        "sampling_method": (
            "deterministic pseudorandom datasets-server pages; "
            "deduplicated by cleaned-text SHA-256"
        ),
        "sources": source_manifests,
        "licensing": {
            "dataset_card": (
                "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus"
            ),
            "dataset_license": "ODC-By-1.0",
            "attribution": (
                "Contains information from HuggingFaceTB/smollm-corpus, "
                "made available under the ODC Attribution License."
            ),
            "content_rights_note": (
                "The database licence may not cover every underlying content "
                "right. Review before public or commercial distribution."
            ),
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _atomic_write(manifest_path, manifest_bytes)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--fineweb-tokens",
        type=int,
        default=DEFAULT_FINEWEB_TOKENS,
    )
    parser.add_argument(
        "--cosmopedia-tokens",
        type=int,
        default=DEFAULT_COSMOPEDIA_TOKENS,
    )
    parser.add_argument("--fineweb-min-score", type=int, default=3)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.0,
        help="Seconds between datasets-server page requests.",
    )
    parser.add_argument("--expected-revision")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        revision = current_dataset_revision()
        if (
            args.expected_revision
            and revision["revision_sha"] != args.expected_revision
        ):
            raise SamplerError(
                "Dataset revision changed: expected "
                f"{args.expected_revision}, found {revision['revision_sha']}"
            )

        specs = [
            SourceSpec(
                config="fineweb-edu-dedup",
                output_name="fineweb_edu_dedup.txt",
                token_budget=args.fineweb_tokens,
                minimum_score=args.fineweb_min_score,
            ),
            SourceSpec(
                config="cosmopedia-v2",
                output_name="cosmopedia_v2.txt",
                token_budget=args.cosmopedia_tokens,
            ),
        ]
        samples = [
            sample_source(
                spec,
                seed=args.seed,
                page_size=args.page_size,
                request_delay=args.request_delay,
            )
            for spec in specs
        ]
        manifest = write_samples(
            samples,
            output_dir=args.output_dir,
            revision=revision,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"Saved bounded SmolLM sample to {args.output_dir.resolve()}")
        for source in manifest["sources"]:
            print(
                f"{source['config']}: {source['token_count']:,} tokens, "
                f"{source['document_count']:,} documents"
            )
        return 0
    except (OSError, SamplerError, ValueError) as exc:
        print(f"SmolLM sampling failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
