"""
Deterministic, privacy-conscious dataset preparation for AuroraLM.

The builder reads only explicitly supplied, allowlisted sources:

* Aurora journal ``entry`` text and legacy ``context`` input/response pairs;
* memory ``summary`` text (never embeddings);
* verified state-aligned JSONL emitted by Aurora's session logger;
* optional UTF-8 general-corpus text files.

It produces separate outputs for clean documents, legacy unverified turns, and
verified stateful turns. Generated data belongs under ``data/`` or
``datasets/``, both of which are ignored by this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config import REGULATOR_NAMES, REGULATOR_RANGES
from state_schema import AuroraState171


DATASET_FORMAT_VERSION = 1
DEFAULT_SEED = "aurora-dataset-v1"
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_MIN_CHARS = 3
SESSION_GLOB = "aurora_training_*.jsonl"

OUTPUT_FILENAMES = (
    "documents.jsonl",
    "legacy_turns.jsonl",
    "stateful_turns.jsonl",
    "tokenizer_corpus.txt",
    "manifest.json",
)


class DatasetError(RuntimeError):
    """Raised when an input cannot safely enter the AuroraLM dataset."""


def clean_text(value: Any) -> str:
    """Return stable NFKC text with internal whitespace collapsed."""
    if not isinstance(value, str):
        return ""
    normalised = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalised).strip()


def _normalised_text_key(text: str) -> str:
    return clean_text(text).casefold()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_fingerprint(text: str) -> str:
    """Fingerprint text for case-insensitive, whitespace-stable deduplication."""
    return _sha256_bytes(_normalised_text_key(text).encode("utf-8"))


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_split(
    group_key: str,
    *,
    seed: str = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> str:
    """Assign a stable train/validation split without global RNG state."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    payload = f"{seed}\0{group_key}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    score = bucket / float(2**64)
    return "validation" if score < validation_fraction else "train"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read valid UTF-8 JSON from {path.name}") from exc


def _source_descriptor(path: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_fingerprint(path),
    }


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise DatasetError("Metadata contains a non-finite float")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _safe_scalar(item())
    raise DatasetError(f"Metadata scalar has unsupported type {type(value).__name__}")


def _safe_subset(mapping: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    return {
        field: _safe_scalar(mapping.get(field))
        for field in fields
        if field in mapping
    }


def _validate_regulators(value: Any, *, location: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DatasetError(f"{location} has no regulator mapping")

    supplied = set(value)
    required = set(REGULATOR_NAMES)
    missing = sorted(required - supplied)
    unknown = sorted(supplied - required)
    if missing or unknown:
        raise DatasetError(
            f"{location} regulator contract mismatch: "
            f"missing={missing}, unknown={unknown}"
        )

    regulators: dict[str, float] = {}
    for name in REGULATOR_NAMES:
        raw_value = value[name]
        if isinstance(raw_value, bool):
            raise DatasetError(
                f"{location} regulator {name!r} must be numeric"
            )
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"{location} regulator {name!r} must be numeric"
            ) from exc
        lower, upper = REGULATOR_RANGES[name]
        if not isfinite(number) or not lower <= number <= upper:
            raise DatasetError(
                f"{location} regulator {name!r}={number!r} must be in "
                f"[{lower}, {upper}]"
            )
        regulators[name] = number
    return regulators


def _day_group(timestamp: Any, fallback: str) -> str:
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        return timestamp[:10]
    return fallback


class DocumentAccumulator:
    """Globally deduplicate clean text while preserving allowlisted provenance."""

    def __init__(self, min_chars: int):
        self.min_chars = min_chars
        self._documents: dict[str, dict[str, Any]] = {}
        self.skipped_short = 0
        self.duplicates = 0

    def add(
        self,
        text: Any,
        *,
        source: str,
        source_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        cleaned = clean_text(text)
        if len(cleaned) < self.min_chars:
            self.skipped_short += 1
            return

        dedup_key = text_fingerprint(cleaned)
        provenance = {
            "source": source,
            "source_index": source_index,
            "metadata": dict(metadata or {}),
        }
        existing = self._documents.get(dedup_key)
        if existing is not None:
            self.duplicates += 1
            if provenance not in existing["provenance"]:
                existing["provenance"].append(provenance)
            return

        self._documents[dedup_key] = {
            "text": cleaned,
            "text_sha256": _sha256_bytes(cleaned.encode("utf-8")),
            "dedup_sha256": dedup_key,
            "provenance": [provenance],
        }

    def records(
        self,
        *,
        seed: str,
        validation_fraction: float,
    ) -> list[dict[str, Any]]:
        records = []
        for dedup_key, document in sorted(self._documents.items()):
            records.append(
                {
                    "format_version": DATASET_FORMAT_VERSION,
                    "record_id": f"doc-{dedup_key[:24]}",
                    "split": assign_split(
                        f"document:{dedup_key}",
                        seed=seed,
                        validation_fraction=validation_fraction,
                    ),
                    **document,
                }
            )
        return records


def _legacy_turn_key(input_text: str, output_text: str) -> str:
    payload = (
        _normalised_text_key(input_text)
        + "\0"
        + _normalised_text_key(output_text)
    )
    return _sha256_bytes(payload.encode("utf-8"))


def extract_journal(
    path: Path,
    documents: DocumentAccumulator,
    *,
    seed: str,
    validation_fraction: float,
    min_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise DatasetError(f"{path.name} must contain a list of journal entries")

    legacy_by_key: dict[str, dict[str, Any]] = {}
    duplicate_turns = 0

    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise DatasetError(f"{path.name} entry {index} is not an object")

        timestamp = item.get("timestamp")
        trigger = item.get("trigger")
        documents.add(
            item.get("entry"),
            source="journal.entry",
            source_index=index,
            metadata=_safe_subset(
                item,
                ("timestamp", "trigger", "pfc_state"),
            ),
        )

        context = item.get("context")
        if not isinstance(context, Mapping):
            continue
        input_text = clean_text(context.get("last_input"))
        output_text = clean_text(context.get("last_response"))
        if len(input_text) < min_chars or len(output_text) < min_chars:
            continue

        pair_key = _legacy_turn_key(input_text, output_text)
        if pair_key in legacy_by_key:
            duplicate_turns += 1
            continue

        group = _day_group(timestamp, f"legacy:{pair_key}")
        legacy_by_key[pair_key] = {
            "format_version": DATASET_FORMAT_VERSION,
            "record_id": f"legacy-turn-{pair_key[:24]}",
            "split": assign_split(
                f"legacy-day:{group}",
                seed=seed,
                validation_fraction=validation_fraction,
            ),
            "source": "journal.context",
            "source_index": index,
            "timestamp": timestamp,
            "input_text": input_text,
            "output_text": output_text,
            "expressed": None,
            "alignment_quality": "legacy_unverified",
            "aurora_state": None,
            "regulators": {
                "tension": _safe_scalar(item.get("tension_at_writing")),
                "coherence": _safe_scalar(item.get("coherence_at_writing")),
                "prediction_bias": _safe_scalar(item.get("bias_at_writing")),
                "pfc_state": _safe_scalar(item.get("pfc_state")),
            },
            "events": {
                "trigger": _safe_scalar(trigger),
                "dominant_emotion": _safe_scalar(
                    context.get("dominant_emotion")
                ),
            },
        }

    return list(sorted(legacy_by_key.values(), key=lambda row: row["record_id"])), duplicate_turns


def extract_memories(path: Path, documents: DocumentAccumulator) -> None:
    data = _load_json(path)
    if not isinstance(data, Mapping) or not isinstance(data.get("memories"), list):
        raise DatasetError(f"{path.name} must contain a 'memories' list")

    for index, item in enumerate(data["memories"]):
        if not isinstance(item, Mapping):
            raise DatasetError(f"{path.name} memory {index} is not an object")
        metadata = item.get("metadata")
        safe_metadata = {
            "memory_type": _safe_scalar(item.get("memory_type")),
            **_safe_subset(
                metadata,
                ("encoding_emotion", "encoded_at", "promoted_at"),
            ),
        }
        documents.add(
            item.get("summary"),
            source="memory.summary",
            source_index=index,
            metadata=safe_metadata,
        )


def _validate_stateful_record(
    record: Any,
    *,
    filename: str,
    line_number: int,
    seed: str,
    validation_fraction: float,
) -> dict[str, Any]:
    location = f"{filename}:{line_number}"
    if not isinstance(record, Mapping):
        raise DatasetError(f"{location} is not a JSON object")

    input_text = clean_text(record.get("input_text"))
    if not input_text:
        raise DatasetError(f"{location} has no usable input_text")

    raw_output = record.get("output_text")
    if raw_output is not None and not isinstance(raw_output, str):
        raise DatasetError(f"{location} output_text must be a string or null")
    output_text = clean_text(raw_output) if raw_output is not None else None
    if raw_output is not None and not output_text:
        output_text = None

    expressed = record.get("expressed")
    if not isinstance(expressed, bool):
        raise DatasetError(f"{location} expressed must be boolean")
    if expressed != (output_text is not None):
        raise DatasetError(f"{location} expressed/output_text disagree")

    state_record = record.get("aurora_state")
    if not isinstance(state_record, Mapping):
        raise DatasetError(f"{location} has no Aurora state record")
    try:
        state = AuroraState171.from_record(state_record)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{location} has an invalid Aurora state") from exc

    session_id = clean_text(record.get("session_id")) or filename
    raw_turn = record.get("turn")
    try:
        turn_number = int(raw_turn)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{location} turn must be an integer") from exc

    timestamp = record.get("timestamp")
    identity_payload = json.dumps(
        {
            "session_id": session_id,
            "turn": turn_number,
            "timestamp": timestamp,
            "input_sha256": text_fingerprint(input_text),
            "state_sha256": _sha256_bytes(
                json.dumps(state.values, separators=(",", ":")).encode("utf-8")
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    record_id = _sha256_bytes(identity_payload.encode("utf-8"))

    regulators = _validate_regulators(
        record.get("regulators"),
        location=location,
    )
    events = _safe_subset(
        record.get("events"),
        ("dominant_emotion", "memory_writes", "curiosity_triggered"),
    )

    return {
        "format_version": DATASET_FORMAT_VERSION,
        "record_id": f"stateful-turn-{record_id[:24]}",
        "split": assign_split(
            f"session:{session_id}",
            seed=seed,
            validation_fraction=validation_fraction,
        ),
        "source": clean_text(record.get("source")) or "unknown",
        "session_id": session_id,
        "turn": turn_number,
        "timestamp": timestamp,
        "input_text": input_text,
        "output_text": output_text,
        "expressed": expressed,
        "alignment_quality": "exact_turn_capture",
        "aurora_state": state.to_record(),
        "regulators": regulators,
        "events": events,
    }


def extract_stateful_sessions(
    sessions_dir: Path | None,
    *,
    seed: str,
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sessions_dir is None:
        return [], []
    if not sessions_dir.is_dir():
        raise DatasetError(f"Session directory not found: {sessions_dir}")

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for path in sorted(sessions_dir.glob(SESSION_GLOB)):
        sources.append(_source_descriptor(path, "stateful_session_jsonl"))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise DatasetError(f"Could not read UTF-8 session file {path.name}") from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"{path.name}:{line_number} is invalid JSON"
                ) from exc
            record = _validate_stateful_record(
                raw_record,
                filename=path.name,
                line_number=line_number,
                seed=seed,
                validation_fraction=validation_fraction,
            )
            if record["record_id"] in seen_ids:
                raise DatasetError(
                    f"Duplicate stateful turn identity in {path.name}:{line_number}"
                )
            seen_ids.add(record["record_id"])
            records.append(record)

    records.sort(key=lambda row: (row["session_id"], row["turn"], row["record_id"]))
    return records, sources


def extract_general_text(
    paths: Iterable[Path],
    documents: DocumentAccumulator,
) -> list[dict[str, Any]]:
    sources = []
    for path in sorted(paths, key=lambda value: str(value)):
        if not path.is_file():
            raise DatasetError(f"General text file not found: {path}")
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DatasetError(f"General text is not valid UTF-8: {path.name}") from exc
        sources.append(_source_descriptor(path, "general_text"))
        paragraphs = re.split(r"\n\s*\n+", raw_text)
        for index, paragraph in enumerate(paragraphs):
            documents.add(
                paragraph,
                source="general.text",
                source_index=index,
                metadata={"source_name": path.name},
            )
    return sources


def _split_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(record.get("split", "unknown") for record in records)
    return dict(sorted(counts.items()))


def _tokenizer_texts(
    documents: Sequence[Mapping[str, Any]],
    legacy_turns: Sequence[Mapping[str, Any]],
    stateful_turns: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Use train-split text only and globally deduplicate individual strings."""
    by_key: dict[str, str] = {}

    def add(text: Any) -> None:
        cleaned = clean_text(text)
        if cleaned:
            by_key.setdefault(text_fingerprint(cleaned), cleaned)

    for document in documents:
        if document["split"] == "train":
            add(document["text"])
    for turn in [*legacy_turns, *stateful_turns]:
        if turn["split"] == "train":
            add(turn["input_text"])
            add(turn.get("output_text"))

    return [by_key[key] for key in sorted(by_key)]


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        return b""
    text = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in records
    )
    return (text + "\n").encode("utf-8")


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


def prepare_dataset(
    *,
    journal_path: Path | None,
    memories_path: Path | None,
    sessions_dir: Path | None = None,
    general_text_paths: Sequence[Path] = (),
    seed: str = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> dict[str, Any]:
    """Load approved sources and return records plus a text-free audit summary."""
    if journal_path is None and memories_path is None and sessions_dir is None and not general_text_paths:
        raise DatasetError("At least one input source is required")
    if min_chars < 1:
        raise ValueError("min_chars must be at least 1")
    # Validate early so empty inputs do not bypass split validation.
    assign_split(
        "validation-check",
        seed=seed,
        validation_fraction=validation_fraction,
    )

    documents = DocumentAccumulator(min_chars=min_chars)
    sources = []
    legacy_turns: list[dict[str, Any]] = []
    legacy_duplicates = 0

    if journal_path is not None:
        if not journal_path.is_file():
            raise DatasetError(f"Journal file not found: {journal_path}")
        sources.append(_source_descriptor(journal_path, "aurora_journal"))
        legacy_turns, legacy_duplicates = extract_journal(
            journal_path,
            documents,
            seed=seed,
            validation_fraction=validation_fraction,
            min_chars=min_chars,
        )

    if memories_path is not None:
        if not memories_path.is_file():
            raise DatasetError(f"Memories file not found: {memories_path}")
        sources.append(_source_descriptor(memories_path, "aurora_memories"))
        extract_memories(memories_path, documents)

    sources.extend(extract_general_text(general_text_paths, documents))
    stateful_turns, session_sources = extract_stateful_sessions(
        sessions_dir,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    sources.extend(session_sources)

    document_records = documents.records(
        seed=seed,
        validation_fraction=validation_fraction,
    )
    tokenizer_texts = _tokenizer_texts(
        document_records,
        legacy_turns,
        stateful_turns,
    )

    summary = {
        "documents": {
            "total": len(document_records),
            "splits": _split_counts(document_records),
            "duplicates_removed": documents.duplicates,
            "short_or_empty_skipped": documents.skipped_short,
        },
        "legacy_turns": {
            "total": len(legacy_turns),
            "splits": _split_counts(legacy_turns),
            "duplicates_removed": legacy_duplicates,
            "alignment_quality": "legacy_unverified",
        },
        "stateful_turns": {
            "total": len(stateful_turns),
            "splits": _split_counts(stateful_turns),
            "alignment_quality": "exact_turn_capture",
        },
        "tokenizer_train_texts": len(tokenizer_texts),
    }

    return {
        "documents": document_records,
        "legacy_turns": legacy_turns,
        "stateful_turns": stateful_turns,
        "tokenizer_texts": tokenizer_texts,
        "sources": sources,
        "summary": summary,
        "settings": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "min_chars": min_chars,
        },
    }


def write_dataset(
    prepared: Mapping[str, Any],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write all outputs atomically and return the complete manifest."""
    targets = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise DatasetError(
            f"Output already exists ({names}); pass --overwrite explicitly"
        )

    documents_bytes = _jsonl_bytes(prepared["documents"])
    legacy_bytes = _jsonl_bytes(prepared["legacy_turns"])
    stateful_bytes = _jsonl_bytes(prepared["stateful_turns"])
    tokenizer_bytes = (
        "\n".join(prepared["tokenizer_texts"]) + "\n"
        if prepared["tokenizer_texts"]
        else ""
    ).encode("utf-8")

    payloads = {
        "documents.jsonl": documents_bytes,
        "legacy_turns.jsonl": legacy_bytes,
        "stateful_turns.jsonl": stateful_bytes,
        "tokenizer_corpus.txt": tokenizer_bytes,
    }
    for name, payload in payloads.items():
        _atomic_write(targets[name], payload)

    manifest = {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": prepared["settings"],
        "sources": prepared["sources"],
        "summary": prepared["summary"],
        "privacy_contract": {
            "allowlisted_text_fields": [
                "journal.entry",
                "journal.context.last_input",
                "journal.context.last_response",
                "memories.summary",
                "session.input_text",
                "session.output_text",
                "general_text",
            ],
            "excluded_fields": [
                "embedding",
                "raw_fragments",
                "memory metadata outside the explicit allowlist",
                "journal context outside the explicit allowlist",
                "backup files",
                "live state files",
                "Minecraft logs",
            ],
            "legacy_turn_policy": (
                "kept separate and tagged legacy_unverified; never represented "
                "as exact state-aligned supervision"
            ),
        },
        "outputs": {
            name: {
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in payloads.items()
        },
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(targets["manifest.json"], manifest_bytes)
    return manifest


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--memories", type=Path)
    parser.add_argument("--sessions-dir", type=Path)
    parser.add_argument(
        "--general-text",
        type=Path,
        action="append",
        default=[],
        help="UTF-8 general-corpus text file; repeat for multiple files.",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Inspect counts, splits, duplicates, and hashes without writing text.",
    )
    _add_source_arguments(audit)

    build = subparsers.add_parser(
        "build",
        help="Build deterministic JSONL datasets and tokenizer corpus.",
    )
    _add_source_arguments(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    return parser


def _prepare_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_dataset(
        journal_path=args.journal,
        memories_path=args.memories,
        sessions_dir=args.sessions_dir,
        general_text_paths=args.general_text,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        min_chars=args.min_chars,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        prepared = _prepare_from_args(args)
        if args.command == "audit":
            report = {
                "settings": prepared["settings"],
                "sources": prepared["sources"],
                "summary": prepared["summary"],
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        manifest = write_dataset(
            prepared,
            args.output,
            overwrite=args.overwrite,
        )
        print(f"Built AuroraLM dataset in {args.output.resolve()}")
        print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"Dataset build failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
