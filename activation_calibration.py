"""Freeze AuroraLM's shared thalamic activation ceiling from real captures.

Calibration deliberately reads only the ``aurora_state`` portion of Aurora
session logs.  Regulator payloads and conversation text are irrelevant to the
raw 171-node scale and are never copied into the public calibration receipt.

The ceiling is the maximum observed positive activation.  AuroraLM does not
silently clip values during training or readiness checks, so a later capture
above the frozen ceiling forces an explicit recalibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from state_schema import AuroraState171, SCHEMA_FINGERPRINT, SCHEMA_VERSION


CALIBRATION_FORMAT_VERSION = 1
DEFAULT_SESSION_GLOB = "aurora_training_*.jsonl"
CALIBRATION_METHOD = "maximum observed positive activation; no clipping"


class CalibrationError(ValueError):
    """Raised when session logs cannot support a trustworthy calibration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise CalibrationError("Cannot calculate a percentile without values.")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_weight = upper - position
    upper_weight = position - lower
    return (
        sorted_values[lower] * lower_weight
        + sorted_values[upper] * upper_weight
    )


def _source_digest(descriptors: Iterable[dict[str, Any]]) -> str:
    encoded = json.dumps(
        list(descriptors),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibrate_session_logs(
    paths: Sequence[Path],
    *,
    frozen_at: str | None = None,
) -> dict[str, Any]:
    """Return a privacy-preserving calibration receipt for session logs."""

    sources = sorted((Path(path) for path in paths), key=lambda path: str(path))
    if not sources:
        raise CalibrationError("No Aurora session logs were supplied.")

    active_scores: list[float] = []
    record_count = 0
    source_descriptors: list[dict[str, Any]] = []

    for path in sources:
        if not path.is_file():
            raise CalibrationError(f"Session log does not exist: {path}")

        file_records = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("record is not an object")
                        state_record = record.get("aurora_state")
                        if not isinstance(state_record, dict):
                            raise ValueError("aurora_state is missing")
                        state = AuroraState171.from_record(state_record)
                        if state.source != "thalamus":
                            raise ValueError("state source is not thalamus")
                        if record.get("timestamp") != state.captured_at:
                            raise ValueError(
                                "turn/state capture timestamps do not match"
                            )
                    except (TypeError, ValueError) as exc:
                        raise CalibrationError(
                            f"{path.name}:{line_number}: {exc}"
                        ) from exc

                    active_scores.extend(
                        value for value in state.values if value > 0.0
                    )
                    file_records += 1
                    record_count += 1
        except OSError as exc:
            raise CalibrationError(f"Could not read {path}: {exc}") from exc

        source_descriptors.append(
            {
                "sha256": _sha256(path),
                "record_count": file_records,
            }
        )

    if record_count == 0:
        raise CalibrationError("Calibration logs contain no state records.")
    if not active_scores:
        raise CalibrationError(
            "Calibration logs contain no positive thalamic activations."
        )

    active_scores.sort()
    ceiling = max(active_scores)
    timestamp = frozen_at or datetime.now(timezone.utc).isoformat()
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise CalibrationError("frozen_at must be a non-empty timestamp.")

    return {
        "format_version": CALIBRATION_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "activation_ceiling": ceiling,
        "method": CALIBRATION_METHOD,
        "frozen_at": timestamp,
        "record_count": record_count,
        "active_score_count": len(active_scores),
        "source_log_count": len(source_descriptors),
        "source_digest": _source_digest(source_descriptors),
        "source_digest_contract": (
            "sha256(canonical JSON of ordered "
            "{sha256,record_count} descriptors)"
        ),
        "statistics": {
            "minimum_positive": min(active_scores),
            "median": _percentile(active_scores, 0.50),
            "p95": _percentile(active_scores, 0.95),
            "p99": _percentile(active_scores, 0.99),
            "maximum": ceiling,
        },
        "privacy": (
            "Aggregate receipt only; raw session logs and conversation text "
            "are excluded from Git."
        ),
    }


def write_calibration(path: Path, calibration: dict[str, Any]) -> None:
    """Atomically write one stable, human-readable calibration receipt."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = (
        json.dumps(
            calibration,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate AuroraLM's activation ceiling from genuine Aurora "
            "session logs without exporting private turn text."
        )
    )
    parser.add_argument("--sessions-dir", type=Path, required=True)
    parser.add_argument("--glob", default=DEFAULT_SESSION_GLOB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-at")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = sorted(args.sessions_dir.glob(args.glob))
    calibration = calibrate_session_logs(paths, frozen_at=args.frozen_at)
    write_calibration(args.output, calibration)
    print(
        "Calibrated "
        f"{calibration['record_count']} records; "
        f"activation ceiling={calibration['activation_ceiling']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
