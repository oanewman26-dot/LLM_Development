"""
Two-gate readiness audit before AuroraLM state-conditioned experimentation.

Gate 1 asks whether the dense base model is ready to freeze. Gate 2 asks
whether genuine state-aligned material and a frozen calibration ceiling exist.
The audit never runs correct-vs-shuffled-state training; it only authorises
that later experiment when both prerequisites pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional

from activation_calibration import (
    CALIBRATION_FORMAT_VERSION,
    CALIBRATION_METHOD,
)
from config import ACTIVATION_CEILING
from state_schema import (
    AuroraState171,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
)


AUDIT_FORMAT_VERSION = 1
DEFAULT_CALIBRATION_MANIFEST = (
    Path(__file__).resolve().parent
    / "calibration"
    / "activation_calibration_v1.json"
)
SCORECARD_FORMAT_VERSION = 1
RECENT_EVALUATIONS = 5
MAX_RECENT_VALIDATION_IMPROVEMENT = 0.01
MINIMUM_SEMANTIC_MEAN = 2.5
SEMANTIC_CRITERIA = (
    "prompt_intent_retention",
    "clause_to_clause_coherence",
    "semantic_stability",
)
BLIND_SEED = "auroralm-base-readiness-v1"

QUALITATIVE_FILES = {
    "sampled": (
        "qualitative_step_5000.json",
        "qualitative_phase2_sampled.json",
    ),
    "greedy": (
        "qualitative_step_5000_greedy.json",
        "qualitative_phase2_greedy.json",
    ),
    "conservative": (
        "qualitative_step_5000_conservative.json",
        "qualitative_phase2_conservative.json",
    ),
}


class ReadinessGateError(RuntimeError):
    """Raised when a readiness input is malformed rather than merely absent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessGateError(f"Could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ReadinessGateError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    metrics = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReadinessGateError(
                        f"{path}:{line_number} is not a JSON object."
                    )
                metrics.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessGateError(f"Could not read metrics: {path}") from exc
    return metrics


def _metric_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [
        metric
        for metric in metrics
        if isinstance(metric.get("step"), int)
        and isinstance(metric.get("validation_loss"), (int, float))
        and math.isfinite(float(metric["validation_loss"]))
    ]
    if not validation:
        return {
            "evaluations": 0,
            "latest_step": None,
            "latest_validation_loss": None,
            "recent_validation_improvement": None,
            "recent_window": 0,
        }
    recent = validation[-RECENT_EVALUATIONS:]
    improvement = (
        float(recent[0]["validation_loss"])
        - float(recent[-1]["validation_loss"])
    )
    return {
        "evaluations": len(validation),
        "latest_step": validation[-1]["step"],
        "latest_validation_loss": float(validation[-1]["validation_loss"]),
        "latest_validation_perplexity": float(
            validation[-1].get("validation_perplexity", float("nan"))
        ),
        "recent_validation_improvement": improvement,
        "recent_window": len(recent),
        "recent_steps": [metric["step"] for metric in recent],
    }


def _candidate_is_a(item_id: str) -> bool:
    digest = hashlib.sha256(f"{BLIND_SEED}:{item_id}".encode("utf-8")).digest()
    return digest[0] % 2 == 0


def _result_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ReadinessGateError("Qualitative report has no results list.")
    mapped = {}
    for result in results:
        if not isinstance(result, Mapping) or not isinstance(
            result.get("name"), str
        ):
            raise ReadinessGateError("Qualitative result is malformed.")
        mapped[result["name"]] = result
    return mapped


def build_blinded_scorecard(
    *,
    baseline_reports: Mapping[str, Mapping[str, Any]],
    candidate_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    items = []
    for decoding in ("sampled", "conservative"):
        baseline = _result_map(baseline_reports[decoding])
        candidate = _result_map(candidate_reports[decoding])
        if set(baseline) != set(candidate):
            raise ReadinessGateError(
                f"{decoding} baseline/candidate prompts do not match."
            )
        for name in sorted(baseline):
            baseline_result = baseline[name]
            candidate_result = candidate[name]
            if baseline_result.get("prompt") != candidate_result.get("prompt"):
                raise ReadinessGateError(
                    f"{decoding}:{name} prompt text has drifted."
                )
            item_id = f"{decoding}:{name}"
            candidate_a = _candidate_is_a(item_id)
            continuation_a = (
                candidate_result["text"]
                if candidate_a
                else baseline_result["text"]
            )
            continuation_b = (
                baseline_result["text"]
                if candidate_a
                else candidate_result["text"]
            )
            empty_scores = {
                criterion: None for criterion in SEMANTIC_CRITERIA
            }
            items.append(
                {
                    "id": item_id,
                    "decoding": decoding,
                    "prompt": baseline_result["prompt"],
                    "continuation_a": continuation_a,
                    "continuation_b": continuation_b,
                    "scores": {
                        "a": dict(empty_scores),
                        "b": dict(empty_scores),
                    },
                }
            )
    return {
        "format_version": SCORECARD_FORMAT_VERSION,
        "blind_seed": BLIND_SEED,
        "review_status": "pending",
        "instructions": {
            "scale": {
                "0": "fails completely",
                "1": "mostly fails",
                "2": "mixed or fragile",
                "3": "mostly succeeds",
                "4": "succeeds clearly",
            },
            "criteria": {
                "prompt_intent_retention": (
                    "Does the continuation remain responsive to the prompt?"
                ),
                "clause_to_clause_coherence": (
                    "Do successive clauses form one intelligible thought?"
                ),
                "semantic_stability": (
                    "Does meaning remain stable without contradiction or drift?"
                ),
            },
            "completion": (
                "Score every A and B field from 0 to 4, then set "
                "review_status to complete. Do not try to identify models."
            ),
        },
        "items": items,
    }


def _score_semantic_review(
    scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    if scorecard.get("format_version") != SCORECARD_FORMAT_VERSION:
        raise ReadinessGateError("Unsupported semantic scorecard format.")
    if scorecard.get("blind_seed") != BLIND_SEED:
        raise ReadinessGateError("Semantic scorecard blind seed has drifted.")
    if scorecard.get("review_status") != "complete":
        return {
            "status": "pending",
            "threshold": MINIMUM_SEMANTIC_MEAN,
        }
    items = scorecard.get("items")
    if not isinstance(items, list) or not items:
        raise ReadinessGateError("Completed scorecard has no items.")

    candidate_scores = {criterion: [] for criterion in SEMANTIC_CRITERIA}
    baseline_scores = {criterion: [] for criterion in SEMANTIC_CRITERIA}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ReadinessGateError("Scorecard item is malformed.")
        scores = item.get("scores")
        if not isinstance(scores, Mapping):
            raise ReadinessGateError("Scorecard item has no scores.")
        candidate_label = "a" if _candidate_is_a(item["id"]) else "b"
        baseline_label = "b" if candidate_label == "a" else "a"
        for label, destination in (
            (candidate_label, candidate_scores),
            (baseline_label, baseline_scores),
        ):
            labelled = scores.get(label)
            if not isinstance(labelled, Mapping):
                raise ReadinessGateError("Scorecard side is malformed.")
            for criterion in SEMANTIC_CRITERIA:
                value = labelled.get(criterion)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= float(value) <= 4
                ):
                    raise ReadinessGateError(
                        f"Score {item['id']}:{label}:{criterion} "
                        "must be between 0 and 4."
                    )
                destination[criterion].append(float(value))

    candidate_means = {
        criterion: mean(values)
        for criterion, values in candidate_scores.items()
    }
    baseline_means = {
        criterion: mean(values)
        for criterion, values in baseline_scores.items()
    }
    passed = all(
        value >= MINIMUM_SEMANTIC_MEAN
        for value in candidate_means.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "threshold": MINIMUM_SEMANTIC_MEAN,
        "candidate_means": candidate_means,
        "baseline_means": baseline_means,
    }


def evaluate_base_gate(
    *,
    candidate_run: Path,
    baseline_run: Path,
    scorecard_path: Optional[Path] = None,
    scorecard_output: Optional[Path] = None,
) -> dict[str, Any]:
    candidate_run = Path(candidate_run)
    baseline_run = Path(baseline_run)
    run_config = _read_json(candidate_run / "run_config.json")
    max_steps = run_config.get("settings", {}).get("max_steps")
    if not isinstance(max_steps, int):
        raise ReadinessGateError("Candidate max_steps is missing.")

    latest_path = candidate_run / "latest.json"
    latest = _read_json(latest_path) if latest_path.is_file() else {}
    completed_steps = latest.get("completed_steps", 0)
    complete = completed_steps == max_steps

    candidate_metrics = _metric_summary(
        _read_metrics(candidate_run / "metrics.jsonl")
    )
    baseline_metrics = _metric_summary(
        _read_metrics(baseline_run / "metrics.jsonl")
    )
    recent_improvement = candidate_metrics["recent_validation_improvement"]
    plateau_pass = (
        complete
        and candidate_metrics["recent_window"] == RECENT_EVALUATIONS
        and recent_improvement is not None
        and recent_improvement <= MAX_RECENT_VALIDATION_IMPROVEMENT
    )
    candidate_loss = candidate_metrics["latest_validation_loss"]
    baseline_loss = baseline_metrics["latest_validation_loss"]
    improves_on_baseline = (
        candidate_loss is not None
        and baseline_loss is not None
        and candidate_loss < baseline_loss
    )

    baseline_reports: dict[str, dict[str, Any]] = {}
    candidate_reports: dict[str, dict[str, Any]] = {}
    missing_qualitative = []
    for decoding, (baseline_name, candidate_name) in QUALITATIVE_FILES.items():
        baseline_path = baseline_run / baseline_name
        candidate_path = candidate_run / candidate_name
        if not baseline_path.is_file():
            missing_qualitative.append(str(baseline_path))
        else:
            baseline_reports[decoding] = _read_json(baseline_path)
        if not candidate_path.is_file():
            missing_qualitative.append(str(candidate_path))
        else:
            candidate_reports[decoding] = _read_json(candidate_path)

    semantic_review = {
        "status": "pending",
        "threshold": MINIMUM_SEMANTIC_MEAN,
    }
    scorecard_created = False
    if not missing_qualitative:
        template = build_blinded_scorecard(
            baseline_reports=baseline_reports,
            candidate_reports=candidate_reports,
        )
        if scorecard_path is not None and Path(scorecard_path).is_file():
            semantic_review = _score_semantic_review(
                _read_json(Path(scorecard_path))
            )
        elif scorecard_output is not None:
            scorecard_output = Path(scorecard_output)
            if not scorecard_output.exists():
                _write_json(scorecard_output, template)
                scorecard_created = True

    reasons = []
    if not complete:
        reasons.append(
            f"candidate training is incomplete ({completed_steps}/{max_steps})"
        )
    if complete and not plateau_pass:
        reasons.append(
            "validation is still improving materially over the final "
            f"{RECENT_EVALUATIONS} evaluations"
        )
    if complete and not improves_on_baseline:
        reasons.append("candidate validation loss does not beat the baseline")
    if complete and missing_qualitative:
        reasons.append("candidate qualitative suite is incomplete")
    if complete and semantic_review["status"] == "pending":
        reasons.append("blinded semantic scorecard is not complete")
    if semantic_review["status"] == "fail":
        reasons.append("candidate failed the blinded semantic threshold")

    if not complete or missing_qualitative or semantic_review["status"] == "pending":
        status = "PENDING"
    elif (
        plateau_pass
        and improves_on_baseline
        and semantic_review["status"] == "pass"
    ):
        status = "PASS"
    else:
        status = "NOT_READY"

    return {
        "gate": "base_model_readiness",
        "status": status,
        "candidate_complete": complete,
        "completed_steps": completed_steps,
        "target_steps": max_steps,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "criteria": {
            "recent_evaluations": RECENT_EVALUATIONS,
            "maximum_recent_validation_improvement": (
                MAX_RECENT_VALIDATION_IMPROVEMENT
            ),
            "plateau_pass": plateau_pass,
            "improves_on_baseline": improves_on_baseline,
            "semantic_review": semantic_review,
        },
        "missing_qualitative_reports": missing_qualitative,
        "scorecard_created": scorecard_created,
        "reasons": reasons,
    }


def _manifest_descriptor(
    dataset_manifest: Mapping[str, Any],
    filename: str,
) -> Optional[Mapping[str, Any]]:
    outputs = dataset_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return None
    descriptor = outputs.get(filename)
    return descriptor if isinstance(descriptor, Mapping) else None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def evaluate_state_material_gate(
    *,
    dataset_dir: Path,
    calibration_manifest_path: Optional[Path] = None,
    activation_ceiling: Any = ACTIVATION_CEILING,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    stateful_path = dataset_dir / "stateful_turns.jsonl"
    dataset_manifest_path = dataset_dir / "manifest.json"
    reasons = []
    invalid_rows = []
    valid_rows = 0
    usable_by_split = {"train": 0, "validation": 0}
    sessions_by_split = {"train": set(), "validation": set()}
    state_hashes = set()

    if not dataset_manifest_path.is_file():
        reasons.append("dataset manifest is missing")
        dataset_manifest: dict[str, Any] = {}
    else:
        dataset_manifest = _read_json(dataset_manifest_path)

    descriptor = _manifest_descriptor(
        dataset_manifest,
        "stateful_turns.jsonl",
    )
    descriptor_valid = False
    stateful_sha256 = None
    if not stateful_path.is_file():
        reasons.append("stateful_turns.jsonl is missing")
    else:
        stateful_sha256 = _sha256(stateful_path)
        if descriptor is None:
            reasons.append("dataset manifest does not describe stateful turns")
        else:
            descriptor_valid = (
                descriptor.get("bytes") == stateful_path.stat().st_size
                and descriptor.get("sha256") == stateful_sha256
            )
            if not descriptor_valid:
                reasons.append("stateful turns do not match the dataset manifest")

        try:
            with stateful_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("row is not an object")
                        if row.get("alignment_quality") != "exact_turn_capture":
                            raise ValueError("alignment is not exact_turn_capture")
                        state_record = row.get("aurora_state")
                        if not isinstance(state_record, Mapping):
                            raise ValueError("Aurora state is missing")
                        state = AuroraState171.from_record(state_record)
                        if state.source != "thalamus":
                            raise ValueError("state source is not thalamus")
                        if row.get("timestamp") != state.captured_at:
                            raise ValueError(
                                "turn/state capture timestamps do not match"
                            )
                        split = row.get("split")
                        if split not in usable_by_split:
                            raise ValueError("split is not train or validation")
                        session_id = row.get("session_id")
                        if not isinstance(session_id, str) or not session_id:
                            raise ValueError("session_id is missing")
                        valid_rows += 1
                        state_hashes.add(
                            hashlib.sha256(
                                json.dumps(
                                    state.values,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                        )
                        if (
                            row.get("expressed") is True
                            and isinstance(row.get("output_text"), str)
                            and row["output_text"].strip()
                            and isinstance(row.get("input_text"), str)
                            and row["input_text"].strip()
                        ):
                            usable_by_split[split] += 1
                            sessions_by_split[split].add(session_id)
                    except (TypeError, ValueError) as exc:
                        invalid_rows.append(
                            {
                                "line": line_number,
                                "error": str(exc),
                            }
                        )
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"stateful turns could not be parsed: {exc}")

    ceiling_valid = (
        isinstance(activation_ceiling, (int, float))
        and not isinstance(activation_ceiling, bool)
        and math.isfinite(float(activation_ceiling))
        and float(activation_ceiling) > 0.0
    )
    if not ceiling_valid:
        reasons.append("ACTIVATION_CEILING is not a positive frozen value")

    calibration_path = (
        Path(calibration_manifest_path)
        if calibration_manifest_path is not None
        else DEFAULT_CALIBRATION_MANIFEST
    )
    calibration_valid = False
    calibration: Optional[dict[str, Any]] = None
    if not calibration_path.is_file():
        reasons.append("activation calibration manifest is missing")
    else:
        calibration = _read_json(calibration_path)
        statistics = calibration.get("statistics")
        calibration_valid = (
            ceiling_valid
            and calibration.get("format_version")
            == CALIBRATION_FORMAT_VERSION
            and calibration.get("schema_version") == SCHEMA_VERSION
            and calibration.get("schema_fingerprint") == SCHEMA_FINGERPRINT
            and calibration.get("activation_ceiling")
            == float(activation_ceiling)
            and calibration.get("method") == CALIBRATION_METHOD
            and isinstance(calibration.get("frozen_at"), str)
            and bool(calibration["frozen_at"].strip())
            and isinstance(calibration.get("record_count"), int)
            and not isinstance(calibration.get("record_count"), bool)
            and calibration["record_count"] > 0
            and isinstance(calibration.get("active_score_count"), int)
            and not isinstance(calibration.get("active_score_count"), bool)
            and calibration["active_score_count"] > 0
            and isinstance(calibration.get("source_log_count"), int)
            and not isinstance(calibration.get("source_log_count"), bool)
            and calibration["source_log_count"] > 0
            and _is_sha256(calibration.get("source_digest"))
            and isinstance(statistics, Mapping)
            and statistics.get("maximum") == float(activation_ceiling)
        )
        if not calibration_valid:
            reasons.append(
                "activation calibration manifest is invalid or mismatched"
            )

    if valid_rows == 0:
        reasons.append("no genuine state-aligned turns exist")
    if invalid_rows:
        reasons.append("one or more state-aligned rows are invalid")
    if usable_by_split["train"] == 0:
        reasons.append("no usable state-aligned training text exists")
    if usable_by_split["validation"] == 0:
        reasons.append("no usable state-aligned validation text exists")
    if len(state_hashes) < 2:
        reasons.append("fewer than two distinct state vectors exist to shuffle")

    if ceiling_valid and calibration_valid and stateful_path.is_file():
        try:
            with stateful_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    AuroraState171.from_record(
                        row["aurora_state"]
                    ).normalised_values(
                        activation_ceiling=float(activation_ceiling),
                        clip=False,
                    )
        except (KeyError, TypeError, ValueError) as exc:
            calibration_valid = False
            reasons.append(
                f"recorded state exceeds or violates calibration: {exc}"
            )

    status = "PASS" if not reasons else "BLOCKED"
    return {
        "gate": "state_material_readiness",
        "status": status,
        "stateful_turns": {
            "path": str(stateful_path),
            "manifest_descriptor_valid": descriptor_valid,
            "sha256": stateful_sha256,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "usable_text_pairs": usable_by_split,
            "sessions": {
                split: len(sessions)
                for split, sessions in sessions_by_split.items()
            },
            "distinct_state_vectors": len(state_hashes),
        },
        "calibration": {
            "config_activation_ceiling": activation_ceiling,
            "config_value_valid": ceiling_valid,
            "manifest_path": str(calibration_path),
            "manifest_valid": calibration_valid,
            "manifest": calibration,
        },
        "reasons": list(dict.fromkeys(reasons)),
    }


def run_readiness_audit(
    *,
    candidate_run: Path,
    baseline_run: Path,
    dataset_dir: Path,
    scorecard_path: Optional[Path] = None,
    scorecard_output: Optional[Path] = None,
    calibration_manifest_path: Optional[Path] = None,
    activation_ceiling: Any = ACTIVATION_CEILING,
) -> dict[str, Any]:
    base_gate = evaluate_base_gate(
        candidate_run=candidate_run,
        baseline_run=baseline_run,
        scorecard_path=scorecard_path,
        scorecard_output=scorecard_output,
    )
    state_gate = evaluate_state_material_gate(
        dataset_dir=dataset_dir,
        calibration_manifest_path=calibration_manifest_path,
        activation_ceiling=activation_ceiling,
    )
    authorised = (
        base_gate["status"] == "PASS"
        and state_gate["status"] == "PASS"
    )
    return {
        "format_version": AUDIT_FORMAT_VERSION,
        "test": "pre_state_conditioning_two_gate_readiness",
        "gate_1": base_gate,
        "gate_2": state_gate,
        "decision": {
            "state_ablation_authorised": authorised,
            "state_conditioned_training_authorised": authorised,
            "dense_base_pretraining_only": not authorised,
            "status": "PASS" if authorised else "BLOCKED",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit base-model and state-material readiness without running "
            "state-conditioned training."
        )
    )
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path)
    parser.add_argument("--scorecard-output", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_readiness_audit(
        candidate_run=args.candidate_run,
        baseline_run=args.baseline_run,
        dataset_dir=args.dataset_dir,
        scorecard_path=args.scorecard,
        scorecard_output=args.scorecard_output,
        calibration_manifest_path=args.calibration_manifest,
    )
    _write_json(args.output, report)
    print(
        f"Gate 1: {report['gate_1']['status']} | "
        f"Gate 2: {report['gate_2']['status']} | "
        "state ablation authorised: "
        f"{report['decision']['state_ablation_authorised']}"
    )
    print(f"Report: {args.output}")
    return 0 if report["decision"]["state_ablation_authorised"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
