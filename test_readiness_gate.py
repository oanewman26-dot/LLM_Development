import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from activation_calibration import (
    CALIBRATION_FORMAT_VERSION,
    CALIBRATION_METHOD,
)
from readiness_gate import (
    evaluate_base_gate,
    evaluate_state_material_gate,
)
from state_schema import (
    EMOTION_COUNT,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
)


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root, *, completed, maximum, validation_losses):
    root.mkdir()
    _write_json(
        root / "run_config.json",
        {"settings": {"max_steps": maximum}},
    )
    if completed:
        _write_json(
            root / "latest.json",
            {
                "checkpoint": f"step_{completed:08d}.pt",
                "completed_steps": completed,
                "sha256": "unused-in-readiness-test",
            },
        )
    with (root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for index, loss in enumerate(validation_losses, start=1):
            handle.write(
                json.dumps(
                    {
                        "step": index * 100,
                        "validation_loss": loss,
                        "validation_perplexity": 100.0,
                    }
                )
                + "\n"
            )


def _state_record(value, captured_at):
    values = [0.0] * EMOTION_COUNT
    values[0] = value
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "captured_at": captured_at,
        "source": "thalamus",
        "values": values,
    }


class TestReadinessGate(unittest.TestCase):
    def test_incomplete_base_run_is_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            baseline = root / "baseline"
            _write_run(
                candidate,
                completed=400,
                maximum=500,
                validation_losses=[5.2, 5.1, 5.0, 4.9],
            )
            _write_run(
                baseline,
                completed=500,
                maximum=500,
                validation_losses=[5.5, 5.4, 5.3, 5.2, 5.1],
            )

            result = evaluate_base_gate(
                candidate_run=candidate,
                baseline_run=baseline,
            )

            self.assertEqual(result["status"], "PENDING")
            self.assertFalse(result["candidate_complete"])
            self.assertIn("incomplete", result["reasons"][0])

    def test_empty_state_data_and_missing_ceiling_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            stateful = dataset / "stateful_turns.jsonl"
            stateful.write_text("", encoding="utf-8")
            _write_json(
                dataset / "manifest.json",
                {
                    "outputs": {
                        "stateful_turns.jsonl": {
                            "bytes": 0,
                            "sha256": _sha256(stateful),
                        }
                    }
                },
            )

            result = evaluate_state_material_gate(
                dataset_dir=dataset,
                activation_ceiling=None,
            )

            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(
                result["stateful_turns"]["valid_rows"],
                0,
            )
            self.assertFalse(
                result["calibration"]["config_value_valid"]
            )

    def test_exact_state_data_and_frozen_calibration_can_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            rows = []
            for split, session, value in (
                ("train", "session-train", 0.5),
                ("validation", "session-validation", 0.75),
            ):
                timestamp = f"2026-07-28T0{len(rows)}:00:00+00:00"
                rows.append(
                    {
                        "split": split,
                        "session_id": session,
                        "timestamp": timestamp,
                        "input_text": "What do you notice?",
                        "output_text": "I notice a change.",
                        "expressed": True,
                        "alignment_quality": "exact_turn_capture",
                        "aurora_state": _state_record(value, timestamp),
                    }
                )
            stateful = dataset / "stateful_turns.jsonl"
            stateful.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            stateful_hash = _sha256(stateful)
            _write_json(
                dataset / "manifest.json",
                {
                    "outputs": {
                        "stateful_turns.jsonl": {
                            "bytes": stateful.stat().st_size,
                            "sha256": stateful_hash,
                        }
                    }
                },
            )
            calibration = dataset / "activation_calibration.json"
            _write_json(
                calibration,
                {
                    "format_version": CALIBRATION_FORMAT_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "schema_fingerprint": SCHEMA_FINGERPRINT,
                    "record_count": 2,
                    "active_score_count": 2,
                    "source_log_count": 1,
                    "source_digest": "a" * 64,
                    "activation_ceiling": 1.0,
                    "method": CALIBRATION_METHOD,
                    "frozen_at": "2026-07-28T02:00:00+00:00",
                    "statistics": {
                        "minimum_positive": 0.5,
                        "median": 0.625,
                        "p95": 0.7375,
                        "p99": 0.7475,
                        "maximum": 1.0,
                    },
                },
            )

            result = evaluate_state_material_gate(
                dataset_dir=dataset,
                calibration_manifest_path=calibration,
                activation_ceiling=1.0,
            )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                result["stateful_turns"]["distinct_state_vectors"],
                2,
            )
            self.assertTrue(result["calibration"]["manifest_valid"])


if __name__ == "__main__":
    unittest.main()
