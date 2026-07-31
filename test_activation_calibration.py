import json
import tempfile
import unittest
from pathlib import Path

from activation_calibration import (
    CALIBRATION_METHOD,
    CalibrationError,
    calibrate_session_logs,
    write_calibration,
)
from state_schema import (
    EMOTION_COUNT,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
)


def _record(timestamp, values):
    return {
        "timestamp": timestamp,
        "input_text": "private text is not copied",
        "aurora_state": {
            "schema_version": SCHEMA_VERSION,
            "schema_fingerprint": SCHEMA_FINGERPRINT,
            "captured_at": timestamp,
            "source": "thalamus",
            "values": values,
        },
    }


class ActivationCalibrationTests(unittest.TestCase):
    def test_calibration_uses_observed_maximum_and_redacts_turn_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "session_a.jsonl"
            second = root / "session_b.jsonl"
            values_a = [0.0] * EMOTION_COUNT
            values_b = [0.0] * EMOTION_COUNT
            values_a[0] = 0.5
            values_b[1] = 0.75
            first.write_text(
                json.dumps(_record("2026-07-31T00:00:00+00:00", values_a))
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(_record("2026-07-31T00:01:00+00:00", values_b))
                + "\n",
                encoding="utf-8",
            )

            result = calibrate_session_logs(
                [second, first],
                frozen_at="2026-07-31T01:00:00+00:00",
            )

            self.assertEqual(result["activation_ceiling"], 0.75)
            self.assertEqual(result["method"], CALIBRATION_METHOD)
            self.assertEqual(result["record_count"], 2)
            self.assertEqual(result["active_score_count"], 2)
            self.assertEqual(result["source_log_count"], 2)
            self.assertNotIn("private text", json.dumps(result))

            output = root / "receipt.json"
            write_calibration(output, result)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_invalid_state_record_is_rejected_with_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.jsonl"
            values = [0.0] * EMOTION_COUNT
            record = _record("2026-07-31T00:00:00+00:00", values)
            record["aurora_state"]["schema_fingerprint"] = "wrong"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CalibrationError,
                r"broken\.jsonl:1",
            ):
                calibrate_session_logs([path])

    def test_all_zero_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "silent.jsonl"
            path.write_text(
                json.dumps(
                    _record(
                        "2026-07-31T00:00:00+00:00",
                        [0.0] * EMOTION_COUNT,
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                CalibrationError,
                "no positive thalamic activations",
            ):
                calibrate_session_logs([path])


if __name__ == "__main__":
    unittest.main()
