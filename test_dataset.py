import json
import tempfile
import unittest
from pathlib import Path

from config import REGULATOR_NAMES
from dataset import DatasetError, prepare_dataset, write_dataset
from state_schema import (
    EMOTION_COUNT,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def valid_state_record():
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "captured_at": "2026-07-27T12:00:00+00:00",
        "source": "thalamus",
        "values": [0.0] * EMOTION_COUNT,
    }


def valid_regulators():
    values = {
        name: 0.5
        for name in REGULATOR_NAMES
    }
    values["dominant_valence"] = -0.25
    return values


class DatasetFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.journal = self.root / "journal.json"
        self.memories = self.root / "memories.json"
        self.sessions = self.root / "sessions"
        self.sessions.mkdir()

        write_json(
            self.journal,
            [
                {
                    "timestamp": "2026-07-01T10:00:00+00:00",
                    "trigger": "settling",
                    "entry": "  Aurora   notices a signal. ",
                    "tension_at_writing": 0.1,
                    "coherence_at_writing": 0.8,
                    "bias_at_writing": 0.0,
                    "pfc_state": "express",
                    "raw_fragments": ["must never enter output"],
                    "context": {
                        "last_input": "What did you notice?",
                        "last_response": "A quiet signal.",
                        "private_metadata": "excluded",
                    },
                },
                {
                    "timestamp": "2026-07-01T10:01:00+00:00",
                    "trigger": "settled",
                    "entry": "aurora notices a signal.",
                    "tension_at_writing": 0.0,
                    "bias_at_writing": 0.0,
                    "context": {},
                },
            ],
        )
        write_json(
            self.memories,
            {
                "prediction_bias": 0.0,
                "memories": [
                    {
                        "summary": "A remembered garden.",
                        "memory_type": "self",
                        "weight": 0.8,
                        "embedding": [0.1, 0.2, 0.3],
                        "metadata": {
                            "encoding_emotion": "calm",
                            "secret": "excluded",
                        },
                    }
                ],
            },
        )

        session = {
            "format_version": 1,
            "session_id": "session-1",
            "timestamp": "2026-07-27T12:00:00+00:00",
            "turn": 1,
            "source": "omi",
            "input_text": "Are you there?",
            "output_text": "I am listening.",
            "expressed": True,
            "aurora_state": valid_state_record(),
            "regulators": valid_regulators(),
            "events": {
                "dominant_emotion": "calm",
                "unapproved": "excluded",
            },
        }
        (self.sessions / "aurora_training_session-1.jsonl").write_text(
            json.dumps(session) + "\n",
            encoding="utf-8",
        )

    def prepare(self, **overrides):
        options = {
            "journal_path": self.journal,
            "memories_path": self.memories,
            "sessions_dir": self.sessions,
            "seed": "test-seed",
            "validation_fraction": 0.25,
            "min_chars": 3,
        }
        options.update(overrides)
        return prepare_dataset(**options)


class TestDatasetBuilder(unittest.TestCase):
    def test_allowlist_excludes_embeddings_fragments_and_private_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DatasetFixture(temporary)
            prepared = fixture.prepare()
            serialised = json.dumps(prepared, sort_keys=True)

            self.assertNotIn("embedding", serialised)
            self.assertNotIn("must never enter output", serialised)
            self.assertNotIn("private_metadata", serialised)
            self.assertNotIn('"secret"', serialised)
            self.assertNotIn('"unapproved"', serialised)

    def test_documents_are_deduplicated_after_text_normalisation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DatasetFixture(temporary)
            prepared = fixture.prepare()

            self.assertEqual(len(prepared["documents"]), 2)
            self.assertEqual(
                prepared["summary"]["documents"]["duplicates_removed"],
                1,
            )

    def test_legacy_and_stateful_turns_are_kept_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DatasetFixture(temporary)
            prepared = fixture.prepare()

            self.assertEqual(len(prepared["legacy_turns"]), 1)
            self.assertEqual(
                prepared["legacy_turns"][0]["alignment_quality"],
                "legacy_unverified",
            )
            self.assertIsNone(prepared["legacy_turns"][0]["aurora_state"])

            self.assertEqual(len(prepared["stateful_turns"]), 1)
            self.assertEqual(
                prepared["stateful_turns"][0]["alignment_quality"],
                "exact_turn_capture",
            )
            self.assertEqual(
                len(prepared["stateful_turns"][0]["aurora_state"]["values"]),
                EMOTION_COUNT,
            )
            self.assertEqual(
                tuple(prepared["stateful_turns"][0]["regulators"]),
                REGULATOR_NAMES,
            )

    def test_splits_and_data_outputs_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DatasetFixture(temporary)
            first = fixture.prepare()
            second = fixture.prepare()

            for key in ("documents", "legacy_turns", "stateful_turns", "tokenizer_texts"):
                self.assertEqual(first[key], second[key])

            output_one = Path(temporary) / "out-one"
            output_two = Path(temporary) / "out-two"
            write_dataset(first, output_one)
            write_dataset(second, output_two)
            for name in (
                "documents.jsonl",
                "legacy_turns.jsonl",
                "stateful_turns.jsonl",
                "tokenizer_corpus.txt",
            ):
                self.assertEqual(
                    (output_one / name).read_bytes(),
                    (output_two / name).read_bytes(),
                )

    def test_invalid_stateful_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = DatasetFixture(temporary)
            session_path = fixture.sessions / "aurora_training_session-1.jsonl"
            record = json.loads(session_path.read_text(encoding="utf-8"))
            record["aurora_state"]["values"] = [0.0] * 170
            session_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, "invalid Aurora state"):
                fixture.prepare()

            record["aurora_state"] = valid_state_record()
            record["regulators"]["confidence"] = 1.1
            session_path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "regulator 'confidence'"):
                fixture.prepare()


if __name__ == "__main__":
    unittest.main()
