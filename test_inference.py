import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from config import AuroraLMConfig
from inference import (
    GenerationSettings,
    InferenceContractError,
    _current_runtime_contract,
    audit_exact_shingles,
    generate,
    load_inference_session,
    validate_generation_settings,
)
from NN import AuroraLM
from state_schema import SCHEMA_FINGERPRINT, SCHEMA_VERSION
from tokenizer import BOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID, AuroraTokenizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_config() -> AuroraLMConfig:
    return AuroraLMConfig(
        name="inference_test_tiny",
        vocab_size=280,
        context_length=8,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
        state_size=0,
        state_hidden_dim=0,
        num_experts=0,
    )


def _write_fixture(root: Path) -> tuple[Path, Path]:
    tokenizer_dir = root / "tokenizer"
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    training_texts = [
        "Aurora notices rain, uncertainty, and the quiet garden.",
        "A model continues text one token at a time.",
        "Unicode remains whole: café, naïve, stars ✨, and an em dash —.",
    ] * 30
    tokenizer = AuroraTokenizer.train(
        training_texts,
        vocab_size=280,
        min_frequency=1,
        show_progress=False,
        length=len(training_texts),
    )
    tokenizer.save(tokenizer_dir)

    config = _tiny_config()
    torch.manual_seed(91)
    model = AuroraLM(config)
    checkpoint = {
        "checkpoint_format_version": 1,
        "completed_steps": 7,
        "contract": {
            "runtime_contract": _current_runtime_contract(),
            "model_config": config.__dict__,
            "contracts": {
                "training_mode": "dense_neutral_state",
                "tokenizer_fingerprint": tokenizer.fingerprint,
                "state_schema_version": SCHEMA_VERSION,
                "state_schema_fingerprint": SCHEMA_FINGERPRINT,
            },
        },
        "model_state": model.state_dict(),
    }
    checkpoint_path = run_dir / "step_00000007.pt"
    torch.save(checkpoint, checkpoint_path)
    (run_dir / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_path.name,
                "completed_steps": 7,
                "sha256": _sha256(checkpoint_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, tokenizer_dir


class TestInference(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir, self.tokenizer_dir = _write_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def _load(self):
        return load_inference_session(
            self.run_dir,
            self.tokenizer_dir,
            device=torch.device("cpu"),
        )

    def test_strict_loader_reconstructs_checkpoint(self):
        session = self._load()
        self.assertEqual(session.completed_steps, 7)
        self.assertEqual(session.config.vocab_size, 280)
        self.assertEqual(session.config.context_length, 8)
        self.assertEqual(
            session.checkpoint_sha256,
            _sha256(session.checkpoint_path),
        )

    def test_seeded_generation_is_exactly_repeatable(self):
        session = self._load()
        settings = GenerationSettings(
            max_new_tokens=6,
            min_new_tokens=6,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            seed=1234,
        )
        first = generate(session, "Aurora", settings)
        second = generate(session, "Aurora", settings)

        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.generated_tokens, 6)
        self.assertFalse(
            {PAD_TOKEN_ID, BOS_TOKEN_ID, UNK_TOKEN_ID}.intersection(
                first.token_ids
            )
        )

    def test_long_prompt_is_safely_truncated_to_context(self):
        session = self._load()
        result = generate(
            session,
            "a very long prompt " * 100,
            GenerationSettings(
                max_new_tokens=2,
                min_new_tokens=2,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                seed=1,
            ),
        )
        self.assertEqual(result.prompt_tokens, session.config.context_length)
        self.assertEqual(result.generated_tokens, 2)

    def test_latest_checkpoint_hash_tampering_is_rejected(self):
        checkpoint = self.run_dir / "step_00000007.pt"
        with checkpoint.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(
            InferenceContractError,
            "hash does not match",
        ):
            self._load()

    def test_runtime_drift_is_rejected(self):
        checkpoint = self.run_dir / "step_00000007.pt"
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        payload["contract"]["runtime_contract"]["torch"] = "0.0.0"
        torch.save(payload, checkpoint)
        latest = json.loads(
            (self.run_dir / "latest.json").read_text(encoding="utf-8")
        )
        latest["sha256"] = _sha256(checkpoint)
        (self.run_dir / "latest.json").write_text(
            json.dumps(latest) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "runtime contract",
        ):
            self._load()

    def test_exact_shingle_audit_reports_matches_without_source_text(self):
        documents = self.root / "documents.jsonl"
        text = (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen"
        )
        documents.write_text(
            json.dumps({"text": text}) + "\n",
            encoding="utf-8",
        )
        result = audit_exact_shingles(
            [text],
            documents,
            shingle_words=16,
        )
        self.assertEqual(result["candidate_shingles"], 2)
        self.assertEqual(result["exact_matches"], 2)

    def test_invalid_generation_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_generation_settings(
                GenerationSettings(max_new_tokens=0)
            )


if __name__ == "__main__":
    unittest.main()
