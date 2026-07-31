import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from config import AuroraLMConfig
from tokenizer import AuroraTokenizer
from train import (
    ContractError,
    DeterministicBatchSampler,
    TrainingSettings,
    build_packed_dataset,
    format_progress,
    learning_rate_for_step,
    load_training_contracts,
    run_training,
)


def _descriptor(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tiny_config() -> AuroraLMConfig:
    return AuroraLMConfig(
        name="test_tiny",
        vocab_size=280,
        context_length=8,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
        state_size=0,
        state_hidden_dim=8,
        num_experts=0,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=0,
        max_steps=2,
        batch_size=1,
        gradient_clip=1.0,
        checkpoint_every=1,
    )


def _tiny_settings(**overrides) -> TrainingSettings:
    settings = TrainingSettings(
        max_steps=2,
        batch_size=1,
        gradient_accumulation=1,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=0,
        weight_decay=0.01,
        gradient_clip=1.0,
        seed=123,
        amp="none",
        deterministic=True,
        max_train_documents=None,
        max_validation_documents=None,
    )
    return replace(settings, **overrides)


def _write_training_fixture(root: Path) -> tuple[Path, Path, AuroraLMConfig]:
    dataset_dir = root / "dataset"
    tokenizer_dir = root / "tokenizer"
    dataset_dir.mkdir(parents=True)

    tokenizer_texts = [
        "Aurora observes tension, coherence, salience, and novelty.",
        "The garden is quiet while the system continues listening.",
        "Unicode remains whole: café, naïve, em dash —, and stars ✨.",
        "A causal language model predicts the next token only.",
    ] * 20
    tokenizer = AuroraTokenizer.train(
        tokenizer_texts,
        vocab_size=280,
        min_frequency=1,
        show_progress=False,
        length=len(tokenizer_texts),
    )
    tokenizer.save(tokenizer_dir)
    _write_json(
        tokenizer_dir / "training_manifest.json",
        {
            "artifact": {
                "fingerprint": tokenizer.fingerprint,
                "tokenizer_json": _descriptor(
                    tokenizer_dir / "tokenizer.json"
                ),
            }
        },
    )

    records = []
    for split, count in (("train", 5), ("validation", 3)):
        for index in range(count):
            text = (
                f"{split} document {index}. "
                "Aurora follows a deterministic thread of thought. "
            ) * 5
            records.append(
                {
                    "format_version": 1,
                    "record_id": f"{split}-{index}",
                    "split": split,
                    "text": text,
                }
            )
    documents_path = dataset_dir / "documents.jsonl"
    documents_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    _write_json(
        dataset_dir / "manifest.json",
        {
            "outputs": {
                "documents.jsonl": _descriptor(documents_path),
            }
        },
    )
    return dataset_dir, tokenizer_dir, _tiny_config()


class _NumericTokenizer:
    def encode(self, text, *, add_special_tokens=True):
        content = [10 + int(part) for part in text.split()]
        return [1, *content, 2] if add_special_tokens else content


class TestTrainingPipeline(unittest.TestCase):
    def test_progress_indicator_contains_percent_throughput_and_eta(self):
        rendered = format_progress(
            completed_steps=50,
            max_steps=100,
            train_loss=4.25,
            learning_rate=1e-4,
            tokens_per_second=700.0,
            eta_seconds=125.0,
            width=10,
        )
        self.assertIn("[#####-----]", rendered)
        self.assertIn("50.00%", rendered)
        self.assertIn("700 tok/s", rendered)
        self.assertIn("ETA 2m05s", rendered)

    def test_learning_rate_warmup_and_cosine_endpoints(self):
        values = [
            learning_rate_for_step(
                step,
                max_steps=6,
                warmup_steps=2,
                max_lr=1.0,
                min_lr=0.1,
            )
            for step in range(6)
        ]
        self.assertEqual(values[0], 0.5)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[2], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)

    def test_packing_preserves_next_token_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "documents.jsonl"
            rows = [
                {"split": "train", "text": "0 1 2 3"},
                {"split": "validation", "text": "9 9 9 9"},
                {"split": "train", "text": "4 5 6 7"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dataset = build_packed_dataset(
                path,
                _NumericTokenizer(),
                "train",
                context_length=4,
            )

            self.assertEqual(dataset.document_count, 2)
            self.assertEqual(dataset.token_count, 12)
            self.assertEqual(len(dataset), 2)
            first_inputs, first_labels = dataset[0]
            second_inputs, _ = dataset[1]
            self.assertEqual(first_labels[-1].item(), second_inputs[0].item())
            self.assertNotIn(19, first_inputs.tolist() + first_labels.tolist())

    def test_batch_sampler_state_resumes_exactly(self):
        sampler = DeterministicBatchSampler(11, 3, seed=77)
        sampler.next_indices()
        state = sampler.state_dict()
        expected = [sampler.next_indices(), sampler.next_indices()]

        resumed = DeterministicBatchSampler(11, 3, seed=999)
        resumed.load_state_dict(state)
        actual = [resumed.next_indices(), resumed.next_indices()]

        for expected_indices, actual_indices in zip(expected, actual):
            self.assertTrue(torch.equal(expected_indices, actual_indices))

    def test_documents_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir, tokenizer_dir, model_config = (
                _write_training_fixture(root)
            )
            with (dataset_dir / "documents.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write("{}\n")

            with self.assertRaisesRegex(
                ContractError,
                "does not match its manifest",
            ):
                load_training_contracts(
                    dataset_dir,
                    tokenizer_dir,
                    model_config,
                )

    def test_interrupted_training_resumes_byte_identically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir, tokenizer_dir, model_config = (
                _write_training_fixture(root)
            )
            settings = _tiny_settings()
            device = torch.device("cpu")

            uninterrupted_dir = root / "uninterrupted"
            uninterrupted = run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=uninterrupted_dir,
                settings=settings,
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
            )

            resumed_dir = root / "resumed"
            interrupted = run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=resumed_dir,
                settings=settings,
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                stop_after_step=1,
            )
            resumed = run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=resumed_dir,
                settings=settings,
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                resume="latest",
            )

            self.assertEqual(uninterrupted["completed_steps"], 2)
            self.assertEqual(interrupted["completed_steps"], 1)
            self.assertEqual(resumed["completed_steps"], 2)

            full_checkpoint = torch.load(
                uninterrupted_dir / "step_00000002.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed_checkpoint = torch.load(
                resumed_dir / "step_00000002.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, expected in full_checkpoint["model_state"].items():
                actual = resumed_checkpoint["model_state"][name]
                self.assertTrue(torch.equal(expected, actual), name)
            self.assertEqual(
                full_checkpoint["last_metrics"]["train_loss"],
                resumed_checkpoint["last_metrics"]["train_loss"],
            )
            self.assertEqual(
                full_checkpoint["last_metrics"]["validation_loss"],
                resumed_checkpoint["last_metrics"]["validation_loss"],
            )

    def test_new_phase_initialises_from_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir, tokenizer_dir, model_config = (
                _write_training_fixture(root)
            )
            device = torch.device("cpu")
            source_dir = root / "source"
            run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=source_dir,
                settings=_tiny_settings(max_steps=1),
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                progress_every=1,
            )

            phase_two_dir = root / "phase_two"
            result = run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=phase_two_dir,
                settings=_tiny_settings(
                    max_steps=1,
                    seed=456,
                    max_lr=5e-4,
                    min_lr=5e-5,
                ),
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                progress_every=1,
                init_from=source_dir,
            )

            self.assertEqual(
                result["initialization"]["source_completed_steps"],
                1,
            )
            checkpoint = torch.load(
                phase_two_dir / "step_00000001.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                checkpoint["contract"]["initialization"]["kind"],
                "checkpoint",
            )
            self.assertEqual(
                checkpoint["contract"]["initialization"][
                    "source_completed_steps"
                ],
                1,
            )

    def test_checkpoint_initialised_phase_resumes_its_original_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir, tokenizer_dir, model_config = (
                _write_training_fixture(root)
            )
            device = torch.device("cpu")
            source_dir = root / "source"
            run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=source_dir,
                settings=_tiny_settings(max_steps=1),
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                progress_every=1,
            )

            phase_two_dir = root / "phase_two"
            phase_two_settings = _tiny_settings(
                max_steps=2,
                seed=456,
                max_lr=5e-4,
                min_lr=5e-5,
            )
            interrupted = run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=phase_two_dir,
                settings=phase_two_settings,
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                progress_every=1,
                init_from=source_dir,
                stop_after_step=1,
            )
            resumed = run_training(
                model_config=model_config,
                dataset_dir=dataset_dir,
                tokenizer_dir=tokenizer_dir,
                run_dir=phase_two_dir,
                settings=phase_two_settings,
                device=device,
                eval_every=1,
                eval_batches=1,
                checkpoint_every=1,
                progress_every=1,
                resume="latest",
            )

            self.assertEqual(interrupted["completed_steps"], 1)
            self.assertEqual(resumed["completed_steps"], 2)
            self.assertEqual(
                resumed["initialization"]["kind"],
                "checkpoint",
            )
            self.assertEqual(
                resumed["initialization"]["source_completed_steps"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
