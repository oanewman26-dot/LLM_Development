"""
Deterministic dense-language-model training for AuroraLM's PILOT stage.

This trainer intentionally uses only clean ``documents.jsonl`` records and
passes ``aurora_state=None``. Exact state-conditioned training remains a later
stage, after Aurora has captured enough state-aligned turns and the activation
ceiling has been calibrated.

The trainer binds every run and checkpoint to:

* the complete model configuration and state-schema fingerprint;
* the frozen tokenizer artifact fingerprint;
* the dataset manifest and documents-file hashes;
* the packed-data statistics and optimisation settings.

Example:

    python train.py \
        --run-dir runs/pilot_dense_v1 \
        --device auto
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from config import PILOT, AuroraLMConfig
from NN import AuroraLM
from tokenizer import AuroraTokenizer


TRAINER_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1
TRAINING_MODE = "dense_neutral_state"

DEFAULT_DATASET_DIR = Path("data/aurora_smollm_v1")
DEFAULT_TOKENIZER_DIR = Path("artifacts/aurora_bpe_8192_v1")
DOCUMENTS_FILENAME = "documents.jsonl"
DATASET_MANIFEST_FILENAME = "manifest.json"
TOKENIZER_TRAINING_MANIFEST_FILENAME = "training_manifest.json"
LATEST_CHECKPOINT_FILENAME = "latest.json"
RUN_CONFIG_FILENAME = "run_config.json"
METRICS_FILENAME = "metrics.jsonl"


class TrainingError(RuntimeError):
    """Base error for unsafe or invalid training requests."""


class ContractError(TrainingError):
    """Raised when a dataset, tokenizer, run, or checkpoint contract drifts."""


@dataclass(frozen=True)
class TrainingSettings:
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    seed: int
    amp: str
    deterministic: bool
    max_train_documents: Optional[int]
    max_validation_documents: Optional[int]


@dataclass(frozen=True)
class TrainingContracts:
    training_mode: str
    tokenizer_fingerprint: str
    tokenizer_training_manifest_sha256: str
    dataset_manifest_sha256: str
    documents_sha256: str
    state_schema_version: str
    state_schema_fingerprint: str


@dataclass(frozen=True)
class PackedDataStats:
    split: str
    documents: int
    tokens: int
    blocks: int
    context_length: int
    trailing_tokens_dropped: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read valid JSON from {path}.") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}.")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_settings(settings: TrainingSettings) -> None:
    if settings.max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if settings.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if settings.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1.")
    if settings.max_lr <= 0.0 or settings.min_lr < 0.0:
        raise ValueError("Learning rates must be non-negative and max_lr > 0.")
    if settings.min_lr > settings.max_lr:
        raise ValueError("min_lr cannot exceed max_lr.")
    if settings.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if settings.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative.")
    if settings.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive.")
    for name, value in (
        ("max_train_documents", settings.max_train_documents),
        ("max_validation_documents", settings.max_validation_documents),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1 when supplied.")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def resolve_amp(requested: str, device: torch.device) -> str:
    if requested == "auto":
        if device.type != "cuda":
            return "none"
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if requested == "float16" and device.type != "cuda":
        raise TrainingError("float16 AMP is supported only on CUDA by this trainer.")
    if requested == "bfloat16" and device.type not in {"cpu", "cuda"}:
        raise TrainingError(f"bfloat16 AMP is not supported on {device.type}.")
    return requested


def set_reproducible_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _validate_file_descriptor(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> str:
    if not path.is_file():
        raise ContractError(f"{label} is missing: {path}")
    expected_bytes = descriptor.get("bytes")
    expected_hash = descriptor.get("sha256")
    if not isinstance(expected_bytes, int) or not isinstance(expected_hash, str):
        raise ContractError(f"{label} manifest entry is incomplete.")
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_bytes != expected_bytes or actual_hash != expected_hash:
        raise ContractError(
            f"{label} does not match its manifest: expected "
            f"{expected_bytes} bytes/{expected_hash}, found "
            f"{actual_bytes} bytes/{actual_hash}."
        )
    return actual_hash


def load_training_contracts(
    dataset_dir: Path,
    tokenizer_dir: Path,
    model_config: AuroraLMConfig,
) -> tuple[AuroraTokenizer, TrainingContracts]:
    dataset_dir = Path(dataset_dir)
    tokenizer_dir = Path(tokenizer_dir)
    dataset_manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    documents_path = dataset_dir / DOCUMENTS_FILENAME
    tokenizer_manifest_path = (
        tokenizer_dir / TOKENIZER_TRAINING_MANIFEST_FILENAME
    )

    dataset_manifest = _read_json(dataset_manifest_path)
    outputs = dataset_manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(
        outputs.get(DOCUMENTS_FILENAME), dict
    ):
        raise ContractError(
            f"{dataset_manifest_path} does not describe {DOCUMENTS_FILENAME}."
        )
    documents_hash = _validate_file_descriptor(
        documents_path,
        outputs[DOCUMENTS_FILENAME],
        label="Documents file",
    )

    tokenizer = AuroraTokenizer.load(tokenizer_dir)
    tokenizer.validate_config(model_config)

    tokenizer_manifest = _read_json(tokenizer_manifest_path)
    artifact = tokenizer_manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ContractError(
            f"{tokenizer_manifest_path} does not contain artifact provenance."
        )
    if artifact.get("fingerprint") != tokenizer.fingerprint:
        raise ContractError(
            "Tokenizer fingerprint does not match its training manifest."
        )

    tokenizer_json_descriptor = artifact.get("tokenizer_json")
    if not isinstance(tokenizer_json_descriptor, dict):
        raise ContractError("Tokenizer training manifest lacks tokenizer_json.")
    _validate_file_descriptor(
        tokenizer_dir / "tokenizer.json",
        tokenizer_json_descriptor,
        label="Tokenizer JSON",
    )

    schema_version = getattr(model_config, "schema_version", None)
    schema_fingerprint = getattr(model_config, "schema_fingerprint", None)
    if not isinstance(schema_version, str) or not schema_version:
        raise ContractError("Model config lacks a state schema version.")
    if not isinstance(schema_fingerprint, str) or not schema_fingerprint:
        raise ContractError("Model config lacks a state schema fingerprint.")

    contracts = TrainingContracts(
        training_mode=TRAINING_MODE,
        tokenizer_fingerprint=tokenizer.fingerprint,
        tokenizer_training_manifest_sha256=sha256_file(
            tokenizer_manifest_path
        ),
        dataset_manifest_sha256=sha256_file(dataset_manifest_path),
        documents_sha256=documents_hash,
        state_schema_version=schema_version,
        state_schema_fingerprint=schema_fingerprint,
    )
    return tokenizer, contracts


def iter_document_texts(
    documents_path: Path,
    split: str,
    *,
    max_documents: Optional[int] = None,
) -> Iterator[str]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'.")

    count = 0
    with Path(documents_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"Invalid JSON in {documents_path}:{line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ContractError(
                    f"Non-object record in {documents_path}:{line_number}."
                )
            if record.get("split") != split:
                continue
            text = record.get("text")
            if not isinstance(text, str) or not text:
                raise ContractError(
                    f"Missing document text in {documents_path}:{line_number}."
                )
            count += 1
            yield text
            if max_documents is not None and count >= max_documents:
                return


class PackedTokenDataset(torch.utils.data.Dataset):
    """
    Fixed-length causal-LM blocks backed by compact 32-bit token storage.

    Adjacent blocks overlap by one token, so no next-token target is lost at a
    block boundary. Only the final incomplete tail of a split is dropped.
    """

    def __init__(
        self,
        token_ids: array,
        *,
        context_length: int,
        split: str,
        document_count: int,
    ):
        if context_length < 2:
            raise ValueError("context_length must be at least 2.")
        if token_ids.typecode != "I":
            raise TypeError("token_ids must use array('I') storage.")

        block_count = (len(token_ids) - 1) // context_length
        if block_count < 1:
            raise TrainingError(
                f"The {split} split does not contain enough tokens for one "
                f"{context_length}-token training block."
            )

        self._storage = token_ids
        self._tokens = torch.frombuffer(
            memoryview(self._storage),
            dtype=torch.int32,
        )
        self.context_length = context_length
        self.split = split
        self.document_count = document_count
        self.block_count = block_count
        self.trailing_tokens_dropped = (
            len(token_ids) - (block_count * context_length + 1)
        )

    @property
    def token_count(self) -> int:
        return len(self._storage)

    @property
    def stats(self) -> PackedDataStats:
        return PackedDataStats(
            split=self.split,
            documents=self.document_count,
            tokens=self.token_count,
            blocks=self.block_count,
            context_length=self.context_length,
            trailing_tokens_dropped=self.trailing_tokens_dropped,
        )

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.block_count
        if index < 0 or index >= self.block_count:
            raise IndexError(index)
        start = index * self.context_length
        chunk = self._tokens[start : start + self.context_length + 1]
        inputs = chunk[:-1].to(dtype=torch.long)
        labels = chunk[1:].to(dtype=torch.long)
        return inputs, labels


def build_packed_dataset(
    documents_path: Path,
    tokenizer: AuroraTokenizer,
    split: str,
    *,
    context_length: int,
    max_documents: Optional[int] = None,
) -> PackedTokenDataset:
    token_ids = array("I")
    document_count = 0
    for text in iter_document_texts(
        documents_path,
        split,
        max_documents=max_documents,
    ):
        encoded = tokenizer.encode(text, add_special_tokens=True)
        if any(token_id < 0 or token_id > 0xFFFFFFFF for token_id in encoded):
            raise TrainingError("Tokenizer emitted an ID outside uint32 range.")
        token_ids.extend(encoded)
        document_count += 1

    if document_count == 0:
        raise TrainingError(f"The dataset contains no {split} documents.")
    return PackedTokenDataset(
        token_ids,
        context_length=context_length,
        split=split,
        document_count=document_count,
    )


class DeterministicBatchSampler:
    """Epoch-style shuffled batches with checkpointable exact sampler state."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int):
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive.")
        if batch_size < 1 or batch_size > dataset_size:
            raise ValueError(
                "batch_size must be between 1 and the number of blocks."
            )
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.order = torch.randperm(
            self.dataset_size,
            generator=self.generator,
        )

    def next_indices(self) -> torch.Tensor:
        if self.cursor + self.batch_size > self.dataset_size:
            self.epoch += 1
            self.cursor = 0
            self.order = torch.randperm(
                self.dataset_size,
                generator=self.generator,
            )
        indices = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("dataset_size") != self.dataset_size:
            raise ContractError("Checkpoint sampler dataset size has drifted.")
        if state.get("batch_size") != self.batch_size:
            raise ContractError("Checkpoint sampler batch size has drifted.")
        order = state.get("order")
        generator_state = state.get("generator_state")
        cursor = state.get("cursor")
        epoch = state.get("epoch")
        if (
            not isinstance(order, torch.Tensor)
            or order.numel() != self.dataset_size
            or not isinstance(generator_state, torch.Tensor)
            or not isinstance(cursor, int)
            or not 0 <= cursor <= self.dataset_size
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ContractError("Checkpoint sampler state is invalid.")
        self.order = order.to(device="cpu", dtype=torch.long)
        self.cursor = cursor
        self.epoch = epoch
        self.generator.set_state(generator_state.to(device="cpu"))


def _stack_batch(
    dataset: PackedTokenDataset,
    indices: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [dataset[int(index)] for index in indices]
    inputs = torch.stack([row[0] for row in rows]).to(
        device=device,
        non_blocking=device.type == "cuda",
    )
    labels = torch.stack([row[1] for row in rows]).to(
        device=device,
        non_blocking=device.type == "cuda",
    )
    return inputs, labels


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T].")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label batch/sequence dimensions differ.")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )


def learning_rate_for_step(
    step_index: int,
    *,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step_index < 0 or step_index >= max_steps:
        raise ValueError("step_index must be inside the training schedule.")
    if warmup_steps > 0 and step_index < warmup_steps:
        return max_lr * float(step_index + 1) / float(warmup_steps)
    decay_steps = max_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr if step_index == max_steps - 1 else max_lr
    progress = (step_index - warmup_steps) / float(decay_steps - 1)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def _autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _create_grad_scaler(amp: str):
    enabled = amp == "float16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(
    model: AuroraLM,
    dataset: PackedTokenDataset,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    amp: str,
) -> float:
    if max_batches < 1:
        raise ValueError("max_batches must be at least 1.")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_index, start in enumerate(range(0, len(dataset), batch_size)):
        if batch_index >= max_batches:
            break
        end = min(start + batch_size, len(dataset))
        inputs, labels = _stack_batch(dataset, range(start, end), device)
        with _autocast_context(device, amp):
            logits = model(inputs, aurora_state=None)
            loss = causal_lm_loss(logits, labels)
        token_count = labels.numel()
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
    if was_training:
        model.train()
    if total_tokens == 0:
        raise TrainingError("Validation did not evaluate any tokens.")
    return total_loss / total_tokens


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    torch_cpu_state = state.get("torch_cpu")
    if python_state is None or not isinstance(torch_cpu_state, torch.Tensor):
        raise ContractError("Checkpoint RNG state is incomplete.")
    random.setstate(python_state)
    torch.set_rng_state(torch_cpu_state.to(device="cpu"))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _runtime_contract() -> dict[str, str]:
    return {
        "python_major_minor": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        "torch": torch.__version__.split("+", 1)[0],
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def _contract_payload(
    *,
    model_config: AuroraLMConfig,
    settings: TrainingSettings,
    contracts: TrainingContracts,
    train_stats: PackedDataStats,
    validation_stats: PackedDataStats,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "trainer_format_version": TRAINER_FORMAT_VERSION,
        "runtime_contract": _runtime_contract(),
        "model_config": asdict(model_config),
        "settings": asdict(settings),
        "contracts": asdict(contracts),
        "initialization": dict(initialization),
        "packed_data": {
            "train": asdict(train_stats),
            "validation": asdict(validation_stats),
        },
    }


def _require_exact_contract(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if dict(actual) != dict(expected):
        raise ContractError(
            f"{label} contract does not match the current training request."
        )


def save_checkpoint(
    *,
    run_dir: Path,
    completed_steps: int,
    model: AuroraLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    sampler: DeterministicBatchSampler,
    contract: Mapping[str, Any],
    last_metrics: Mapping[str, Any],
) -> Path:
    checkpoint_path = run_dir / f"step_{completed_steps:08d}.pt"
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "completed_steps": completed_steps,
        "contract": dict(contract),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "rng_state": _capture_rng_state(),
        "last_metrics": dict(last_metrics),
    }
    _atomic_torch_save(checkpoint_path, payload)
    latest = {
        "checkpoint": checkpoint_path.name,
        "completed_steps": completed_steps,
        "sha256": sha256_file(checkpoint_path),
    }
    _atomic_write_bytes(
        run_dir / LATEST_CHECKPOINT_FILENAME,
        _json_bytes(latest),
    )
    return checkpoint_path


def resolve_resume_path(run_dir: Path, resume: str) -> Path:
    if resume != "latest":
        return Path(resume)
    latest_path = run_dir / LATEST_CHECKPOINT_FILENAME
    latest = _read_json(latest_path)
    name = latest.get("checkpoint")
    expected_hash = latest.get("sha256")
    if not isinstance(name, str) or Path(name).name != name:
        raise ContractError(f"{latest_path} contains an invalid checkpoint name.")
    checkpoint_path = run_dir / name
    if not checkpoint_path.is_file():
        raise ContractError(f"Latest checkpoint is missing: {checkpoint_path}")
    if not isinstance(expected_hash, str) or sha256_file(
        checkpoint_path
    ) != expected_hash:
        raise ContractError("Latest checkpoint hash does not match latest.json.")
    return checkpoint_path


def resolve_initial_checkpoint(source: Path) -> tuple[Path, str]:
    source = Path(source)
    if source.is_dir():
        latest_path = source / LATEST_CHECKPOINT_FILENAME
        latest = _read_json(latest_path)
        name = latest.get("checkpoint")
        expected_hash = latest.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected_hash, str)
        ):
            raise ContractError(
                f"{latest_path} contains an invalid checkpoint pointer."
            )
        checkpoint_path = source / name
        if not checkpoint_path.is_file():
            raise ContractError(
                f"Initial checkpoint is missing: {checkpoint_path}"
            )
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != expected_hash:
            raise ContractError(
                "Initial checkpoint hash does not match latest.json."
            )
        return checkpoint_path, actual_hash
    if not source.is_file():
        raise ContractError(f"Initial checkpoint does not exist: {source}")
    return source, sha256_file(source)


def inspect_initial_checkpoint(
    source: Path,
    *,
    model_config: AuroraLMConfig,
    contracts: TrainingContracts,
    device: torch.device,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    checkpoint_path, checkpoint_hash = resolve_initial_checkpoint(source)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError(
            f"Could not load initial checkpoint {checkpoint_path}."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ContractError("Initial checkpoint payload is not a dictionary.")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ContractError("Unsupported initial checkpoint format version.")
    source_contract = checkpoint.get("contract")
    if not isinstance(source_contract, dict):
        raise ContractError("Initial checkpoint contract is missing.")
    if source_contract.get("model_config") != asdict(model_config):
        raise ContractError(
            "Initial checkpoint model configuration has drifted."
        )
    if source_contract.get("contracts") != asdict(contracts):
        raise ContractError(
            "Initial checkpoint data/tokenizer/schema contracts have drifted."
        )
    if source_contract.get("runtime_contract") != _runtime_contract():
        raise ContractError("Initial checkpoint runtime contract has drifted.")

    completed_steps = checkpoint.get("completed_steps")
    model_state = checkpoint.get("model_state")
    if (
        not isinstance(completed_steps, int)
        or completed_steps < 0
        or not isinstance(model_state, dict)
    ):
        raise ContractError("Initial checkpoint model state is incomplete.")
    descriptor = {
        "kind": "checkpoint",
        "source_file": checkpoint_path.name,
        "source_sha256": checkpoint_hash,
        "source_completed_steps": completed_steps,
    }
    return descriptor, model_state


def load_checkpoint(
    checkpoint_path: Path,
    *,
    model: AuroraLM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    sampler: DeterministicBatchSampler,
    expected_contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError(
            f"Could not load checkpoint {checkpoint_path}."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ContractError("Checkpoint payload is not a dictionary.")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ContractError("Unsupported checkpoint format version.")
    contract = checkpoint.get("contract")
    if not isinstance(contract, dict):
        raise ContractError("Checkpoint contract is missing.")
    _require_exact_contract(
        expected_contract,
        contract,
        label="Checkpoint",
    )
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        sampler.load_state_dict(checkpoint["sampler_state"])
        _restore_rng_state(checkpoint["rng_state"])
    except (KeyError, RuntimeError, ValueError, TypeError) as exc:
        raise ContractError("Checkpoint training state is incomplete.") from exc
    completed_steps = checkpoint.get("completed_steps")
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise ContractError("Checkpoint completed_steps is invalid.")
    last_metrics = checkpoint.get("last_metrics")
    return completed_steps, (
        dict(last_metrics) if isinstance(last_metrics, dict) else {}
    )


def _append_metric(path: Path, metric: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(
            (json.dumps(dict(metric), sort_keys=True) + "\n").encode("utf-8")
        )
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_run_directory(
    run_dir: Path,
    *,
    contract: Mapping[str, Any],
    resume: Optional[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / RUN_CONFIG_FILENAME
    if resume is None:
        existing = [path for path in run_dir.iterdir()]
        if existing:
            raise TrainingError(
                f"Run directory is not empty: {run_dir}. "
                "Choose a new directory or use --resume."
            )
        run_config = {
            **dict(contract),
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
        }
        _atomic_write_bytes(run_config_path, _json_bytes(run_config))
        return

    existing_config = _read_json(run_config_path)
    existing_contract = {
        key: existing_config.get(key)
        for key in contract
    }
    _require_exact_contract(
        contract,
        existing_contract,
        label="Run directory",
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def format_progress(
    *,
    completed_steps: int,
    max_steps: int,
    train_loss: float,
    learning_rate: float,
    tokens_per_second: float,
    eta_seconds: float,
    width: int = 24,
) -> str:
    fraction = completed_steps / max_steps
    filled = min(width, max(0, int(round(fraction * width))))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {fraction * 100:6.2f}% "
        f"{completed_steps}/{max_steps} "
        f"loss={train_loss:.4f} lr={learning_rate:.3e} "
        f"{tokens_per_second:.0f} tok/s "
        f"ETA {_format_duration(eta_seconds)}"
    )


def run_training(
    *,
    model_config: AuroraLMConfig,
    dataset_dir: Path,
    tokenizer_dir: Path,
    run_dir: Path,
    settings: TrainingSettings,
    device: torch.device,
    eval_every: int,
    eval_batches: int,
    checkpoint_every: int,
    progress_every: int = 10,
    resume: Optional[str] = None,
    init_from: Optional[Path] = None,
    stop_after_step: Optional[int] = None,
) -> dict[str, Any]:
    validate_settings(settings)
    if (
        eval_every < 1
        or eval_batches < 1
        or checkpoint_every < 1
        or progress_every < 1
    ):
        raise ValueError(
            "eval_every, eval_batches, checkpoint_every, and progress_every "
            "must be positive."
        )
    if resume is not None and init_from is not None:
        raise ValueError("--resume and --init-from are mutually exclusive.")
    if stop_after_step is not None and not 1 <= stop_after_step <= settings.max_steps:
        raise ValueError("stop_after_step must be between 1 and max_steps.")

    resolved_amp = resolve_amp(settings.amp, device)
    if resolved_amp != settings.amp:
        settings = TrainingSettings(
            **{**asdict(settings), "amp": resolved_amp}
        )
    set_reproducible_seed(settings.seed, settings.deterministic)

    tokenizer, contracts = load_training_contracts(
        Path(dataset_dir),
        Path(tokenizer_dir),
        model_config,
    )
    documents_path = Path(dataset_dir) / DOCUMENTS_FILENAME
    print("Packing deterministic train split...")
    train_data = build_packed_dataset(
        documents_path,
        tokenizer,
        "train",
        context_length=model_config.context_length,
        max_documents=settings.max_train_documents,
    )
    print("Packing deterministic validation split...")
    validation_data = build_packed_dataset(
        documents_path,
        tokenizer,
        "validation",
        context_length=model_config.context_length,
        max_documents=settings.max_validation_documents,
    )
    if settings.batch_size > len(train_data):
        raise TrainingError(
            f"batch_size={settings.batch_size} exceeds "
            f"{len(train_data)} packed train blocks."
        )

    run_dir = Path(run_dir)
    initial_model_state: Optional[Mapping[str, torch.Tensor]] = None
    if resume is not None:
        run_config_path = run_dir / RUN_CONFIG_FILENAME
        if not run_config_path.is_file():
            raise ContractError(
                f"Cannot resume without {run_config_path}."
            )
        stored_initialization = _read_json(run_config_path).get(
            "initialization"
        )
        if not isinstance(stored_initialization, dict):
            raise ContractError(
                "Run contract has no valid initialization record."
            )
        initialization = dict(stored_initialization)
    elif init_from is None:
        initialization = {
            "kind": "random_seed",
            "seed": settings.seed,
        }
    else:
        initialization, initial_model_state = inspect_initial_checkpoint(
            init_from,
            model_config=model_config,
            contracts=contracts,
            device=device,
        )

    contract = _contract_payload(
        model_config=model_config,
        settings=settings,
        contracts=contracts,
        train_stats=train_data.stats,
        validation_stats=validation_data.stats,
        initialization=initialization,
    )
    _prepare_run_directory(run_dir, contract=contract, resume=resume)

    model = AuroraLM(model_config).to(device)
    if initial_model_state is not None:
        try:
            model.load_state_dict(initial_model_state, strict=True)
        except RuntimeError as exc:
            raise ContractError(
                "Initial checkpoint weights do not match the model."
            ) from exc
        print(
            f"Initialised from {initialization['source_file']} at source "
            f"step {initialization['source_completed_steps']}."
        )
        del initial_model_state
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.max_lr,
        weight_decay=settings.weight_decay,
    )
    scaler = _create_grad_scaler(settings.amp)
    sampler = DeterministicBatchSampler(
        len(train_data),
        settings.batch_size,
        settings.seed + 1,
    )

    completed_steps = 0
    last_metrics: dict[str, Any] = {}
    if resume is not None:
        checkpoint_path = resolve_resume_path(run_dir, resume)
        completed_steps, last_metrics = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            expected_contract=contract,
            device=device,
        )
        if completed_steps >= settings.max_steps:
            raise TrainingError(
                f"Checkpoint already completed {completed_steps} steps; "
                f"max_steps is {settings.max_steps}."
            )
        print(f"Resumed from {checkpoint_path} at step {completed_steps}.")

    total_parameters, trainable_parameters = model.count_parameters()
    print(
        f"Device={device}; AMP={settings.amp}; "
        f"parameters={total_parameters:,}; trainable={trainable_parameters:,}"
    )
    print(
        f"Packed blocks: train={len(train_data):,}, "
        f"validation={len(validation_data):,}"
    )

    metrics_path = run_dir / METRICS_FILENAME
    model.train()
    progress_started = time.perf_counter()
    progress_origin = completed_steps
    while completed_steps < settings.max_steps:
        step_index = completed_steps
        learning_rate = learning_rate_for_step(
            step_index,
            max_steps=settings.max_steps,
            warmup_steps=settings.warmup_steps,
            max_lr=settings.max_lr,
            min_lr=settings.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_started = time.perf_counter()
        for _ in range(settings.gradient_accumulation):
            indices = sampler.next_indices()
            inputs, labels = _stack_batch(train_data, indices, device)
            with _autocast_context(device, settings.amp):
                logits = model(inputs, aurora_state=None)
                loss = causal_lm_loss(logits, labels)
                scaled_loss = loss / settings.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            step_loss += float(loss.detach().item())

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            settings.gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        completed_steps += 1

        elapsed = max(time.perf_counter() - step_started, 1e-9)
        tokens_processed = (
            settings.batch_size
            * model_config.context_length
            * settings.gradient_accumulation
        )
        last_metrics = {
            "step": completed_steps,
            "train_loss": step_loss / settings.gradient_accumulation,
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm),
            "tokens_per_second": tokens_processed / elapsed,
            "sampler_epoch": sampler.epoch,
        }

        should_report_progress = (
            completed_steps == progress_origin + 1
            or completed_steps % progress_every == 0
            or completed_steps == settings.max_steps
            or completed_steps == stop_after_step
        )
        if should_report_progress:
            progress_steps = max(1, completed_steps - progress_origin)
            average_step_seconds = (
                time.perf_counter() - progress_started
            ) / progress_steps
            eta_seconds = (
                settings.max_steps - completed_steps
            ) * average_step_seconds
            print(
                format_progress(
                    completed_steps=completed_steps,
                    max_steps=settings.max_steps,
                    train_loss=last_metrics["train_loss"],
                    learning_rate=learning_rate,
                    tokens_per_second=last_metrics["tokens_per_second"],
                    eta_seconds=eta_seconds,
                ),
                flush=True,
            )

        should_evaluate = (
            completed_steps == 1
            or completed_steps % eval_every == 0
            or completed_steps == settings.max_steps
            or completed_steps == stop_after_step
        )
        if should_evaluate:
            validation_loss = evaluate(
                model,
                validation_data,
                batch_size=min(settings.batch_size, len(validation_data)),
                max_batches=eval_batches,
                device=device,
                amp=settings.amp,
            )
            last_metrics["validation_loss"] = validation_loss
            last_metrics["validation_perplexity"] = (
                math.exp(validation_loss)
                if validation_loss < 50.0
                else float("inf")
            )
            _append_metric(metrics_path, last_metrics)
            print(
                f"step {completed_steps:>6}/{settings.max_steps} "
                f"train={last_metrics['train_loss']:.4f} "
                f"validation={validation_loss:.4f} "
                f"lr={learning_rate:.3e}"
            )

        should_checkpoint = (
            completed_steps % checkpoint_every == 0
            or completed_steps == settings.max_steps
            or completed_steps == stop_after_step
        )
        if should_checkpoint:
            checkpoint_path = save_checkpoint(
                run_dir=run_dir,
                completed_steps=completed_steps,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                sampler=sampler,
                contract=contract,
                last_metrics=last_metrics,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        if completed_steps == stop_after_step:
            break

    return {
        "completed_steps": completed_steps,
        "last_metrics": last_metrics,
        "contracts": asdict(contracts),
        "train_stats": asdict(train_data.stats),
        "validation_stats": asdict(validation_data.stats),
        "initialization": initialization,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "none", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--max-steps", type=int, default=PILOT.max_steps)
    parser.add_argument("--batch-size", type=int, default=PILOT.batch_size)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-lr", type=float, default=PILOT.max_lr)
    parser.add_argument("--min-lr", type=float, default=PILOT.min_lr)
    parser.add_argument("--warmup-steps", type=int, default=PILOT.warmup_steps)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=PILOT.gradient_clip,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=PILOT.checkpoint_every,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress, throughput, and ETA every N optimiser steps.",
    )
    parser.add_argument(
        "--resume",
        help="Checkpoint path or 'latest' within --run-dir.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help=(
            "Start a new optimisation phase from a verified checkpoint or "
            "run directory; optimizer, scheduler, sampler, and RNG start fresh."
        ),
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Cleanly stop after this step; useful for pre-emption tests.",
    )
    parser.add_argument("--max-train-documents", type=int)
    parser.add_argument("--max-validation-documents", type=int)
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Permit non-deterministic torch kernels.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Set torch CPU worker threads for local smoke runs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be at least 1.")
        torch.set_num_threads(args.cpu_threads)

    device = resolve_device(args.device)
    settings = TrainingSettings(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        amp=args.amp,
        deterministic=not args.allow_nondeterministic,
        max_train_documents=args.max_train_documents,
        max_validation_documents=args.max_validation_documents,
    )
    result = run_training(
        model_config=PILOT,
        dataset_dir=args.dataset_dir,
        tokenizer_dir=args.tokenizer_dir,
        run_dir=args.run_dir,
        settings=settings,
        device=device,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        resume=args.resume,
        init_from=args.init_from,
        stop_after_step=args.stop_after_step,
    )
    print(
        f"Training stopped cleanly at step {result['completed_steps']}. "
        f"Mode={TRAINING_MODE}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
