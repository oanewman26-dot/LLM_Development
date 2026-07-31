"""
Strict checkpoint loading and autoregressive generation for AuroraLM.

The current trained checkpoint is a dense, neutral-state language model rather
than an instruction-tuned chatbot. Generation therefore continues text with
``aurora_state=None`` and should be evaluated as base-model completion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

from config import AuroraLMConfig
from NN import AuroraLM
from state_schema import SCHEMA_FINGERPRINT, SCHEMA_VERSION
from tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    AuroraTokenizer,
)


INFERENCE_FORMAT_VERSION = 1
SUPPORTED_CHECKPOINT_FORMAT_VERSION = 1
SUPPORTED_TRAINING_MODE = "dense_neutral_state"
DEFAULT_TOKENIZER_DIR = Path("artifacts/aurora_bpe_8192_v1")

DEFAULT_QUALITATIVE_PROMPTS = (
    {
        "name": "narrative_continuation",
        "prompt": "The rain settled over the garden, and Aurora",
    },
    {
        "name": "learning_concept",
        "prompt": "Learning from experience means",
    },
    {
        "name": "uncertainty_response",
        "prompt": "Aurora noticed uncertainty and chose to",
    },
    {
        "name": "safety_judgement",
        "prompt": "When a situation may be dangerous, the careful response is to",
    },
    {
        "name": "factual_continuation",
        "prompt": "The Moon orbits the Earth because",
    },
    {
        "name": "code_diagnostic",
        "prompt": "def add_numbers(a, b):\n    ",
    },
)


class InferenceError(RuntimeError):
    """Base error for invalid checkpoints or generation requests."""


class InferenceContractError(InferenceError):
    """Raised when checkpoint, runtime, tokenizer, or schema contracts drift."""


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 48
    min_new_tokens: int = 4
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    seed: int = 42


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    token_ids: tuple[int, ...]
    stop_reason: str
    elapsed_seconds: float
    settings: GenerationSettings

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["token_ids"] = list(self.token_ids)
        return value


@dataclass
class InferenceSession:
    model: AuroraLM
    tokenizer: AuroraTokenizer
    config: AuroraLMConfig
    checkpoint_path: Path
    completed_steps: int
    checkpoint_sha256: str
    device: torch.device


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
        raise InferenceContractError(
            f"Could not read valid JSON from {path}."
        ) from exc
    if not isinstance(value, dict):
        raise InferenceContractError(f"Expected a JSON object in {path}.")
    return value


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise InferenceError(
            "CUDA was requested, but torch.cuda.is_available() is false."
        )
    return torch.device(requested)


def resolve_checkpoint(path: Path) -> tuple[Path, Optional[str]]:
    path = Path(path)
    if path.is_dir():
        latest_path = path / "latest.json"
        latest = _read_json(latest_path)
        checkpoint_name = latest.get("checkpoint")
        expected_hash = latest.get("sha256")
        if (
            not isinstance(checkpoint_name, str)
            or Path(checkpoint_name).name != checkpoint_name
            or not isinstance(expected_hash, str)
        ):
            raise InferenceContractError(
                f"{latest_path} contains an invalid checkpoint pointer."
            )
        checkpoint_path = path / checkpoint_name
        if not checkpoint_path.is_file():
            raise InferenceContractError(
                f"Latest checkpoint is missing: {checkpoint_path}"
            )
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != expected_hash:
            raise InferenceContractError(
                "Latest checkpoint hash does not match latest.json."
            )
        return checkpoint_path, actual_hash

    if path.name == "latest.json":
        return resolve_checkpoint(path.parent)
    if not path.is_file():
        raise InferenceContractError(f"Checkpoint does not exist: {path}")
    return path, None


def _current_runtime_contract() -> dict[str, str]:
    return {
        "python_major_minor": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        "torch": torch.__version__.split("+", 1)[0],
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def _config_from_checkpoint(value: Any) -> AuroraLMConfig:
    if not isinstance(value, dict):
        raise InferenceContractError("Checkpoint model configuration is missing.")
    known_fields = {field.name for field in fields(AuroraLMConfig)}
    unknown_fields = set(value) - known_fields
    missing_fields = known_fields - set(value)
    if unknown_fields or missing_fields:
        raise InferenceContractError(
            "Checkpoint model configuration fields do not match this runtime. "
            f"Missing={sorted(missing_fields)}, unknown={sorted(unknown_fields)}."
        )
    try:
        return AuroraLMConfig(**value)
    except (TypeError, ValueError) as exc:
        raise InferenceContractError(
            "Checkpoint model configuration is invalid."
        ) from exc


def load_inference_session(
    checkpoint: Path,
    tokenizer_dir: Path,
    *,
    device: torch.device,
    strict_runtime: bool = True,
) -> InferenceSession:
    checkpoint_path, verified_hash = resolve_checkpoint(Path(checkpoint))
    checkpoint_hash = verified_hash or sha256_file(checkpoint_path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise InferenceContractError(
            f"Could not load checkpoint {checkpoint_path}."
        ) from exc
    if not isinstance(payload, dict):
        raise InferenceContractError("Checkpoint payload is not a dictionary.")
    if (
        payload.get("checkpoint_format_version")
        != SUPPORTED_CHECKPOINT_FORMAT_VERSION
    ):
        raise InferenceContractError("Unsupported checkpoint format version.")

    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise InferenceContractError("Checkpoint contract is missing.")
    runtime_contract = contract.get("runtime_contract")
    if strict_runtime and runtime_contract != _current_runtime_contract():
        raise InferenceContractError(
            "Checkpoint runtime contract does not match this environment."
        )

    training_contracts = contract.get("contracts")
    if not isinstance(training_contracts, dict):
        raise InferenceContractError("Checkpoint training contracts are missing.")
    if training_contracts.get("training_mode") != SUPPORTED_TRAINING_MODE:
        raise InferenceContractError(
            "This inference path supports only dense_neutral_state checkpoints."
        )

    config = _config_from_checkpoint(contract.get("model_config"))
    if (
        config.schema_version != SCHEMA_VERSION
        or config.schema_fingerprint != SCHEMA_FINGERPRINT
        or training_contracts.get("state_schema_version") != SCHEMA_VERSION
        or training_contracts.get("state_schema_fingerprint")
        != SCHEMA_FINGERPRINT
    ):
        raise InferenceContractError(
            "Checkpoint state schema does not match the current Aurora schema."
        )

    tokenizer = AuroraTokenizer.load(tokenizer_dir)
    tokenizer.validate_config(config)
    if (
        training_contracts.get("tokenizer_fingerprint")
        != tokenizer.fingerprint
    ):
        raise InferenceContractError(
            "Checkpoint tokenizer fingerprint does not match the artifact."
        )

    model = AuroraLM(config).to(device)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise InferenceContractError(
            "Checkpoint model state does not match its configuration."
        ) from exc
    nonfinite = [
        name
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all()
    ]
    if nonfinite:
        raise InferenceContractError(
            f"Checkpoint contains non-finite tensors: {nonfinite[:3]}"
        )
    model.eval()

    completed_steps = payload.get("completed_steps")
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise InferenceContractError("Checkpoint completed_steps is invalid.")
    return InferenceSession(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint_path=checkpoint_path,
        completed_steps=completed_steps,
        checkpoint_sha256=checkpoint_hash,
        device=device,
    )


def validate_generation_settings(settings: GenerationSettings) -> None:
    if settings.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if not 0 <= settings.min_new_tokens <= settings.max_new_tokens:
        raise ValueError(
            "min_new_tokens must be between 0 and max_new_tokens."
        )
    if settings.temperature < 0.0:
        raise ValueError("temperature cannot be negative.")
    if settings.top_k < 0:
        raise ValueError("top_k cannot be negative.")
    if not 0.0 < settings.top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if settings.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    previous_ids: Iterable[int],
    penalty: float,
) -> None:
    if penalty == 1.0:
        return
    for token_id in set(previous_ids):
        value = logits[token_id]
        logits[token_id] = (
            value * penalty if value < 0 else value / penalty
        )


def _sample_token(
    logits: torch.Tensor,
    *,
    settings: GenerationSettings,
    generator: torch.Generator,
) -> int:
    if settings.temperature == 0.0:
        return int(torch.argmax(logits).item())

    filtered = logits / settings.temperature
    if settings.top_k > 0:
        top_k = min(settings.top_k, filtered.numel())
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if settings.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative > settings.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)

    probabilities = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise InferenceError("Sampling probabilities became invalid.")
    return int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).item()
    )


@torch.inference_mode()
def generate(
    session: InferenceSession,
    prompt: str,
    settings: GenerationSettings,
) -> GenerationResult:
    validate_generation_settings(settings)
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    content_ids = session.tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    prompt_ids = [
        BOS_TOKEN_ID,
        *content_ids[-(session.config.context_length - 1) :],
    ]
    sequence = list(prompt_ids)
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=session.device.type)
    generator.manual_seed(settings.seed)
    started = time.perf_counter()

    for _ in range(settings.max_new_tokens):
        context = sequence[-session.config.context_length :]
        inputs = torch.tensor(
            [context],
            dtype=torch.long,
            device=session.device,
        )
        logits = session.model(inputs, aurora_state=None)[0, -1].float()
        _apply_repetition_penalty(
            logits,
            context,
            settings.repetition_penalty,
        )
        logits[PAD_TOKEN_ID] = float("-inf")
        logits[BOS_TOKEN_ID] = float("-inf")
        logits[UNK_TOKEN_ID] = float("-inf")
        if len(generated) < settings.min_new_tokens:
            logits[EOS_TOKEN_ID] = float("-inf")

        next_id = _sample_token(
            logits,
            settings=settings,
            generator=generator,
        )
        if next_id == EOS_TOKEN_ID:
            stop_reason = "eos"
            break
        generated.append(next_id)
        sequence.append(next_id)

    elapsed = time.perf_counter() - started
    text = session.tokenizer.decode(generated)
    return GenerationResult(
        prompt=prompt,
        text=text,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        token_ids=tuple(generated),
        stop_reason=stop_reason,
        elapsed_seconds=elapsed,
        settings=settings,
    )


def _normalised_words(text: str) -> list[str]:
    return re.findall(r"\w+(?:['’-]\w+)?", text.casefold())


def repetition_metrics(token_ids: Sequence[int]) -> dict[str, float]:
    if not token_ids:
        return {
            "distinct_token_ratio": 0.0,
            "repeated_trigram_ratio": 0.0,
        }
    distinct = len(set(token_ids)) / len(token_ids)
    trigrams = [
        tuple(token_ids[index : index + 3])
        for index in range(max(0, len(token_ids) - 2))
    ]
    repeated = (
        1.0 - len(set(trigrams)) / len(trigrams)
        if trigrams
        else 0.0
    )
    return {
        "distinct_token_ratio": distinct,
        "repeated_trigram_ratio": repeated,
    }


def audit_exact_shingles(
    generated_texts: Sequence[str],
    documents_path: Path,
    *,
    shingle_words: int = 16,
) -> dict[str, Any]:
    if shingle_words < 4:
        raise ValueError("shingle_words must be at least 4.")
    candidates: set[tuple[str, ...]] = set()
    for text in generated_texts:
        words = _normalised_words(text)
        candidates.update(
            tuple(words[index : index + shingle_words])
            for index in range(len(words) - shingle_words + 1)
        )
    if not candidates:
        return {
            "shingle_words": shingle_words,
            "candidate_shingles": 0,
            "exact_matches": 0,
        }

    matched: set[tuple[str, ...]] = set()
    with Path(documents_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                continue
            words = _normalised_words(text)
            document_shingles = {
                tuple(words[index : index + shingle_words])
                for index in range(len(words) - shingle_words + 1)
            }
            matched.update(candidates.intersection(document_shingles))
            if matched == candidates:
                break
    return {
        "shingle_words": shingle_words,
        "candidate_shingles": len(candidates),
        "exact_matches": len(matched),
    }


def run_qualitative_suite(
    session: InferenceSession,
    prompts: Sequence[Mapping[str, str]],
    settings: GenerationSettings,
    *,
    audit_documents: Optional[Path] = None,
) -> dict[str, Any]:
    results = []
    for index, item in enumerate(prompts):
        name = item.get("name")
        prompt = item.get("prompt")
        if not isinstance(name, str) or not isinstance(prompt, str):
            raise ValueError("Each qualitative prompt needs name and prompt.")
        prompt_settings = GenerationSettings(
            **{**asdict(settings), "seed": settings.seed + index}
        )
        generated = generate(session, prompt, prompt_settings)
        results.append(
            {
                "name": name,
                **generated.to_dict(),
                "diagnostics": repetition_metrics(generated.token_ids),
            }
        )
        print(
            f"[{name}] {generated.generated_tokens} tokens in "
            f"{generated.elapsed_seconds:.2f}s ({generated.stop_reason})"
        )

    report: dict[str, Any] = {
        "format_version": INFERENCE_FORMAT_VERSION,
        "checkpoint": {
            "file": session.checkpoint_path.name,
            "sha256": session.checkpoint_sha256,
            "completed_steps": session.completed_steps,
        },
        "tokenizer_fingerprint": session.tokenizer.fingerprint,
        "model_config": asdict(session.config),
        "generation_settings": asdict(settings),
        "results": results,
    }
    if audit_documents is not None:
        report["memorisation_audit"] = audit_exact_shingles(
            [result["text"] for result in results],
            audit_documents,
        )
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(report), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)


def _settings_from_args(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("prompt")
    _add_generation_arguments(generate_parser)

    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument("--output", type=Path, required=True)
    suite_parser.add_argument("--audit-documents", type=Path)
    _add_generation_arguments(suite_parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    device = resolve_device(args.device)
    session = load_inference_session(
        args.checkpoint,
        args.tokenizer_dir,
        device=device,
    )
    settings = _settings_from_args(args)
    print(
        f"Loaded {session.checkpoint_path.name} at step "
        f"{session.completed_steps} on {device}."
    )

    if args.command == "generate":
        result = generate(session, args.prompt, settings)
        print(result.text)
        return 0

    if args.command == "suite":
        report = run_qualitative_suite(
            session,
            DEFAULT_QUALITATIVE_PROMPTS,
            settings,
            audit_documents=args.audit_documents,
        )
        _write_report(args.output, report)
        print(f"Saved qualitative report to {args.output.resolve()}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
