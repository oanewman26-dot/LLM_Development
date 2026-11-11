"""
Byte-level BPE tokenizer for AuroraLM.

The tokenizer is deliberately trained separately from the language model.
Train it once on a representative mix of the general corpus and Aurora's own
text, save the resulting directory, and use that exact artifact for training,
inference, and checkpoint validation.

The four special-token IDs are part of the checkpoint contract:

    <pad> = 0
    <bos> = 1
    <eos> = 2
    <unk> = 3

Ordinary text is encoded as UTF-8 bytes before BPE merges are applied. The
complete byte alphabet is seeded during training, so previously unseen text,
Unicode punctuation, names, and emoji remain representable without falling
back to <unk>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union


TOKENIZER_FORMAT_VERSION = 1
TOKENIZER_FILENAME = "tokenizer.json"
METADATA_FILENAME = "tokenizer_config.json"

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
SPECIAL_TOKEN_IDS = {
    PAD_TOKEN: PAD_TOKEN_ID,
    BOS_TOKEN: BOS_TOKEN_ID,
    EOS_TOKEN: EOS_TOKEN_ID,
    UNK_TOKEN: UNK_TOKEN_ID,
}

DEFAULT_VOCAB_SIZE = 8192
DEFAULT_CONTEXT_LENGTH = 512


class TokenizerError(RuntimeError):
    """Base error for tokenizer contract failures."""


class TokenizerTrainingError(TokenizerError):
    """Raised when a tokenizer cannot satisfy the requested training contract."""


def _require_tokenizers() -> Any:
    try:
        import tokenizers
    except ImportError as exc:
        raise TokenizerError(
            "AuroraTokenizer requires the 'tokenizers' package. "
            "Install the project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return tokenizers


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a file completely before replacing any existing artifact."""
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


class AuroraTokenizer:
    """
    Stable wrapper around a Hugging Face byte-level BPE tokenizer.

    Use :meth:`train` to create an instance, :meth:`save` to persist the
    checkpoint-compatible artifact, and :meth:`load` everywhere else.
    """

    def __init__(
        self,
        backend: Any,
        *,
        artifact_fingerprint: Optional[str] = None,
    ):
        self._backend = backend
        self._artifact_fingerprint = artifact_fingerprint
        self._validate_special_tokens()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = 2,
        show_progress: bool = True,
        length: Optional[int] = None,
    ) -> "AuroraTokenizer":
        """
        Train byte-level BPE merges from an iterable of text samples.

        The iterable may stream a corpus; it is not materialised in memory.
        ``vocab_size`` is exact rather than advisory because the model's
        embedding and output-head dimensions must match it.
        """
        tokenizers = _require_tokenizers()
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer

        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1.")

        byte_alphabet = pre_tokenizers.ByteLevel.alphabet()
        minimum_vocab_size = len(SPECIAL_TOKENS) + len(byte_alphabet)
        if vocab_size < minimum_vocab_size:
            raise ValueError(
                f"vocab_size must be at least {minimum_vocab_size} for the "
                "four special tokens and complete byte alphabet."
            )

        backend = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        backend.decoder = decoders.ByteLevel()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=byte_alphabet,
        )

        seen = 0

        def checked_texts() -> Iterator[str]:
            nonlocal seen
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError(
                        "Tokenizer training samples must be strings; "
                        f"received {type(text).__name__}."
                    )
                if text:
                    seen += 1
                    yield text

        backend.train_from_iterator(
            checked_texts(),
            trainer=trainer,
            length=length,
        )

        if seen == 0:
            raise TokenizerTrainingError(
                "The training corpus did not contain any non-empty text."
            )

        backend.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
            special_tokens=[
                (BOS_TOKEN, BOS_TOKEN_ID),
                (EOS_TOKEN, EOS_TOKEN_ID),
            ],
        )

        tokenizer = cls(backend)
        if tokenizer.vocab_size != vocab_size:
            raise TokenizerTrainingError(
                f"Requested a vocabulary of {vocab_size} tokens, but the "
                f"corpus produced only {tokenizer.vocab_size}. Supply a larger "
                "or more varied representative corpus, lower min_frequency, "
                "or explicitly choose the smaller size in AuroraLMConfig."
            )

        # Keep the imported module live until training has completed. This
        # assignment also makes the dependency check explicit to type checkers.
        _ = tokenizers
        return tokenizer

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "AuroraTokenizer":
        """Load and validate a saved Aurora tokenizer directory."""
        _require_tokenizers()
        from tokenizers import Tokenizer

        directory = Path(directory)
        tokenizer_path = directory / TOKENIZER_FILENAME
        metadata_path = directory / METADATA_FILENAME

        if not tokenizer_path.is_file() or not metadata_path.is_file():
            raise TokenizerError(
                f"{directory} is not a complete Aurora tokenizer directory; "
                f"expected {TOKENIZER_FILENAME} and {METADATA_FILENAME}."
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(
                f"Could not read tokenizer metadata from {metadata_path}."
            ) from exc

        if metadata.get("format_version") != TOKENIZER_FORMAT_VERSION:
            raise TokenizerError(
                "Unsupported tokenizer format version "
                f"{metadata.get('format_version')!r}; expected "
                f"{TOKENIZER_FORMAT_VERSION}."
            )

        tokenizer_bytes = tokenizer_path.read_bytes()
        actual_fingerprint = _sha256_bytes(tokenizer_bytes)
        expected_fingerprint = metadata.get("fingerprint")
        if actual_fingerprint != expected_fingerprint:
            raise TokenizerError(
                "Tokenizer fingerprint mismatch. The tokenizer JSON and its "
                "metadata do not belong to the same saved artifact."
            )

        backend = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        tokenizer = cls(
            backend,
            artifact_fingerprint=expected_fingerprint,
        )

        if tokenizer.vocab_size != metadata.get("vocab_size"):
            raise TokenizerError(
                "Tokenizer vocabulary size does not match its metadata."
            )
        if tokenizer.special_token_ids != metadata.get("special_token_ids"):
            raise TokenizerError(
                "Tokenizer special-token IDs do not match its metadata."
            )
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Number of IDs, including special tokens."""
        return self._backend.get_vocab_size(with_added_tokens=True)

    @property
    def special_token_ids(self) -> Dict[str, int]:
        return {
            token: self._backend.token_to_id(token)
            for token in SPECIAL_TOKENS
        }

    @property
    def fingerprint(self) -> str:
        """Content hash saved with checkpoints to prevent tokenizer drift."""
        if self._artifact_fingerprint is not None:
            return self._artifact_fingerprint
        return _sha256_bytes(self._serialised_backend())

    def _serialised_backend(self) -> bytes:
        return self._backend.to_str(pretty=True).encode("utf-8")

    def _validate_special_tokens(self) -> None:
        actual = self.special_token_ids
        if actual != SPECIAL_TOKEN_IDS:
            raise TokenizerError(
                "Special-token ID contract mismatch. Expected "
                f"{SPECIAL_TOKEN_IDS}, got {actual}."
            )

    def token_to_id(self, token: str) -> Optional[int]:
        return self._backend.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._backend.id_to_token(token_id)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode one string.

        When truncating, BOS and EOS are retained and only content tokens are
        removed. This avoids training examples that silently lose the
        end-of-sequence target.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}.")
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be at least 1.")
        if truncation and max_length is None:
            raise ValueError("truncation=True requires max_length.")

        content_ids = list(
            self._backend.encode(text, add_special_tokens=False).ids
        )
        special_count = 2 if add_special_tokens else 0

        if truncation and len(content_ids) + special_count > max_length:
            content_budget = max_length - special_count
            if content_budget < 0:
                raise ValueError(
                    f"max_length={max_length} cannot fit the requested "
                    f"{special_count} special tokens."
                )
            content_ids = content_ids[:content_budget]

        if add_special_tokens:
            return [BOS_TOKEN_ID, *content_ids, EOS_TOKEN_ID]
        return content_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode IDs back to UTF-8 text."""
        return self._backend.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch and return framework-neutral IDs and attention masks.

        If ``padding`` is true, sequences are padded to ``max_length`` when it
        is supplied, otherwise to the longest sequence in the batch.
        """
        sequences = [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
            )
            for text in texts
        ]

        if not sequences:
            return {"input_ids": [], "attention_mask": []}

        if padding:
            target_length = (
                max_length
                if max_length is not None
                else max(len(sequence) for sequence in sequences)
            )
            if any(len(sequence) > target_length for sequence in sequences):
                raise ValueError(
                    "A sequence exceeds the padding target. Enable truncation "
                    "or increase max_length."
                )
        else:
            target_length = None

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for sequence in sequences:
            if target_length is None:
                input_ids.append(sequence)
                attention_mask.append([1] * len(sequence))
                continue

            pad_count = target_length - len(sequence)
            input_ids.append(sequence + [PAD_TOKEN_ID] * pad_count)
            attention_mask.append([1] * len(sequence) + [0] * pad_count)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def save(self, directory: Union[str, Path]) -> None:
        """Atomically save the standard tokenizer JSON and Aurora metadata."""
        directory = Path(directory)
        tokenizer_bytes = self._serialised_backend()
        fingerprint = _sha256_bytes(tokenizer_bytes)
        metadata = {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "model_type": "byte_level_bpe",
            "vocab_size": self.vocab_size,
            "special_token_ids": self.special_token_ids,
            "add_bos_eos": True,
            "fingerprint": fingerprint,
        }
        metadata_bytes = (
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        _atomic_write(directory / TOKENIZER_FILENAME, tokenizer_bytes)
        _atomic_write(directory / METADATA_FILENAME, metadata_bytes)
        self._artifact_fingerprint = fingerprint

    def validate_config(self, config: Any) -> None:
        """Refuse a model config whose embedding table cannot fit these IDs."""
        config_vocab_size = getattr(config, "vocab_size", None)
        if config_vocab_size != self.vocab_size:
            raise TokenizerError(
                f"Tokenizer has {self.vocab_size} tokens, but model config "
                f"requests vocab_size={config_vocab_size}."
            )

        context_length = getattr(config, "context_length", None)
        if not isinstance(context_length, int) or context_length < 2:
            raise TokenizerError(
                "Model config context_length must be an integer of at least 2 "
                "to hold BOS and EOS."
            )


def iter_text_files(paths: Sequence[Union[str, Path]]) -> Iterator[str]:
    """
    Stream UTF-8 text files line by line for bounded-memory training.

    Structured JSON should be converted to clean text fields by the future
    dataset builder; feeding raw embeddings or metadata into the tokenizer
    would waste merge capacity.
    """
    for raw_path in paths:
        if str(raw_path) == "-":
            yield from sys.stdin
            continue

        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer corpus file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            yield from handle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train and save a byte-level BPE tokenizer.",
    )
    train_parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="UTF-8 text files. Use '-' to read text from standard input.",
    )
    train_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for tokenizer.json and tokenizer_config.json.",
    )
    train_parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )
    train_parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
    )
    train_parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Validate and describe a saved tokenizer.",
    )
    inspect_parser.add_argument("directory")

    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode a short string with a saved tokenizer.",
    )
    encode_parser.add_argument("directory")
    encode_parser.add_argument("text")
    encode_parser.add_argument(
        "--no-special-tokens",
        action="store_true",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        tokenizer = AuroraTokenizer.train(
            iter_text_files(args.input),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            show_progress=not args.no_progress,
        )
        tokenizer.save(args.output)
        print(
            f"Saved {tokenizer.vocab_size}-token Aurora tokenizer to "
            f"{Path(args.output).resolve()}"
        )
        print(f"Fingerprint: {tokenizer.fingerprint}")
        return 0

    if args.command == "inspect":
        tokenizer = AuroraTokenizer.load(args.directory)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Special token IDs: {tokenizer.special_token_ids}")
        print(f"Fingerprint: {tokenizer.fingerprint}")
        return 0

    if args.command == "encode":
        tokenizer = AuroraTokenizer.load(args.directory)
        token_ids = tokenizer.encode(
            args.text,
            add_special_tokens=not args.no_special_tokens,
        )
        print(json.dumps(token_ids))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
