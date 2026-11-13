"""Versioned state contract for AuroraLM.

Aurora's thalamus produces sparse activations over the same 171 emotion nodes
defined by ``emotion_nodes.EMOTION_WORDS`` in the live Aurora system.  This
module fixes their order, validates a turn's state, and serialises it safely for
datasets and checkpoints.

The raw vector is deliberately kept separate from normalisation.  Do not
normalise each turn by its own maximum: that would erase the difference between
a quiet state and an intense one.  Calibrate one fixed ceiling from recorded
training data, save it with the run configuration, then use that same value for
training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "aurora-state-171/v1"

# This order is a model contract.  It must stay aligned with EMOTION_WORDS in
# Aurora's live emotion_nodes.py.  New state meanings require a new schema
# version rather than an in-place reorder.
EMOTION_NAMES: tuple[str, ...] = (
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed",
    "anxious", "aroused", "ashamed", "astonished", "at ease", "awestruck",
    "bewildered", "bitter", "blissful", "bored", "brooding", "calm",
    "cheerful", "compassionate", "contemptuous", "content", "defiant",
    "delighted", "dependent", "depressed", "desperate", "disdainful",
    "disgusted", "disoriented", "dispirited", "distressed", "disturbed",
    "docile", "droopy", "dumbstruck", "eager", "ecstatic", "elated",
    "embarrassed", "empathetic", "energized", "enraged", "enthusiastic",
    "envious", "euphoric", "exasperated", "excited", "exuberant",
    "frightened", "frustrated", "fulfilled", "furious", "gloomy",
    "grateful", "greedy", "grief-stricken", "grumpy", "guilty", "happy",
    "hateful", "heartbroken", "hope", "hopeful", "horrified", "hostile",
    "humiliated", "hurt", "hysterical", "impatient", "indifferent",
    "indignant", "infatuated", "inspired", "insulted", "invigorated",
    "irate", "irritated", "jealous", "joyful", "jubilant", "kind",
    "lazy", "listless", "lonely", "loving", "mad", "melancholy",
    "miserable", "mortified", "mystified", "nervous", "nostalgic",
    "obstinate", "offended", "on edge", "optimistic", "outraged",
    "overwhelmed", "panicked", "paranoid", "patient", "peaceful",
    "perplexed", "playful", "pleased", "proud", "puzzled", "rattled",
    "reflective", "refreshed", "regretful", "rejuvenated", "relaxed",
    "relieved", "remorseful", "resentful", "resigned", "restless",
    "sad", "safe", "satisfied", "scared", "scornful", "self-confident",
    "self-conscious", "self-critical", "sensitive", "sentimental",
    "serene", "shaken", "shocked", "skeptical", "sleepy", "sluggish",
    "smug", "sorry", "spiteful", "stimulated", "stressed", "stubborn",
    "stuck", "sullen", "surprised", "suspicious", "sympathetic", "tense",
    "terrified", "thankful", "thrilled", "tired", "tormented", "trapped",
    "triumphant", "troubled", "uneasy", "unhappy", "unnerved", "unsettled",
    "upset", "valiant", "vengeful", "vibrant", "vigilant", "vindictive",
    "vulnerable", "weary", "worn out", "worried", "worthless",
)

EMOTION_COUNT = len(EMOTION_NAMES)
EMOTION_INDEX = {name: index for index, name in enumerate(EMOTION_NAMES)}
SCHEMA_FINGERPRINT = sha256("\0".join(EMOTION_NAMES).encode("utf-8")).hexdigest()

if EMOTION_COUNT != 171:
    raise RuntimeError(f"AuroraLM requires 171 emotion dimensions, found {EMOTION_COUNT}.")
if len(EMOTION_INDEX) != EMOTION_COUNT:
    raise RuntimeError("Emotion names must be unique.")


class StateSchemaError(ValueError):
    """Raised when a state cannot safely enter the AuroraLM data pipeline."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_values(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != EMOTION_COUNT:
        raise StateSchemaError(
            f"Expected {EMOTION_COUNT} emotion values, received {len(values)}."
        )

    checked: list[float] = []
    for index, raw_value in enumerate(values):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise StateSchemaError(
                f"{EMOTION_NAMES[index]!r} must be numeric, got {raw_value!r}."
            ) from exc
        if not isfinite(value):
            raise StateSchemaError(
                f"{EMOTION_NAMES[index]!r} must be finite, got {value!r}."
            )
        if value < 0.0:
            raise StateSchemaError(
                f"{EMOTION_NAMES[index]!r} must be non-negative, got {value}."
            )
        checked.append(value)
    return tuple(checked)


@dataclass(frozen=True, slots=True)
class AuroraState171:
    """A single validated, sparse-or-dense thalamic state at one point in time.

    ``values`` are raw activation scores in the canonical ``EMOTION_NAMES``
    order.  They are not per-turn normalised.  Sparse turns are represented by
    zero values for nodes that did not fire.
    """

    values: tuple[float, ...]
    captured_at: str
    source: str = "thalamus"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise StateSchemaError(
                f"Expected schema {SCHEMA_VERSION!r}, got {self.schema_version!r}."
            )
        if not self.source or not self.source.strip():
            raise StateSchemaError("State source must be a non-empty string.")
        object.__setattr__(self, "values", _validate_values(self.values))

    @classmethod
    def zero(
        cls,
        *,
        captured_at: str | None = None,
        source: str = "thalamus",
    ) -> "AuroraState171":
        """Return an explicit no-node-fired state, not a missing state."""
        return cls(
            values=(0.0,) * EMOTION_COUNT,
            captured_at=captured_at or _utc_now(),
            source=source,
        )

    @classmethod
    def from_mapping(
        cls,
        activations: Mapping[str, float],
        *,
        captured_at: str | None = None,
        source: str = "thalamus",
        strict: bool = True,
    ) -> "AuroraState171":
        """Build a dense vector from ``{emotion_name: activation}`` pairs."""
        unknown = set(activations).difference(EMOTION_INDEX)
        if strict and unknown:
            raise StateSchemaError(
                "Unknown emotion names: " + ", ".join(sorted(map(str, unknown)))
            )

        values = [0.0] * EMOTION_COUNT
        for name, raw_value in activations.items():
            index = EMOTION_INDEX.get(name)
            if index is not None:
                values[index] = raw_value

        return cls(
            values=tuple(values),
            captured_at=captured_at or _utc_now(),
            source=source,
        )

    @classmethod
    def from_active_nodes(
        cls,
        active_nodes: Iterable[tuple[Any, float]],
        *,
        captured_at: str | None = None,
        source: str = "thalamus",
        strict: bool = True,
    ) -> "AuroraState171":
        """Adapt Aurora's ``[(EmotionNode, activation_score), ...]`` output.

        If a node appears twice, the strongest score wins.  That preserves the
        sparse thalamic interpretation while producing a deterministic dense
        model input.
        """
        activations: dict[str, float] = {}
        for node, raw_score in active_nodes:
            name = getattr(node, "name", node)
            if not isinstance(name, str):
                raise StateSchemaError(f"Emotion node has no usable name: {node!r}.")
            if name not in EMOTION_INDEX:
                if strict:
                    raise StateSchemaError(f"Unknown emotion node: {name!r}.")
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise StateSchemaError(
                    f"Activation for {name!r} must be numeric, got {raw_score!r}."
                ) from exc
            activations[name] = max(activations.get(name, 0.0), score)

        return cls.from_mapping(
            activations,
            captured_at=captured_at,
            source=source,
            strict=strict,
        )

    @property
    def active_count(self) -> int:
        return sum(value > 0.0 for value in self.values)

    def active_items(self) -> tuple[tuple[str, float], ...]:
        """Return the active nodes in canonical order."""
        return tuple(
            (name, value)
            for name, value in zip(EMOTION_NAMES, self.values)
            if value > 0.0
        )

    def normalised_values(
        self,
        *,
        activation_ceiling: float,
        clip: bool = False,
    ) -> tuple[float, ...]:
        """Apply one fixed calibration ceiling to all turns.

        Choose ``activation_ceiling`` from a recorded calibration set (for
        example a high percentile of valid thalamic scores) and store it in the
        experiment config.  With ``clip=False`` an out-of-range live score is
        an error, not a hidden data change.
        """
        try:
            ceiling = float(activation_ceiling)
        except (TypeError, ValueError) as exc:
            raise StateSchemaError("activation_ceiling must be a positive number.") from exc
        if not isfinite(ceiling) or ceiling <= 0.0:
            raise StateSchemaError("activation_ceiling must be a positive finite number.")

        normalised: list[float] = []
        for name, value in zip(EMOTION_NAMES, self.values):
            if value > ceiling:
                if not clip:
                    raise StateSchemaError(
                        f"{name!r}={value} exceeds calibration ceiling {ceiling}. "
                        "Recalibrate or pass clip=True explicitly."
                    )
                normalised.append(1.0)
            else:
                normalised.append(value / ceiling)
        return tuple(normalised)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe record for a dataset row or an inference audit log."""
        return {
            "schema_version": self.schema_version,
            "schema_fingerprint": SCHEMA_FINGERPRINT,
            "captured_at": self.captured_at,
            "source": self.source,
            "values": list(self.values),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AuroraState171":
        """Load a serialised state and reject incompatible vector orders."""
        version = record.get("schema_version")
        fingerprint = record.get("schema_fingerprint")
        if version != SCHEMA_VERSION:
            raise StateSchemaError(
                f"Expected schema {SCHEMA_VERSION!r}, got {version!r}."
            )
        if fingerprint != SCHEMA_FINGERPRINT:
            raise StateSchemaError("State schema fingerprint does not match this model.")

        values = record.get("values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise StateSchemaError("State record must contain a numeric values sequence.")

        captured_at = record.get("captured_at")
        source = record.get("source", "thalamus")
        if not isinstance(captured_at, str) or not captured_at:
            raise StateSchemaError("State record must contain a captured_at timestamp.")
        if not isinstance(source, str):
            raise StateSchemaError("State record source must be a string.")

        return cls(values=tuple(values), captured_at=captured_at, source=source)

    def to_torch(
        self,
        *,
        activation_ceiling: float | None = None,
        clip: bool = False,
        batch: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> Any:
        """Return a PyTorch tensor without making PyTorch a schema dependency.

        The default shape is ``(1, 171)`` for a single turn.  Pass
        ``activation_ceiling`` in both training and inference once calibration
        has been chosen.
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("to_torch() requires PyTorch to be installed.") from exc

        values = (
            self.normalised_values(
                activation_ceiling=activation_ceiling,
                clip=clip,
            )
            if activation_ceiling is not None
            else self.values
        )
        tensor = torch.tensor(values, dtype=dtype or torch.float32, device=device)
        return tensor.unsqueeze(0) if batch else tensor


def schema_metadata() -> dict[str, Any]:
    """Metadata to save beside every dataset and checkpoint."""
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "emotion_count": EMOTION_COUNT,
        "emotion_names": list(EMOTION_NAMES),
    }
