"""The 171-state contract shared with the Aurora repository.

Aurora produces the state and owns the schema; AuroraLM consumes it. Both
repositories carry the same versioned artifact,
``docs/contracts/aurora-state-171-effective-v4.json``, and the same pinned
values below, so a change made on one side only fails here or there. The
Aurora repository has a test of the same name that checks its own definitions
against the artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from auroralm import config, state_schema
from auroralm.state_schema import AuroraState171, StateSchemaError

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "contracts" / "aurora-state-171-effective-v4.json"

# Pinned in both repositories. Changing any of these is a schema change and
# is made in Aurora and AuroraLM in the same piece of work.
EXPECTED_VERSION = "aurora-state-171-effective/v4"
EXPECTED_COUNT = 171
EXPECTED_FINGERPRINT = (
    "d66d0993e77af71b3af245165c0f12acba1602b532619698464d35a0f7fd523a"
)
# SHA-256 of the artifact as canonical JSON (sorted keys, compact separators,
# ASCII), so line endings and indentation cannot change it.
EXPECTED_ARTIFACT_DIGEST = (
    "96427c689e48f890a482b6e5f415bc991e5327ce5b61ba737655f9f57549cbf1"
)
EXPECTED_SUPERSEDED = (
    "aurora-state-171/v1",
    "aurora-state-171x2/v2",
    "aurora-state-171-weighted/v3",
)


def _artifact() -> dict[str, Any]:
    loaded = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _fingerprint(names: list[str]) -> str:
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _record(**changes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": EXPECTED_VERSION,
        "schema_fingerprint": EXPECTED_FINGERPRINT,
        "captured_at": "2026-11-14T12:00:00+00:00",
        "source": "affect_pipeline",
        "values": [0.0] * EXPECTED_COUNT,
    }
    record.update(changes)
    return record


def test_artifact_is_the_published_version():
    canonical = json.dumps(
        _artifact(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == EXPECTED_ARTIFACT_DIGEST


def test_artifact_is_self_consistent():
    artifact = _artifact()
    names = artifact["emotion_names"]
    assert artifact["schema_version"] == EXPECTED_VERSION
    assert artifact["emotion_count"] == EXPECTED_COUNT
    assert len(names) == EXPECTED_COUNT
    assert len(set(names)) == EXPECTED_COUNT
    assert artifact["schema_fingerprint"] == EXPECTED_FINGERPRINT
    assert _fingerprint(names) == EXPECTED_FINGERPRINT
    assert tuple(artifact["superseded_versions"]) == EXPECTED_SUPERSEDED


def test_code_declares_the_canonical_version():
    assert state_schema.SCHEMA_VERSION == EXPECTED_VERSION
    assert config.STATE_SCHEMA_VERSION == EXPECTED_VERSION
    assert config.AuroraLMConfig().schema_version == EXPECTED_VERSION
    assert state_schema.SUPERSEDED_SCHEMA_VERSIONS == EXPECTED_SUPERSEDED


def test_code_matches_the_artifact_names_order_and_fingerprint():
    artifact = _artifact()
    assert list(state_schema.EMOTION_NAMES) == artifact["emotion_names"]
    assert config.STATE_DIMENSIONS == artifact["emotion_names"]
    assert state_schema.EMOTION_COUNT == EXPECTED_COUNT
    assert state_schema.SCHEMA_FINGERPRINT == EXPECTED_FINGERPRINT
    assert config.AuroraLMConfig().schema_fingerprint == EXPECTED_FINGERPRINT
    metadata = state_schema.schema_metadata()
    assert metadata["schema_version"] == EXPECTED_VERSION
    assert metadata["emotion_names"] == artifact["emotion_names"]


def test_a_current_record_round_trips():
    values = [0.0] * EXPECTED_COUNT
    # An effective value may exceed one: thalamic affinity can push the
    # evidence above one and the contract sets no upper bound.
    values[0], values[1] = 0.375, 1.05
    state = AuroraState171.from_record(_record(values=values))
    assert state.schema_version == EXPECTED_VERSION
    assert state.source == "affect_pipeline"
    assert state.values[:2] == (0.375, 1.05)
    assert AuroraState171.from_record(state.to_record()) == state


def test_an_aurora_capture_payload_is_accepted():
    """Aurora's capture adds per-node detail beside the vector; it is ignored."""
    record = _record(
        active_nodes=[{"name": "afraid", "activation": 0.75, "weighted_value": 0.375}],
        raw_active_nodes=[],
        mood_nodes=[],
        effective_nodes=[],
    )
    assert AuroraState171.from_record(record).active_count == 0


@pytest.mark.parametrize("version", EXPECTED_SUPERSEDED)
def test_superseded_versions_are_rejected(version):
    with pytest.raises(StateSchemaError, match="Expected schema"):
        AuroraState171.from_record(_record(schema_version=version))
    with pytest.raises(StateSchemaError, match="Expected schema"):
        AuroraState171(
            values=(0.0,) * EXPECTED_COUNT,
            captured_at="2026-11-14T12:00:00+00:00",
            schema_version=version,
        )


@pytest.mark.parametrize("channel", ["activations", "weights", "regulators"])
def test_superseded_channels_are_rejected_in_a_current_record(channel):
    record = _record(**{channel: [0.5] * EXPECTED_COUNT})
    with pytest.raises(StateSchemaError, match="Superseded state channels"):
        AuroraState171.from_record(record)


def test_a_reordered_vocabulary_is_rejected():
    names = list(state_schema.EMOTION_NAMES)
    names[0], names[1] = names[1], names[0]
    record = _record(schema_fingerprint=_fingerprint(names))
    with pytest.raises(StateSchemaError, match="fingerprint"):
        AuroraState171.from_record(record)


@pytest.mark.parametrize("bad", [True, -0.1, float("nan"), float("inf"), "high"])
def test_invalid_values_are_rejected(bad):
    values: list[Any] = [0.0] * EXPECTED_COUNT
    values[3] = bad
    with pytest.raises(StateSchemaError):
        AuroraState171.from_record(_record(values=values))


def test_wrong_length_is_rejected():
    with pytest.raises(StateSchemaError):
        AuroraState171.from_record(_record(values=[0.0] * (EXPECTED_COUNT - 1)))
