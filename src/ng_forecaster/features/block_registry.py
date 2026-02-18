"""Feature-block registry contracts used by CP1 block-tagged runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation

_DEFAULT_RULE = "available_timestamp_lte_asof"
_REQUIRED_FIELDS = ("asof_rule", "max_staleness_days")


@dataclass(frozen=True)
class FeatureBlock:
    """Single feature block definition."""

    block_id: str
    enabled: bool
    asof_rule: str
    max_staleness_days: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class FeatureBlockRegistry:
    """Normalized feature-block registry."""

    version: int
    defaults: dict[str, Any]
    blocks: tuple[FeatureBlock, ...]
    feature_to_block: dict[str, str]

    def enabled_blocks(self) -> tuple[FeatureBlock, ...]:
        """Return enabled blocks only."""

        return tuple(block for block in self.blocks if block.enabled)

    def block_for_feature(self, feature_name: str) -> str | None:
        """Return block id for a feature, or ``None`` when unregistered."""

        return self.feature_to_block.get(str(feature_name))


def _as_positive_int(value: object, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolation(
            "invalid_feature_block_registry",
            key=key,
            detail="value must be an integer >= 0",
        )
    return int(value)


def _as_non_empty_string(value: object, *, key: str) -> str:
    token = str(value).strip()
    if not token:
        raise ContractViolation(
            "invalid_feature_block_registry",
            key=key,
            detail="value must be a non-empty string",
        )
    return token


def validate_feature_block_registry(payload: Mapping[str, Any]) -> FeatureBlockRegistry:
    """Validate and normalize block registry payload."""

    version = payload.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ContractViolation(
            "invalid_feature_block_registry",
            key="version",
            detail="version must be integer >= 1",
        )

    raw_defaults = payload.get("defaults", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, Mapping):
        raise ContractViolation(
            "invalid_feature_block_registry",
            key="defaults",
            detail="defaults must be a mapping",
        )
    defaults: dict[str, Any] = dict(raw_defaults)
    defaults["asof_rule"] = _as_non_empty_string(
        defaults.get("asof_rule", _DEFAULT_RULE),
        key="defaults.asof_rule",
    )
    defaults["max_staleness_days"] = _as_positive_int(
        defaults.get("max_staleness_days", 180),
        key="defaults.max_staleness_days",
    )

    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, Mapping) or not raw_blocks:
        raise ContractViolation(
            "invalid_feature_block_registry",
            key="blocks",
            detail="blocks must be a non-empty mapping",
        )

    blocks: list[FeatureBlock] = []
    feature_to_block: dict[str, str] = {}
    for block_id, raw in sorted(raw_blocks.items()):
        block_key = f"blocks.{block_id}"
        if not isinstance(raw, Mapping):
            raise ContractViolation(
                "invalid_feature_block_registry",
                key=block_key,
                detail="block payload must be a mapping",
            )

        enabled = bool(raw.get("enabled", True))
        asof_rule = _as_non_empty_string(
            raw.get("asof_rule", defaults["asof_rule"]),
            key=f"{block_key}.asof_rule",
        )
        max_staleness_days = _as_positive_int(
            raw.get("max_staleness_days", defaults["max_staleness_days"]),
            key=f"{block_key}.max_staleness_days",
        )
        features_raw = raw.get("features", [])
        if not isinstance(features_raw, list) or not features_raw:
            raise ContractViolation(
                "invalid_feature_block_registry",
                key=f"{block_key}.features",
                detail="features must be a non-empty list",
            )

        normalized_features: list[str] = []
        for idx, item in enumerate(features_raw):
            feature_name = _as_non_empty_string(
                item,
                key=f"{block_key}.features[{idx}]",
            )
            normalized_features.append(feature_name)
            if feature_name in feature_to_block:
                raise ContractViolation(
                    "invalid_feature_block_registry",
                    key=f"{block_key}.features",
                    detail=(
                        f"feature '{feature_name}' is assigned to multiple blocks: "
                        f"{feature_to_block[feature_name]} and {block_id}"
                    ),
                )
            feature_to_block[feature_name] = str(block_id)

        blocks.append(
            FeatureBlock(
                block_id=str(block_id),
                enabled=enabled,
                asof_rule=asof_rule,
                max_staleness_days=max_staleness_days,
                features=tuple(normalized_features),
            )
        )

    return FeatureBlockRegistry(
        version=int(version),
        defaults=defaults,
        blocks=tuple(blocks),
        feature_to_block=feature_to_block,
    )


def load_feature_block_registry(
    path: str | Path = "configs/feature_blocks.yaml",
) -> FeatureBlockRegistry:
    """Load and validate feature block registry YAML."""

    payload = load_yaml(path)
    return validate_feature_block_registry(payload)


def enabled_feature_policies(
    registry: FeatureBlockRegistry,
) -> dict[str, dict[str, Any]]:
    """Return per-feature metadata for enabled blocks."""

    policies: dict[str, dict[str, Any]] = {}
    for block in registry.enabled_blocks():
        for feature_name in block.features:
            policies[feature_name] = {
                "block_id": block.block_id,
                "asof_rule": block.asof_rule,
                "max_staleness_days": int(block.max_staleness_days),
            }
    return policies


def required_block_metadata_keys() -> tuple[str, ...]:
    """Return the required policy metadata keys for enabled feature policies."""

    return _REQUIRED_FIELDS
