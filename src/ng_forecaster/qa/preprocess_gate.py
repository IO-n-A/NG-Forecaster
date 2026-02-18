"""Preprocessing gate checker used before replay/model execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.artifact_schema import (
    ArtifactValidationError,
    validate_preprocess_artifact_bundle,
)

_REQUIRED_ARTIFACTS = (
    "preprocess_summary.json",
    "missing_value_flags.csv",
    "outlier_flags.csv",
)


@dataclass(frozen=True)
class PreprocessGateResult:
    """Outcome for preprocess gate checks."""

    passed: bool
    artifact_dir: Path
    summary_status: str
    checked_files: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "artifact_dir": str(self.artifact_dir),
            "summary_status": self.summary_status,
            "checked_files": list(self.checked_files),
        }


def resolve_latest_nowcast_artifact_dir(
    root: str | Path = "data/artifacts/nowcast",
) -> Path:
    """Return the latest nowcast artifact directory by lexicographic timestamp."""

    base = Path(root)
    if not base.exists():
        raise ContractViolation(
            "missing_preprocess_artifacts",
            key="artifact_root",
            detail=f"artifact root does not exist: {base}",
        )

    candidates = sorted(path for path in base.iterdir() if path.is_dir())
    if not candidates:
        raise ContractViolation(
            "missing_preprocess_artifacts",
            key="artifact_root",
            detail=f"no run directories found under {base}",
        )
    return candidates[-1]


def check_preprocess_gate(artifact_dir: str | Path) -> PreprocessGateResult:
    """Validate preprocess artifact presence, schema, and pass/fail summary."""

    base = Path(artifact_dir)
    missing = [name for name in _REQUIRED_ARTIFACTS if not (base / name).exists()]
    if missing:
        raise ContractViolation(
            "missing_preprocess_artifacts",
            key="artifact_bundle",
            detail="missing artifacts: " + ", ".join(sorted(missing)),
        )

    summary_path = base / "preprocess_summary.json"
    missing_path = base / "missing_value_flags.csv"
    outlier_path = base / "outlier_flags.csv"

    try:
        validate_preprocess_artifact_bundle(summary_path, missing_path, outlier_path)
    except ArtifactValidationError as exc:
        raise ContractViolation(
            "preprocess_schema_failed",
            key=exc.key,
            detail=str(exc),
        ) from exc

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    status = str(summary.get("status", "<missing>"))
    if status != "passed":
        raise ContractViolation(
            "preprocess_status_failed",
            key="status",
            detail=f"expected passed but found {status}",
        )

    return PreprocessGateResult(
        passed=True,
        artifact_dir=base,
        summary_status=status,
        checked_files=_REQUIRED_ARTIFACTS,
    )
