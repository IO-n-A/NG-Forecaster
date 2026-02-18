"""QA and integration governance helpers for Sprint 1."""

from ng_forecaster.qa.artifact_schema import (
    ArtifactValidationError,
    validate_preprocess_artifact_bundle,
)
from ng_forecaster.qa.handoff_validator import (
    HandoffValidationResult,
    validate_handoff_evidence,
)
from ng_forecaster.qa.n4_calibration_gate import N4GateResult, check_n4_acceptance
from ng_forecaster.qa.n5_policy_audit import N5AuditResult, audit_n5_policy
from ng_forecaster.qa.n6_adoption_gate import (
    N6AdoptionResult,
    check_n6_adoption_readiness,
)
from ng_forecaster.qa.ops_readiness import OpsReadinessResult, check_ops_readiness
from ng_forecaster.qa.preprocess_gate import PreprocessGateResult, check_preprocess_gate
from ng_forecaster.qa.target_month_gate import (
    TargetMonthGateResult,
    check_target_month_gate,
)

__all__ = [
    "ArtifactValidationError",
    "HandoffValidationResult",
    "N4GateResult",
    "N5AuditResult",
    "N6AdoptionResult",
    "OpsReadinessResult",
    "PreprocessGateResult",
    "TargetMonthGateResult",
    "audit_n5_policy",
    "check_n4_acceptance",
    "check_n6_adoption_readiness",
    "check_ops_readiness",
    "check_preprocess_gate",
    "check_target_month_gate",
    "validate_handoff_evidence",
    "validate_preprocess_artifact_bundle",
]
