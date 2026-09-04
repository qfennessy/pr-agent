"""Provider-neutral contracts for checkpoint review evaluation.

The contracts in this module deliberately contain no model, provider, retry, or
publication implementation.  They describe immutable inputs and outputs so the
production review cascade can be replayed later without giving benchmark truth to
the model-visible path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from pr_agent.algo.review_snapshot import ReviewEvent, ReviewResultState, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails, SpecialistRunDetails

EVALUATION_SCHEMA_VERSION = "checkpoint-evaluation-v2"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_FAILURE_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STAGE_FAILURE_REASON_CODE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_STAGE_STATE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STAGE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SPECIALIST_STAGE_STATES = frozenset({
    "aggregate_budget_exhausted",
    "cached",
    "cancelled",
    "disabled",
    "input_budget_exhausted",
    "low_confidence",
    "malformed_output",
    "not_required",
    "partial",
    "provider_failure",
    "stale",
    "success",
    "timeout",
    "unavailable",
})
_COMPLETE_SPECIALIST_STAGE_STATES = frozenset({"cached", "not_required", "success"})
_PARTIAL_SPECIALIST_STAGE_STATES = frozenset({"partial"})
_ANSWER_ONLY_KEYS = frozenset({
    "adjudication",
    "adjudication_hash",
    "answer",
    "clean_control",
    "earliest_opportunity",
    "expected_withdrawn_fingerprints",
    "expected_finding",
    "expected_findings",
    "finding_truth",
    "ground_truth",
    "is_clean",
    "label",
    "required_context",
    "severity",
    "truth",
    "verdict",
})
_MODEL_VISIBLE_METADATA_KEYS = frozenset({
    "change_size",
    "language",
    "repository_context_hash",
    "stage",
    "subsystem",
    "task_intent_hash",
})
_MODEL_VISIBLE_HASH_KEYS = frozenset({"repository_context_hash", "task_intent_hash"})
_MODEL_VISIBLE_SLUG = re.compile(r"^[a-z][a-z0-9_.+#-]{0,63}$")


class EvaluationValidationError(ValueError):
    """Raised when an evaluation artifact is incomplete or unsafe."""


class EvaluationCohort(str, Enum):
    CALIBRATION = "calibration"
    THRESHOLD = "threshold"
    TEMPORAL = "temporal"
    HOLDOUT = "holdout"
    CLEAN_CONTROL = "clean_control"


class EvaluationArmKind(str, Enum):
    DETERMINISTIC = "deterministic"
    GENERAL_REVIEW = "general_review"
    SPECIALISTS = "specialists"
    VERIFIED_SPECIALISTS = "verified_specialists"
    FULL_CASCADE = "full_cascade"


class EvaluationRunState(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PROVIDER_FAILURE = "provider_failure"
    COVERAGE_UNAVAILABLE = "coverage_unavailable"
    CANCELLED = "cancelled"
    STALE = "stale"


class MeasurementStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUABLE = "not_evaluable"


class FindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingLifecycleState(str, Enum):
    ACTIVE = "active"
    WITHDRAWN = "withdrawn"


_EVALUATION_SCHEMA_DESCRIPTOR = {
    "version": EVALUATION_SCHEMA_VERSION,
    "artifacts": {
        "arm": (
            "arm_id", "kind", "configuration_hash", "prompt_hash", "model_id", "provider_id",
            "model_revision", "fallback_models", "stage_plan", "enabled",
        ),
        "model_identity": ("model_id", "provider_id", "model_revision"),
        "stage_model_identity": (
            "model_id", "provider_id", "model_revision", "deployment_id_hash",
        ),
        "stage_plan": (
            "stage", "model_route", "configuration_hash", "prompt_hash", "prompt_version",
            "input_schema_version", "output_schema_version",
        ),
        "checkpoint_case": (
            "case_id", "snapshot_id", "snapshot_artifact_hash", "event", "cohort", "parent_case_id",
            "lineage_elapsed_seconds", "developer_elapsed_seconds", "model_visible_metadata",
        ),
        "finding_truth": (
            "finding_id", "fingerprint", "severity", "earliest_opportunity", "required_context",
            "earliest_case_id", "withdrawn_at_case_id",
        ),
        "checkpoint_truth": ("case_id", "is_clean", "adjudication_hash", "findings"),
        "manifest": (
            "schema_version", "schema_hash", "name", "corpus_hash", "policy_hash",
            "configuration_hash", "cases", "arms", "manifest_id",
        ),
        "truth_artifact": ("schema_version", "manifest_id", "truths", "truth_artifact_id"),
        "measurement": ("status", "value"),
        "stage_run": (
            "stage", "state", "coverage_status", "model_id", "provider_id", "model_revision",
            "deployment_id_hash", "fallback_used", "configuration_hash", "prompt_hash",
            "prompt_version", "input_schema_version", "output_schema_version", "latency_seconds",
            "tokens", "cost_usd", "cost_by_model_usd", "ai_call_count", "cached", "confidence",
            "failure_reason_code",
        ),
        "observed_finding": (
            "fingerprint", "severity", "lifecycle_state", "deterministic_overlap", "stage",
        ),
        "run_record": (
            "schema_version", "manifest_id", "case_id", "arm_id", "snapshot_id", "attempt", "state",
            "terminal", "findings", "snapshot_result_state", "latency_seconds", "tokens", "cost_usd",
            "retry_count", "cached", "escalated", "stage_latencies_seconds", "model_id", "provider_id",
            "model_revision", "stage_runs", "failure_reason_code", "record_id",
        ),
        "gate_rule": ("metric", "comparator", "threshold", "minimum_support"),
        "score_metric": ("status", "value", "support"),
        "arm_scorecard": (
            "schema_version", "arm_id", "case_count", "attempt_count", "failed_attempt_count",
            "completed_case_count", "duplicate_finding_count", "true_positive_count",
            "false_positive_count", "false_negative_count", "clean_checkpoint_count",
            "false_interruption_count", "escalated_case_count", "high_critical_case_count",
            "high_critical_escalated_count", "stale_finding_count", "deterministic_overlap_count",
            "metrics", "cohort_metrics",
        ),
        "paired_comparison": (
            "baseline_arm_id", "arm_id", "metric", "support", "delta", "lower_95", "upper_95",
        ),
        "matched_scorecard": (
            "schema_version", "manifest_id", "truth_artifact_id", "arms", "paired_comparisons",
            "scorecard_id",
        ),
        "gate_rule_result": ("rule", "status", "observed", "reason"),
        "gate_decision": (
            "schema_version", "gate_name", "arm_id", "scorecard_id", "gate_spec_hash", "status",
            "rule_results", "decision_id",
        ),
    },
    "enums": {
        "arm_kind": tuple(item.value for item in EvaluationArmKind),
        "cohort": tuple(item.value for item in EvaluationCohort),
        "run_state": tuple(item.value for item in EvaluationRunState),
        "measurement_status": tuple(item.value for item in MeasurementStatus),
        "gate_status": tuple(item.value for item in GateStatus),
        "severity": tuple(item.value for item in FindingSeverity),
        "finding_lifecycle": tuple(item.value for item in FindingLifecycleState),
        "review_event": tuple(item.value for item in ReviewEvent),
        "snapshot_result_state": tuple(item.value for item in ReviewResultState),
        "gate_comparator": ("at_least", "at_most"),
    },
    "model_visible_metadata_keys": tuple(sorted(_MODEL_VISIBLE_METADATA_KEYS)),
    "model_visible_metadata_constraints": {
        "hash_keys": tuple(sorted(_MODEL_VISIBLE_HASH_KEYS)),
        "hash_pattern": _SHA256_PATTERN.pattern,
        "identifier_pattern": _MODEL_VISIBLE_SLUG.pattern,
        "change_size_range": (0, 1_000_000),
        "stage_values": tuple(item.value for item in ReviewEvent),
    },
    "answer_only_keys": tuple(sorted(_ANSWER_ONLY_KEYS)),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    """Return a stable sha256 identity for a JSON-compatible value."""
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def deployment_identity_hash(deployment_id: Optional[str]) -> Optional[str]:
    """Return a source-free identity for a configured production deployment."""
    if deployment_id is None:
        return None
    if not isinstance(deployment_id, str) or not deployment_id.strip():
        raise EvaluationValidationError("deployment_id must be a non-empty string or null")
    return content_hash({"deployment_id": deployment_id})


def evaluation_schema_hash() -> str:
    """Hash the concrete serialized contract, not just its version label."""
    return content_hash(_EVALUATION_SCHEMA_DESCRIPTOR)


def _validate_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise EvaluationValidationError(f"{name} must be a sha256:<64 lowercase hex> identity")


def _validate_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationValidationError(f"{name} must be a non-empty string")


def _reject_unknown_fields(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvaluationValidationError(f"{name} contains unknown fields: {unknown}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


def _copy_json_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{name} must be a JSON object")
    try:
        copied = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError(f"{name} must contain only JSON-compatible values") from exc
    return _freeze_json(copied)


def _answer_only_paths(value: Any, path: str = "model_visible_metadata") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child_path = f"{path}.{key}"
            if normalized in _ANSWER_ONLY_KEYS:
                paths.append(child_path)
            paths.extend(_answer_only_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_answer_only_paths(child, f"{path}[{index}]"))
    return paths


def _validate_model_visible_metadata(value: Mapping[str, Any]) -> None:
    unknown = sorted(set(value) - _MODEL_VISIBLE_METADATA_KEYS)
    if unknown:
        raise EvaluationValidationError(f"model_visible_metadata contains unsupported fields: {unknown}")
    for key, child in value.items():
        if key in _MODEL_VISIBLE_HASH_KEYS:
            if not isinstance(child, str) or not _SHA256_PATTERN.fullmatch(child):
                raise EvaluationValidationError(
                    f"model_visible_metadata.{key} must be a sha256:<64 lowercase hex> identity"
                )
        elif key == "change_size":
            if not isinstance(child, int) or isinstance(child, bool) or not 0 <= child <= 1_000_000:
                raise EvaluationValidationError(
                    "model_visible_metadata.change_size must be an integer from 0 to 1000000"
                )
        elif key == "stage":
            if child not in {event.value for event in ReviewEvent}:
                raise EvaluationValidationError("model_visible_metadata.stage must be a ReviewEvent value")
        elif not isinstance(child, str) or not _MODEL_VISIBLE_SLUG.fullmatch(child):
            raise EvaluationValidationError(
                f"model_visible_metadata.{key} must be a bounded lowercase identifier"
            )


@dataclass(frozen=True)
class EvaluationModelIdentity:
    """One provider/model/revision triple that production fallback may select."""

    model_id: str
    provider_id: str
    model_revision: str

    def __post_init__(self) -> None:
        _validate_identifier("fallback model_id", self.model_id)
        _validate_identifier("fallback provider_id", self.provider_id)
        _validate_identifier("fallback model_revision", self.model_revision)

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationModelIdentity":
        _reject_unknown_fields(
            "evaluation model identity",
            value,
            {"model_id", "provider_id", "model_revision"},
        )
        return cls(
            model_id=value["model_id"],
            provider_id=value["provider_id"],
            model_revision=value["model_revision"],
        )


@dataclass(frozen=True)
class EvaluationStageModelIdentity:
    """One exact primary or fallback deployment allowed for a required stage."""

    model_id: str
    provider_id: str
    model_revision: str
    deployment_id_hash: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_identifier("stage model_id", self.model_id)
        _validate_identifier("stage provider_id", self.provider_id)
        _validate_identifier("stage model_revision", self.model_revision)
        if self.deployment_id_hash is not None:
            _validate_hash("stage deployment_id_hash", self.deployment_id_hash)

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
            "deployment_id_hash": self.deployment_id_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationStageModelIdentity":
        _reject_unknown_fields(
            "evaluation stage model identity",
            value,
            {"model_id", "provider_id", "model_revision", "deployment_id_hash"},
        )
        return cls(
            model_id=value["model_id"],
            provider_id=value["provider_id"],
            model_revision=value["model_revision"],
            deployment_id_hash=value.get("deployment_id_hash"),
        )


@dataclass(frozen=True)
class EvaluationStagePlan:
    """Immutable production contract for one required stage and its fallback route."""

    stage: str
    model_route: tuple[EvaluationStageModelIdentity, ...]
    configuration_hash: str
    prompt_hash: str
    prompt_version: str
    input_schema_version: str
    output_schema_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not _STAGE_STATE_PATTERN.fullmatch(self.stage):
            raise EvaluationValidationError("stage plan name must be a bounded machine-readable identifier")
        object.__setattr__(self, "model_route", tuple(self.model_route))
        if not self.model_route or any(
            not isinstance(identity, EvaluationStageModelIdentity) for identity in self.model_route
        ):
            raise EvaluationValidationError("stage plan model_route requires stage model identities")
        model_ids = [identity.model_id for identity in self.model_route]
        if len(model_ids) != len(set(model_ids)):
            raise EvaluationValidationError("stage plan primary and fallback model ids must be unique")
        _validate_hash("stage plan configuration_hash", self.configuration_hash)
        _validate_hash("stage plan prompt_hash", self.prompt_hash)
        for name in ("prompt_version", "input_schema_version", "output_schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _STAGE_VERSION_PATTERN.fullmatch(value):
                raise EvaluationValidationError(f"stage plan {name} must be a bounded version identifier")

    def resolve_model(self, model_id: str) -> tuple[int, EvaluationStageModelIdentity]:
        matches = [
            (index, identity)
            for index, identity in enumerate(self.model_route)
            if identity.model_id == model_id
        ]
        if len(matches) != 1:
            raise EvaluationValidationError(
                f"stage {self.stage} selected an unpinned model identity: {model_id}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model_route": [identity.to_dict() for identity in self.model_route],
            "configuration_hash": self.configuration_hash,
            "prompt_hash": self.prompt_hash,
            "prompt_version": self.prompt_version,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationStagePlan":
        _reject_unknown_fields(
            "evaluation stage plan",
            value,
            {
                "stage", "model_route", "configuration_hash", "prompt_hash", "prompt_version",
                "input_schema_version", "output_schema_version",
            },
        )
        return cls(
            stage=value["stage"],
            model_route=tuple(
                EvaluationStageModelIdentity.from_dict(identity)
                for identity in value["model_route"]
            ),
            configuration_hash=value["configuration_hash"],
            prompt_hash=value["prompt_hash"],
            prompt_version=value["prompt_version"],
            input_schema_version=value["input_schema_version"],
            output_schema_version=value["output_schema_version"],
        )


@dataclass(frozen=True)
class EvaluationArm:
    """One production-backed arm in a paired evaluation."""

    arm_id: str
    kind: EvaluationArmKind
    configuration_hash: str
    prompt_hash: str
    model_id: Optional[str] = None
    provider_id: Optional[str] = None
    model_revision: Optional[str] = None
    fallback_models: tuple[EvaluationModelIdentity, ...] = field(default_factory=tuple)
    stage_plan: tuple[EvaluationStagePlan, ...] = field(default_factory=tuple)
    enabled: bool = True

    def __post_init__(self) -> None:
        _validate_identifier("arm_id", self.arm_id)
        if not isinstance(self.kind, EvaluationArmKind):
            raise EvaluationValidationError("kind must be an EvaluationArmKind")
        _validate_hash("configuration_hash", self.configuration_hash)
        _validate_hash("prompt_hash", self.prompt_hash)
        if not isinstance(self.enabled, bool):
            raise EvaluationValidationError("evaluation arm enabled must be a boolean")
        object.__setattr__(self, "fallback_models", tuple(self.fallback_models))
        if any(not isinstance(identity, EvaluationModelIdentity) for identity in self.fallback_models):
            raise EvaluationValidationError("fallback_models must use EvaluationModelIdentity")
        object.__setattr__(self, "stage_plan", tuple(self.stage_plan))
        if any(not isinstance(stage, EvaluationStagePlan) for stage in self.stage_plan):
            raise EvaluationValidationError("stage_plan must use EvaluationStagePlan")
        stage_names = [stage.stage for stage in self.stage_plan]
        if len(stage_names) != len(set(stage_names)):
            raise EvaluationValidationError("stage_plan must contain unique required stage names")
        if self.kind is EvaluationArmKind.DETERMINISTIC:
            if (
                self.model_id is not None
                or self.provider_id is not None
                or self.model_revision is not None
                or self.fallback_models
                or self.stage_plan
            ):
                raise EvaluationValidationError(
                    "the deterministic arm cannot name a model, provider, revision, fallback, or stage"
                )
        elif not self.model_id or not self.provider_id:
            raise EvaluationValidationError("model-backed arms require immutable model_id and provider_id values")
        else:
            _validate_identifier("model_id", self.model_id)
            _validate_identifier("provider_id", self.provider_id)
            if self.model_revision is not None:
                _validate_identifier("model_revision", self.model_revision)
            model_ids = [self.model_id, *(identity.model_id for identity in self.fallback_models)]
            if len(model_ids) != len(set(model_ids)):
                raise EvaluationValidationError("primary and fallback model ids must be unique within an arm")
        stage_backed = self.kind in {
            EvaluationArmKind.SPECIALISTS,
            EvaluationArmKind.VERIFIED_SPECIALISTS,
            EvaluationArmKind.FULL_CASCADE,
        }
        if stage_backed and not self.stage_plan:
            raise EvaluationValidationError("specialist-backed arms require an immutable nonempty stage_plan")
        if not stage_backed and self.stage_plan:
            raise EvaluationValidationError("deterministic and general-review arms cannot define a stage_plan")

    def required_stage(self, stage: str) -> EvaluationStagePlan:
        matches = [item for item in self.stage_plan if item.stage == stage]
        if len(matches) != 1:
            raise EvaluationValidationError(f"run emitted an unexpected stage: {stage}")
        return matches[0]

    def model_identities(self) -> tuple[tuple[Optional[str], Optional[str], Optional[str]], ...]:
        if self.kind is EvaluationArmKind.DETERMINISTIC:
            return ((None, None, None),)
        identities = (
            (self.model_id, self.provider_id, self.model_revision),
            *(
                (identity.model_id, identity.provider_id, identity.model_revision)
                for identity in self.fallback_models
            ),
            *(
                (identity.model_id, identity.provider_id, identity.model_revision)
                for stage in self.stage_plan
                for identity in stage.model_route
            ),
        )
        return tuple(dict.fromkeys(identities))

    def aggregate_model_identities(self) -> tuple[tuple[Optional[str], Optional[str], Optional[str]], ...]:
        if self.kind is EvaluationArmKind.DETERMINISTIC:
            return ((None, None, None),)
        return (
            (self.model_id, self.provider_id, self.model_revision),
            *(
                (identity.model_id, identity.provider_id, identity.model_revision)
                for identity in self.fallback_models
            ),
        )

    def accepts_run_identity(
        self,
        model_id: Optional[str],
        provider_id: Optional[str],
        model_revision: Optional[str],
    ) -> bool:
        return (model_id, provider_id, model_revision) in self.aggregate_model_identities()

    def resolve_model_identity(self, model_id: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        matches = [identity for identity in self.aggregate_model_identities() if identity[0] == model_id]
        if len(matches) != 1:
            raise EvaluationValidationError(f"run selected an unpinned model identity: {model_id}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "kind": self.kind.value,
            "configuration_hash": self.configuration_hash,
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
            "fallback_models": [identity.to_dict() for identity in self.fallback_models],
            "stage_plan": [stage.to_dict() for stage in self.stage_plan],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationArm":
        _reject_unknown_fields(
            "evaluation arm",
            value,
            {
                "arm_id", "kind", "configuration_hash", "prompt_hash", "model_id", "provider_id",
                "model_revision", "fallback_models", "stage_plan", "enabled",
            },
        )
        return cls(
            arm_id=value["arm_id"],
            kind=EvaluationArmKind(value["kind"]),
            configuration_hash=value["configuration_hash"],
            prompt_hash=value["prompt_hash"],
            model_id=value.get("model_id"),
            provider_id=value.get("provider_id"),
            model_revision=value.get("model_revision"),
            fallback_models=tuple(
                EvaluationModelIdentity.from_dict(identity)
                for identity in value.get("fallback_models", [])
            ),
            stage_plan=tuple(
                EvaluationStagePlan.from_dict(stage)
                for stage in value.get("stage_plan", [])
            ),
            enabled=value.get("enabled", True),
        )


@dataclass(frozen=True)
class CheckpointCase:
    """A source-free reference to one serialized :class:`ReviewSnapshot`."""

    case_id: str
    snapshot_id: str
    snapshot_artifact_hash: str
    event: ReviewEvent
    cohort: EvaluationCohort
    parent_case_id: Optional[str] = None
    lineage_elapsed_seconds: Optional[float] = None
    developer_elapsed_seconds: Optional[float] = None
    model_visible_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("case_id", self.case_id)
        _validate_hash("snapshot_id", self.snapshot_id)
        _validate_hash("snapshot_artifact_hash", self.snapshot_artifact_hash)
        if not isinstance(self.event, ReviewEvent):
            raise EvaluationValidationError("event must be a ReviewEvent")
        if not isinstance(self.cohort, EvaluationCohort):
            raise EvaluationValidationError("cohort must be an EvaluationCohort")
        if self.parent_case_id is not None:
            _validate_identifier("parent_case_id", self.parent_case_id)
            if self.parent_case_id == self.case_id:
                raise EvaluationValidationError("a checkpoint cannot be its own parent")
        for field_name in ("lineage_elapsed_seconds", "developer_elapsed_seconds"):
            field_value = getattr(self, field_name)
            if field_value is not None:
                numeric = isinstance(field_value, (int, float)) and not isinstance(field_value, bool)
                if not numeric or not math.isfinite(field_value) or field_value < 0:
                    raise EvaluationValidationError(f"{field_name} must be a finite non-negative number")
        copied_metadata = _copy_json_mapping("model_visible_metadata", self.model_visible_metadata)
        leaked_paths = _answer_only_paths(copied_metadata)
        if leaked_paths:
            raise EvaluationValidationError(
                "answer-only fields are forbidden in model_visible_metadata: " + ", ".join(leaked_paths)
            )
        _validate_model_visible_metadata(copied_metadata)
        object.__setattr__(self, "model_visible_metadata", copied_metadata)

    def model_visible_payload(self) -> dict[str, Any]:
        """Return the only checkpoint fields permitted to reach production orchestration."""
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "snapshot_artifact_hash": self.snapshot_artifact_hash,
            "event": self.event.value,
            "metadata": _thaw_json(self.model_visible_metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_artifact_hash": self.snapshot_artifact_hash,
            "event": self.event.value,
            "cohort": self.cohort.value,
            "parent_case_id": self.parent_case_id,
            "lineage_elapsed_seconds": self.lineage_elapsed_seconds,
            "developer_elapsed_seconds": self.developer_elapsed_seconds,
            "model_visible_metadata": _thaw_json(self.model_visible_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointCase":
        _reject_unknown_fields(
            "checkpoint case",
            value,
            {
                "case_id", "snapshot_id", "snapshot_artifact_hash", "event", "cohort", "parent_case_id",
                "lineage_elapsed_seconds", "developer_elapsed_seconds", "model_visible_metadata",
            },
        )
        return cls(
            case_id=value["case_id"],
            snapshot_id=value["snapshot_id"],
            snapshot_artifact_hash=value["snapshot_artifact_hash"],
            event=ReviewEvent.parse(value["event"]),
            cohort=EvaluationCohort(value["cohort"]),
            parent_case_id=value.get("parent_case_id"),
            lineage_elapsed_seconds=value.get("lineage_elapsed_seconds"),
            developer_elapsed_seconds=value.get("developer_elapsed_seconds"),
            model_visible_metadata=value.get("model_visible_metadata", {}),
        )


@dataclass(frozen=True)
class FindingTruth:
    """Independently adjudicated, answer-only truth for one finding."""

    finding_id: str
    fingerprint: str
    severity: FindingSeverity
    earliest_opportunity: ReviewEvent
    required_context: tuple[str, ...]
    earliest_case_id: Optional[str] = None
    withdrawn_at_case_id: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_identifier("finding_id", self.finding_id)
        _validate_identifier("fingerprint", self.fingerprint)
        if not isinstance(self.severity, FindingSeverity):
            raise EvaluationValidationError("severity must be a FindingSeverity")
        if not isinstance(self.earliest_opportunity, ReviewEvent):
            raise EvaluationValidationError("earliest_opportunity must be a ReviewEvent")
        if isinstance(self.required_context, str) or not isinstance(self.required_context, (list, tuple)):
            raise EvaluationValidationError("required_context must be a list or tuple")
        normalized_context = tuple(dict.fromkeys(self.required_context))
        if not normalized_context or any(not isinstance(item, str) or not item.strip() for item in normalized_context):
            raise EvaluationValidationError("required_context must contain non-empty context identifiers")
        object.__setattr__(self, "required_context", normalized_context)
        if self.withdrawn_at_case_id is not None:
            _validate_identifier("withdrawn_at_case_id", self.withdrawn_at_case_id)
        if self.earliest_case_id is not None:
            _validate_identifier("earliest_case_id", self.earliest_case_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "fingerprint": self.fingerprint,
            "severity": self.severity.value,
            "earliest_opportunity": self.earliest_opportunity.value,
            "required_context": list(self.required_context),
            "earliest_case_id": self.earliest_case_id,
            "withdrawn_at_case_id": self.withdrawn_at_case_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FindingTruth":
        _reject_unknown_fields(
            "finding truth",
            value,
            {
                "finding_id", "fingerprint", "severity", "earliest_opportunity", "required_context",
                "earliest_case_id", "withdrawn_at_case_id",
            },
        )
        return cls(
            finding_id=value["finding_id"],
            fingerprint=value["fingerprint"],
            severity=FindingSeverity(value["severity"]),
            earliest_opportunity=ReviewEvent.parse(value["earliest_opportunity"]),
            required_context=tuple(value["required_context"]),
            earliest_case_id=value.get("earliest_case_id"),
            withdrawn_at_case_id=value.get("withdrawn_at_case_id"),
        )


@dataclass(frozen=True)
class CheckpointTruth:
    """Answer-only truth for one checkpoint, stored outside the manifest and plan."""

    case_id: str
    is_clean: bool
    adjudication_hash: str
    findings: tuple[FindingTruth, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_identifier("case_id", self.case_id)
        _validate_hash("adjudication_hash", self.adjudication_hash)
        if not isinstance(self.is_clean, bool):
            raise EvaluationValidationError("is_clean must be a boolean")
        object.__setattr__(self, "findings", tuple(self.findings))
        if any(not isinstance(finding, FindingTruth) for finding in self.findings):
            raise EvaluationValidationError("checkpoint truth findings must use FindingTruth")
        if self.is_clean and self.findings:
            raise EvaluationValidationError("a clean checkpoint cannot contain finding truth")
        if not self.is_clean and not self.findings:
            raise EvaluationValidationError("a non-clean checkpoint requires at least one finding truth")
        fingerprints = [finding.fingerprint for finding in self.findings]
        if len(fingerprints) != len(set(fingerprints)):
            raise EvaluationValidationError(f"checkpoint {self.case_id} contains duplicate truth fingerprints")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "is_clean": self.is_clean,
            "adjudication_hash": self.adjudication_hash,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointTruth":
        _reject_unknown_fields(
            "checkpoint truth",
            value,
            {"case_id", "is_clean", "adjudication_hash", "findings"},
        )
        return cls(
            case_id=value["case_id"],
            is_clean=value["is_clean"],
            adjudication_hash=value["adjudication_hash"],
            findings=tuple(FindingTruth.from_dict(item) for item in value.get("findings", [])),
        )


@dataclass(frozen=True)
class EvaluationManifest:
    """Immutable evaluation inventory; answer-only truth is intentionally absent."""

    name: str
    corpus_hash: str
    policy_hash: str
    configuration_hash: str
    cases: tuple[CheckpointCase, ...]
    arms: tuple[EvaluationArm, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION
    schema_hash: str = field(default_factory=evaluation_schema_hash)
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported evaluation schema_version: {self.schema_version}")
        _validate_hash("schema_hash", self.schema_hash)
        expected_schema_hash = evaluation_schema_hash()
        if self.schema_hash != expected_schema_hash:
            raise EvaluationValidationError("schema_hash does not match the evaluation schema version")
        _validate_identifier("name", self.name)
        _validate_hash("corpus_hash", self.corpus_hash)
        _validate_hash("policy_hash", self.policy_hash)
        _validate_hash("configuration_hash", self.configuration_hash)
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "arms", tuple(self.arms))
        if any(not isinstance(case, CheckpointCase) for case in self.cases):
            raise EvaluationValidationError("manifest cases must use CheckpointCase")
        if any(not isinstance(arm, EvaluationArm) for arm in self.arms):
            raise EvaluationValidationError("manifest arms must use EvaluationArm")
        if not self.cases:
            raise EvaluationValidationError("an evaluation manifest requires at least one checkpoint case")
        if not self.arms:
            raise EvaluationValidationError("an evaluation manifest requires at least one arm")
        self._validate_cases()
        self._validate_arms()
        object.__setattr__(self, "manifest_id", content_hash(self._identity_payload()))

    def _validate_cases(self) -> None:
        case_ids = [case.case_id for case in self.cases]
        snapshot_ids = [case.snapshot_id for case in self.cases]
        artifact_hashes = [case.snapshot_artifact_hash for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError("checkpoint case ids must be unique")
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise EvaluationValidationError("snapshot identities must be unique across checkpoint cases")
        if len(artifact_hashes) != len(set(artifact_hashes)):
            raise EvaluationValidationError("snapshot artifact hashes must be unique across checkpoint cases")
        cases_by_id = {case.case_id: case for case in self.cases}
        for case in self.cases:
            if case.parent_case_id is None:
                continue
            parent = cases_by_id.get(case.parent_case_id)
            if parent is None:
                raise EvaluationValidationError(f"checkpoint {case.case_id} names an unknown parent")
            if parent.cohort is not case.cohort:
                raise EvaluationValidationError("checkpoint lineages cannot cross evaluation cohorts")
            if (
                parent.lineage_elapsed_seconds is not None
                and case.lineage_elapsed_seconds is not None
                and case.lineage_elapsed_seconds < parent.lineage_elapsed_seconds
            ):
                raise EvaluationValidationError("checkpoint lineage time cannot move backwards")
        for case in self.cases:
            visited: set[str] = set()
            current = case
            while current.parent_case_id is not None:
                if current.case_id in visited:
                    raise EvaluationValidationError("checkpoint parent relationships contain a cycle")
                visited.add(current.case_id)
                current = cases_by_id[current.parent_case_id]

    def _validate_arms(self) -> None:
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise EvaluationValidationError("evaluation arm ids must be unique")
        if not any(arm.enabled for arm in self.arms):
            raise EvaluationValidationError("an evaluation manifest requires at least one enabled arm")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "name": self.name,
            "corpus_hash": self.corpus_hash,
            "policy_hash": self.policy_hash,
            "configuration_hash": self.configuration_hash,
            "cases": [case.to_dict() for case in self.cases],
            "arms": [arm.to_dict() for arm in self.arms],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "manifest_id": self.manifest_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationManifest":
        _reject_unknown_fields(
            "evaluation manifest",
            value,
            {
                "schema_version", "schema_hash", "name", "corpus_hash", "policy_hash", "configuration_hash",
                "cases", "arms", "manifest_id",
            },
        )
        manifest = cls(
            name=value["name"],
            corpus_hash=value["corpus_hash"],
            policy_hash=value["policy_hash"],
            configuration_hash=value["configuration_hash"],
            cases=tuple(CheckpointCase.from_dict(item) for item in value["cases"]),
            arms=tuple(EvaluationArm.from_dict(item) for item in value["arms"]),
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
            schema_hash=value.get(
                "schema_hash",
                evaluation_schema_hash(),
            ),
        )
        supplied_manifest_id = value.get("manifest_id")
        if supplied_manifest_id is not None and supplied_manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("manifest_id does not match the manifest content")
        return manifest


@dataclass(frozen=True)
class TruthArtifact:
    """Separately loaded answer-only artifact for one immutable manifest."""

    manifest_id: str
    truths: tuple[CheckpointTruth, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION
    truth_artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported truth schema_version: {self.schema_version}")
        _validate_hash("manifest_id", self.manifest_id)
        object.__setattr__(self, "truths", tuple(self.truths))
        if any(not isinstance(truth, CheckpointTruth) for truth in self.truths):
            raise EvaluationValidationError("truths must use CheckpointTruth")
        case_ids = [truth.case_id for truth in self.truths]
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationValidationError("truth case ids must be unique")
        payload = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "truths": [truth.to_dict() for truth in self.truths],
        }
        object.__setattr__(self, "truth_artifact_id", content_hash(payload))

    def validate_for_manifest(self, manifest: EvaluationManifest) -> None:
        if self.manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("truth artifact belongs to a different manifest")
        manifest_cases = {case.case_id for case in manifest.cases}
        truth_cases = {truth.case_id for truth in self.truths}
        missing = sorted(manifest_cases - truth_cases)
        extra = sorted(truth_cases - manifest_cases)
        if missing or extra:
            raise EvaluationValidationError(f"truth cases do not match manifest; missing={missing}, extra={extra}")
        case_by_id = {case.case_id: case for case in manifest.cases}

        def is_descendant(candidate_id: str, ancestor_id: str) -> bool:
            current = case_by_id[candidate_id]
            while current.parent_case_id is not None:
                if current.parent_case_id == ancestor_id:
                    return True
                current = case_by_id[current.parent_case_id]
            return False

        def lineage_root(case_id: str) -> str:
            current = case_by_id[case_id]
            while current.parent_case_id is not None:
                current = case_by_id[current.parent_case_id]
            return current.case_id

        defect_lineages = {
            lineage_root(truth.case_id)
            for truth in self.truths
            if not truth.is_clean
        }

        for truth in self.truths:
            case = case_by_id[truth.case_id]
            if (
                truth.is_clean
                and case.cohort is not EvaluationCohort.CLEAN_CONTROL
                and lineage_root(truth.case_id) not in defect_lineages
            ):
                raise EvaluationValidationError(
                    "clean truth outside clean_control must share a lineage with defect truth"
                )
            if not truth.is_clean and case.cohort is EvaluationCohort.CLEAN_CONTROL:
                raise EvaluationValidationError("clean_control cohort cannot contain defect truth")
            for finding in truth.findings:
                earliest_case_id = finding.earliest_case_id
                if earliest_case_id is not None:
                    if earliest_case_id not in manifest_cases:
                        raise EvaluationValidationError("finding truth names an unknown earliest checkpoint")
                    if case_by_id[earliest_case_id].event is not finding.earliest_opportunity:
                        raise EvaluationValidationError(
                            "finding earliest checkpoint event does not match earliest_opportunity"
                        )
                    if earliest_case_id != truth.case_id and not is_descendant(truth.case_id, earliest_case_id):
                        raise EvaluationValidationError(
                            "finding earliest checkpoint must be in the same ancestor lineage"
                        )
                elif case.event is not finding.earliest_opportunity:
                    raise EvaluationValidationError(
                        "finding earliest_opportunity must match its checkpoint when earliest_case_id is omitted"
                    )
                if finding.withdrawn_at_case_id:
                    if finding.withdrawn_at_case_id not in manifest_cases:
                        raise EvaluationValidationError("finding truth names an unknown withdrawal checkpoint")
                    if not is_descendant(finding.withdrawn_at_case_id, truth.case_id):
                        raise EvaluationValidationError(
                            "finding withdrawal checkpoint must be a later lineage descendant"
                        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "truths": [truth.to_dict() for truth in self.truths],
        }
        return {**payload, "truth_artifact_id": self.truth_artifact_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TruthArtifact":
        _reject_unknown_fields(
            "truth artifact",
            value,
            {"schema_version", "manifest_id", "truths", "truth_artifact_id"},
        )
        artifact = cls(
            manifest_id=value["manifest_id"],
            truths=tuple(CheckpointTruth.from_dict(item) for item in value["truths"]),
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
        )
        supplied_id = value.get("truth_artifact_id")
        if supplied_id is not None and supplied_id != artifact.truth_artifact_id:
            raise EvaluationValidationError("truth_artifact_id does not match the truth content")
        return artifact


@dataclass(frozen=True)
class NumericMeasurement:
    """A numeric value that cannot confuse unavailable data with zero."""

    status: MeasurementStatus
    value: Optional[float]

    def __post_init__(self) -> None:
        if not isinstance(self.status, MeasurementStatus):
            raise EvaluationValidationError("measurement status must be a MeasurementStatus")
        if self.status is MeasurementStatus.UNAVAILABLE and self.value is not None:
            raise EvaluationValidationError("an unavailable measurement cannot have a value")
        if self.status is not MeasurementStatus.UNAVAILABLE and self.value is None:
            raise EvaluationValidationError("complete and partial measurements require a value")
        if self.value is not None:
            numeric_value = isinstance(self.value, (int, float)) and not isinstance(self.value, bool)
            if not numeric_value or not math.isfinite(self.value):
                raise EvaluationValidationError("measurement values must be finite numbers")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "value": self.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NumericMeasurement":
        _reject_unknown_fields("numeric measurement", value, {"status", "value"})
        return cls(status=MeasurementStatus(value["status"]), value=value.get("value"))


def _sum_measurements(measurements: Sequence[NumericMeasurement]) -> NumericMeasurement:
    """Sum known components without turning an absent component into zero."""
    measurements = tuple(measurements)
    known = tuple(item for item in measurements if item.status is not MeasurementStatus.UNAVAILABLE)
    if not known:
        return NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    status = (
        MeasurementStatus.COMPLETE
        if len(known) == len(measurements)
        and all(item.status is MeasurementStatus.COMPLETE for item in known)
        else MeasurementStatus.PARTIAL
    )
    return NumericMeasurement(status, sum(float(item.value) for item in known if item.value is not None))


@dataclass(frozen=True)
class EvaluationStageRun:
    """Source-free telemetry for one production model stage or specialist role."""

    stage: str
    state: str
    coverage_status: MeasurementStatus
    model_id: str
    provider_id: str
    model_revision: str
    deployment_id_hash: Optional[str]
    configuration_hash: str
    prompt_hash: str
    prompt_version: str
    input_schema_version: str
    output_schema_version: str
    latency_seconds: NumericMeasurement
    tokens: NumericMeasurement
    cost_usd: NumericMeasurement
    cost_by_model_usd: Mapping[str, float] = field(default_factory=dict)
    ai_call_count: int = 0
    cached: bool = False
    fallback_used: bool = False
    confidence: Optional[float] = None
    failure_reason_code: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("model_id", "provider_id", "model_revision"):
            _validate_identifier(f"stage run {name}", getattr(self, name))
        if self.deployment_id_hash is not None:
            _validate_hash("stage run deployment_id_hash", self.deployment_id_hash)
        _validate_hash("stage run configuration_hash", self.configuration_hash)
        _validate_hash("stage run prompt_hash", self.prompt_hash)
        if not isinstance(self.stage, str) or not _STAGE_STATE_PATTERN.fullmatch(self.stage):
            raise EvaluationValidationError("stage run stage must be a bounded machine-readable identifier")
        for name in ("prompt_version", "input_schema_version", "output_schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _STAGE_VERSION_PATTERN.fullmatch(value):
                raise EvaluationValidationError(f"stage run {name} must be a bounded version identifier")
        if not isinstance(self.state, str) or not _STAGE_STATE_PATTERN.fullmatch(self.state):
            raise EvaluationValidationError("stage run state must be a bounded machine-readable code")
        if self.state not in _SPECIALIST_STAGE_STATES:
            raise EvaluationValidationError(f"stage run state is unsupported: {self.state}")
        if not isinstance(self.coverage_status, MeasurementStatus):
            raise EvaluationValidationError("stage run coverage_status must be a MeasurementStatus")
        expected_coverage = (
            MeasurementStatus.COMPLETE
            if self.state in _COMPLETE_SPECIALIST_STAGE_STATES
            else (
                MeasurementStatus.PARTIAL
                if self.state in _PARTIAL_SPECIALIST_STAGE_STATES
                else MeasurementStatus.UNAVAILABLE
            )
        )
        if self.coverage_status is not expected_coverage:
            raise EvaluationValidationError("stage run coverage_status contradicts its state")
        for name, measurement in (
            ("latency_seconds", self.latency_seconds),
            ("tokens", self.tokens),
            ("cost_usd", self.cost_usd),
        ):
            if not isinstance(measurement, NumericMeasurement):
                raise EvaluationValidationError(f"stage run {name} must use NumericMeasurement")
            if measurement.value is not None and measurement.value <= 0:
                raise EvaluationValidationError(f"stage run known {name} must be positive")
        if self.tokens.value is not None and not float(self.tokens.value).is_integer():
            raise EvaluationValidationError("stage run tokens must be an integer count")
        if not isinstance(self.ai_call_count, int) or isinstance(self.ai_call_count, bool) or self.ai_call_count < 0:
            raise EvaluationValidationError("stage run ai_call_count must be a non-negative integer")
        if not isinstance(self.cached, bool) or not isinstance(self.fallback_used, bool):
            raise EvaluationValidationError("stage run cache and fallback flags must be booleans")
        if self.state == "cached" and not self.cached:
            raise EvaluationValidationError("a cached stage run must set cached")
        if self.state != "cached" and self.cached:
            raise EvaluationValidationError("only a cached stage run may set cached")
        if self.state == "success" and self.ai_call_count < 1:
            raise EvaluationValidationError("a successful stage run requires at least one observed AI call")
        if self.state == "cached" and self.ai_call_count:
            raise EvaluationValidationError("a cached stage run cannot claim a new AI call")
        if self.confidence is not None:
            if (
                not isinstance(self.confidence, (int, float))
                or isinstance(self.confidence, bool)
                or not math.isfinite(self.confidence)
                or not 0 <= self.confidence <= 1
            ):
                raise EvaluationValidationError("stage run confidence must be a finite value from 0 to 1")
        if self.state in _COMPLETE_SPECIALIST_STAGE_STATES:
            if self.failure_reason_code is not None:
                raise EvaluationValidationError("a covered stage run cannot have a failure_reason_code")
        elif (
            not isinstance(self.failure_reason_code, str)
            or not _STAGE_FAILURE_REASON_CODE_PATTERN.fullmatch(self.failure_reason_code)
        ):
            raise EvaluationValidationError("an uncovered stage run requires a bounded machine-readable reason code")
        if not isinstance(self.cost_by_model_usd, Mapping):
            raise EvaluationValidationError("stage run cost_by_model_usd must be a JSON object")
        normalized_costs: dict[str, float] = {}
        for model_id, value in self.cost_by_model_usd.items():
            _validate_identifier("stage run cost model_id", model_id)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise EvaluationValidationError("stage run per-model costs must be finite positive numbers")
            normalized_costs[model_id] = float(value)
        if self.cost_usd.status is MeasurementStatus.UNAVAILABLE and normalized_costs:
            raise EvaluationValidationError("an unavailable stage cost cannot contain per-model costs")
        if self.cost_usd.value is not None and not math.isclose(
            self.cost_usd.value,
            sum(normalized_costs.values()),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvaluationValidationError("stage run per-model costs must sum to cost_usd")
        object.__setattr__(self, "cost_by_model_usd", MappingProxyType(dict(sorted(normalized_costs.items()))))

    @classmethod
    def from_specialist_details(
        cls,
        arm: EvaluationArm,
        stage: str,
        details: SpecialistRunDetails,
    ) -> "EvaluationStageRun":
        """Normalize one mutable production role record against its frozen arm."""
        if not isinstance(details, SpecialistRunDetails):
            raise EvaluationValidationError("specialist stage telemetry must use SpecialistRunDetails")
        if details.role != stage:
            raise EvaluationValidationError("specialist stage key contradicts its role")
        if not details.model_used:
            raise EvaluationValidationError(f"specialist stage {stage} has no observed model identity")
        stage_plan = arm.required_stage(stage)
        route_index, stage_identity = stage_plan.resolve_model(details.model_used)
        observed_deployment_hash = deployment_identity_hash(details.deployment_id)
        if observed_deployment_hash != stage_identity.deployment_id_hash:
            raise EvaluationValidationError(f"specialist stage {stage} selected an unpinned deployment identity")
        if details.fallback_used is not (route_index > 0):
            raise EvaluationValidationError(f"specialist stage {stage} fallback telemetry contradicts its route")
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "num_ai_calls",
            "known_cost_call_count",
        ):
            value = getattr(details, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EvaluationValidationError(f"specialist stage {stage} has invalid {name}")
        if details.known_cost_call_count > details.num_ai_calls:
            raise EvaluationValidationError(f"specialist stage {stage} prices more calls than it observed")
        if details.num_ai_calls == 0 and (
            details.prompt_tokens or details.completion_tokens or details.total_tokens
        ):
            raise EvaluationValidationError(f"specialist stage {stage} has token usage without an AI call")
        if details.total_tokens == 0 and (details.prompt_tokens or details.completion_tokens):
            raise EvaluationValidationError(f"specialist stage {stage} has inconsistent token totals")
        token_measurement = (
            NumericMeasurement(MeasurementStatus.PARTIAL, float(details.total_tokens))
            if details.total_tokens > 0
            else NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
        )

        try:
            total_cost = float(details.total_cost_usd)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationValidationError(f"specialist stage {stage} has invalid cost telemetry") from exc
        if not math.isfinite(total_cost) or total_cost < 0:
            raise EvaluationValidationError(f"specialist stage {stage} has invalid cost telemetry")
        if not isinstance(details.model_costs_usd, Mapping):
            raise EvaluationValidationError(f"specialist stage {stage} has invalid per-model cost telemetry")
        normalized_costs: dict[str, float] = {}
        for cost_model_id, cost in details.model_costs_usd.items():
            stage_plan.resolve_model(cost_model_id)
            try:
                numeric_cost = float(cost)
            except (TypeError, ValueError, OverflowError) as exc:
                raise EvaluationValidationError(
                    f"specialist stage {stage} has invalid per-model cost telemetry"
                ) from exc
            if not math.isfinite(numeric_cost) or numeric_cost <= 0:
                raise EvaluationValidationError(f"specialist stage {stage} has invalid per-model cost telemetry")
            normalized_costs[cost_model_id] = numeric_cost
        if details.known_cost_call_count == 0:
            if total_cost != 0 or normalized_costs:
                raise EvaluationValidationError(f"specialist stage {stage} has cost without a priced call")
            cost_measurement = NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
        else:
            if total_cost <= 0 or not normalized_costs:
                raise EvaluationValidationError(f"specialist stage {stage} omits priced-call cost telemetry")
            if not math.isclose(total_cost, sum(normalized_costs.values()), rel_tol=1e-12, abs_tol=1e-12):
                raise EvaluationValidationError(f"specialist stage {stage} has inconsistent per-model costs")
            cost_measurement = NumericMeasurement(MeasurementStatus(details.cost_status), total_cost)

        latency = details.latency_seconds
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise EvaluationValidationError(f"specialist stage {stage} has invalid latency telemetry")
        latency_measurement = (
            NumericMeasurement(MeasurementStatus.COMPLETE, float(latency))
            if latency > 0
            else NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
        )
        state = details.state
        if not isinstance(state, str) or state not in _SPECIALIST_STAGE_STATES:
            raise EvaluationValidationError(f"specialist stage {stage} has unsupported state telemetry")
        coverage_status = (
            MeasurementStatus.COMPLETE
            if state in _COMPLETE_SPECIALIST_STAGE_STATES
            else (
                MeasurementStatus.PARTIAL
                if state in _PARTIAL_SPECIALIST_STAGE_STATES
                else MeasurementStatus.UNAVAILABLE
            )
        )
        for name in ("prompt_version", "input_schema_version", "schema_version"):
            value = getattr(details, name)
            if not isinstance(value, str) or not _STAGE_VERSION_PATTERN.fullmatch(value):
                raise EvaluationValidationError(f"specialist stage {stage} has invalid {name}")
        observed_versions = (
            details.prompt_version,
            details.input_schema_version,
            details.schema_version,
        )
        expected_versions = (
            stage_plan.prompt_version,
            stage_plan.input_schema_version,
            stage_plan.output_schema_version,
        )
        if observed_versions != expected_versions:
            raise EvaluationValidationError(f"specialist stage {stage} versions do not match its frozen plan")
        return cls(
            stage=stage,
            state=state,
            coverage_status=coverage_status,
            model_id=stage_identity.model_id,
            provider_id=stage_identity.provider_id,
            model_revision=stage_identity.model_revision,
            deployment_id_hash=stage_identity.deployment_id_hash,
            configuration_hash=stage_plan.configuration_hash,
            prompt_hash=stage_plan.prompt_hash,
            prompt_version=stage_plan.prompt_version,
            input_schema_version=stage_plan.input_schema_version,
            output_schema_version=stage_plan.output_schema_version,
            latency_seconds=latency_measurement,
            tokens=token_measurement,
            cost_usd=cost_measurement,
            cost_by_model_usd=normalized_costs,
            ai_call_count=details.num_ai_calls,
            cached=details.cached,
            fallback_used=details.fallback_used,
            confidence=details.confidence,
            failure_reason_code=details.failure_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "state": self.state,
            "coverage_status": self.coverage_status.value,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
            "deployment_id_hash": self.deployment_id_hash,
            "fallback_used": self.fallback_used,
            "configuration_hash": self.configuration_hash,
            "prompt_hash": self.prompt_hash,
            "prompt_version": self.prompt_version,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
            "latency_seconds": self.latency_seconds.to_dict(),
            "tokens": self.tokens.to_dict(),
            "cost_usd": self.cost_usd.to_dict(),
            "cost_by_model_usd": dict(self.cost_by_model_usd),
            "ai_call_count": self.ai_call_count,
            "cached": self.cached,
            "confidence": self.confidence,
            "failure_reason_code": self.failure_reason_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationStageRun":
        _reject_unknown_fields(
            "evaluation stage run",
            value,
            {
                "stage", "state", "coverage_status", "model_id", "provider_id", "model_revision",
                "deployment_id_hash", "fallback_used", "configuration_hash", "prompt_hash",
                "prompt_version", "input_schema_version", "output_schema_version", "latency_seconds",
                "tokens", "cost_usd", "cost_by_model_usd", "ai_call_count", "cached", "confidence",
                "failure_reason_code",
            },
        )
        return cls(
            stage=value["stage"],
            state=value["state"],
            coverage_status=MeasurementStatus(value["coverage_status"]),
            model_id=value["model_id"],
            provider_id=value["provider_id"],
            model_revision=value["model_revision"],
            deployment_id_hash=value.get("deployment_id_hash"),
            fallback_used=value["fallback_used"],
            configuration_hash=value["configuration_hash"],
            prompt_hash=value["prompt_hash"],
            prompt_version=value["prompt_version"],
            input_schema_version=value["input_schema_version"],
            output_schema_version=value["output_schema_version"],
            latency_seconds=NumericMeasurement.from_dict(value["latency_seconds"]),
            tokens=NumericMeasurement.from_dict(value["tokens"]),
            cost_usd=NumericMeasurement.from_dict(value["cost_usd"]),
            cost_by_model_usd=value.get("cost_by_model_usd", {}),
            ai_call_count=value["ai_call_count"],
            cached=value["cached"],
            confidence=value.get("confidence"),
            failure_reason_code=value.get("failure_reason_code"),
        )


@dataclass(frozen=True)
class ObservedFinding:
    fingerprint: str
    severity: FindingSeverity
    lifecycle_state: FindingLifecycleState = FindingLifecycleState.ACTIVE
    deterministic_overlap: Optional[bool] = None
    stage: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_identifier("fingerprint", self.fingerprint)
        if not isinstance(self.severity, FindingSeverity):
            raise EvaluationValidationError("observed finding severity must be a FindingSeverity")
        if not isinstance(self.lifecycle_state, FindingLifecycleState):
            raise EvaluationValidationError("observed finding lifecycle_state must be a FindingLifecycleState")
        if self.deterministic_overlap is not None and not isinstance(self.deterministic_overlap, bool):
            raise EvaluationValidationError("deterministic_overlap must be a boolean or null")
        if self.stage is not None:
            _validate_identifier("observed finding stage", self.stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "severity": self.severity.value,
            "lifecycle_state": self.lifecycle_state.value,
            "deterministic_overlap": self.deterministic_overlap,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservedFinding":
        _reject_unknown_fields(
            "observed finding",
            value,
            {"fingerprint", "severity", "lifecycle_state", "deterministic_overlap", "stage"},
        )
        return cls(
            fingerprint=value["fingerprint"],
            severity=FindingSeverity(value["severity"]),
            lifecycle_state=FindingLifecycleState(value.get("lifecycle_state", FindingLifecycleState.ACTIVE.value)),
            deterministic_overlap=value.get("deterministic_overlap"),
            stage=value.get("stage"),
        )


@dataclass(frozen=True)
class EvaluationRunRecord:
    """One immutable attempt for one manifest/case/arm tuple."""

    manifest_id: str
    case_id: str
    arm_id: str
    snapshot_id: str
    attempt: int
    state: EvaluationRunState
    terminal: bool
    findings: tuple[ObservedFinding, ...] = field(default_factory=tuple)
    snapshot_result_state: Optional[ReviewResultState] = None
    latency_seconds: NumericMeasurement = field(
        default_factory=lambda: NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    )
    tokens: NumericMeasurement = field(
        default_factory=lambda: NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    )
    cost_usd: NumericMeasurement = field(
        default_factory=lambda: NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
    )
    retry_count: int = 0
    cached: bool = False
    escalated: Optional[bool] = None
    stage_latencies_seconds: Mapping[str, NumericMeasurement] = field(default_factory=dict)
    model_id: Optional[str] = None
    provider_id: Optional[str] = None
    model_revision: Optional[str] = None
    stage_runs: tuple[EvaluationStageRun, ...] = field(default_factory=tuple)
    failure_reason_code: Optional[str] = None
    schema_version: str = EVALUATION_SCHEMA_VERSION
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported run schema_version: {self.schema_version}")
        _validate_hash("manifest_id", self.manifest_id)
        _validate_hash("snapshot_id", self.snapshot_id)
        _validate_identifier("case_id", self.case_id)
        _validate_identifier("arm_id", self.arm_id)
        if not isinstance(self.state, EvaluationRunState):
            raise EvaluationValidationError("state must be an EvaluationRunState")
        if not isinstance(self.attempt, int) or self.attempt < 1:
            raise EvaluationValidationError("attempt must be a positive integer")
        if not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise EvaluationValidationError("retry_count must be a non-negative integer")
        if not isinstance(self.terminal, bool):
            raise EvaluationValidationError("terminal must be a boolean")
        if not isinstance(self.cached, bool):
            raise EvaluationValidationError("cached must be a boolean")
        if self.escalated is not None and not isinstance(self.escalated, bool):
            raise EvaluationValidationError("escalated must be a boolean or null")
        if self.snapshot_result_state is not None and not isinstance(self.snapshot_result_state, ReviewResultState):
            raise EvaluationValidationError("snapshot_result_state must be a ReviewResultState or null")
        expected_state_by_snapshot_result = {
            ReviewResultState.FINDINGS: EvaluationRunState.COMPLETED,
            ReviewResultState.NO_FINDINGS: EvaluationRunState.COMPLETED,
            ReviewResultState.COVERAGE_UNAVAILABLE: EvaluationRunState.COVERAGE_UNAVAILABLE,
            ReviewResultState.CANCELLED: EvaluationRunState.CANCELLED,
            ReviewResultState.STALE: EvaluationRunState.STALE,
        }
        if (
            self.snapshot_result_state is not None
            and self.state is not expected_state_by_snapshot_result[self.snapshot_result_state]
        ):
            raise EvaluationValidationError("run state contradicts snapshot_result_state")
        for name, measurement in (
            ("latency_seconds", self.latency_seconds),
            ("tokens", self.tokens),
            ("cost_usd", self.cost_usd),
        ):
            if not isinstance(measurement, NumericMeasurement):
                raise EvaluationValidationError(f"{name} must use NumericMeasurement")
        object.__setattr__(self, "findings", tuple(self.findings))
        if any(not isinstance(finding, ObservedFinding) for finding in self.findings):
            raise EvaluationValidationError("run findings must use ObservedFinding")
        if self.state is not EvaluationRunState.COMPLETED and self.findings:
            raise EvaluationValidationError("only completed run records may contain findings")
        if not isinstance(self.stage_latencies_seconds, Mapping) or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(measurement, NumericMeasurement)
            for name, measurement in self.stage_latencies_seconds.items()
        ):
            raise EvaluationValidationError("stage latencies must map stage names to NumericMeasurement values")
        object.__setattr__(self, "stage_latencies_seconds", MappingProxyType(dict(self.stage_latencies_seconds)))
        identity = (self.model_id, self.provider_id, self.model_revision)
        if any(value is not None for value in identity) and not all(value is not None for value in identity):
            raise EvaluationValidationError("run model identity must be a complete model/provider/revision triple")
        for name, value in zip(("model_id", "provider_id", "model_revision"), identity, strict=True):
            if value is not None:
                _validate_identifier(f"run {name}", value)
        object.__setattr__(self, "stage_runs", tuple(self.stage_runs))
        if any(not isinstance(stage_run, EvaluationStageRun) for stage_run in self.stage_runs):
            raise EvaluationValidationError("stage_runs must use EvaluationStageRun")
        stage_names = [stage_run.stage for stage_run in self.stage_runs]
        if len(stage_names) != len(set(stage_names)):
            raise EvaluationValidationError("stage_runs must contain unique stage identities")
        for stage_run in self.stage_runs:
            stage_latency = self.stage_latencies_seconds.get(stage_run.stage)
            if stage_latency is not None and stage_latency != stage_run.latency_seconds:
                raise EvaluationValidationError("stage latency telemetry contradicts its stage run")
        if (
            self.snapshot_result_state is ReviewResultState.NO_FINDINGS
            and any(stage.coverage_status is not MeasurementStatus.COMPLETE for stage in self.stage_runs)
        ):
            raise EvaluationValidationError("a no-findings run cannot claim clean coverage with an uncovered stage")
        if self.failure_reason_code is not None and (
            not isinstance(self.failure_reason_code, str)
            or not _FAILURE_REASON_CODE_PATTERN.fullmatch(self.failure_reason_code)
        ):
            raise EvaluationValidationError("run failure_reason_code must be a bounded machine-readable code")
        if self.state is EvaluationRunState.COMPLETED and self.failure_reason_code is not None:
            raise EvaluationValidationError("a completed run cannot have a failure_reason_code")
        payload = self._identity_payload()
        object.__setattr__(self, "record_id", content_hash(payload))

    def _identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "snapshot_id": self.snapshot_id,
            "attempt": self.attempt,
            "state": self.state.value,
            "terminal": self.terminal,
            "findings": [finding.to_dict() for finding in self.findings],
            "snapshot_result_state": self.snapshot_result_state.value if self.snapshot_result_state else None,
            "latency_seconds": self.latency_seconds.to_dict(),
            "tokens": self.tokens.to_dict(),
            "cost_usd": self.cost_usd.to_dict(),
            "retry_count": self.retry_count,
            "cached": self.cached,
            "escalated": self.escalated,
            "stage_latencies_seconds": {
                name: measurement.to_dict()
                for name, measurement in sorted(self.stage_latencies_seconds.items())
            },
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
        }
        if self.stage_runs:
            payload["stage_runs"] = [stage_run.to_dict() for stage_run in self.stage_runs]
        if self.failure_reason_code is not None:
            payload["failure_reason_code"] = self.failure_reason_code
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "record_id": self.record_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRunRecord":
        _reject_unknown_fields(
            "evaluation run record",
            value,
            {
                "schema_version", "manifest_id", "case_id", "arm_id", "snapshot_id", "attempt", "state",
                "terminal", "findings", "snapshot_result_state", "latency_seconds", "tokens", "cost_usd",
                "retry_count", "cached", "escalated", "stage_latencies_seconds", "model_id", "provider_id",
                "model_revision", "stage_runs", "failure_reason_code", "record_id",
            },
        )
        snapshot_result_state = value.get("snapshot_result_state")
        record = cls(
            manifest_id=value["manifest_id"],
            case_id=value["case_id"],
            arm_id=value["arm_id"],
            snapshot_id=value["snapshot_id"],
            attempt=value["attempt"],
            state=EvaluationRunState(value["state"]),
            terminal=value["terminal"],
            findings=tuple(ObservedFinding.from_dict(item) for item in value.get("findings", [])),
            snapshot_result_state=ReviewResultState(snapshot_result_state) if snapshot_result_state else None,
            latency_seconds=NumericMeasurement.from_dict(value["latency_seconds"]),
            tokens=NumericMeasurement.from_dict(value["tokens"]),
            cost_usd=NumericMeasurement.from_dict(value["cost_usd"]),
            retry_count=value.get("retry_count", 0),
            cached=value.get("cached", False),
            escalated=value.get("escalated"),
            stage_latencies_seconds={
                name: NumericMeasurement.from_dict(measurement)
                for name, measurement in value.get("stage_latencies_seconds", {}).items()
            },
            model_id=value.get("model_id"),
            provider_id=value.get("provider_id"),
            model_revision=value.get("model_revision"),
            stage_runs=tuple(EvaluationStageRun.from_dict(item) for item in value.get("stage_runs", [])),
            failure_reason_code=value.get("failure_reason_code"),
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
        )
        supplied_id = value.get("record_id")
        if supplied_id is not None and supplied_id != record.record_id:
            raise EvaluationValidationError("record_id does not match the run record content")
        return record

    @classmethod
    def from_snapshot_result(
        cls,
        manifest: EvaluationManifest,
        case: CheckpointCase,
        arm: EvaluationArm,
        result: ReviewSnapshotResult,
        details: Optional[RunDetails],
        *,
        attempt: int,
        terminal: bool,
        findings: Sequence[ObservedFinding] = (),
        retry_count: int = 0,
        escalated: Optional[bool] = None,
        stage_latencies_seconds: Optional[Mapping[str, NumericMeasurement]] = None,
        failure_reason_code: Optional[str] = None,
        no_model_execution: bool = False,
    ) -> "EvaluationRunRecord":
        """Bind shipped snapshot/run telemetry to one evaluation attempt."""
        if case not in manifest.cases:
            raise EvaluationValidationError("checkpoint case does not belong to the evaluation manifest")
        if arm not in manifest.arms:
            raise EvaluationValidationError("evaluation arm does not belong to the evaluation manifest")
        if result.snapshot_id != case.snapshot_id:
            raise EvaluationValidationError("ReviewSnapshotResult does not belong to the checkpoint snapshot")
        state_by_result = {
            ReviewResultState.FINDINGS: EvaluationRunState.COMPLETED,
            ReviewResultState.NO_FINDINGS: EvaluationRunState.COMPLETED,
            ReviewResultState.COVERAGE_UNAVAILABLE: EvaluationRunState.COVERAGE_UNAVAILABLE,
            ReviewResultState.CANCELLED: EvaluationRunState.CANCELLED,
            ReviewResultState.STALE: EvaluationRunState.STALE,
        }
        token_measurement = NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
        cost_measurement = NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
        stage_runs: tuple[EvaluationStageRun, ...] = ()
        if details is not None:
            if not isinstance(details.specialist_runs, Mapping) or any(
                not isinstance(stage, str) or not _STAGE_STATE_PATTERN.fullmatch(stage)
                for stage in details.specialist_runs
            ):
                raise EvaluationValidationError("RunDetails specialist_runs contains an invalid stage identity")
            stage_runs = tuple(
                EvaluationStageRun.from_specialist_details(arm, stage, stage_details)
                for stage, stage_details in sorted(details.specialist_runs.items())
            )
            token_components: list[NumericMeasurement] = []
            cost_components: list[NumericMeasurement] = []
            has_primary_call = bool(details.num_ai_calls or details.has_token_usage or details.model_used)
            if has_primary_call:
                token_total = details.total_tokens or details.prompt_tokens + details.completion_tokens
                token_components.append(
                    NumericMeasurement(MeasurementStatus.PARTIAL, float(token_total))
                    if token_total > 0
                    else NumericMeasurement(MeasurementStatus.UNAVAILABLE, None)
                )
                primary_cost_status = MeasurementStatus(details.cost_status)
                primary_cost_value = (
                    float(details.total_cost_usd)
                    if primary_cost_status is not MeasurementStatus.UNAVAILABLE
                    else None
                )
                cost_components.append(NumericMeasurement(primary_cost_status, primary_cost_value))
            token_components.extend(stage.tokens for stage in stage_runs)
            cost_components.extend(stage.cost_usd for stage in stage_runs)
            token_measurement = _sum_measurements(token_components)
            cost_measurement = _sum_measurements(cost_components)
        if no_model_execution:
            if (
                result.state is not ReviewResultState.NO_FINDINGS
                or any(
                    finding.lifecycle_state is FindingLifecycleState.ACTIVE
                    for finding in findings
                )
                or not terminal
                or retry_count
                or result.cached
                or escalated is not None
                or stage_latencies_seconds
                or details is None
                or details.num_ai_calls
                or details.has_token_usage
                or details.known_cost_call_count
                or details.total_cost_usd
                or details.model_costs_usd
                or details.specialist_runs
                or details.adjudication_runs
                or failure_reason_code is not None
            ):
                raise EvaluationValidationError("no-model run records must be zero-call successful executions")
            token_measurement = NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
            cost_measurement = NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
        selected_model_id = (
            None
            if arm.kind is EvaluationArmKind.DETERMINISTIC
            else (
                details.model_used
                if details and details.model_used
                else None if stage_runs else arm.model_id
            )
        )
        if selected_model_id is None and arm.kind is not EvaluationArmKind.DETERMINISTIC:
            selected_provider_id = None
            selected_model_revision = None
        else:
            selected_model_id, selected_provider_id, selected_model_revision = arm.resolve_model_identity(
                selected_model_id
            )
        merged_stage_latencies = dict(stage_latencies_seconds or {})
        for stage_run in stage_runs:
            supplied = merged_stage_latencies.get(stage_run.stage)
            if supplied is not None and supplied != stage_run.latency_seconds:
                raise EvaluationValidationError("supplied stage latency contradicts RunDetails specialist telemetry")
            merged_stage_latencies[stage_run.stage] = stage_run.latency_seconds
        record = cls(
            manifest_id=manifest.manifest_id,
            case_id=case.case_id,
            arm_id=arm.arm_id,
            snapshot_id=case.snapshot_id,
            attempt=attempt,
            state=state_by_result[result.state],
            terminal=terminal,
            findings=tuple(findings),
            snapshot_result_state=result.state,
            latency_seconds=NumericMeasurement(MeasurementStatus.COMPLETE, result.latency_seconds),
            tokens=token_measurement,
            cost_usd=cost_measurement,
            retry_count=retry_count,
            cached=result.cached,
            escalated=escalated,
            stage_latencies_seconds=merged_stage_latencies,
            model_id=selected_model_id,
            provider_id=selected_provider_id,
            model_revision=selected_model_revision,
            stage_runs=stage_runs,
            failure_reason_code=failure_reason_code,
        )
        validate_run_model_telemetry(arm, record)
        return record


def _is_no_model_execution_record(record: EvaluationRunRecord) -> bool:
    return (
        record.state is EvaluationRunState.COMPLETED
        and record.terminal
        and record.snapshot_result_state is ReviewResultState.NO_FINDINGS
        and not any(
            finding.lifecycle_state is FindingLifecycleState.ACTIVE
            for finding in record.findings
        )
        and record.tokens == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
        and record.cost_usd == NumericMeasurement(MeasurementStatus.COMPLETE, 0.0)
        and not record.retry_count
        and not record.cached
        and record.escalated is None
        and not record.stage_runs
        and not record.stage_latencies_seconds
        and record.failure_reason_code is None
    )


def validate_run_model_telemetry(
    arm: EvaluationArm,
    record: EvaluationRunRecord,
    *,
    context: str = "run record",
) -> None:
    """Validate aggregate and per-stage identities against one frozen arm."""
    if not isinstance(arm, EvaluationArm) or not isinstance(record, EvaluationRunRecord):
        raise EvaluationValidationError("run model telemetry validation requires an arm and run record")
    aggregate_identity = (record.model_id, record.provider_id, record.model_revision)
    if aggregate_identity == (None, None, None):
        telemetry_free_failure = (
            record.state is not EvaluationRunState.COMPLETED
            and not record.stage_runs
            and record.tokens.status is MeasurementStatus.UNAVAILABLE
            and record.cost_usd.status is MeasurementStatus.UNAVAILABLE
            and not record.stage_latencies_seconds
        )
        if (
            arm.kind is not EvaluationArmKind.DETERMINISTIC
            and not record.stage_runs
            and not telemetry_free_failure
        ):
            raise EvaluationValidationError(
                f"{context} omits both aggregate and per-stage model identities"
            )
    elif not arm.accepts_run_identity(*aggregate_identity):
        raise EvaluationValidationError(f"{context} model identity does not match its frozen arm")
    expected_by_stage = {stage.stage: stage for stage in arm.stage_plan}
    actual_by_stage = {stage.stage: stage for stage in record.stage_runs}
    missing = sorted(set(expected_by_stage) - set(actual_by_stage))
    unexpected = sorted(set(actual_by_stage) - set(expected_by_stage))
    if missing or unexpected:
        if _is_no_model_execution_record(record) and missing and not unexpected:
            return
        if (
            record.state is not EvaluationRunState.COMPLETED
            and missing
            and not unexpected
            and not actual_by_stage
        ):
            return
        raise EvaluationValidationError(
            f"{context} stages do not match its frozen plan; missing={missing}, unexpected={unexpected}"
        )
    for stage_name, stage_run in actual_by_stage.items():
        stage_plan = expected_by_stage[stage_name]
        route_index, expected_identity = stage_plan.resolve_model(stage_run.model_id)
        observed_identity = (
            stage_run.model_id,
            stage_run.provider_id,
            stage_run.model_revision,
            stage_run.deployment_id_hash,
        )
        planned_identity = (
            expected_identity.model_id,
            expected_identity.provider_id,
            expected_identity.model_revision,
            expected_identity.deployment_id_hash,
        )
        if observed_identity != planned_identity:
            raise EvaluationValidationError(
                f"{context} stage model or deployment identity does not match its frozen plan"
            )
        if stage_run.fallback_used is not (route_index > 0):
            raise EvaluationValidationError(f"{context} stage fallback telemetry contradicts its frozen plan")
        observed_contract = (
            stage_run.configuration_hash,
            stage_run.prompt_hash,
            stage_run.prompt_version,
            stage_run.input_schema_version,
            stage_run.output_schema_version,
        )
        planned_contract = (
            stage_plan.configuration_hash,
            stage_plan.prompt_hash,
            stage_plan.prompt_version,
            stage_plan.input_schema_version,
            stage_plan.output_schema_version,
        )
        if observed_contract != planned_contract:
            raise EvaluationValidationError(f"{context} stage versions do not match its frozen plan")
        for cost_model_id in stage_run.cost_by_model_usd:
            stage_plan.resolve_model(cost_model_id)


@dataclass(frozen=True)
class EvaluationPlanItem:
    case_id: str
    arm_id: str
    snapshot_id: str
    snapshot_artifact_hash: str
    event: ReviewEvent
    configuration_hash: str
    prompt_hash: str
    model_id: Optional[str]
    provider_id: Optional[str]
    model_revision: Optional[str]
    fallback_models: tuple[EvaluationModelIdentity, ...]
    stage_plan: tuple[EvaluationStagePlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_artifact_hash": self.snapshot_artifact_hash,
            "event": self.event.value,
            "configuration_hash": self.configuration_hash,
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
            "fallback_models": [identity.to_dict() for identity in self.fallback_models],
            "stage_plan": [stage.to_dict() for stage in self.stage_plan],
        }


@dataclass(frozen=True)
class EvaluationPlan:
    """Credential-free deterministic expansion of one manifest."""

    manifest_id: str
    schema_hash: str
    items: tuple[EvaluationPlanItem, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "schema_hash": self.schema_hash,
            "network_calls": 0,
            "model_calls": 0,
            "items": [item.to_dict() for item in self.items],
        }


def build_evaluation_plan(manifest: EvaluationManifest) -> EvaluationPlan:
    """Expand cases and enabled arms without contacting a provider or model."""
    items = tuple(
        EvaluationPlanItem(
            case_id=case.case_id,
            arm_id=arm.arm_id,
            snapshot_id=case.snapshot_id,
            snapshot_artifact_hash=case.snapshot_artifact_hash,
            event=case.event,
            configuration_hash=arm.configuration_hash,
            prompt_hash=arm.prompt_hash,
            model_id=arm.model_id,
            provider_id=arm.provider_id,
            model_revision=arm.model_revision,
            fallback_models=arm.fallback_models,
            stage_plan=arm.stage_plan,
        )
        for case in sorted(manifest.cases, key=lambda item: item.case_id)
        for arm in sorted((item for item in manifest.arms if item.enabled), key=lambda item: item.arm_id)
    )
    return EvaluationPlan(
        manifest_id=manifest.manifest_id,
        schema_hash=manifest.schema_hash,
        items=items,
    )
