"""Fail-closed production bindings for checkpoint evaluation replay.

The evaluator owns pairing, authorization, immutable snapshot loading, and durable
attempt records.  It deliberately does not own a model client, prompt renderer,
retry loop, token counter, diff provider, router, specialist, or verifier.  Each
``ProductionArmBinding`` must adapt the corresponding production orchestration to
``ReviewSnapshotResult`` and ``RunDetails`` without publishing output.

Bindings for specialists must remain unavailable until their concurrent role model
identities and normalized findings fit the evaluation run schema.  Treating those
models as fallbacks for one selected model would lose production evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationManifest,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationValidationError,
    NumericMeasurement,
    ObservedFinding,
)
from pr_agent.algo.checkpoint_evaluation_execution import (
    EvaluationArtifactStore,
    PaidExecutionDecision,
    PaidExecutionRequest,
    evaluate_paid_execution,
)
from pr_agent.algo.checkpoint_evaluation_snapshot import load_review_snapshot_artifact
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewResultState, ReviewSnapshot, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails

ModelIdentity = tuple[Optional[str], Optional[str], Optional[str]]
ProductionArmAdapter = Callable[[ReviewSnapshot, "ProductionArmContext"], Awaitable["ProductionArmResult"]]
_FAILURE_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ProductionDependencyUnavailable(EvaluationValidationError):
    """Raised before any production or artifact-store call when an arm is unavailable."""


class ModelTelemetryShape(str, Enum):
    """Model identity shape emitted by one production arm."""

    NONE = "none"
    SINGLE_SELECTED = "single_selected"
    PER_STAGE = "per_stage"


@dataclass(frozen=True)
class ProductionArmContext:
    """Answer-free metadata passed beside the exact loaded snapshot object."""

    manifest_id: str
    case_id: str
    arm_id: str
    event: str
    snapshot_artifact_hash: str
    configuration_hash: str
    prompt_hash: str
    model_visible_metadata: Mapping[str, object]
    publish_output: bool = False

    def __post_init__(self) -> None:
        if self.publish_output:
            raise EvaluationValidationError("checkpoint production bindings cannot publish output")
        object.__setattr__(
            self,
            "model_visible_metadata",
            MappingProxyType(dict(self.model_visible_metadata)),
        )


@dataclass(frozen=True)
class ProductionArmResult:
    """Source-free result of one production arm execution.

    Expected provider, timeout, and malformed-output failures use ``failure_state``
    with a ``COVERAGE_UNAVAILABLE`` snapshot result.  This lets the runner retain
    their exact attempt state while still deriving all shared snapshot, token, cost,
    cache, and selected-model fields through ``from_snapshot_result``.
    """

    snapshot_result: ReviewSnapshotResult
    run_details: Optional[RunDetails]
    findings: tuple[ObservedFinding, ...] = field(default_factory=tuple)
    terminal: bool = True
    retry_count: int = 0
    escalated: Optional[bool] = None
    stage_latencies_seconds: Mapping[str, NumericMeasurement] = field(default_factory=dict)
    failure_state: Optional[EvaluationRunState] = None
    failure_reason_code: Optional[str] = None
    latency_measurement: Optional[NumericMeasurement] = None
    model_identity: Optional[ModelIdentity] = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_result, ReviewSnapshotResult):
            raise EvaluationValidationError("production arm result must use ReviewSnapshotResult")
        if self.run_details is not None and not isinstance(self.run_details, RunDetails):
            raise EvaluationValidationError("production arm telemetry must use RunDetails")
        if self.run_details is not None and self.run_details.specialist_runs:
            raise EvaluationValidationError(
                "production arm result contains per-stage or per-role model telemetry "
                "that EvaluationRunRecord cannot preserve"
            )
        if not isinstance(self.snapshot_result.state, ReviewResultState):
            raise EvaluationValidationError("production snapshot result state is invalid")
        latency = self.snapshot_result.latency_seconds
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise EvaluationValidationError("production snapshot latency must be finite and non-negative")
        if self.latency_measurement is not None:
            if not isinstance(self.latency_measurement, NumericMeasurement):
                raise EvaluationValidationError("production latency override must use NumericMeasurement")
            if self.latency_measurement.value is not None and self.latency_measurement.value < 0:
                raise EvaluationValidationError("production latency override cannot be negative")
        if self.snapshot_result.state not in {ReviewResultState.FINDINGS, ReviewResultState.NO_FINDINGS}:
            if self.latency_measurement is None:
                raise EvaluationValidationError("non-completed production results require explicit latency status")
            if not isinstance(self.failure_reason_code, str) or not _FAILURE_REASON_CODE.fullmatch(
                self.failure_reason_code
            ):
                raise EvaluationValidationError("non-completed production results require a bounded reason code")
        elif self.failure_reason_code is not None:
            raise EvaluationValidationError("completed production results cannot have a failure reason")
        if self.model_identity is not None:
            identity = tuple(self.model_identity)
            if len(identity) != 3:
                raise EvaluationValidationError("production result model identity must be a triple")
            object.__setattr__(self, "model_identity", identity)
        object.__setattr__(self, "findings", tuple(self.findings))
        if any(not isinstance(finding, ObservedFinding) for finding in self.findings):
            raise EvaluationValidationError("production arm findings must use ObservedFinding")
        if not isinstance(self.retry_count, int) or isinstance(self.retry_count, bool) or self.retry_count < 0:
            raise EvaluationValidationError("production arm retry_count must be a non-negative integer")
        if not isinstance(self.terminal, bool):
            raise EvaluationValidationError("production arm terminal must be a boolean")
        if self.snapshot_result.state is ReviewResultState.FINDINGS and not self.findings:
            raise EvaluationValidationError("a findings result requires normalized finding fingerprints")
        if self.snapshot_result.state is not ReviewResultState.FINDINGS and self.findings:
            raise EvaluationValidationError("only a findings result may include normalized findings")
        if (
            self.snapshot_result.state in {ReviewResultState.FINDINGS, ReviewResultState.NO_FINDINGS}
            and not self.terminal
        ):
            raise EvaluationValidationError("completed production results must be terminal")
        allowed_failure_states = {
            EvaluationRunState.TIMEOUT,
            EvaluationRunState.MALFORMED,
            EvaluationRunState.PROVIDER_FAILURE,
        }
        if self.failure_state is not None:
            if self.failure_state not in allowed_failure_states:
                raise EvaluationValidationError("production failure_state is not an adapter failure")
            if self.snapshot_result.state is not ReviewResultState.COVERAGE_UNAVAILABLE:
                raise EvaluationValidationError("adapter failures require coverage_unavailable snapshot state")
            if self.terminal:
                raise EvaluationValidationError("adapter failures must remain resumable")
        if not isinstance(self.stage_latencies_seconds, Mapping) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(measurement, NumericMeasurement)
            for name, measurement in self.stage_latencies_seconds.items()
        ):
            raise EvaluationValidationError("production stage latency telemetry is invalid")
        object.__setattr__(
            self,
            "stage_latencies_seconds",
            MappingProxyType(dict(self.stage_latencies_seconds)),
        )


@dataclass(frozen=True)
class ProductionArmBinding:
    """One metadata-pinned adapter to shipped production orchestration."""

    kind: EvaluationArmKind
    configuration_hash: str
    prompt_hash: str
    model_identities: tuple[ModelIdentity, ...]
    telemetry_shape: ModelTelemetryShape
    adapter: Optional[ProductionArmAdapter]
    available: bool
    unavailable_reason: Optional[str] = None
    publish_output: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvaluationArmKind):
            raise EvaluationValidationError("production binding kind must be an EvaluationArmKind")
        object.__setattr__(self, "model_identities", tuple(tuple(identity) for identity in self.model_identities))
        if any(len(identity) != 3 for identity in self.model_identities):
            raise EvaluationValidationError("production binding model identities must be triples")
        if not isinstance(self.telemetry_shape, ModelTelemetryShape):
            raise EvaluationValidationError("production binding telemetry_shape is invalid")
        if not isinstance(self.available, bool) or not isinstance(self.publish_output, bool):
            raise EvaluationValidationError("production binding flags must be booleans")
        if self.available:
            if self.adapter is None:
                raise EvaluationValidationError("an available production binding requires an adapter")
            if self.unavailable_reason is not None:
                raise EvaluationValidationError("an available production binding cannot have an unavailable reason")
        else:
            if self.adapter is not None:
                raise EvaluationValidationError("an unavailable production binding cannot expose an adapter")
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason.strip():
                raise EvaluationValidationError("an unavailable production binding requires an explicit reason")


@dataclass(frozen=True)
class ProductionEvaluationPreflight:
    manifest_id: str
    snapshots_by_case_id: Mapping[str, ReviewSnapshot]
    arms_by_kind: Mapping[EvaluationArmKind, EvaluationArm]
    bindings_by_kind: Mapping[EvaluationArmKind, ProductionArmBinding]

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots_by_case_id", MappingProxyType(dict(self.snapshots_by_case_id)))
        object.__setattr__(self, "arms_by_kind", MappingProxyType(dict(self.arms_by_kind)))
        object.__setattr__(self, "bindings_by_kind", MappingProxyType(dict(self.bindings_by_kind)))


@dataclass(frozen=True)
class ProductionEvaluationResult:
    manifest_id: str
    records: tuple[EvaluationRunRecord, ...]


class ProductionEvaluationRunner:
    """Run one authorized, resumable five-arm replay without duplicating production logic."""

    def __init__(
        self,
        manifest: EvaluationManifest,
        *,
        snapshot_paths: Mapping[str, str | Path],
        bindings: Sequence[ProductionArmBinding],
        artifact_store: EvaluationArtifactStore,
        paid_request: PaidExecutionRequest,
        paid_decision: PaidExecutionDecision,
        evaluation_enabled: bool,
        allow_paid_execution: bool,
        publish_output: bool,
    ) -> None:
        self.manifest = manifest
        self.snapshot_paths = MappingProxyType(dict(snapshot_paths))
        self.bindings = tuple(bindings)
        self.artifact_store = artifact_store
        self.paid_request = paid_request
        self.paid_decision = paid_decision
        self.evaluation_enabled = evaluation_enabled
        self.allow_paid_execution = allow_paid_execution
        self.publish_output = publish_output

    def preflight(self) -> ProductionEvaluationPreflight:
        """Validate the complete run before any adapter or artifact-store call."""
        arms_by_kind = _validate_five_arm_manifest(self.manifest)
        bindings_by_kind = _validate_bindings(arms_by_kind, self.bindings)
        _validate_paid_authorization(
            self.manifest,
            self.paid_request,
            self.paid_decision,
            evaluation_enabled=self.evaluation_enabled,
            allow_paid_execution=self.allow_paid_execution,
            publish_output=self.publish_output,
        )
        snapshots = _load_all_snapshots(self.manifest, self.snapshot_paths)
        return ProductionEvaluationPreflight(
            manifest_id=self.manifest.manifest_id,
            snapshots_by_case_id=snapshots,
            arms_by_kind=arms_by_kind,
            bindings_by_kind=bindings_by_kind,
        )

    async def run(self) -> ProductionEvaluationResult:
        preflight = self.preflight()
        cases_by_id = {case.case_id: case for case in self.manifest.cases}
        arms_by_id = {arm.arm_id: arm for arm in self.manifest.arms if arm.enabled}
        bindings_by_arm_id = {
            arm.arm_id: preflight.bindings_by_kind[kind] for kind, arm in preflight.arms_by_kind.items()
        }
        records: list[EvaluationRunRecord] = []
        for resume_item in self.artifact_store.resume_plan(self.manifest):
            item = resume_item.plan_item
            case = cases_by_id[item.case_id]
            arm = arms_by_id[item.arm_id]
            binding = bindings_by_arm_id[item.arm_id]
            snapshot = preflight.snapshots_by_case_id[item.case_id]
            context = ProductionArmContext(
                manifest_id=self.manifest.manifest_id,
                case_id=case.case_id,
                arm_id=arm.arm_id,
                event=case.event.value,
                snapshot_artifact_hash=case.snapshot_artifact_hash,
                configuration_hash=arm.configuration_hash,
                prompt_hash=arm.prompt_hash,
                model_visible_metadata=case.model_visible_metadata,
            )
            adapter = binding.adapter
            if adapter is None:  # guarded by preflight; keep the paid boundary explicit
                raise ProductionDependencyUnavailable(f"{binding.kind.value} production adapter is unavailable")
            outcome = await adapter(snapshot, context)
            if not isinstance(outcome, ProductionArmResult):
                raise EvaluationValidationError("production adapter returned an unsupported result type")
            record = _record_from_production_result(
                self.manifest,
                case,
                arm,
                snapshot,
                outcome,
                attempt=resume_item.next_attempt,
            )
            self.artifact_store.append_record(self.manifest, record)
            records.append(record)
        return ProductionEvaluationResult(self.manifest.manifest_id, tuple(records))


def _validate_five_arm_manifest(manifest: EvaluationManifest) -> Mapping[EvaluationArmKind, EvaluationArm]:
    if not isinstance(manifest, EvaluationManifest):
        raise EvaluationValidationError("production evaluation requires a validated EvaluationManifest")
    enabled = tuple(arm for arm in manifest.arms if arm.enabled)
    by_kind: dict[EvaluationArmKind, list[EvaluationArm]] = {}
    for arm in enabled:
        by_kind.setdefault(arm.kind, []).append(arm)
    missing = [kind.value for kind in EvaluationArmKind if not by_kind.get(kind)]
    duplicate = [kind.value for kind, arms in by_kind.items() if len(arms) != 1]
    if len(enabled) != len(EvaluationArmKind) or missing or duplicate:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if duplicate:
            details.append("duplicated " + ", ".join(sorted(duplicate)))
        raise EvaluationValidationError(
            "paid replay requires exactly one enabled arm of every EvaluationArmKind"
            + (": " + "; ".join(details) if details else "")
        )
    return MappingProxyType({kind: arms[0] for kind, arms in by_kind.items()})


def _validate_bindings(
    arms_by_kind: Mapping[EvaluationArmKind, EvaluationArm],
    bindings: Sequence[ProductionArmBinding],
) -> Mapping[EvaluationArmKind, ProductionArmBinding]:
    by_kind: dict[EvaluationArmKind, list[ProductionArmBinding]] = {}
    for binding in bindings:
        if not isinstance(binding, ProductionArmBinding):
            raise EvaluationValidationError("production bindings must use ProductionArmBinding")
        by_kind.setdefault(binding.kind, []).append(binding)
    if len(bindings) != len(EvaluationArmKind) or any(len(by_kind.get(kind, ())) != 1 for kind in EvaluationArmKind):
        raise EvaluationValidationError("paid replay requires exactly one production binding for every arm kind")

    unavailable: list[str] = []
    validated: dict[EvaluationArmKind, ProductionArmBinding] = {}
    for kind in EvaluationArmKind:
        arm = arms_by_kind[kind]
        binding = by_kind[kind][0]
        if binding.configuration_hash != arm.configuration_hash:
            raise EvaluationValidationError(f"{kind.value} binding configuration hash does not match its arm")
        if binding.prompt_hash != arm.prompt_hash:
            raise EvaluationValidationError(f"{kind.value} binding prompt hash does not match its arm")
        if binding.model_identities != arm.model_identities():
            raise EvaluationValidationError(f"{kind.value} binding model identities do not match its arm")
        if binding.publish_output:
            raise EvaluationValidationError(f"{kind.value} binding can publish output")
        expected_shape = (
            ModelTelemetryShape.NONE if kind is EvaluationArmKind.DETERMINISTIC else ModelTelemetryShape.SINGLE_SELECTED
        )
        if binding.telemetry_shape is ModelTelemetryShape.PER_STAGE:
            unavailable.append(
                f"{kind.value}: EvaluationRunRecord cannot preserve per-stage or per-role model identities"
            )
        elif binding.telemetry_shape is not expected_shape:
            raise EvaluationValidationError(f"{kind.value} binding has the wrong model telemetry shape")
        if not binding.available:
            unavailable.append(f"{kind.value}: {binding.unavailable_reason}")
        validated[kind] = binding
    if unavailable:
        raise ProductionDependencyUnavailable(
            "production evaluation dependencies unavailable: " + "; ".join(unavailable)
        )
    return MappingProxyType(validated)


def _validate_paid_authorization(
    manifest: EvaluationManifest,
    request: PaidExecutionRequest,
    decision: PaidExecutionDecision,
    *,
    evaluation_enabled: bool,
    allow_paid_execution: bool,
    publish_output: bool,
) -> None:
    if request.manifest_id != manifest.manifest_id:
        raise EvaluationValidationError("paid execution request belongs to a different manifest")
    expected = evaluate_paid_execution(
        manifest,
        request,
        evaluation_enabled=evaluation_enabled,
        allow_paid_execution=allow_paid_execution,
        publish_output=publish_output,
    )
    if decision.to_dict() != expected.to_dict():
        raise EvaluationValidationError("paid execution decision does not match the current request and settings")
    expected.require_authorized()


def _load_all_snapshots(
    manifest: EvaluationManifest,
    snapshot_paths: Mapping[str, str | Path],
) -> Mapping[str, ReviewSnapshot]:
    expected_case_ids = {case.case_id for case in manifest.cases}
    supplied_case_ids = set(snapshot_paths)
    if supplied_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - supplied_case_ids)
        extra = sorted(supplied_case_ids - expected_case_ids)
        raise EvaluationValidationError(
            f"snapshot paths must match manifest cases exactly; missing={missing}, extra={extra}"
        )
    snapshots: dict[str, ReviewSnapshot] = {}
    for case in sorted(manifest.cases, key=lambda item: item.case_id):
        snapshots[case.case_id] = load_review_snapshot_artifact(snapshot_paths[case.case_id], case)
    return MappingProxyType(snapshots)


def _record_from_production_result(
    manifest: EvaluationManifest,
    case: CheckpointCase,
    arm: EvaluationArm,
    snapshot: ReviewSnapshot,
    outcome: ProductionArmResult,
    *,
    attempt: int,
) -> EvaluationRunRecord:
    if outcome.run_details is not None and outcome.run_details.specialist_runs:
        # RunDetails is mutable even though ProductionArmResult is frozen. Recheck at
        # the persistence boundary so an adapter cannot add role telemetry after the
        # result was constructed and silently collapse it into one selected model.
        raise EvaluationValidationError(
            "production arm result contains per-stage or per-role model telemetry "
            "that EvaluationRunRecord cannot preserve"
        )
    result = outcome.snapshot_result
    if result.snapshot_id != snapshot.snapshot_id:
        raise EvaluationValidationError("production result names a different immutable snapshot")
    if result.state in {ReviewResultState.FINDINGS, ReviewResultState.NO_FINDINGS}:
        if result.current_snapshot_id != snapshot.snapshot_id:
            raise EvaluationValidationError("completed production result is stale relative to its snapshot")
    if not result.advisory or not result.shadow_capable:
        raise EvaluationValidationError("production evaluation results must remain advisory and shadow-capable")
    record = EvaluationRunRecord.from_snapshot_result(
        manifest,
        case,
        arm,
        result,
        outcome.run_details,
        attempt=attempt,
        terminal=outcome.terminal,
        findings=outcome.findings,
        retry_count=outcome.retry_count,
        escalated=outcome.escalated,
        stage_latencies_seconds=outcome.stage_latencies_seconds,
        failure_reason_code=outcome.failure_reason_code,
    )
    replacements = {}
    if outcome.latency_measurement is not None:
        replacements["latency_seconds"] = outcome.latency_measurement
    if arm.kind is EvaluationArmKind.DETERMINISTIC:
        if outcome.model_identity is not None:
            raise EvaluationValidationError("deterministic results cannot name a model identity")
    else:
        model_identity = outcome.model_identity
        details_model = outcome.run_details.model_used if outcome.run_details is not None else None
        if model_identity is None:
            if not details_model:
                raise EvaluationValidationError("model-backed results require an observed model identity")
            model_identity = arm.resolve_model_identity(details_model)
        elif not arm.accepts_run_identity(*model_identity):
            raise EvaluationValidationError("production result selected an unpinned model identity")
        if details_model is not None and details_model != model_identity[0]:
            raise EvaluationValidationError("production result model identity contradicts RunDetails")
        replacements.update(
            model_id=model_identity[0],
            provider_id=model_identity[1],
            model_revision=model_identity[2],
        )
    if outcome.failure_state is not None:
        replacements.update(
            state=outcome.failure_state,
            snapshot_result_state=None,
        )
    if replacements:
        record = replace(record, **replacements)
    return record


def failed_production_arm_result(
    snapshot: ReviewSnapshot,
    *,
    state: EvaluationRunState,
    reason_code: str,
    latency_seconds: NumericMeasurement,
    retry_count: int,
    run_details: Optional[RunDetails] = None,
    model_identity: Optional[ModelIdentity] = None,
    stage_latencies_seconds: Optional[Mapping[str, NumericMeasurement]] = None,
) -> ProductionArmResult:
    """Build a source-free, resumable adapter failure with explicit retry telemetry."""
    if not isinstance(reason_code, str) or not _FAILURE_REASON_CODE.fullmatch(reason_code):
        raise EvaluationValidationError("production failure reason_code must be a bounded machine-readable code")
    if not isinstance(latency_seconds, NumericMeasurement):
        raise EvaluationValidationError("production failure latency must use NumericMeasurement")
    if latency_seconds.value is not None and latency_seconds.value < 0:
        raise EvaluationValidationError("production failure latency cannot be negative")
    result = ReviewSnapshotResult(
        snapshot_id=snapshot.snapshot_id,
        state=ReviewResultState.COVERAGE_UNAVAILABLE,
        current_snapshot_id=snapshot.snapshot_id,
        review=None,
        coverage_issues=(CoverageIssue(reason=reason_code),),
        # ReviewSnapshotResult predates unavailable measurements. The normalized
        # run record below always replaces this placeholder with the explicit value.
        latency_seconds=latency_seconds.value if latency_seconds.value is not None else 0.0,
    )
    return ProductionArmResult(
        snapshot_result=result,
        run_details=run_details,
        terminal=False,
        retry_count=retry_count,
        stage_latencies_seconds=stage_latencies_seconds or {},
        failure_state=state,
        failure_reason_code=reason_code,
        latency_measurement=latency_seconds,
        model_identity=model_identity,
    )
