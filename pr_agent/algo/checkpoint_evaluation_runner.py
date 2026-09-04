"""Fail-closed production bindings for checkpoint evaluation replay.

The evaluator owns pairing, authorization, immutable snapshot loading, and durable
attempt records.  It deliberately does not own a model client, prompt renderer,
retry loop, token counter, diff provider, router, specialist, or verifier.  Each
``ProductionArmBinding`` must adapt the corresponding production orchestration to
``ReviewSnapshotResult`` and ``RunDetails`` without publishing output.

Concurrent specialist role identities are retained as per-stage run telemetry. Treating
those models as fallbacks for one selected model would lose production evidence.
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

from pr_agent.algo.checkpoint_cost_authority import FrozenCostAuthority, validate_cost_authorities
from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationManifest,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationStagePlan,
    EvaluationValidationError,
    FindingLifecycleState,
    MeasurementStatus,
    NumericMeasurement,
    ObservedFinding,
    content_hash,
)
from pr_agent.algo.checkpoint_evaluation_execution import (
    EvaluationArtifactStore,
    PaidExecutionDecision,
    PaidExecutionRequest,
    evaluate_paid_execution,
)
from pr_agent.algo.checkpoint_evaluation_findings import (
    carry_forward_active_findings,
    derive_finding_lifecycle,
)
from pr_agent.algo.checkpoint_evaluation_snapshot import (
    LoadedReviewSnapshotAndConfiguration,
    load_review_snapshot_and_configuration_artifacts,
)
from pr_agent.algo.review_configuration import ReviewConfigurationBundle
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewResultState, ReviewSnapshot, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails

ModelIdentity = tuple[Optional[str], Optional[str], Optional[str]]
ProductionArmAdapter = Callable[[ReviewSnapshot, "ProductionArmContext"], Awaitable["ProductionArmResult"]]
_FAILURE_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
PRE_EXECUTION_ZERO_COST_FAILURE_CODES = frozenset({
    "invalid_request",
    "invalid_snapshot",
    "model_execution_not_authorized",
    "request_too_large",
    "review_configuration_unverified",
    "review_configuration_mismatch",
    "stage_sources_unverified",
    "cost_authority_unverified",
    "worker_start_failed",
})


def _no_model_coverage_accounts_for_snapshot(
    snapshot: ReviewSnapshot,
    coverage_issues: Sequence[CoverageIssue],
) -> bool:
    """Require every changed path when a non-empty snapshot skipped model execution."""

    if not snapshot.diff.strip():
        return True
    covered_paths = {issue.path for issue in coverage_issues if issue.path}
    return bool(snapshot.changed_paths) and set(snapshot.changed_paths) <= covered_paths


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
    review_configuration: ReviewConfigurationBundle = field(repr=False)
    cost_authority: Optional[FrozenCostAuthority] = field(default=None, repr=False)
    stage_plan: tuple[EvaluationStagePlan, ...] = field(default_factory=tuple)
    hard_cost_cap_usd: Optional[float] = None
    publish_output: bool = False

    def __post_init__(self) -> None:
        if self.publish_output:
            raise EvaluationValidationError("checkpoint production bindings cannot publish output")
        if not isinstance(self.review_configuration, ReviewConfigurationBundle):
            raise EvaluationValidationError("production adapter context requires a review configuration bundle")
        if self.cost_authority is not None and not isinstance(self.cost_authority, FrozenCostAuthority):
            raise EvaluationValidationError("production adapter context requires a frozen cost authority")
        object.__setattr__(self, "stage_plan", tuple(self.stage_plan))
        if any(not isinstance(stage, EvaluationStagePlan) for stage in self.stage_plan):
            raise EvaluationValidationError("production adapter context requires an evaluation stage plan")
        if self.hard_cost_cap_usd is not None and (
            not isinstance(self.hard_cost_cap_usd, (int, float))
            or isinstance(self.hard_cost_cap_usd, bool)
            or not math.isfinite(self.hard_cost_cap_usd)
            or self.hard_cost_cap_usd <= 0
        ):
            raise EvaluationValidationError("production adapter hard cost cap must be finite and positive")
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
    no_model_execution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_result, ReviewSnapshotResult):
            raise EvaluationValidationError("production arm result must use ReviewSnapshotResult")
        if self.run_details is not None and not isinstance(self.run_details, RunDetails):
            raise EvaluationValidationError("production arm telemetry must use RunDetails")
        if not isinstance(self.no_model_execution, bool):
            raise EvaluationValidationError("production no-model marker must be a boolean")
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
        has_active_findings = any(
            finding.lifecycle_state is FindingLifecycleState.ACTIVE
            for finding in self.findings
        )
        if self.snapshot_result.state is ReviewResultState.FINDINGS and not has_active_findings:
            raise EvaluationValidationError("a findings result requires an active normalized finding")
        if self.snapshot_result.state is ReviewResultState.NO_FINDINGS and has_active_findings:
            raise EvaluationValidationError("a no-findings result may include only withdrawn findings")
        if self.no_model_execution:
            details = self.run_details
            if (
                self.snapshot_result.state not in {
                    ReviewResultState.NO_FINDINGS,
                    ReviewResultState.COVERAGE_UNAVAILABLE,
                }
                or has_active_findings
                or (
                    self.failure_state is not None
                    and self.failure_reason_code not in PRE_EXECUTION_ZERO_COST_FAILURE_CODES
                )
                or self.model_identity is not None
                or self.terminal is not (
                    self.snapshot_result.state is ReviewResultState.NO_FINDINGS
                )
                or details is None
                or details.num_ai_calls != 0
                or details.route_attempts not in {0, 1}
                or (self.failure_state is not None and details.route_attempts != 0)
                or details.model_retry_attempts != 0
                or details.has_token_usage
                or details.known_cost_call_count != 0
                or details.total_cost_usd != 0
                or details.model_costs_usd
                or details.specialist_runs
                or details.adjudication_runs
            ):
                raise EvaluationValidationError(
                    "no-model production results must be empty successful or pre-execution failure outcomes"
                )
        if (
            self.snapshot_result.state not in {ReviewResultState.FINDINGS, ReviewResultState.NO_FINDINGS}
            and self.findings
        ):
            raise EvaluationValidationError("only completed results may include normalized findings")
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
    stage_plan: tuple[EvaluationStagePlan, ...]
    telemetry_shape: ModelTelemetryShape
    adapter: Optional[ProductionArmAdapter]
    available: bool
    enforces_hard_cost_cap: bool = False
    unavailable_reason: Optional[str] = None
    publish_output: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvaluationArmKind):
            raise EvaluationValidationError("production binding kind must be an EvaluationArmKind")
        object.__setattr__(self, "model_identities", tuple(tuple(identity) for identity in self.model_identities))
        if any(len(identity) != 3 for identity in self.model_identities):
            raise EvaluationValidationError("production binding model identities must be triples")
        object.__setattr__(self, "stage_plan", tuple(self.stage_plan))
        if any(not isinstance(stage, EvaluationStagePlan) for stage in self.stage_plan):
            raise EvaluationValidationError("production binding stage_plan must use EvaluationStagePlan")
        stage_names = [stage.stage for stage in self.stage_plan]
        if len(stage_names) != len(set(stage_names)):
            raise EvaluationValidationError("production binding stage_plan must contain unique stages")
        if not isinstance(self.telemetry_shape, ModelTelemetryShape):
            raise EvaluationValidationError("production binding telemetry_shape is invalid")
        if (
            not isinstance(self.available, bool)
            or not isinstance(self.enforces_hard_cost_cap, bool)
            or not isinstance(self.publish_output, bool)
        ):
            raise EvaluationValidationError("production binding flags must be booleans")
        if self.available:
            if self.adapter is None:
                raise EvaluationValidationError("an available production binding requires an adapter")
            if self.unavailable_reason is not None:
                raise EvaluationValidationError("an available production binding cannot have an unavailable reason")
        else:
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason.strip():
                raise EvaluationValidationError("an unavailable production binding requires an explicit reason")


@dataclass(frozen=True)
class ProductionEvaluationPreflight:
    manifest_id: str
    snapshots_by_case_id: Mapping[str, ReviewSnapshot]
    review_configurations_by_case_id: Mapping[str, ReviewConfigurationBundle] = field(repr=False)
    arms_by_kind: Mapping[EvaluationArmKind, EvaluationArm]
    bindings_by_kind: Mapping[EvaluationArmKind, ProductionArmBinding]
    cost_authorities_by_pair: Mapping[tuple[str, str], FrozenCostAuthority] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots_by_case_id", MappingProxyType(dict(self.snapshots_by_case_id)))
        object.__setattr__(
            self,
            "review_configurations_by_case_id",
            MappingProxyType(dict(self.review_configurations_by_case_id)),
        )
        object.__setattr__(self, "arms_by_kind", MappingProxyType(dict(self.arms_by_kind)))
        object.__setattr__(self, "bindings_by_kind", MappingProxyType(dict(self.bindings_by_kind)))
        object.__setattr__(
            self,
            "cost_authorities_by_pair",
            MappingProxyType(dict(self.cost_authorities_by_pair)),
        )


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
        review_configuration_artifact_hashes: Optional[Mapping[str, str]] = None,
        bindings: Sequence[ProductionArmBinding],
        artifact_store: EvaluationArtifactStore,
        paid_request: PaidExecutionRequest,
        paid_decision: PaidExecutionDecision,
        evaluation_enabled: bool,
        allow_paid_execution: bool,
        publish_output: bool,
        cost_authorities: Sequence[FrozenCostAuthority] = (),
    ) -> None:
        self.manifest = manifest
        self.snapshot_paths = MappingProxyType(dict(snapshot_paths))
        self.review_configuration_artifact_hashes = MappingProxyType(
            dict(review_configuration_artifact_hashes or {})
        )
        self.bindings = tuple(bindings)
        self.artifact_store = artifact_store
        self.paid_request = paid_request
        self.paid_decision = paid_decision
        self.evaluation_enabled = evaluation_enabled
        self.allow_paid_execution = allow_paid_execution
        self.publish_output = publish_output
        self.cost_authorities = tuple(cost_authorities)

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
        loaded_pairs = _load_all_snapshot_configuration_pairs(
            self.manifest,
            self.snapshot_paths,
            self.review_configuration_artifact_hashes,
        )
        _validate_loaded_stage_sources(arms_by_kind, loaded_pairs)
        cost_authorities_by_pair = (
            validate_cost_authorities(self.manifest, self.paid_request, self.cost_authorities)
            if self.cost_authorities
            else MappingProxyType({})
        )
        cases_by_id = {case.case_id: case for case in self.manifest.cases}
        arms_by_id = {arm.arm_id: arm for arm in self.manifest.arms if arm.enabled}
        for (case_id, arm_id), authority in cost_authorities_by_pair.items():
            case = cases_by_id[case_id]
            arm = arms_by_id[arm_id]
            authority.require_context(
                manifest_id=self.manifest.manifest_id,
                case_id=case_id,
                arm_id=arm_id,
                snapshot_id=case.snapshot_id,
                arm_configuration_hash=arm.configuration_hash,
                review_configuration_hash=loaded_pairs[case_id].review_configuration.configuration_hash,
            )
        self.artifact_store.bind_paid_request(self.manifest, self.paid_request)
        return ProductionEvaluationPreflight(
            manifest_id=self.manifest.manifest_id,
            snapshots_by_case_id={case_id: pair.snapshot for case_id, pair in loaded_pairs.items()},
            review_configurations_by_case_id={
                case_id: pair.review_configuration for case_id, pair in loaded_pairs.items()
            },
            arms_by_kind=arms_by_kind,
            bindings_by_kind=bindings_by_kind,
            cost_authorities_by_pair=cost_authorities_by_pair,
        )

    async def run(self) -> ProductionEvaluationResult:
        preflight = self.preflight()
        cases_by_id = {case.case_id: case for case in self.manifest.cases}
        arms_by_id = {arm.arm_id: arm for arm in self.manifest.arms if arm.enabled}
        bindings_by_arm_id = {
            arm.arm_id: preflight.bindings_by_kind[kind] for kind, arm in preflight.arms_by_kind.items()
        }
        retained_terminal = {
            (record.case_id, record.arm_id): record
            for record in self.artifact_store.load_records(self.manifest)
            if record.terminal
        }

        def lineage_depth(case: CheckpointCase) -> int:
            depth = 0
            current = case
            while current.parent_case_id is not None:
                depth += 1
                current = cases_by_id[current.parent_case_id]
            return depth

        records: list[EvaluationRunRecord] = []
        resume_items = sorted(
            self.artifact_store.resume_plan(self.manifest, self.paid_request),
            key=lambda item: (
                lineage_depth(cases_by_id[item.plan_item.case_id]),
                item.plan_item.case_id,
                item.plan_item.arm_id,
            ),
        )
        for resume_item in resume_items:
            item = resume_item.plan_item
            case = cases_by_id[item.case_id]
            arm = arms_by_id[item.arm_id]
            binding = bindings_by_arm_id[item.arm_id]
            snapshot = preflight.snapshots_by_case_id[item.case_id]
            review_configuration = preflight.review_configurations_by_case_id[item.case_id]
            adapter = binding.adapter
            if adapter is None:  # guarded by preflight; keep the paid boundary explicit
                raise ProductionDependencyUnavailable(f"{binding.kind.value} production adapter is unavailable")
            parent_record = None
            if case.parent_case_id is not None:
                parent_record = retained_terminal.get((case.parent_case_id, arm.arm_id))
                if parent_record is None or parent_record.state is not EvaluationRunState.COMPLETED:
                    continue
            paid_budget = next(
                (
                    budget
                    for budget in self.paid_request.plan_item_budgets
                    if budget.case_id == case.case_id and budget.arm_id == arm.arm_id
                ),
                None,
            )
            if arm.kind is not EvaluationArmKind.DETERMINISTIC and paid_budget is None:
                raise EvaluationValidationError(f"pair {case.case_id}/{arm.arm_id} has no immutable paid budget")
            context = ProductionArmContext(
                manifest_id=self.manifest.manifest_id,
                case_id=case.case_id,
                arm_id=arm.arm_id,
                event=case.event.value,
                snapshot_artifact_hash=case.snapshot_artifact_hash,
                configuration_hash=arm.configuration_hash,
                prompt_hash=arm.prompt_hash,
                model_visible_metadata=case.model_visible_metadata,
                review_configuration=review_configuration,
                cost_authority=preflight.cost_authorities_by_pair.get((case.case_id, arm.arm_id)),
                stage_plan=arm.stage_plan,
                hard_cost_cap_usd=(
                    paid_budget.hard_cost_cap_per_attempt_usd
                    if paid_budget is not None
                    else None
                ),
            )
            max_attempts = paid_budget.max_attempts if paid_budget is not None else None
            if max_attempts is not None and resume_item.next_attempt > max_attempts:
                raise EvaluationValidationError(
                    f"pair {case.case_id}/{arm.arm_id} exhausted its immutable attempt limit"
                )
            if arm.kind is not EvaluationArmKind.DETERMINISTIC:
                _require_remaining_paid_capacity(
                    self.manifest,
                    self.artifact_store,
                    self.paid_request,
                )
                if not self.artifact_store.reserve_paid_attempt(
                    self.manifest,
                    self.paid_request,
                    item,
                    resume_item.next_attempt,
                ):
                    raise EvaluationValidationError(
                        f"pair {case.case_id}/{arm.arm_id} attempt is already reserved"
                    )
            outcome = await adapter(snapshot, context)
            if not isinstance(outcome, ProductionArmResult):
                raise EvaluationValidationError("production adapter returned an unsupported result type")
            if case.parent_case_id is not None and outcome.snapshot_result.state in {
                ReviewResultState.FINDINGS,
                ReviewResultState.NO_FINDINGS,
            }:
                if parent_record is None or parent_record.state is not EvaluationRunState.COMPLETED:
                    raise EvaluationValidationError(
                        f"pair {case.case_id}/{arm.arm_id} requires a completed terminal parent record"
                    )
                lifecycle_deriver = (
                    derive_finding_lifecycle
                    if _has_complete_lifecycle_coverage(arm, outcome)
                    else carry_forward_active_findings
                )
                outcome = replace(
                    outcome,
                    findings=lifecycle_deriver(
                        outcome.findings,
                        parent_record.findings,
                        arm_id=arm.arm_id,
                        parent_arm_id=parent_record.arm_id,
                    ),
                )
            record = _record_from_production_result(
                self.manifest,
                case,
                arm,
                snapshot,
                outcome,
                binding.telemetry_shape,
                attempt=resume_item.next_attempt,
            )
            if (
                paid_budget is not None
                and record.cost_usd.status is MeasurementStatus.COMPLETE
                and record.cost_usd.value is not None
                and record.cost_usd.value > paid_budget.hard_cost_cap_per_attempt_usd
            ):
                raise EvaluationValidationError(
                    f"pair {case.case_id}/{arm.arm_id} adapter exceeded its hard cost cap"
                )
            if not record.terminal and max_attempts is not None and resume_item.next_attempt == max_attempts:
                record = replace(record, terminal=True)
            self.artifact_store.append_record(self.manifest, record)
            records.append(record)
            if record.terminal:
                retained_terminal[(record.case_id, record.arm_id)] = record
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
        if binding.stage_plan != arm.stage_plan:
            raise EvaluationValidationError(f"{kind.value} binding stage plan does not match its arm")
        if binding.publish_output:
            raise EvaluationValidationError(f"{kind.value} binding can publish output")
        if kind is not EvaluationArmKind.DETERMINISTIC and not binding.enforces_hard_cost_cap:
            unavailable.append(f"{kind.value}: adapter cannot enforce a hard per-call cost cap")
        expected_shape = {
            EvaluationArmKind.DETERMINISTIC: ModelTelemetryShape.NONE,
            EvaluationArmKind.GENERAL_REVIEW: ModelTelemetryShape.SINGLE_SELECTED,
            EvaluationArmKind.SPECIALISTS: ModelTelemetryShape.PER_STAGE,
            EvaluationArmKind.VERIFIED_SPECIALISTS: ModelTelemetryShape.PER_STAGE,
            EvaluationArmKind.FULL_CASCADE: ModelTelemetryShape.PER_STAGE,
        }[kind]
        if binding.telemetry_shape is not expected_shape:
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


def _require_remaining_paid_capacity(
    manifest: EvaluationManifest,
    artifact_store: EvaluationArtifactStore,
    request: PaidExecutionRequest,
) -> None:
    """Recheck cumulative spend and worst-case remaining work before each paid call."""
    paid_arm_ids = {
        arm.arm_id
        for arm in manifest.arms
        if arm.enabled and arm.kind is not EvaluationArmKind.DETERMINISTIC
    }
    records = artifact_store.load_records(manifest)
    paid_records = tuple(record for record in records if record.arm_id in paid_arm_ids)
    if any(
        record.cost_usd.status is not MeasurementStatus.COMPLETE
        or record.cost_usd.value is None
        for record in paid_records
    ):
        raise EvaluationValidationError("retained paid attempts require complete cost telemetry before resuming")
    cumulative_cost = sum(float(record.cost_usd.value) for record in paid_records)
    terminal_pairs = {
        (record.case_id, record.arm_id)
        for record in paid_records
        if record.terminal
    }
    attempts_by_pair: dict[tuple[str, str], int] = {}
    for record in paid_records:
        pair = (record.case_id, record.arm_id)
        attempts_by_pair[pair] = attempts_by_pair.get(pair, 0) + 1
    budget_by_pair = {
        (budget.case_id, budget.arm_id): budget
        for budget in request.plan_item_budgets
    }
    remaining_reserved_cost = sum(
        budget.hard_cost_cap_per_attempt_usd
        * (budget.max_attempts - attempts_by_pair.get(pair, 0))
        for pair, budget in budget_by_pair.items()
        if pair not in terminal_pairs
    )
    if remaining_reserved_cost <= 0:
        raise EvaluationValidationError("paid replay has no bounded model attempts remaining")
    worst_case_total = cumulative_cost + remaining_reserved_cost
    if worst_case_total > request.cost_cap_usd + 1e-12:
        raise EvaluationValidationError(
            "retained cumulative spend plus remaining hard-capped work exceeds the explicit cost cap"
        )


def _load_all_snapshot_configuration_pairs(
    manifest: EvaluationManifest,
    snapshot_paths: Mapping[str, str | Path],
    review_configuration_artifact_hashes: Mapping[str, str],
) -> Mapping[str, LoadedReviewSnapshotAndConfiguration]:
    expected_case_ids = {case.case_id for case in manifest.cases}
    supplied_case_ids = set(snapshot_paths)
    if supplied_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - supplied_case_ids)
        extra = sorted(supplied_case_ids - expected_case_ids)
        raise EvaluationValidationError(
            f"snapshot paths must match manifest cases exactly; missing={missing}, extra={extra}"
        )
    supplied_configuration_case_ids = set(review_configuration_artifact_hashes)
    if supplied_configuration_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - supplied_configuration_case_ids)
        extra = sorted(supplied_configuration_case_ids - expected_case_ids)
        raise EvaluationValidationError(
            "review configuration artifact hashes must match manifest cases exactly; "
            f"missing={missing}, extra={extra}"
        )
    loaded_pairs: dict[str, LoadedReviewSnapshotAndConfiguration] = {}
    for case in sorted(manifest.cases, key=lambda item: item.case_id):
        loaded_pairs[case.case_id] = load_review_snapshot_and_configuration_artifacts(
            snapshot_paths[case.case_id],
            case,
            review_configuration_artifact_hash=review_configuration_artifact_hashes[case.case_id],
        )
    for case in manifest.cases:
        snapshot = loaded_pairs[case.case_id].snapshot
        expected_parent_snapshot_id = (
            loaded_pairs[case.parent_case_id].snapshot.snapshot_id
            if case.parent_case_id is not None
            else None
        )
        if snapshot.parent_snapshot_id != expected_parent_snapshot_id:
            raise EvaluationValidationError(
                f"snapshot lineage for case {case.case_id} does not match its manifest parent"
            )
        _validate_loaded_snapshot_metadata(case, snapshot)
    return MappingProxyType(loaded_pairs)


def _validate_loaded_stage_sources(
    arms_by_kind: Mapping[EvaluationArmKind, EvaluationArm],
    loaded_pairs: Mapping[str, LoadedReviewSnapshotAndConfiguration],
) -> None:
    """Validate extended private bundles before touching the artifact store.

    Legacy v1 bundles have no stage-source extension and remain loadable only
    when the manifest has no stage-backed arm. Every stage-backed arm requires
    an exact typed source contract.
    """

    stage_plans = tuple(arm.stage_plan for arm in arms_by_kind.values() if arm.stage_plan)
    for case_id, pair in loaded_pairs.items():
        sources = pair.review_configuration.stage_sources
        if sources is None:
            if stage_plans:
                raise EvaluationValidationError(
                    f"checkpoint {case_id} stage sources are unavailable for its evaluation plan"
                )
            continue
        for kind in (
            EvaluationArmKind.SPECIALISTS,
            EvaluationArmKind.VERIFIED_SPECIALISTS,
            EvaluationArmKind.FULL_CASCADE,
        ):
            expected = sources.required_stage_names(kind)
            actual = tuple(stage.stage for stage in arms_by_kind[kind].stage_plan)
            if actual != expected:
                raise EvaluationValidationError(
                    f"checkpoint {case_id} {kind.value} stage plan does not match its required cascade order"
                )
        for kind in (
            EvaluationArmKind.SPECIALISTS,
            EvaluationArmKind.VERIFIED_SPECIALISTS,
            EvaluationArmKind.FULL_CASCADE,
        ):
            stage_plan = arms_by_kind[kind].stage_plan
            try:
                sources.validate_stage_plan(stage_plan, arm_kind=kind)
            except EvaluationValidationError as exc:
                raise EvaluationValidationError(
                    f"checkpoint {case_id} stage sources do not match its evaluation plan"
                ) from exc


def _validate_loaded_snapshot_metadata(case: CheckpointCase, snapshot: ReviewSnapshot) -> None:
    """Recompute model-visible facts that have an authoritative snapshot source."""
    metadata = case.model_visible_metadata
    derived = {
        "change_size": len(snapshot.changed_paths),
        "stage": snapshot.event.value,
        "task_intent_hash": content_hash({"task_intent": snapshot.task_intent}),
        "repository_context_hash": content_hash({
            "base_revision": snapshot.base_revision,
            "base_selector": snapshot.base_selector,
            "changed_paths": list(snapshot.changed_paths),
            "focus_path": snapshot.focus_path,
        }),
    }
    for key, expected in derived.items():
        if key in metadata and metadata[key] != expected:
            raise EvaluationValidationError(
                f"model_visible_metadata.{key} does not match the loaded snapshot"
            )


def _record_from_production_result(
    manifest: EvaluationManifest,
    case: CheckpointCase,
    arm: EvaluationArm,
    snapshot: ReviewSnapshot,
    outcome: ProductionArmResult,
    telemetry_shape: ModelTelemetryShape,
    *,
    attempt: int,
) -> EvaluationRunRecord:
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
        no_model_execution=outcome.no_model_execution,
    )
    no_model_execution = outcome.no_model_execution
    if (
        no_model_execution
        and outcome.failure_state is None
        and not _no_model_coverage_accounts_for_snapshot(snapshot, result.coverage_issues)
    ):
        raise EvaluationValidationError(
            "zero-model production results must account for every changed path"
        )
    has_stage_runs = bool(record.stage_runs)
    if telemetry_shape is ModelTelemetryShape.NONE and has_stage_runs:
        raise EvaluationValidationError("deterministic production bindings cannot emit model stage telemetry")
    if telemetry_shape is ModelTelemetryShape.SINGLE_SELECTED and has_stage_runs:
        raise EvaluationValidationError("single-model production bindings cannot emit per-stage model telemetry")
    if (
        telemetry_shape is ModelTelemetryShape.PER_STAGE
        and not has_stage_runs
        and not no_model_execution
        and outcome.failure_state is None
    ):
        raise EvaluationValidationError("per-stage production bindings require model stage telemetry")
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
            if details_model:
                model_identity = arm.resolve_model_identity(details_model)
            elif has_stage_runs:
                model_identity = (None, None, None)
            elif no_model_execution:
                model_identity = arm.resolve_model_identity(record.model_id)
            elif outcome.failure_state is not None:
                model_identity = (None, None, None)
            else:
                raise EvaluationValidationError("model-backed results require an observed model identity")
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
    no_model_execution: bool = False,
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
        no_model_execution=no_model_execution,
    )


def _has_complete_lifecycle_coverage(
    arm: EvaluationArm,
    outcome: ProductionArmResult,
) -> bool:
    """Require complete planned stages before inferring that a parent finding disappeared."""

    if outcome.snapshot_result.coverage_issues:
        return False
    if outcome.no_model_execution:
        return True
    if not arm.stage_plan:
        return True
    details = outcome.run_details
    if details is None or set(details.specialist_runs) != {stage.stage for stage in arm.stage_plan}:
        return False
    return all(
        stage.state in {"cached", "not_required", "success"}
        for stage in details.specialist_runs.values()
    )
