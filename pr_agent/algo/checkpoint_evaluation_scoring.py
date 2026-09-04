"""Pure matched-arm scoring and rollout-gate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from pr_agent.algo.checkpoint_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    CheckpointCase,
    CheckpointTruth,
    EvaluationArmKind,
    EvaluationCohort,
    EvaluationManifest,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationValidationError,
    FindingLifecycleState,
    FindingSeverity,
    GateStatus,
    MeasurementStatus,
    TruthArtifact,
    content_hash,
    validate_run_model_telemetry,
)
from pr_agent.algo.review_snapshot import ReviewEvent

_TWO_SIDED_95_T_CRITICAL = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


class GateComparator(str, Enum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


_SEVERITY_WEIGHT = {
    FindingSeverity.LOW: 1.0,
    FindingSeverity.MEDIUM: 2.0,
    FindingSeverity.HIGH: 4.0,
    FindingSeverity.CRITICAL: 8.0,
}


@dataclass(frozen=True)
class ScoreMetric:
    status: MeasurementStatus
    value: Optional[float]
    support: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, MeasurementStatus):
            raise EvaluationValidationError("score metric status must be a MeasurementStatus")
        if not isinstance(self.support, int) or self.support < 0:
            raise EvaluationValidationError("metric support must be a non-negative integer")
        if self.status is MeasurementStatus.UNAVAILABLE and self.value is not None:
            raise EvaluationValidationError("an unavailable score metric cannot have a value")
        if self.status is not MeasurementStatus.UNAVAILABLE and self.value is None:
            raise EvaluationValidationError("a complete or partial score metric requires a value")
        if self.value is not None and not math.isfinite(self.value):
            raise EvaluationValidationError("score metric values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "value": self.value, "support": self.support}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreMetric":
        return cls(
            status=MeasurementStatus(value["status"]),
            value=value.get("value"),
            support=value["support"],
        )


@dataclass(frozen=True)
class ArmScorecard:
    arm_id: str
    case_count: int
    attempt_count: int
    failed_attempt_count: int
    completed_case_count: int
    duplicate_finding_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    clean_checkpoint_count: int
    false_interruption_count: int
    metrics: Mapping[str, ScoreMetric]
    escalated_case_count: int = 0
    high_critical_case_count: int = 0
    high_critical_escalated_count: int = 0
    stale_finding_count: int = 0
    deterministic_overlap_count: int = 0
    cohort_metrics: Mapping[str, Mapping[str, ScoreMetric]] = field(default_factory=dict)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported scorecard schema_version: {self.schema_version}")
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise EvaluationValidationError("scorecard arm_id must be a non-empty string")
        counts = (
            self.case_count,
            self.attempt_count,
            self.failed_attempt_count,
            self.completed_case_count,
            self.duplicate_finding_count,
            self.true_positive_count,
            self.false_positive_count,
            self.false_negative_count,
            self.clean_checkpoint_count,
            self.false_interruption_count,
            self.escalated_case_count,
            self.high_critical_case_count,
            self.high_critical_escalated_count,
            self.stale_finding_count,
            self.deterministic_overlap_count,
        )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise EvaluationValidationError("scorecard counts must be non-negative integers")
        if not isinstance(self.metrics, Mapping) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(metric, ScoreMetric)
            for name, metric in self.metrics.items()
        ):
            raise EvaluationValidationError("scorecard metrics must map names to ScoreMetric values")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        frozen_cohorts: dict[str, Mapping[str, ScoreMetric]] = {}
        for cohort, metrics in self.cohort_metrics.items():
            if not isinstance(cohort, str) or not cohort.strip() or not isinstance(metrics, Mapping):
                raise EvaluationValidationError("cohort metrics must map cohort names to metric mappings")
            if any(
                not isinstance(name, str) or not name.strip() or not isinstance(metric, ScoreMetric)
                for name, metric in metrics.items()
            ):
                raise EvaluationValidationError("cohort metrics must contain ScoreMetric values")
            frozen_cohorts[cohort] = MappingProxyType(dict(metrics))
        object.__setattr__(self, "cohort_metrics", MappingProxyType(frozen_cohorts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "case_count": self.case_count,
            "attempt_count": self.attempt_count,
            "failed_attempt_count": self.failed_attempt_count,
            "completed_case_count": self.completed_case_count,
            "duplicate_finding_count": self.duplicate_finding_count,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "clean_checkpoint_count": self.clean_checkpoint_count,
            "false_interruption_count": self.false_interruption_count,
            "escalated_case_count": self.escalated_case_count,
            "high_critical_case_count": self.high_critical_case_count,
            "high_critical_escalated_count": self.high_critical_escalated_count,
            "stale_finding_count": self.stale_finding_count,
            "deterministic_overlap_count": self.deterministic_overlap_count,
            "metrics": {name: metric.to_dict() for name, metric in sorted(self.metrics.items())},
            "cohort_metrics": {
                cohort: {name: metric.to_dict() for name, metric in sorted(metrics.items())}
                for cohort, metrics in sorted(self.cohort_metrics.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArmScorecard":
        return cls(
            arm_id=value["arm_id"],
            case_count=value["case_count"],
            attempt_count=value["attempt_count"],
            failed_attempt_count=value["failed_attempt_count"],
            completed_case_count=value["completed_case_count"],
            duplicate_finding_count=value["duplicate_finding_count"],
            true_positive_count=value["true_positive_count"],
            false_positive_count=value["false_positive_count"],
            false_negative_count=value["false_negative_count"],
            clean_checkpoint_count=value["clean_checkpoint_count"],
            false_interruption_count=value["false_interruption_count"],
            escalated_case_count=value.get("escalated_case_count", 0),
            high_critical_case_count=value.get("high_critical_case_count", 0),
            high_critical_escalated_count=value.get("high_critical_escalated_count", 0),
            stale_finding_count=value.get("stale_finding_count", 0),
            deterministic_overlap_count=value.get("deterministic_overlap_count", 0),
            metrics={name: ScoreMetric.from_dict(metric) for name, metric in value["metrics"].items()},
            cohort_metrics={
                cohort: {name: ScoreMetric.from_dict(metric) for name, metric in metrics.items()}
                for cohort, metrics in value.get("cohort_metrics", {}).items()
            },
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class PairedArmComparison:
    baseline_arm_id: str
    arm_id: str
    metric: str
    support: int
    delta: float
    lower_95: float
    upper_95: float

    def __post_init__(self) -> None:
        if not self.baseline_arm_id or not self.arm_id or not self.metric:
            raise EvaluationValidationError("paired comparison identifiers must be non-empty")
        if not isinstance(self.support, int) or self.support < 1:
            raise EvaluationValidationError("paired comparison support must be positive")
        if any(not math.isfinite(value) for value in (self.delta, self.lower_95, self.upper_95)):
            raise EvaluationValidationError("paired comparison values must be finite")
        if self.lower_95 > self.delta or self.delta > self.upper_95:
            raise EvaluationValidationError("paired comparison confidence interval must contain the delta")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_arm_id": self.baseline_arm_id,
            "arm_id": self.arm_id,
            "metric": self.metric,
            "support": self.support,
            "delta": self.delta,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairedArmComparison":
        return cls(
            baseline_arm_id=value["baseline_arm_id"],
            arm_id=value["arm_id"],
            metric=value["metric"],
            support=value["support"],
            delta=value["delta"],
            lower_95=value["lower_95"],
            upper_95=value["upper_95"],
        )


@dataclass(frozen=True)
class MatchedArmScorecard:
    manifest_id: str
    truth_artifact_id: str
    arms: tuple[ArmScorecard, ...]
    paired_comparisons: tuple[PairedArmComparison, ...] = field(default_factory=tuple)
    schema_version: str = EVALUATION_SCHEMA_VERSION
    scorecard_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported scorecard schema_version: {self.schema_version}")
        object.__setattr__(self, "arms", tuple(self.arms))
        if not self.arms:
            raise EvaluationValidationError("matched-arm scorecard requires at least one arm")
        if any(not isinstance(arm, ArmScorecard) for arm in self.arms):
            raise EvaluationValidationError("matched-arm scorecard arms must use ArmScorecard")
        object.__setattr__(self, "paired_comparisons", tuple(self.paired_comparisons))
        if any(not isinstance(item, PairedArmComparison) for item in self.paired_comparisons):
            raise EvaluationValidationError("paired comparisons must use PairedArmComparison")
        payload = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "truth_artifact_id": self.truth_artifact_id,
            "arms": [arm.to_dict() for arm in self.arms],
            "paired_comparisons": [item.to_dict() for item in self.paired_comparisons],
        }
        object.__setattr__(self, "scorecard_id", content_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "truth_artifact_id": self.truth_artifact_id,
            "arms": [arm.to_dict() for arm in self.arms],
            "paired_comparisons": [item.to_dict() for item in self.paired_comparisons],
            "scorecard_id": self.scorecard_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchedArmScorecard":
        scorecard = cls(
            manifest_id=value["manifest_id"],
            truth_artifact_id=value["truth_artifact_id"],
            arms=tuple(ArmScorecard.from_dict(arm) for arm in value["arms"]),
            paired_comparisons=tuple(
                PairedArmComparison.from_dict(item) for item in value.get("paired_comparisons", [])
            ),
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
        )
        supplied_id = value.get("scorecard_id")
        if supplied_id is not None and supplied_id != scorecard.scorecard_id:
            raise EvaluationValidationError("scorecard_id does not match the scorecard content")
        return scorecard


@dataclass(frozen=True)
class GateRule:
    metric: str
    comparator: GateComparator
    threshold: float
    minimum_support: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise EvaluationValidationError("gate rule metric must be a non-empty string")
        if not isinstance(self.threshold, (int, float)) or isinstance(self.threshold, bool):
            raise EvaluationValidationError("gate threshold must be numeric")
        if not isinstance(self.comparator, GateComparator):
            raise EvaluationValidationError("gate comparator must be a GateComparator")
        if not math.isfinite(self.threshold):
            raise EvaluationValidationError("gate threshold must be finite")
        if not isinstance(self.minimum_support, int) or self.minimum_support < 1:
            raise EvaluationValidationError("gate minimum_support must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "comparator": self.comparator.value,
            "threshold": self.threshold,
            "minimum_support": self.minimum_support,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateRule":
        return cls(
            metric=value["metric"],
            comparator=GateComparator(value["comparator"]),
            threshold=value["threshold"],
            minimum_support=value.get("minimum_support", 1),
        )


def rollout_gate_spec_hash(gate_name: str, arm_id: str, rules: Sequence[GateRule]) -> str:
    """Freeze the maintainer-approved gate identity separately from measured evidence."""
    return content_hash({
        "gate_name": gate_name,
        "arm_id": arm_id,
        "rules": [rule.to_dict() for rule in rules],
    })


@dataclass(frozen=True)
class GateRuleResult:
    rule: GateRule
    status: GateStatus
    observed: Optional[ScoreMetric]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule, GateRule):
            raise EvaluationValidationError("gate rule result rule must be a GateRule")
        if not isinstance(self.status, GateStatus):
            raise EvaluationValidationError("gate rule result status must be a GateStatus")
        if self.observed is not None and not isinstance(self.observed, ScoreMetric):
            raise EvaluationValidationError("gate rule result observed value must be a ScoreMetric")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise EvaluationValidationError("gate rule result reason must be a non-empty string")
        if (
            self.observed is None
            or self.observed.status is not MeasurementStatus.COMPLETE
            or self.observed.support < self.rule.minimum_support
        ):
            expected_status = GateStatus.NOT_EVALUABLE
        else:
            assert self.observed.value is not None
            passed = (
                self.observed.value >= self.rule.threshold
                if self.rule.comparator is GateComparator.AT_LEAST
                else self.observed.value <= self.rule.threshold
            )
            expected_status = GateStatus.PASSED if passed else GateStatus.FAILED
        if self.status is not expected_status:
            raise EvaluationValidationError("gate rule result status does not match its evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "status": self.status.value,
            "observed": self.observed.to_dict() if self.observed else None,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateRuleResult":
        return cls(
            rule=GateRule.from_dict(value["rule"]),
            status=GateStatus(value["status"]),
            observed=ScoreMetric.from_dict(value["observed"]) if value.get("observed") else None,
            reason=value["reason"],
        )


@dataclass(frozen=True)
class RolloutGateDecision:
    gate_name: str
    arm_id: str
    scorecard_id: str
    status: GateStatus
    rule_results: tuple[GateRuleResult, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION
    gate_spec_hash: str = field(init=False)
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationValidationError(f"unsupported gate schema_version: {self.schema_version}")
        if not isinstance(self.gate_name, str) or not self.gate_name.strip():
            raise EvaluationValidationError("gate_name must be a non-empty string")
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise EvaluationValidationError("gate arm_id must be a non-empty string")
        if not isinstance(self.status, GateStatus):
            raise EvaluationValidationError("gate status must be a GateStatus")
        object.__setattr__(self, "rule_results", tuple(self.rule_results))
        if not self.rule_results:
            raise EvaluationValidationError("rollout gate decision requires rule results")
        if any(not isinstance(result, GateRuleResult) for result in self.rule_results):
            raise EvaluationValidationError("rollout gate decision results must use GateRuleResult")
        expected_status = (
            GateStatus.FAILED
            if any(result.status is GateStatus.FAILED for result in self.rule_results)
            else GateStatus.NOT_EVALUABLE
            if any(result.status is GateStatus.NOT_EVALUABLE for result in self.rule_results)
            else GateStatus.PASSED
        )
        if self.status is not expected_status:
            raise EvaluationValidationError("rollout gate status does not match its rule results")
        object.__setattr__(
            self,
            "gate_spec_hash",
            rollout_gate_spec_hash(
                self.gate_name,
                self.arm_id,
                tuple(result.rule for result in self.rule_results),
            ),
        )
        payload = {
            "schema_version": self.schema_version,
            "gate_name": self.gate_name,
            "arm_id": self.arm_id,
            "scorecard_id": self.scorecard_id,
            "gate_spec_hash": self.gate_spec_hash,
            "status": self.status.value,
            "rule_results": [result.to_dict() for result in self.rule_results],
        }
        object.__setattr__(self, "decision_id", content_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_name": self.gate_name,
            "arm_id": self.arm_id,
            "scorecard_id": self.scorecard_id,
            "gate_spec_hash": self.gate_spec_hash,
            "status": self.status.value,
            "rule_results": [result.to_dict() for result in self.rule_results],
            "decision_id": self.decision_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutGateDecision":
        decision = cls(
            gate_name=value["gate_name"],
            arm_id=value["arm_id"],
            scorecard_id=value["scorecard_id"],
            status=GateStatus(value["status"]),
            rule_results=tuple(GateRuleResult.from_dict(item) for item in value["rule_results"]),
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
        )
        supplied_id = value.get("decision_id")
        supplied_spec_hash = value.get("gate_spec_hash")
        if supplied_spec_hash is not None and supplied_spec_hash != decision.gate_spec_hash:
            raise EvaluationValidationError("gate_spec_hash does not match the gate rules")
        if supplied_id is not None and supplied_id != decision.decision_id:
            raise EvaluationValidationError("decision_id does not match the gate decision content")
        return decision


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ratio(numerator: float, denominator: float, support: int) -> ScoreMetric:
    if denominator == 0:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, support)
    return ScoreMetric(MeasurementStatus.COMPLETE, numerator / denominator, support)


def _summed_measurement(
    records: Iterable[EvaluationRunRecord], attribute: str, *, expected_count: Optional[int] = None
) -> ScoreMetric:
    records = tuple(records)
    values: list[float] = []
    partial = False
    for record in records:
        measurement = getattr(record, attribute)
        if measurement.status is MeasurementStatus.UNAVAILABLE:
            partial = True
            continue
        values.append(float(measurement.value))
        if measurement.status is MeasurementStatus.PARTIAL:
            partial = True
    if not values:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
    if expected_count is not None and len({record.case_id for record in records}) < expected_count:
        partial = True
    status = MeasurementStatus.PARTIAL if partial else MeasurementStatus.COMPLETE
    return ScoreMetric(status, sum(values), len(values))


def _latency_metric(
    records: Sequence[EvaluationRunRecord], percentile: float, *, expected_count: Optional[int] = None
) -> ScoreMetric:
    values = [
        float(record.latency_seconds.value)
        for record in records
        if record.latency_seconds.status is MeasurementStatus.COMPLETE
        and record.latency_seconds.value is not None
    ]
    if not values:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
    complete = len(values) == len(records) and (expected_count is None or len(records) == expected_count)
    status = MeasurementStatus.COMPLETE if complete else MeasurementStatus.PARTIAL
    return ScoreMetric(status, _percentile(values, percentile), len(values))


def _measurement_ratio(numerator: ScoreMetric, denominator: float, support: int) -> ScoreMetric:
    if numerator.status is MeasurementStatus.UNAVAILABLE or numerator.value is None or denominator <= 0:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
    return ScoreMetric(numerator.status, numerator.value / denominator, support)


def _developer_hours(cases: Sequence[CheckpointCase]) -> ScoreMetric:
    values = [case.developer_elapsed_seconds for case in cases if case.developer_elapsed_seconds is not None]
    if not values:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
    status = MeasurementStatus.COMPLETE if len(values) == len(cases) else MeasurementStatus.PARTIAL
    return ScoreMetric(status, sum(values) / 3600.0, len(values))


def _stage_latency_metric(
    records: Sequence[EvaluationRunRecord], stage: str, percentile: float, *, expected_count: Optional[int] = None
) -> ScoreMetric:
    values: list[float] = []
    eligible = 0
    for record in records:
        measurement = record.stage_latencies_seconds.get(stage)
        if measurement is None:
            continue
        eligible += 1
        if measurement.status is MeasurementStatus.COMPLETE and measurement.value is not None:
            values.append(float(measurement.value))
    if not values:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
    complete = (
        len(values) == eligible == len(records)
        and (expected_count is None or len(records) == expected_count)
    )
    status = MeasurementStatus.COMPLETE if complete else MeasurementStatus.PARTIAL
    return ScoreMetric(status, _percentile(values, percentile), len(values))


def _lineage_roots(manifest: EvaluationManifest) -> dict[str, str]:
    case_by_id = {case.case_id: case for case in manifest.cases}
    roots: dict[str, str] = {}
    for case in manifest.cases:
        current = case
        while current.parent_case_id is not None:
            current = case_by_id[current.parent_case_id]
        roots[case.case_id] = current.case_id
    return roots


def _basic_case_metrics(
    cases: Sequence[CheckpointCase],
    truth_by_case: Mapping[str, CheckpointTruth],
    terminals_by_case: Mapping[str, EvaluationRunRecord],
) -> dict[str, ScoreMetric]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    weighted_true_positive = 0.0
    weighted_truth = 0.0
    completed = 0
    clean_count = 0
    false_interruptions = 0
    unavailable_coverage = 0
    failed_or_missing = 0
    clean_evidence_missing = 0
    partial_coverage = 0
    for case in cases:
        truth = truth_by_case[case.case_id]
        terminal = terminals_by_case.get(case.case_id)
        observations = (
            tuple(
                finding for finding in terminal.findings
                if finding.lifecycle_state is FindingLifecycleState.ACTIVE
            )
            if terminal and terminal.state is EvaluationRunState.COMPLETED
            else ()
        )
        if terminal and terminal.state is EvaluationRunState.COMPLETED:
            completed += 1
        observed = {finding.fingerprint for finding in observations}
        expected = {finding.fingerprint for finding in truth.findings}
        expected_by_fingerprint = {finding.fingerprint: finding for finding in truth.findings}
        true_positive += len(observed & expected)
        false_positive += len(observed - expected)
        false_negative += len(expected - observed)
        weighted_true_positive += sum(
            _SEVERITY_WEIGHT[expected_by_fingerprint[fingerprint].severity]
            for fingerprint in observed & expected
        )
        weighted_truth += sum(_SEVERITY_WEIGHT[finding.severity] for finding in truth.findings)
        failed_or_missing += terminal is None or terminal.state is not EvaluationRunState.COMPLETED
        unavailable_coverage += (
            terminal is not None
            and (
                terminal.state is EvaluationRunState.COVERAGE_UNAVAILABLE
                or bool(terminal.coverage_issues)
            )
        )
        partial_coverage += terminal is not None and bool(terminal.coverage_issues)
        if truth.is_clean:
            clean_count += 1
            false_interruptions += bool(observed)
            clean_evidence_missing += (
                terminal is None
                or terminal.state is not EvaluationRunState.COMPLETED
                or bool(terminal.coverage_issues)
            )
    false_interruption_metric = _ratio(false_interruptions, clean_count, clean_count)
    if clean_evidence_missing and false_interruption_metric.status is MeasurementStatus.COMPLETE:
        false_interruption_metric = ScoreMetric(
            MeasurementStatus.PARTIAL,
            false_interruption_metric.value,
            false_interruption_metric.support,
        )
    metrics = {
        "case_support": ScoreMetric(MeasurementStatus.COMPLETE, float(len(cases)), len(cases)),
        "structured_output_rate": _ratio(completed, len(cases), len(cases)),
        "verified_precision": _ratio(
            true_positive, true_positive + false_positive, true_positive + false_positive
        ),
        "verified_recall": _ratio(
            true_positive, true_positive + false_negative, true_positive + false_negative
        ),
        "severity_weighted_recall": _ratio(
            weighted_true_positive, weighted_truth, true_positive + false_negative
        ),
        "false_interruptions_per_clean_checkpoint": false_interruption_metric,
        "failure_or_missing_case_rate": _ratio(
            failed_or_missing, len(cases), len(cases)
        ),
        "unavailable_coverage_rate": _ratio(
            unavailable_coverage, len(cases), len(cases)
        ),
    }
    if partial_coverage:
        for name in ("verified_recall", "severity_weighted_recall"):
            metric = metrics[name]
            if metric.status is MeasurementStatus.COMPLETE:
                metrics[name] = ScoreMetric(MeasurementStatus.PARTIAL, metric.value, metric.support)
    return metrics


def _time_and_lineage_metrics(
    manifest: EvaluationManifest,
    truth_by_case: Mapping[str, CheckpointTruth],
    terminals_by_case: Mapping[str, EvaluationRunRecord],
) -> tuple[dict[str, ScoreMetric], int, int]:
    roots = _lineage_roots(manifest)
    case_by_id = {case.case_id: case for case in manifest.cases}

    def lineage_depth(case: CheckpointCase) -> int:
        depth = 0
        while case.parent_case_id is not None:
            depth += 1
            case = case_by_id[case.parent_case_id]
        return depth

    def is_at_or_after(case_id: str, earliest_case_id: str) -> bool:
        current = case_by_id[case_id]
        while True:
            if current.case_id == earliest_case_id:
                return True
            if current.parent_case_id is None:
                return False
            current = case_by_id[current.parent_case_id]

    def earlier_case_id(candidate_id: str, current_id: str) -> str:
        """Choose the earlier opportunity without depending on manifest order."""
        if candidate_id == current_id:
            return current_id
        if is_at_or_after(current_id, candidate_id):
            return candidate_id
        if is_at_or_after(candidate_id, current_id):
            return current_id
        candidate = case_by_id[candidate_id]
        current = case_by_id[current_id]
        if (
            candidate.lineage_elapsed_seconds is not None
            and current.lineage_elapsed_seconds is not None
            and candidate.lineage_elapsed_seconds != current.lineage_elapsed_seconds
        ):
            return (
                candidate_id
                if candidate.lineage_elapsed_seconds < current.lineage_elapsed_seconds
                else current_id
            )
        candidate_key = (lineage_depth(candidate), candidate.case_id)
        current_key = (lineage_depth(current), current.case_id)
        return candidate_id if candidate_key < current_key else current_id

    ordered_cases = sorted(
        manifest.cases,
        key=lambda case: (
            roots[case.case_id],
            case.lineage_elapsed_seconds if case.lineage_elapsed_seconds is not None else math.inf,
            lineage_depth(case),
            case.case_id,
        ),
    )
    truth_instances: dict[tuple[str, str], tuple[Any, str]] = {}
    withdrawals: dict[tuple[str, str], str] = {}
    for case in manifest.cases:
        for finding in truth_by_case[case.case_id].findings:
            key = (roots[case.case_id], finding.fingerprint)
            earliest_case_id = finding.earliest_case_id or case.case_id
            previous = truth_instances.get(key)
            if previous is None:
                truth_instances[key] = (finding, earliest_case_id)
            else:
                previous_finding, previous_case_id = previous
                if (
                    finding.severity,
                    finding.earliest_opportunity,
                    finding.required_context,
                ) != (
                    previous_finding.severity,
                    previous_finding.earliest_opportunity,
                    previous_finding.required_context,
                ):
                    raise EvaluationValidationError(
                        "repeated finding fingerprint has inconsistent truth metadata"
                    )
                if earlier_case_id(earliest_case_id, previous_case_id) == earliest_case_id:
                    truth_instances[key] = (finding, earliest_case_id)
            if finding.withdrawn_at_case_id:
                previous_withdrawal = withdrawals.get(key)
                if (
                    previous_withdrawal is not None
                    and previous_withdrawal != finding.withdrawn_at_case_id
                ):
                    raise EvaluationValidationError(
                        "repeated finding fingerprint has inconsistent withdrawal checkpoint"
                    )
                withdrawals[key] = finding.withdrawn_at_case_id

    first_detection: dict[tuple[str, str], CheckpointCase] = {}
    for case in ordered_cases:
        terminal = terminals_by_case.get(case.case_id)
        if not terminal or terminal.state is not EvaluationRunState.COMPLETED:
            continue
        for observation in terminal.findings:
            key = (roots[case.case_id], observation.fingerprint)
            truth_instance = truth_instances.get(key)
            withdrawal_case_id = withdrawals.get(key)
            precedes_withdrawal = (
                withdrawal_case_id is None
                or (
                    case.case_id != withdrawal_case_id
                    and is_at_or_after(withdrawal_case_id, case.case_id)
                )
            )
            if (
                observation.lifecycle_state is FindingLifecycleState.ACTIVE
                and truth_instance is not None
                and is_at_or_after(case.case_id, truth_instance[1])
                and precedes_withdrawal
            ):
                first_detection.setdefault(key, case)

    delays: list[float] = []
    timing_incomplete = False
    first_events: dict[ReviewEvent, int] = {event: 0 for event in ReviewEvent}
    for key, (_, earliest_case_id) in truth_instances.items():
        detection_case = first_detection.get(key)
        if detection_case is None:
            timing_incomplete = True
            continue
        first_events[detection_case.event] += 1
        earliest_case = case_by_id[earliest_case_id]
        terminal = terminals_by_case[detection_case.case_id]
        if earliest_case.lineage_elapsed_seconds is None or detection_case.lineage_elapsed_seconds is None:
            timing_incomplete = True
            continue
        latency = (
            float(terminal.latency_seconds.value)
            if terminal.latency_seconds.status is MeasurementStatus.COMPLETE
            and terminal.latency_seconds.value is not None
            else None
        )
        if latency is None:
            timing_incomplete = True
            continue
        delays.append(max(
            0.0,
            detection_case.lineage_elapsed_seconds - earliest_case.lineage_elapsed_seconds + latency,
        ))

    metrics: dict[str, ScoreMetric] = {}
    total_truth = len(truth_instances)
    for event, count in first_events.items():
        metrics[f"incremental_recall_{event.value}"] = _ratio(count, total_truth, total_truth)
    if delays:
        status = (
            MeasurementStatus.PARTIAL
            if timing_incomplete or len(delays) != total_truth
            else MeasurementStatus.COMPLETE
        )
        metrics["time_to_first_valid_finding_p50_seconds"] = ScoreMetric(
            status, _percentile(delays, 0.50), len(delays)
        )
        metrics["time_to_first_valid_finding_p95_seconds"] = ScoreMetric(
            status, _percentile(delays, 0.95), len(delays)
        )
    else:
        metrics["time_to_first_valid_finding_p50_seconds"] = ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
        metrics["time_to_first_valid_finding_p95_seconds"] = ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)

    stale_count = 0
    withdrawal_evidence_incomplete = False
    withdrawn_count = len(withdrawals)
    for key, withdrawal_case_id in withdrawals.items():
        withdrawal_terminal = terminals_by_case.get(withdrawal_case_id)
        if withdrawal_terminal is None or withdrawal_terminal.state is not EvaluationRunState.COMPLETED:
            withdrawal_evidence_incomplete = True
        for case in ordered_cases:
            if roots[case.case_id] != key[0]:
                continue
            after_withdrawal = is_at_or_after(case.case_id, withdrawal_case_id)
            if not after_withdrawal:
                continue
            terminal = terminals_by_case.get(case.case_id)
            if terminal is None or terminal.state is not EvaluationRunState.COMPLETED:
                withdrawal_evidence_incomplete = True
            if terminal and terminal.state is EvaluationRunState.COMPLETED and any(
                finding.fingerprint == key[1] and finding.lifecycle_state is FindingLifecycleState.ACTIVE
                for finding in terminal.findings
            ):
                stale_count += 1
                break
    metrics["stale_findings_withdrawn_rate"] = _ratio(
        withdrawn_count - stale_count, withdrawn_count, withdrawn_count
    )
    withdrawal_metric = metrics["stale_findings_withdrawn_rate"]
    if withdrawal_evidence_incomplete and withdrawal_metric.status is MeasurementStatus.COMPLETE:
        metrics["stale_findings_withdrawn_rate"] = ScoreMetric(
            MeasurementStatus.PARTIAL,
            withdrawal_metric.value,
            withdrawal_metric.support,
        )
    return metrics, stale_count, len(first_detection)


def _paired_interval(values: Sequence[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom <= len(_TWO_SIDED_95_T_CRITICAL):
        critical_value = _TWO_SIDED_95_T_CRITICAL[degrees_of_freedom - 1]
    else:
        # Cornish-Fisher expansion for t(0.975, df), retaining the finite-sample
        # correction instead of silently switching to the narrower normal bound.
        z = 1.959963984540054
        critical_value = (
            z
            + (z ** 3 + z) / (4 * degrees_of_freedom)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * degrees_of_freedom ** 2)
        )
    margin = critical_value * math.sqrt(variance / len(values))
    return mean, mean - margin, mean + margin


def _paired_comparisons(
    manifest: EvaluationManifest,
    truth_by_case: Mapping[str, CheckpointTruth],
    terminal_by_arm: Mapping[str, Mapping[str, EvaluationRunRecord]],
    baseline_arm_id: Optional[str],
) -> tuple[PairedArmComparison, ...]:
    arm_ids = sorted(terminal_by_arm)
    if len(arm_ids) < 2:
        return ()
    incumbent_ids = [
        arm.arm_id for arm in manifest.arms
        if arm.enabled and arm.kind is EvaluationArmKind.GENERAL_REVIEW
    ]
    deterministic_ids = [
        arm.arm_id for arm in manifest.arms
        if arm.enabled and arm.kind is EvaluationArmKind.DETERMINISTIC
    ]
    baseline = baseline_arm_id or (
        incumbent_ids[0] if incumbent_ids else deterministic_ids[0] if deterministic_ids else arm_ids[0]
    )
    if baseline not in arm_ids:
        raise EvaluationValidationError(f"paired baseline arm is disabled or unknown: {baseline}")

    def outcomes(arm_id: str, case: CheckpointCase) -> dict[str, Optional[float]]:
        terminal = terminal_by_arm[arm_id].get(case.case_id)
        completed = terminal is not None and terminal.state is EvaluationRunState.COMPLETED
        observations = {
            finding.fingerprint
            for finding in (terminal.findings if completed else ())
            if finding.lifecycle_state is FindingLifecycleState.ACTIVE
        }
        expected = {finding.fingerprint for finding in truth_by_case[case.case_id].findings}
        recall = len(observations & expected) / len(expected) if expected else 1.0
        result: dict[str, Optional[float]] = {
            "structured_output_rate": float(completed),
            "case_recall": recall,
            "false_interruption_rate": (
                float(bool(observations))
                if completed and truth_by_case[case.case_id].is_clean
                else None
            ),
        }
        for metric, measurement in (
            ("latency_seconds", terminal.latency_seconds if terminal is not None else None),
            ("tokens", terminal.tokens if terminal is not None else None),
            ("cost_usd", terminal.cost_usd if terminal is not None else None),
        ):
            result[metric] = (
                float(measurement.value)
                if measurement is not None
                and measurement.status is MeasurementStatus.COMPLETE
                and measurement.value is not None
                else None
            )
        return result

    comparisons: list[PairedArmComparison] = []
    for arm_id in arm_ids:
        if arm_id == baseline:
            continue
        by_metric: dict[str, list[float]] = {}
        for case in manifest.cases:
            baseline_outcomes = outcomes(baseline, case)
            arm_outcomes = outcomes(arm_id, case)
            for metric in baseline_outcomes:
                if metric == "case_recall" and truth_by_case[case.case_id].is_clean:
                    continue
                if metric == "false_interruption_rate" and not truth_by_case[case.case_id].is_clean:
                    continue
                baseline_value = baseline_outcomes[metric]
                arm_value = arm_outcomes[metric]
                if baseline_value is None or arm_value is None:
                    continue
                by_metric.setdefault(metric, []).append(arm_value - baseline_value)
        for metric, values in sorted(by_metric.items()):
            delta, lower, upper = _paired_interval(values)
            comparisons.append(PairedArmComparison(
                baseline_arm_id=baseline,
                arm_id=arm_id,
                metric=metric,
                support=len(values),
                delta=delta,
                lower_95=lower,
                upper_95=upper,
            ))
    return tuple(comparisons)


def _validate_records(
    manifest: EvaluationManifest,
    records: Sequence[EvaluationRunRecord],
) -> dict[tuple[str, str], list[EvaluationRunRecord]]:
    case_by_id = {case.case_id: case for case in manifest.cases}
    arms_by_id = {arm.arm_id: arm for arm in manifest.arms if arm.enabled}
    arm_ids = set(arms_by_id)
    grouped: dict[tuple[str, str], list[EvaluationRunRecord]] = {}
    record_ids: set[str] = set()
    attempts: set[tuple[str, str, int]] = set()
    for record in records:
        if record.record_id in record_ids:
            raise EvaluationValidationError("duplicate run record identity")
        record_ids.add(record.record_id)
        if record.manifest_id != manifest.manifest_id:
            raise EvaluationValidationError("run record belongs to a different manifest")
        case = case_by_id.get(record.case_id)
        if case is None:
            raise EvaluationValidationError(f"run record names unknown case {record.case_id}")
        if record.arm_id not in arm_ids:
            raise EvaluationValidationError(f"run record names disabled or unknown arm {record.arm_id}")
        if record.snapshot_id != case.snapshot_id:
            raise EvaluationValidationError("run record snapshot does not match its checkpoint case")
        arm = arms_by_id[record.arm_id]
        validate_run_model_telemetry(arm, record)
        attempt_key = (record.case_id, record.arm_id, record.attempt)
        if attempt_key in attempts:
            raise EvaluationValidationError("run attempts must be unique per case and arm")
        attempts.add(attempt_key)
        grouped.setdefault((record.case_id, record.arm_id), []).append(record)
    for pair, pair_records in grouped.items():
        ordered = sorted(pair_records, key=lambda record: record.attempt)
        if [record.attempt for record in ordered] != list(range(1, len(ordered) + 1)):
            raise EvaluationValidationError(f"case/arm pair {pair} omits a retained attempt")
        terminals = [record for record in ordered if record.terminal]
        if len(terminals) > 1:
            raise EvaluationValidationError(f"case/arm pair {pair} has multiple terminal records")
        if terminals and terminals[0] is not ordered[-1]:
            raise EvaluationValidationError(f"case/arm pair {pair} has attempts after its terminal record")
        if any(record.state is EvaluationRunState.COMPLETED and not record.terminal for record in ordered):
            raise EvaluationValidationError(f"case/arm pair {pair} has a non-terminal completed result")
    return grouped


def score_matched_arms(
    manifest: EvaluationManifest,
    truth_artifact: TruthArtifact,
    records: Sequence[EvaluationRunRecord],
    *,
    baseline_arm_id: Optional[str] = None,
) -> MatchedArmScorecard:
    """Score every enabled arm over the same manifest cases.

    Missing terminal records, failures, malformed output, and unavailable coverage
    remain in the case denominator instead of being silently replaced or dropped.
    """
    truth_artifact.validate_for_manifest(manifest)
    truth_by_case: dict[str, CheckpointTruth] = {truth.case_id: truth for truth in truth_artifact.truths}
    grouped = _validate_records(manifest, records)
    arm_scorecards: list[ArmScorecard] = []
    terminal_by_arm: dict[str, dict[str, EvaluationRunRecord]] = {}
    for arm in sorted((item for item in manifest.arms if item.enabled), key=lambda item: item.arm_id):
        arm_records = [record for record in records if record.arm_id == arm.arm_id]
        terminals_by_case: dict[str, EvaluationRunRecord] = {}
        for case in manifest.cases:
            pair_records = grouped.get((case.case_id, arm.arm_id), [])
            terminal = next((record for record in pair_records if record.terminal), None)
            if terminal is not None:
                terminals_by_case[case.case_id] = terminal
        terminal_by_arm[arm.arm_id] = terminals_by_case

        true_positive_count = 0
        false_positive_count = 0
        false_negative_count = 0
        duplicate_finding_count = 0
        false_interruption_count = 0
        weighted_true_positive = 0.0
        weighted_truth = 0.0
        clean_checkpoint_count = 0
        escalated_case_count = 0
        high_critical_case_count = 0
        high_critical_escalated_count = 0
        deterministic_overlap_count = 0
        deterministic_overlap_known = 0
        observed_finding_count = 0
        completed_records: list[EvaluationRunRecord] = []
        terminal_records: list[EvaluationRunRecord] = []
        escalation_known = 0
        clean_evidence_missing = 0
        partial_coverage_case_count = 0
        for case in manifest.cases:
            truth = truth_by_case[case.case_id]
            truth_by_fingerprint = {finding.fingerprint: finding for finding in truth.findings}
            weighted_truth += sum(_SEVERITY_WEIGHT[finding.severity] for finding in truth.findings)
            terminal = terminals_by_case.get(case.case_id)
            observations = (
                tuple(
                    finding for finding in terminal.findings
                    if finding.lifecycle_state is FindingLifecycleState.ACTIVE
                )
                if terminal and terminal.state is EvaluationRunState.COMPLETED
                else ()
            )
            observation_fingerprints = [finding.fingerprint for finding in observations]
            unique_observations = set(observation_fingerprints)
            duplicate_finding_count += len(observation_fingerprints) - len(unique_observations)
            truth_fingerprints = set(truth_by_fingerprint)
            matched = unique_observations & truth_fingerprints
            true_positive_count += len(matched)
            false_positive_count += len(unique_observations - truth_fingerprints)
            false_negative_count += len(truth_fingerprints - unique_observations)
            weighted_true_positive += sum(_SEVERITY_WEIGHT[truth_by_fingerprint[item].severity] for item in matched)
            if truth.is_clean:
                clean_checkpoint_count += 1
                false_interruption_count += bool(unique_observations)
                clean_evidence_missing += (
                    terminal is None
                    or terminal.state is not EvaluationRunState.COMPLETED
                    or bool(terminal.coverage_issues)
                )
            partial_coverage_case_count += terminal is not None and bool(terminal.coverage_issues)
            high_critical = any(
                finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
                for finding in truth.findings
            )
            high_critical_case_count += high_critical
            if terminal and terminal.escalated is not None:
                escalation_known += 1
                escalated_case_count += terminal.escalated
                high_critical_escalated_count += terminal.escalated and high_critical
            for finding in observations:
                observed_finding_count += 1
                if finding.deterministic_overlap is not None:
                    deterministic_overlap_known += 1
                    deterministic_overlap_count += finding.deterministic_overlap
            if terminal and terminal.state is EvaluationRunState.COMPLETED:
                completed_records.append(terminal)
            if terminal is not None:
                terminal_records.append(terminal)

        failed_attempt_count = sum(record.state is not EvaluationRunState.COMPLETED for record in arm_records)
        completed_case_count = len(completed_records)
        total_cost = _summed_measurement(
            arm_records, "cost_usd", expected_count=len(manifest.cases)
        )
        developer_hours = _developer_hours(manifest.cases)
        if developer_hours.value is None or developer_hours.value <= 0:
            false_per_hour = ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
            cost_per_hour = ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
        else:
            false_per_hour = ScoreMetric(
                developer_hours.status,
                false_interruption_count / developer_hours.value,
                developer_hours.support,
            )
            cost_per_hour = _measurement_ratio(total_cost, developer_hours.value, total_cost.support)
            if (
                developer_hours.status is MeasurementStatus.PARTIAL
                and cost_per_hour.status is MeasurementStatus.COMPLETE
            ):
                cost_per_hour = ScoreMetric(MeasurementStatus.PARTIAL, cost_per_hour.value, cost_per_hour.support)
        lineage_metrics, stale_finding_count, unique_detected_count = _time_and_lineage_metrics(
            manifest, truth_by_case, terminals_by_case
        )
        false_interruption_metric = _ratio(
            false_interruption_count,
            clean_checkpoint_count,
            clean_checkpoint_count,
        )
        if clean_evidence_missing and false_interruption_metric.status is MeasurementStatus.COMPLETE:
            false_interruption_metric = ScoreMetric(
                MeasurementStatus.PARTIAL,
                false_interruption_metric.value,
                false_interruption_metric.support,
            )
        if clean_evidence_missing and false_per_hour.status is MeasurementStatus.COMPLETE:
            false_per_hour = ScoreMetric(
                MeasurementStatus.PARTIAL,
                false_per_hour.value,
                false_per_hour.support,
            )
        missing_or_failed_cases = sum(
            case.case_id not in terminals_by_case
            or terminals_by_case[case.case_id].state is not EvaluationRunState.COMPLETED
            for case in manifest.cases
        )
        unavailable_coverage_cases = sum(
            case.case_id not in terminals_by_case
            or terminals_by_case[case.case_id].state is not EvaluationRunState.COMPLETED
            or bool(terminals_by_case[case.case_id].coverage_issues)
            for case in manifest.cases
        )
        retry_status = (
            MeasurementStatus.UNAVAILABLE
            if not arm_records
            else MeasurementStatus.PARTIAL
            if len({record.case_id for record in arm_records}) < len(manifest.cases)
            else MeasurementStatus.COMPLETE
        )
        metrics = {
            "structured_output_rate": _ratio(completed_case_count, len(manifest.cases), len(manifest.cases)),
            "verified_precision": _ratio(
                true_positive_count,
                true_positive_count + false_positive_count,
                true_positive_count + false_positive_count,
            ),
            "verified_recall": _ratio(
                true_positive_count,
                true_positive_count + false_negative_count,
                true_positive_count + false_negative_count,
            ),
            "severity_weighted_recall": _ratio(
                weighted_true_positive,
                weighted_truth,
                true_positive_count + false_negative_count,
            ),
            "false_interruptions_per_clean_checkpoint": false_interruption_metric,
            "false_interruptions_per_developer_hour": false_per_hour,
            "developer_hours": developer_hours,
            "escalation_precision": _ratio(
                high_critical_escalated_count, escalated_case_count, escalated_case_count
            ),
            "high_critical_escalation_recall": _ratio(
                high_critical_escalated_count, high_critical_case_count, high_critical_case_count
            ),
            "unavailable_coverage_rate": _ratio(
                unavailable_coverage_cases,
                len(manifest.cases),
                len(manifest.cases),
            ),
            "failure_or_missing_case_rate": _ratio(
                missing_or_failed_cases,
                len(manifest.cases),
                len(manifest.cases),
            ),
            "failure_attempt_rate": _ratio(failed_attempt_count, len(arm_records), len(arm_records)),
            "latency_p50_seconds": _latency_metric(
                terminal_records, 0.50, expected_count=len(manifest.cases)
            ),
            "latency_p95_seconds": _latency_metric(
                terminal_records, 0.95, expected_count=len(manifest.cases)
            ),
            "total_tokens": _summed_measurement(
                arm_records, "tokens", expected_count=len(manifest.cases)
            ),
            "total_cost_usd": total_cost,
            "total_retries": ScoreMetric(
                retry_status,
                float(sum(record.retry_count for record in arm_records)) if arm_records else None,
                len(arm_records),
            ),
            "cost_per_developer_hour": cost_per_hour,
            "cost_per_verified_finding": _measurement_ratio(
                total_cost, unique_detected_count, unique_detected_count
            ),
            "deterministic_overlap_rate": _ratio(
                deterministic_overlap_count, deterministic_overlap_known, deterministic_overlap_known
            ),
            **lineage_metrics,
        }
        if escalation_known < len(manifest.cases):
            for name in ("escalation_precision", "high_critical_escalation_recall"):
                metric = metrics[name]
                if metric.status is MeasurementStatus.COMPLETE:
                    metrics[name] = ScoreMetric(MeasurementStatus.PARTIAL, metric.value, metric.support)
        if partial_coverage_case_count:
            for name in ("verified_recall", "severity_weighted_recall"):
                metric = metrics[name]
                if metric.status is MeasurementStatus.COMPLETE:
                    metrics[name] = ScoreMetric(MeasurementStatus.PARTIAL, metric.value, metric.support)
        overlap_metric = metrics["deterministic_overlap_rate"]
        if (
            deterministic_overlap_known < observed_finding_count
            and overlap_metric.status is MeasurementStatus.COMPLETE
        ):
            metrics["deterministic_overlap_rate"] = ScoreMetric(
                MeasurementStatus.PARTIAL,
                overlap_metric.value,
                overlap_metric.support,
            )
        for event in (ReviewEvent.FILE_SAVE, ReviewEvent.WORKTREE_IDLE, ReviewEvent.PRE_COMMIT):
            event_case_count = sum(case.event is event for case in manifest.cases)
            event_records = [
                record for case_id, record in terminals_by_case.items()
                if next(case for case in manifest.cases if case.case_id == case_id).event is event
            ]
            metrics[f"latency_{event.value}_p50_seconds"] = _latency_metric(
                event_records, 0.50, expected_count=event_case_count
            )
            metrics[f"latency_{event.value}_p95_seconds"] = _latency_metric(
                event_records, 0.95, expected_count=event_case_count
            )
        stages = sorted({stage for record in terminal_records for stage in record.stage_latencies_seconds})
        for stage in stages:
            metrics[f"latency_stage_{stage}_p50_seconds"] = _stage_latency_metric(
                terminal_records, stage, 0.50, expected_count=len(manifest.cases)
            )
            metrics[f"latency_stage_{stage}_p95_seconds"] = _stage_latency_metric(
                terminal_records, stage, 0.95, expected_count=len(manifest.cases)
            )
        cohort_metrics = {
            cohort.value: _basic_case_metrics(
                [case for case in manifest.cases if case.cohort is cohort], truth_by_case, terminals_by_case
            )
            for cohort in EvaluationCohort
            if any(case.cohort is cohort for case in manifest.cases)
        }
        arm_scorecards.append(ArmScorecard(
            arm_id=arm.arm_id,
            case_count=len(manifest.cases),
            attempt_count=len(arm_records),
            failed_attempt_count=failed_attempt_count,
            completed_case_count=completed_case_count,
            duplicate_finding_count=duplicate_finding_count,
            true_positive_count=true_positive_count,
            false_positive_count=false_positive_count,
            false_negative_count=false_negative_count,
            clean_checkpoint_count=clean_checkpoint_count,
            false_interruption_count=false_interruption_count,
            escalated_case_count=escalated_case_count,
            high_critical_case_count=high_critical_case_count,
            high_critical_escalated_count=high_critical_escalated_count,
            stale_finding_count=stale_finding_count,
            deterministic_overlap_count=deterministic_overlap_count,
            metrics=metrics,
            cohort_metrics=cohort_metrics,
        ))
    return MatchedArmScorecard(
        manifest_id=manifest.manifest_id,
        truth_artifact_id=truth_artifact.truth_artifact_id,
        arms=tuple(arm_scorecards),
        paired_comparisons=_paired_comparisons(
            manifest,
            truth_by_case,
            terminal_by_arm,
            baseline_arm_id,
        ),
    )


def evaluate_rollout_gate(
    gate_name: str,
    scorecard: MatchedArmScorecard,
    arm_id: str,
    rules: Sequence[GateRule],
) -> RolloutGateDecision:
    """Evaluate explicit thresholds without treating missing evidence as permission."""
    if not gate_name.strip():
        raise EvaluationValidationError("gate_name must be a non-empty string")
    if not rules:
        raise EvaluationValidationError("a rollout gate requires at least one rule")
    arm = next((item for item in scorecard.arms if item.arm_id == arm_id), None)
    if arm is None:
        raise EvaluationValidationError(f"scorecard does not contain arm {arm_id}")

    results: list[GateRuleResult] = []
    for rule in rules:
        observed = arm.metrics.get(rule.metric)
        if observed is None and rule.metric.startswith("paired."):
            parts = rule.metric.split(".")
            if len(parts) == 3 and parts[2] in {"delta", "lower_95", "upper_95"}:
                comparisons = [
                    comparison
                    for comparison in scorecard.paired_comparisons
                    if comparison.arm_id == arm_id and comparison.metric == parts[1]
                ]
                if len(comparisons) == 1:
                    comparison = comparisons[0]
                    observed = ScoreMetric(
                        MeasurementStatus.COMPLETE,
                        float(getattr(comparison, parts[2])),
                        comparison.support,
                    )
        if observed is None and rule.metric.startswith("cohort."):
            parts = rule.metric.split(".", 2)
            if len(parts) == 3:
                observed = arm.cohort_metrics.get(parts[1], {}).get(parts[2])
        if observed is None:
            results.append(GateRuleResult(rule, GateStatus.NOT_EVALUABLE, None, "metric is absent"))
            continue
        if observed.status is not MeasurementStatus.COMPLETE:
            results.append(GateRuleResult(
                rule,
                GateStatus.NOT_EVALUABLE,
                observed,
                f"metric status is {observed.status.value}",
            ))
            continue
        if observed.support < rule.minimum_support:
            results.append(GateRuleResult(
                rule,
                GateStatus.NOT_EVALUABLE,
                observed,
                "metric support is below the required minimum",
            ))
            continue
        assert observed.value is not None
        passed = (
            observed.value >= rule.threshold
            if rule.comparator is GateComparator.AT_LEAST
            else observed.value <= rule.threshold
        )
        results.append(GateRuleResult(
            rule,
            GateStatus.PASSED if passed else GateStatus.FAILED,
            observed,
            "threshold satisfied" if passed else "threshold not satisfied",
        ))

    if any(result.status is GateStatus.FAILED for result in results):
        status = GateStatus.FAILED
    elif any(result.status is GateStatus.NOT_EVALUABLE for result in results):
        status = GateStatus.NOT_EVALUABLE
    else:
        status = GateStatus.PASSED
    return RolloutGateDecision(
        gate_name=gate_name,
        arm_id=arm_id,
        scorecard_id=scorecard.scorecard_id,
        status=status,
        rule_results=tuple(results),
    )
