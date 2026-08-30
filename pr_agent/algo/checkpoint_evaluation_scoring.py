"""Pure matched-arm scoring and rollout-gate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from pr_agent.algo.checkpoint_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    CheckpointTruth,
    EvaluationManifest,
    EvaluationRunRecord,
    EvaluationRunState,
    EvaluationValidationError,
    FindingSeverity,
    GateStatus,
    MeasurementStatus,
    TruthArtifact,
    content_hash,
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
        )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise EvaluationValidationError("scorecard counts must be non-negative integers")
        if not isinstance(self.metrics, Mapping) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(metric, ScoreMetric)
            for name, metric in self.metrics.items()
        ):
            raise EvaluationValidationError("scorecard metrics must map names to ScoreMetric values")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

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
            "metrics": {name: metric.to_dict() for name, metric in sorted(self.metrics.items())},
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
            metrics={name: ScoreMetric.from_dict(metric) for name, metric in value["metrics"].items()},
            schema_version=value.get("schema_version", EVALUATION_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class MatchedArmScorecard:
    manifest_id: str
    truth_artifact_id: str
    arms: tuple[ArmScorecard, ...]
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
        payload = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "truth_artifact_id": self.truth_artifact_id,
            "arms": [arm.to_dict() for arm in self.arms],
        }
        object.__setattr__(self, "scorecard_id", content_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "truth_artifact_id": self.truth_artifact_id,
            "arms": [arm.to_dict() for arm in self.arms],
            "scorecard_id": self.scorecard_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchedArmScorecard":
        scorecard = cls(
            manifest_id=value["manifest_id"],
            truth_artifact_id=value["truth_artifact_id"],
            arms=tuple(ArmScorecard.from_dict(arm) for arm in value["arms"]),
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
        payload = {
            "schema_version": self.schema_version,
            "gate_name": self.gate_name,
            "arm_id": self.arm_id,
            "scorecard_id": self.scorecard_id,
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


def _summed_measurement(records: Iterable[EvaluationRunRecord], attribute: str) -> ScoreMetric:
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
    status = MeasurementStatus.PARTIAL if partial else MeasurementStatus.COMPLETE
    return ScoreMetric(status, sum(values), len(values))


def _latency_metric(records: Sequence[EvaluationRunRecord], percentile: float) -> ScoreMetric:
    values = [
        float(record.latency_seconds.value)
        for record in records
        if record.latency_seconds.status is MeasurementStatus.COMPLETE
        and record.latency_seconds.value is not None
    ]
    if not values:
        return ScoreMetric(MeasurementStatus.UNAVAILABLE, None, 0)
    status = MeasurementStatus.COMPLETE if len(values) == len(records) else MeasurementStatus.PARTIAL
    return ScoreMetric(status, _percentile(values, percentile), len(values))


def _validate_records(
    manifest: EvaluationManifest,
    records: Sequence[EvaluationRunRecord],
) -> dict[tuple[str, str], list[EvaluationRunRecord]]:
    case_by_id = {case.case_id: case for case in manifest.cases}
    arm_ids = {arm.arm_id for arm in manifest.arms if arm.enabled}
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
        attempt_key = (record.case_id, record.arm_id, record.attempt)
        if attempt_key in attempts:
            raise EvaluationValidationError("run attempts must be unique per case and arm")
        attempts.add(attempt_key)
        grouped.setdefault((record.case_id, record.arm_id), []).append(record)
    for pair, pair_records in grouped.items():
        terminals = [record for record in pair_records if record.terminal]
        if len(terminals) > 1:
            raise EvaluationValidationError(f"case/arm pair {pair} has multiple terminal records")
    return grouped


def score_matched_arms(
    manifest: EvaluationManifest,
    truth_artifact: TruthArtifact,
    records: Sequence[EvaluationRunRecord],
) -> MatchedArmScorecard:
    """Score every enabled arm over the same manifest cases.

    Missing terminal records, failures, malformed output, and unavailable coverage
    remain in the case denominator instead of being silently replaced or dropped.
    """
    truth_artifact.validate_for_manifest(manifest)
    truth_by_case: dict[str, CheckpointTruth] = {truth.case_id: truth for truth in truth_artifact.truths}
    grouped = _validate_records(manifest, records)
    arm_scorecards: list[ArmScorecard] = []
    for arm in sorted((item for item in manifest.arms if item.enabled), key=lambda item: item.arm_id):
        arm_records = [record for record in records if record.arm_id == arm.arm_id]
        terminals_by_case: dict[str, EvaluationRunRecord] = {}
        for case in manifest.cases:
            pair_records = grouped.get((case.case_id, arm.arm_id), [])
            terminal = next((record for record in pair_records if record.terminal), None)
            if terminal is not None:
                terminals_by_case[case.case_id] = terminal

        true_positive_count = 0
        false_positive_count = 0
        false_negative_count = 0
        duplicate_finding_count = 0
        false_interruption_count = 0
        weighted_true_positive = 0.0
        weighted_truth = 0.0
        clean_checkpoint_count = 0
        completed_records: list[EvaluationRunRecord] = []
        for case in manifest.cases:
            truth = truth_by_case[case.case_id]
            truth_by_fingerprint = {finding.fingerprint: finding for finding in truth.findings}
            weighted_truth += sum(_SEVERITY_WEIGHT[finding.severity] for finding in truth.findings)
            terminal = terminals_by_case.get(case.case_id)
            observations = terminal.findings if terminal and terminal.state is EvaluationRunState.COMPLETED else ()
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
                if unique_observations:
                    false_interruption_count += 1
            if terminal and terminal.state is EvaluationRunState.COMPLETED:
                completed_records.append(terminal)

        failed_attempt_count = sum(record.state is not EvaluationRunState.COMPLETED for record in arm_records)
        completed_case_count = len(completed_records)
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
            "false_interruptions_per_clean_checkpoint": _ratio(
                false_interruption_count,
                clean_checkpoint_count,
                clean_checkpoint_count,
            ),
            "unavailable_coverage_rate": _ratio(
                sum(
                    terminals_by_case.get(case.case_id) is not None
                    and terminals_by_case[case.case_id].state is EvaluationRunState.COVERAGE_UNAVAILABLE
                    for case in manifest.cases
                ),
                len(manifest.cases),
                len(manifest.cases),
            ),
            "failure_attempt_rate": _ratio(failed_attempt_count, len(arm_records), len(arm_records)),
            "latency_p50_seconds": _latency_metric(completed_records, 0.50),
            "latency_p95_seconds": _latency_metric(completed_records, 0.95),
            "total_tokens": _summed_measurement(arm_records, "tokens"),
            "total_cost_usd": _summed_measurement(arm_records, "cost_usd"),
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
            metrics=metrics,
        ))
    return MatchedArmScorecard(
        manifest_id=manifest.manifest_id,
        truth_artifact_id=truth_artifact.truth_artifact_id,
        arms=tuple(arm_scorecards),
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
