"""Adapters from isolated production review output to evaluation contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pr_agent.algo.checkpoint_evaluation import (
    EvaluationArmKind,
    EvaluationRunState,
    EvaluationValidationError,
    MeasurementStatus,
    NumericMeasurement,
)
from pr_agent.algo.checkpoint_evaluation_findings import (
    normalize_general_review_findings,
    normalize_verified_findings,
)
from pr_agent.algo.checkpoint_evaluation_runner import (
    ProductionArmContext,
    ProductionArmResult,
    failed_production_arm_result,
)
from pr_agent.algo.checkpoint_review_subprocess import (
    CheckpointReviewSubprocessOutcome,
    CheckpointReviewSubprocessState,
    run_checkpoint_review_subprocess,
)
from pr_agent.algo.review_snapshot import ReviewResultState, ReviewSnapshot, ReviewSnapshotResult
from pr_agent.algo.run_details import RunDetails, deserialize_run_details_for_evaluation

_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_MALFORMED_FAILURES = frozenset({
    "invalid_request",
    "invalid_snapshot",
    "request_too_large",
    "review_snapshot_mismatch",
    "result_serialization_failed",
    "result_too_large",
    "worker_output_too_large",
    "worker_protocol_failed",
    "worker_snapshot_mismatch",
})


def _review_findings(review: Mapping[str, object]) -> list[Mapping[str, object]]:
    review_body = review.get("review")
    if not isinstance(review_body, Mapping):
        raise EvaluationValidationError("production review output omits its review mapping")
    findings = review_body.get("key_issues_to_review")
    if not isinstance(findings, list) or any(not isinstance(item, Mapping) for item in findings):
        raise EvaluationValidationError("production review output omits its finding list")
    return findings


def _verified_findings(review: Mapping[str, object]):
    findings = _review_findings(review)
    artifact = review.get("candidate_verification")
    if not isinstance(artifact, Mapping):
        raise EvaluationValidationError("verified production output omits verifier telemetry")
    status = artifact.get("status")
    if status not in {"complete", "partial", "no_candidates"}:
        raise EvaluationValidationError("verified production output has unavailable verifier coverage")
    if artifact.get("publication_safe") is not True:
        raise EvaluationValidationError("verified production output is not publication-safe")
    decisions = artifact.get("decisions", [])
    if not isinstance(decisions, list) or any(not isinstance(item, Mapping) for item in decisions):
        raise EvaluationValidationError("verified production output has invalid verifier decisions")
    severity_by_key = {}
    for decision in decisions:
        if decision.get("verdict") != "verified" or decision.get("reason") is not None:
            continue
        stable_key = decision.get("trusted_stable_key")
        severity = decision.get("normalized_severity")
        if (
            not isinstance(stable_key, str)
            or not _SHA256_ID.fullmatch(stable_key)
            or severity not in {"low", "medium", "high", "critical"}
        ):
            raise EvaluationValidationError("verified decision omits trusted identity or severity")
        if stable_key in severity_by_key:
            raise EvaluationValidationError("verified decisions contain a duplicate trusted identity")
        severity_by_key[stable_key] = severity

    published_keys = []
    for finding in findings:
        stable_key = finding.get("trusted_stable_key")
        if not isinstance(stable_key, str) or not _SHA256_ID.fullmatch(stable_key):
            raise EvaluationValidationError("verified finding omits its trusted identity")
        if stable_key not in severity_by_key:
            raise EvaluationValidationError("verified finding has no matching verifier decision")
        if finding.get("normalized_severity") != severity_by_key[stable_key]:
            raise EvaluationValidationError("verified finding severity contradicts its verifier decision")
        published_keys.append(stable_key)
    if len(published_keys) != len(set(published_keys)):
        raise EvaluationValidationError("verified findings contain a duplicate trusted identity")
    return normalize_verified_findings(findings, severity_by_fingerprint=severity_by_key)


def adapt_checkpoint_review_outcome(
    snapshot: ReviewSnapshot,
    kind: EvaluationArmKind,
    outcome: CheckpointReviewSubprocessOutcome,
) -> ProductionArmResult:
    """Convert one strict subprocess response without inferring missing evidence."""

    if not isinstance(snapshot, ReviewSnapshot) or not isinstance(kind, EvaluationArmKind):
        raise TypeError("production review adaptation requires a snapshot and arm kind")
    if not isinstance(outcome, CheckpointReviewSubprocessOutcome):
        raise TypeError("production review adaptation requires a subprocess outcome")
    if kind not in {EvaluationArmKind.GENERAL_REVIEW, EvaluationArmKind.VERIFIED_SPECIALISTS}:
        raise EvaluationValidationError(f"{kind.value} production finding semantics are unavailable")
    latency = NumericMeasurement(
        MeasurementStatus.COMPLETE if outcome.latency_seconds is not None else MeasurementStatus.UNAVAILABLE,
        outcome.latency_seconds,
    )
    if outcome.state is not CheckpointReviewSubprocessState.COMPLETED:
        failure_state = (
            EvaluationRunState.TIMEOUT
            if outcome.state is CheckpointReviewSubprocessState.TIMEOUT
            else EvaluationRunState.MALFORMED
            if outcome.failure_reason_code in _MALFORMED_FAILURES
            else EvaluationRunState.PROVIDER_FAILURE
        )
        return failed_production_arm_result(
            snapshot,
            state=failure_state,
            reason_code=outcome.failure_reason_code or "production_review_failed",
            latency_seconds=latency,
            retry_count=0,
        )
    if outcome.snapshot_id != snapshot.snapshot_id:
        raise EvaluationValidationError("production review output names a different snapshot")
    if outcome.review is None:
        raise EvaluationValidationError("completed production review output is incomplete")
    raw_findings = _review_findings(outcome.review)
    if outcome.run_details is None:
        if raw_findings or snapshot.diff.strip():
            raise EvaluationValidationError("only empty-diff production results may omit run telemetry")
        details = RunDetails(start_time=0.0, finish_time=0.0)
        findings = ()
    else:
        details = deserialize_run_details_for_evaluation(outcome.run_details)
        if kind is EvaluationArmKind.GENERAL_REVIEW:
            findings = normalize_general_review_findings(raw_findings)
        else:
            findings = _verified_findings(outcome.review)
    result_state = ReviewResultState.FINDINGS if findings else ReviewResultState.NO_FINDINGS
    return ProductionArmResult(
        snapshot_result=ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=result_state,
            current_snapshot_id=snapshot.snapshot_id,
            review=None,
            coverage_issues=(),
            latency_seconds=outcome.latency_seconds or 0.0,
        ),
        run_details=details,
        findings=findings,
        latency_measurement=latency,
        no_model_execution=outcome.run_details is None,
    )


def build_checkpoint_review_adapter(kind: EvaluationArmKind):
    """Build a no-publish production adapter; caller still owns paid authorization."""

    if kind not in {EvaluationArmKind.GENERAL_REVIEW, EvaluationArmKind.VERIFIED_SPECIALISTS}:
        raise EvaluationValidationError(f"{kind.value} production finding semantics are unavailable")

    async def adapter(snapshot: ReviewSnapshot, context: ProductionArmContext) -> ProductionArmResult:
        outcome = await run_checkpoint_review_subprocess(
            snapshot,
            review_configuration=context.review_configuration,
            evaluation_stage_plan=context.stage_plan,
            allow_model_execution=True,
        )
        try:
            return adapt_checkpoint_review_outcome(snapshot, kind, outcome)
        except EvaluationValidationError:
            if outcome.run_details is None:
                raise
            details = deserialize_run_details_for_evaluation(outcome.run_details)
            return failed_production_arm_result(
                snapshot,
                state=EvaluationRunState.MALFORMED,
                reason_code="production_output_invalid",
                latency_seconds=NumericMeasurement(
                    MeasurementStatus.COMPLETE
                    if outcome.latency_seconds is not None
                    else MeasurementStatus.UNAVAILABLE,
                    outcome.latency_seconds,
                ),
                retry_count=0,
                run_details=details,
            )

    return adapter
