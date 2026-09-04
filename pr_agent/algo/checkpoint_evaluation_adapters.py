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
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewResultState, ReviewSnapshot, ReviewSnapshotResult
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
_COMPLETE_STAGE_STATES = frozenset({"cached", "not_required", "success"})


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


def _verified_output_truncated(review: Mapping[str, object], published_count: int) -> bool:
    artifact = review.get("candidate_verification")
    if artifact is None and published_count == 0:
        return False
    if not isinstance(artifact, Mapping):
        raise EvaluationValidationError("verified production output omits verifier telemetry")
    counts = {}
    for field_name in ("finding_limit_dropped", "verified_count", "verifier_verified_count"):
        value = artifact.get(field_name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationValidationError(f"verified production output has invalid {field_name}")
        counts[field_name] = value
    return (
        counts.get("finding_limit_dropped", 0) > 0
        or counts.get("verified_count", published_count) != published_count
        or counts.get("verifier_verified_count", published_count) > published_count
    )


def _coverage_issues(
    snapshot: ReviewSnapshot,
    kind: EvaluationArmKind,
    review: Mapping[str, object],
    details: RunDetails,
) -> tuple[CoverageIssue, ...]:
    """Retain production coverage gaps without exposing review source."""

    metadata = review.get("metadata")
    omitted_files = []
    deleted_files = []
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise EvaluationValidationError("production review output has invalid metadata")
        omitted_files = metadata.get("omitted_files", [])
        if (
            not isinstance(omitted_files, list)
            or any(not isinstance(path, str) or not path for path in omitted_files)
        ):
            raise EvaluationValidationError("production review output has invalid omitted-file coverage")
        deleted_files = metadata.get("deleted_files", [])
        if (
            not isinstance(deleted_files, list)
            or any(not isinstance(path, str) or not path for path in deleted_files)
        ):
            raise EvaluationValidationError("production review output has invalid deleted-file coverage")
    unexpected_paths = (set(omitted_files) | set(deleted_files)) - set(snapshot.changed_paths)
    if unexpected_paths:
        raise EvaluationValidationError("production review output names an unexpected coverage path")
    issues = list(snapshot.coverage_issues)
    issues.extend(
        CoverageIssue(reason="token_budget_omitted", path=path)
        for path in sorted(set(omitted_files))
    )
    issues.extend(
        CoverageIssue(reason="deleted_file_unsupported", path=path)
        for path in sorted(set(deleted_files))
    )
    if kind is EvaluationArmKind.VERIFIED_SPECIALISTS:
        if _verified_output_truncated(review, len(_review_findings(review))):
            issues.append(
                CoverageIssue(reason="verified_finding_truncated", path="candidate_verification")
            )
        issues.extend(
            CoverageIssue(reason="stage_coverage_unavailable", path=stage)
            for stage, stage_details in sorted(details.specialist_runs.items())
            if stage_details.state not in _COMPLETE_STAGE_STATES
        )
    return tuple(issues)


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
        details = RunDetails(start_time=0.0, finish_time=0.0)
    else:
        details = deserialize_run_details_for_evaluation(outcome.run_details)
    no_model_execution = (
        details.num_ai_calls == 0
        and not details.has_token_usage
        and details.known_cost_call_count == 0
        and details.total_cost_usd == 0
        and not details.model_costs_usd
        and not details.specialist_runs
        and not details.adjudication_runs
    )
    if no_model_execution:
        if raw_findings or snapshot.diff.strip():
            raise EvaluationValidationError("only empty-diff production results may have zero model execution")
        findings = ()
    else:
        if kind is EvaluationArmKind.GENERAL_REVIEW:
            findings = normalize_general_review_findings(raw_findings)
        else:
            findings = _verified_findings(outcome.review)
    coverage_issues = _coverage_issues(snapshot, kind, outcome.review, details)
    result_state = (
        ReviewResultState.FINDINGS
        if findings
        else ReviewResultState.COVERAGE_UNAVAILABLE
        if coverage_issues
        else ReviewResultState.NO_FINDINGS
    )
    return ProductionArmResult(
        snapshot_result=ReviewSnapshotResult(
            snapshot_id=snapshot.snapshot_id,
            state=result_state,
            current_snapshot_id=snapshot.snapshot_id,
            review=None,
            coverage_issues=coverage_issues,
            latency_seconds=outcome.latency_seconds or 0.0,
        ),
        run_details=details,
        findings=findings,
        terminal=result_state is not ReviewResultState.COVERAGE_UNAVAILABLE,
        failure_reason_code=(
            "production_coverage_unavailable"
            if result_state is ReviewResultState.COVERAGE_UNAVAILABLE
            else None
        ),
        latency_measurement=latency,
        no_model_execution=no_model_execution,
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
