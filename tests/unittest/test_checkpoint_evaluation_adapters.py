import hashlib

import pytest

from pr_agent.algo.checkpoint_evaluation import EvaluationArmKind, EvaluationRunState, FindingSeverity
from pr_agent.algo.checkpoint_evaluation_adapters import (
    adapt_checkpoint_review_outcome,
    build_checkpoint_review_adapter,
)
from pr_agent.algo.checkpoint_evaluation_runner import ProductionArmContext
from pr_agent.algo.checkpoint_review_subprocess import (
    CheckpointReviewSubprocessOutcome,
    CheckpointReviewSubprocessState,
)
from pr_agent.algo.review_configuration import materialize_review_configuration
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewEvent, ReviewResultState, ReviewSnapshot
from pr_agent.algo.run_details import RunDetails, SpecialistRunDetails, serialize_run_details_for_evaluation


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _snapshot(*, diff="diff --git a/example.py b/example.py\n+changed = True\n") -> ReviewSnapshot:
    configuration = materialize_review_configuration(repo_context_files={})
    return ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        base_selector="main",
        changed_paths=("example.py",) if diff else (),
        diff=diff,
        policy_version="policy-v1",
        created_at="2026-09-04T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
    )


def _outcome(snapshot: ReviewSnapshot, review: dict) -> CheckpointReviewSubprocessOutcome:
    return CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.COMPLETED,
        snapshot_id=snapshot.snapshot_id,
        review=review,
        run_details=serialize_run_details_for_evaluation(RunDetails(
            model_used="openai/gpt-test",
            review_profile="bugs_only",
            num_ai_calls=1,
            start_time=0.0,
            finish_time=0.25,
        )),
        latency_seconds=0.25,
    )


def test_general_review_adapter_uses_production_root_cause_and_severity():
    snapshot = _snapshot()
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": [{
            "relevant_file": "example.py",
            "issue_header": "Bug",
            "issue_content": "Failure details",
            "start_line": 1,
            "end_line": 1,
            "root_cause": "Shared invariant is broken",
            "normalized_severity": "high",
        }]},
    })

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert len(result.findings) == 1
    assert result.findings[0].severity is FindingSeverity.HIGH
    assert result.findings[0].stage == "general_review"


def test_verified_adapter_joins_trusted_identity_to_verifier_severity():
    snapshot = _snapshot()
    stable_key = _hash("stable-key")
    finding = {
        "relevant_file": "example.py",
        "issue_header": "Bug",
        "issue_content": "Verified failure",
        "start_line": 1,
        "end_line": 1,
        "root_cause_id": _hash("root-cause"),
        "trusted_stable_key": stable_key,
        "normalized_severity": "critical",
    }
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": [finding]},
        "candidate_verification": {
            "status": "complete",
            "publication_safe": True,
            "decisions": [{
                "candidate_id": "candidate-1",
                "verdict": "verified",
                "trusted_stable_key": stable_key,
                "normalized_severity": "critical",
            }],
        },
    })

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.VERIFIED_SPECIALISTS, outcome)

    assert result.findings[0].fingerprint == stable_key
    assert result.findings[0].severity is FindingSeverity.CRITICAL
    assert result.findings[0].stage == "candidate_verification"


def test_adapter_retains_omitted_file_as_unavailable_coverage():
    snapshot = _snapshot()
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": []},
        "metadata": {"omitted_files": ["example.py", "example.py"]},
    })

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.snapshot_result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert [(issue.reason, issue.path) for issue in result.snapshot_result.coverage_issues] == [
        ("token_budget_omitted", "example.py"),
    ]
    assert result.findings == ()
    assert result.terminal is False
    assert result.failure_reason_code == "production_coverage_unavailable"


def test_adapter_retains_preexisting_snapshot_coverage():
    configuration = materialize_review_configuration(repo_context_files={})
    snapshot = ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        changed_paths=("example.py",),
        diff="diff --git a/example.py b/example.py\n+changed = True\n",
        policy_version="policy-v1",
        created_at="2026-09-04T12:00:00Z",
        review_configuration_hash=configuration.configuration_hash,
        coverage_issues=(CoverageIssue(reason="file_too_large", path="example.py"),),
    )
    outcome = _outcome(snapshot, {"review": {"key_issues_to_review": []}})

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.snapshot_result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.snapshot_result.coverage_issues == snapshot.coverage_issues


def test_adapter_retains_deleted_file_as_unavailable_coverage():
    snapshot = _snapshot()
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": []},
        "metadata": {"deleted_files": ["example.py", "example.py"]},
    })

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.snapshot_result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert [(issue.reason, issue.path) for issue in result.snapshot_result.coverage_issues] == [
        ("deleted_file_unsupported", "example.py"),
    ]
    assert result.findings == ()
    assert result.terminal is False
    assert result.failure_reason_code == "production_coverage_unavailable"


def test_verified_adapter_converts_uncovered_clean_stage_to_coverage_unavailable():
    snapshot = _snapshot()
    details = RunDetails(start_time=0.0, finish_time=0.25)
    details.specialist_runs["change_classification"] = SpecialistRunDetails(
        role="change_classification",
        model_used="openai/gpt-test",
        deployment_id="deployment-one",
        fallback_used=False,
        prompt_version="classification-prompt-v1",
        input_schema_version="classification-input-v1",
        schema_version="classification-output-v1",
        state="unavailable",
        failure_reason="specialist_unavailable",
    )
    outcome = CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.COMPLETED,
        snapshot_id=snapshot.snapshot_id,
        review={
            "review": {"key_issues_to_review": []},
            "candidate_verification": {
                "status": "no_candidates",
                "publication_safe": True,
                "decisions": [],
            },
        },
        run_details=serialize_run_details_for_evaluation(details),
        latency_seconds=0.25,
    )

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.VERIFIED_SPECIALISTS, outcome)

    assert result.snapshot_result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert [(issue.reason, issue.path) for issue in result.snapshot_result.coverage_issues] == [
        ("stage_coverage_unavailable", "change_classification"),
    ]
    assert result.findings == ()
    assert result.terminal is False
    assert result.failure_reason_code == "production_coverage_unavailable"


def test_verified_adapter_preserves_findings_from_partial_stage_coverage():
    snapshot = _snapshot()
    stable_key = _hash("stable-key")
    details = RunDetails(start_time=0.0, finish_time=0.25)
    details.specialist_runs["candidate_verification"] = SpecialistRunDetails(
        role="candidate_verification",
        model_used="openai/gpt-test",
        deployment_id="deployment-one",
        fallback_used=False,
        prompt_version="verification-prompt-v1",
        input_schema_version="verification-input-v1",
        schema_version="verification-output-v1",
        state="partial",
        failure_reason="verification_coverage_partial",
    )
    outcome = CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.COMPLETED,
        snapshot_id=snapshot.snapshot_id,
        review={
            "review": {"key_issues_to_review": [{
                "relevant_file": "example.py",
                "issue_header": "Bug",
                "issue_content": "Verified failure",
                "start_line": 1,
                "end_line": 1,
                "root_cause_id": _hash("root-cause"),
                "trusted_stable_key": stable_key,
                "normalized_severity": "high",
            }]},
            "candidate_verification": {
                "status": "partial",
                "publication_safe": True,
                "decisions": [{
                    "candidate_id": "candidate-1",
                    "verdict": "verified",
                    "trusted_stable_key": stable_key,
                    "normalized_severity": "high",
                }],
            },
        },
        run_details=serialize_run_details_for_evaluation(details),
        latency_seconds=0.25,
    )

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.VERIFIED_SPECIALISTS, outcome)

    assert result.snapshot_result.state is ReviewResultState.FINDINGS
    assert [finding.fingerprint for finding in result.findings] == [stable_key]
    assert [(issue.reason, issue.path) for issue in result.snapshot_result.coverage_issues] == [
        ("stage_coverage_unavailable", "candidate_verification"),
    ]
    assert result.terminal is True
    assert result.failure_reason_code is None


@pytest.mark.parametrize("mutation", ("missing", "conflicting", "duplicate"))
def test_verified_adapter_rejects_untrusted_severity_joins(mutation):
    snapshot = _snapshot()
    stable_key = _hash("stable-key")
    decisions = [{
        "candidate_id": "candidate-1",
        "verdict": "verified",
        "trusted_stable_key": stable_key,
        "normalized_severity": "high",
    }]
    finding = {
        "relevant_file": "example.py",
        "issue_header": "Bug",
        "issue_content": "Verified failure",
        "start_line": 1,
        "end_line": 1,
        "root_cause_id": _hash("root-cause"),
        "trusted_stable_key": stable_key,
        "normalized_severity": "high",
    }
    if mutation == "missing":
        decisions.clear()
    elif mutation == "conflicting":
        finding["normalized_severity"] = "low"
    else:
        decisions.append(dict(decisions[0]))
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": [finding]},
        "candidate_verification": {
            "status": "complete",
            "publication_safe": True,
            "decisions": decisions,
        },
    })

    with pytest.raises(ValueError):
        adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.VERIFIED_SPECIALISTS, outcome)


def test_subprocess_failure_becomes_retained_production_failure():
    snapshot = _snapshot()
    outcome = CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.TIMEOUT,
        snapshot_id=snapshot.snapshot_id,
        latency_seconds=2.0,
        failure_reason_code="worker_timeout",
    )

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.failure_state is EvaluationRunState.TIMEOUT
    assert result.failure_reason_code == "worker_timeout"
    assert result.latency_measurement.value == 2.0


@pytest.mark.parametrize(
    "reason_code",
    (
        "invalid_request",
        "invalid_snapshot",
        "request_too_large",
        "review_snapshot_mismatch",
        "result_serialization_failed",
        "result_too_large",
        "worker_output_too_large",
        "worker_protocol_failed",
        "worker_snapshot_mismatch",
    ),
)
def test_structural_subprocess_failures_are_retained_as_malformed(reason_code):
    snapshot = _snapshot()
    outcome = CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.FAILED,
        snapshot_id=snapshot.snapshot_id,
        failure_reason_code=reason_code,
    )

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.failure_state is EvaluationRunState.MALFORMED
    assert result.failure_reason_code == reason_code


def test_execution_subprocess_failure_is_retained_as_provider_failure():
    snapshot = _snapshot()
    outcome = CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.FAILED,
        snapshot_id=snapshot.snapshot_id,
        failure_reason_code="review_execution_failed",
    )

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.failure_state is EvaluationRunState.PROVIDER_FAILURE


def test_subprocess_outcome_repr_does_not_expose_review_text():
    snapshot = _snapshot()
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": [{"issue_content": "private source details"}]},
    })

    assert "private source details" not in repr(outcome)


@pytest.mark.parametrize(
    "kind",
    (EvaluationArmKind.GENERAL_REVIEW, EvaluationArmKind.VERIFIED_SPECIALISTS),
)
def test_adapter_accepts_clean_no_call_outcome_without_false_zero_telemetry(kind):
    snapshot = _snapshot(diff="")
    outcome = CheckpointReviewSubprocessOutcome(
        state=CheckpointReviewSubprocessState.COMPLETED,
        snapshot_id=snapshot.snapshot_id,
        review={"review": {"key_issues_to_review": []}},
        latency_seconds=0.0,
    )

    result = adapt_checkpoint_review_outcome(snapshot, kind, outcome)

    assert result.findings == ()
    assert result.run_details.num_ai_calls == 0
    assert result.run_details.cost_status == "unavailable"
    assert result.snapshot_result.review is None
    assert result.no_model_execution is True


def test_adapter_result_repr_does_not_expose_review_text():
    snapshot = _snapshot()
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": [{
            "relevant_file": "example.py",
            "issue_header": "Bug",
            "issue_content": "private source details",
            "start_line": 1,
            "end_line": 1,
            "root_cause": "private root cause",
            "normalized_severity": "high",
        }]},
    })

    result = adapt_checkpoint_review_outcome(snapshot, EvaluationArmKind.GENERAL_REVIEW, outcome)

    assert result.snapshot_result.review is None
    assert "private source details" not in repr(result)


@pytest.mark.asyncio
async def test_production_adapter_retains_malformed_completed_output(monkeypatch):
    snapshot = _snapshot()
    outcome = _outcome(snapshot, {
        "review": {"key_issues_to_review": [{"issue_content": "missing trusted fields"}]},
    })

    async def run_subprocess(*_args, **_kwargs):
        return outcome

    monkeypatch.setattr(
        "pr_agent.algo.checkpoint_evaluation_adapters.run_checkpoint_review_subprocess",
        run_subprocess,
    )
    adapter = build_checkpoint_review_adapter(EvaluationArmKind.GENERAL_REVIEW)
    context = ProductionArmContext(
        manifest_id=_hash("manifest"),
        case_id="case-one",
        arm_id="general",
        event="pre_commit",
        snapshot_artifact_hash=_hash("artifact"),
        configuration_hash=_hash("configuration"),
        prompt_hash=_hash("prompt"),
        model_visible_metadata={},
        review_configuration=materialize_review_configuration(repo_context_files={}),
    )

    result = await adapter(snapshot, context)

    assert result.failure_state is EvaluationRunState.MALFORMED
    assert result.failure_reason_code == "production_output_invalid"
    assert result.run_details.model_used == "openai/gpt-test"
