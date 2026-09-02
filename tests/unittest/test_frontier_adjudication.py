import asyncio
import json
import threading
import time
from decimal import Decimal

import pytest

from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.frontier_adjudication import (
    FRONTIER_INPUT_SCHEMA_VERSION,
    FRONTIER_OUTPUT_SCHEMA_VERSION,
    FrontierAdjudicationConfig,
    FrontierAdjudicationRequest,
    FrontierCandidate,
    FrontierDecision,
    FrontierEvidence,
    FrontierModelIdentity,
    FrontierSignals,
    FrontierState,
    NormalizedSeverity,
    build_frontier_evidence,
    load_frontier_adjudication_config,
    run_frontier_adjudication,
)
from pr_agent.algo.run_details import (
    get_run_details,
    init_run_details,
    record_ai_call,
    record_model_request_attempt,
    record_provider_request_attempt,
)

SYSTEM_PROMPT = "Adjudicate {{ input_schema_version }} into {{ output_schema_version }}."
USER_PROMPT = "Evidence: {{ adjudication_input_json }}"


class FakeHandler:
    supports_frontier_adjudication_telemetry = True

    def __init__(
        self,
        responses,
        *,
        account=True,
        telemetry_gap=None,
        identity_gap=None,
        provider_override=None,
        revision_override=None,
        model_attempts_per_call=1,
        provider_attempts_per_call=None,
        account_failed_attempts=False,
        delay=0,
    ):
        self.responses = list(responses)
        self.account = account
        self.telemetry_gap = telemetry_gap
        self.identity_gap = identity_gap
        self.provider_override = provider_override
        self.revision_override = revision_override
        self.model_attempts_per_call = model_attempts_per_call
        self.provider_attempts_per_call = provider_attempts_per_call
        self.account_failed_attempts = account_failed_attempts
        self.delay = delay
        self.calls = []

    async def chat_completion(self, model, system, user, temperature):
        self.calls.append(model)
        for _ in range(max(0, self.model_attempts_per_call - 1)):
            record_model_request_attempt()
        provider_attempts = (
            self.model_attempts_per_call
            if self.provider_attempts_per_call is None
            else self.provider_attempts_per_call
        )
        for _ in range(provider_attempts):
            record_provider_request_attempt()
        if self.delay:
            await asyncio.sleep(self.delay)
        response = self.responses.pop(0)
        if isinstance(response, Exception) and not self.account_failed_attempts:
            raise response
        if self.account:
            record_ai_call(
                usage=(
                    None if self.telemetry_gap == "usage"
                    else {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
                ),
                model=model,
                cost_usd=None if self.telemetry_gap == "cost" else Decimal("0.01"),
                provider=(
                    None if self.identity_gap == "provider" else self.provider_override
                    or ("provider-fallback" if model == "frontier-fallback" else "provider-primary")
                ),
                model_revision=(
                    None if self.identity_gap == "revision" else self.revision_override
                    or ("revision-fallback" if model == "frontier-fallback" else "revision-primary")
                ),
            )
        if isinstance(response, Exception):
            raise response
        if self.telemetry_gap == "model_attempts":
            get_run_details().adjudication_runs["sha256:finding"].model_attempts = None
        if self.telemetry_gap == "provider_attempts":
            get_run_details().adjudication_runs["sha256:finding"].provider_attempts = None
        return response, "stop"


def output(decision="confirm", severity="medium", confidence=0.9, citations=None):
    if decision != "confirm":
        severity = None
    return json.dumps({
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
        "decision": decision,
        "normalized_severity": severity,
        "confidence": confidence,
        "evidence_citations": citations if citations is not None else ["evidence-1"],
        "unresolved_questions": [],
    })


def config(
    *,
    enabled=True,
    fallback=False,
    stage_timeout=1,
    minimum_confidence=0,
    collect_cost=True,
    model_attempts_per_model=1,
    provider_retries=0,
):
    models = ("frontier-primary", "frontier-fallback") if fallback else ("frontier-primary",)
    deployments = (None,) * len(models)
    identities = tuple(
        FrontierModelIdentity(
            model=model,
            provider="provider-primary" if index == 0 else "provider-fallback",
            revision="revision-primary" if index == 0 else "revision-fallback",
        )
        for index, model in enumerate(models)
    )
    return FrontierAdjudicationConfig(
        enabled=enabled,
        route=AIModelRoute(
            models=models,
            deployments=deployments,
            timeout_seconds=0.5,
            model_retries=model_attempts_per_model,
            provider_retries=provider_retries,
            max_output_tokens=256,
            collect_cost=collect_cost,
        ),
        model_identities=identities,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        minimum_confidence=minimum_confidence,
        stage_timeout_seconds=stage_timeout,
    )


def request(stage_config, *, signals=None, snapshot_id="head-1", evidence=None):
    return FrontierAdjudicationRequest(
        candidate=FrontierCandidate(
            stable_finding_id="sha256:finding",
            root_cause_id="sha256:root",
            path="src/auth.py",
            side="new",
            start_line=10,
            end_line=12,
            title="Authorization bypass",
            explanation="The guard can be skipped.",
            trigger="Pass an unowned object id.",
            impact="Another tenant's object is returned.",
            verified_severity=NormalizedSeverity.HIGH,
        ),
        evidence=evidence if evidence is not None else (FrontierEvidence(
            evidence_id="evidence-1",
            source="changed_patch",
            path="src/auth.py",
            side="new",
            start_line=10,
            end_line=12,
            content="return object_by_id(value)",
        ),),
        signals=signals or FrontierSignals(sensitive=True),
        snapshot_id=snapshot_id,
        configuration_hash=stage_config.configuration_hash,
        prompt_hash=stage_config.prompt_hash,
        policy_version=stage_config.policy_version,
        risk_policy_version="review-router-v1",
    )


@pytest.mark.asyncio
async def test_no_escalation_does_not_call_provider():
    init_run_details()
    stage_config = config()
    handler = FakeHandler([output()])
    result = await run_frontier_adjudication(
        request(stage_config, signals=FrontierSignals()),
        stage_config,
        handler,
        current_identity=lambda: "head-1",
    )
    assert result.state is FrontierState.NOT_REQUIRED
    assert result.decision is FrontierDecision.UNAVAILABLE
    assert handler.calls == []


@pytest.mark.asyncio
async def test_unsupported_handler_fails_before_provider_call():
    init_run_details()
    stage_config = config()
    handler = FakeHandler([output()])
    handler.supports_frontier_adjudication_telemetry = False

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=lambda: "head-1",
    )

    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "handler_telemetry_unsupported"
    assert result.to_telemetry_dict()["stable_finding_id"] == "sha256:finding"
    assert handler.calls == []


@pytest.mark.asyncio
async def test_deterministic_forced_escalation_preserves_severity_floor():
    init_run_details()
    stage_config = config()
    signals = FrontierSignals(
        deterministic_forced=True,
        deterministic_severity_floor=NormalizedSeverity.HIGH,
    )
    result = await run_frontier_adjudication(
        request(stage_config, signals=signals),
        stage_config,
        FakeHandler([output(severity="low")]),
        current_identity=lambda: "head-1",
    )
    assert result.decision is FrontierDecision.CONFIRM
    assert result.normalized_finding.severity is NormalizedSeverity.HIGH
    assert result.telemetry["provider"] == "provider-primary"
    assert result.telemetry["model_revision"] == "revision-primary"
    assert result.telemetry["usage"]["status"] == "complete"
    assert result.telemetry["cost"]["status"] == "complete"
    assert result.telemetry["route_attempts"] == 1
    assert result.telemetry["retries"]["model"] == {
        "status": "complete",
        "configured_attempts_per_model": 1,
        "attempts": 1,
        "retry_attempts": 0,
    }
    assert result.telemetry["retries"]["provider"] == {
        "status": "complete",
        "configured_retries_per_model_attempt": 0,
        "attempts": 1,
        "retry_attempts": 0,
        "unavailable_reason": None,
    }
    assert get_run_details().specialist_runs == {}
    assert "sha256:finding" in get_run_details().adjudication_runs


@pytest.mark.asyncio
async def test_reject_retains_source_free_lifecycle_telemetry():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output(decision="reject")]),
        current_identity=lambda: "head-1",
    )
    assert result.decision is FrontierDecision.REJECT
    assert result.normalized_finding is None
    assert result.to_telemetry_dict()["stable_finding_id"] == "sha256:finding"
    assert result.to_telemetry_dict()["publication_safe"] is False
    assert "src/auth.py" not in json.dumps(result.to_telemetry_dict())


@pytest.mark.asyncio
async def test_insufficient_evidence_signal_is_eligible():
    init_run_details()
    stage_config = config()
    signals = FrontierSignals(insufficient_evidence=True, unresolved_questions=("Does the caller scope access?",))
    handler = FakeHandler([output()])
    await run_frontier_adjudication(
        request(stage_config, signals=signals),
        stage_config,
        handler,
        current_identity=lambda: "head-1",
    )
    assert handler.calls == ["frontier-primary"]


@pytest.mark.asyncio
async def test_grouped_evidence_id_is_one_valid_citation():
    init_run_details()
    stage_config = config()
    evidence = (
        FrontierEvidence(
            evidence_id="context-request-1",
            source="repository_context",
            path="src/policy.py",
            side="new",
            start_line=30,
            end_line=31,
            content="def can_read(user, record):\n    return same_tenant(user, record)",
        ),
        FrontierEvidence(
            evidence_id="context-request-1",
            source="repository_context",
            path="src/policy.py",
            side="new",
            start_line=10,
            end_line=11,
            content="def same_tenant(user, record):\n    return user.tenant == record.tenant",
        ),
    )
    handler = FakeHandler([output(citations=["context-request-1"])])

    adjudication_request = request(stage_config, evidence=evidence)
    result = await run_frontier_adjudication(
        adjudication_request,
        stage_config,
        handler,
        current_identity=lambda: "head-1",
    )

    assert result.state is FrontierState.CONFIRMED
    assert result.evidence_citations == ("context-request-1",)
    assert handler.calls == ["frontier-primary"]
    serialized_evidence = adjudication_request.to_dict()["evidence"]
    assert [item["evidence_id"] for item in serialized_evidence] == [
        "context-request-1",
        "context-request-1",
    ]
    assert [item["start_line"] for item in serialized_evidence] == [30, 10]
    assert all(item["content_sha256"].startswith("sha256:") for item in serialized_evidence)


@pytest.mark.parametrize(
    "conflicting_identity",
    [
        {"source": "changed_context_head"},
        {"path": "src/other_policy.py"},
        {"side": "old"},
    ],
)
def test_grouped_evidence_id_requires_one_source_location_identity(conflicting_identity):
    stage_config = config()
    base = {
        "evidence_id": "context-request-1",
        "source": "repository_context",
        "path": "src/policy.py",
        "side": "new",
    }
    evidence = (
        FrontierEvidence(
            **base,
            start_line=10,
            end_line=11,
            content="first excerpt",
        ),
        FrontierEvidence(
            **{**base, **conflicting_identity},
            start_line=30,
            end_line=31,
            content="second excerpt",
        ),
    )

    with pytest.raises(ValueError, match="citation groups must share source, path, and side"):
        request(stage_config, evidence=evidence)


def test_grouped_evidence_id_rejects_overlapping_excerpts():
    stage_config = config()
    evidence = (
        FrontierEvidence(
            evidence_id="context-request-1",
            source="repository_context",
            path="src/policy.py",
            side="new",
            start_line=10,
            end_line=12,
            content="first excerpt",
        ),
        FrontierEvidence(
            evidence_id="context-request-1",
            source="repository_context",
            path="src/policy.py",
            side="new",
            start_line=12,
            end_line=14,
            content="overlapping excerpt",
        ),
    )

    with pytest.raises(ValueError, match="citation group excerpts must not overlap"):
        request(stage_config, evidence=evidence)


@pytest.mark.asyncio
async def test_malformed_output_fails_unavailable():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler(["not-json"]),
        current_identity=lambda: "head-1",
    )
    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.state is FrontierState.MALFORMED_OUTPUT
    assert result.failure_reason == "malformed_output"


@pytest.mark.asyncio
async def test_timeout_fails_unavailable():
    init_run_details()
    stage_config = config(stage_timeout=0.01)
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], delay=0.05),
        current_identity=lambda: "head-1",
    )
    assert result.state is FrontierState.TIMEOUT
    assert result.failure_reason == "timeout"


@pytest.mark.asyncio
async def test_slow_synchronous_pre_call_identity_refresh_times_out_promptly():
    init_run_details()
    stage_config = config(stage_timeout=0.05)
    handler = FakeHandler([output()])
    release_refresh = threading.Event()

    def slow_identity_refresh():
        release_refresh.wait(timeout=1)
        return "head-1"

    started_at = time.monotonic()
    try:
        result = await run_frontier_adjudication(
            request(stage_config),
            stage_config,
            handler,
            current_identity=slow_identity_refresh,
            deadline_monotonic=started_at + 1,
        )
        elapsed = time.monotonic() - started_at
    finally:
        release_refresh.set()

    assert elapsed < 0.5
    assert result.state is FrontierState.TIMEOUT
    assert result.failure_reason == "timeout"
    assert result.telemetry["state"] == "timeout"
    assert handler.calls == []
    serialized = json.dumps(result.to_telemetry_dict())
    assert "src/auth.py" not in serialized
    assert "object_by_id" not in serialized


@pytest.mark.asyncio
async def test_slow_synchronous_post_call_identity_refresh_times_out_promptly():
    init_run_details()
    stage_config = config(stage_timeout=0.1)
    handler = FakeHandler([output()])
    release_refresh = threading.Event()
    refresh_count = 0

    def slow_second_identity_refresh():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 1:
            return "head-1"
        release_refresh.wait(timeout=1)
        return "head-1"

    started_at = time.monotonic()
    try:
        result = await run_frontier_adjudication(
            request(stage_config),
            stage_config,
            handler,
            current_identity=slow_second_identity_refresh,
        )
        elapsed = time.monotonic() - started_at
    finally:
        release_refresh.set()

    assert elapsed < 0.5
    assert refresh_count == 2
    assert handler.calls == ["frontier-primary"]
    assert result.state is FrontierState.TIMEOUT
    assert result.failure_reason == "timeout"
    assert result.normalized_finding is None
    serialized = json.dumps(result.to_telemetry_dict())
    assert "src/auth.py" not in serialized
    assert "object_by_id" not in serialized


@pytest.mark.asyncio
async def test_slow_async_identity_refresh_times_out_and_is_cancelled():
    init_run_details()
    stage_config = config(stage_timeout=0.02)
    handler = FakeHandler([output()])
    cancelled = asyncio.Event()

    async def slow_async_identity_refresh():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    started_at = time.monotonic()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=slow_async_identity_refresh,
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.2
    assert cancelled.is_set()
    assert handler.calls == []
    assert result.state is FrontierState.TIMEOUT
    serialized = json.dumps(result.to_telemetry_dict())
    assert "src/auth.py" not in serialized
    assert "object_by_id" not in serialized


@pytest.mark.asyncio
async def test_async_identity_refresh_remains_supported():
    init_run_details()
    stage_config = config()
    refresh_count = 0

    async def async_identity_refresh():
        nonlocal refresh_count
        refresh_count += 1
        await asyncio.sleep(0)
        return "head-1"

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()]),
        current_identity=async_identity_refresh,
    )

    assert refresh_count == 2
    assert result.state is FrontierState.CONFIRMED


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_refresh", [1, 2])
@pytest.mark.parametrize(
    ("failure_value", "expected_state", "expected_reason"),
    [
        (None, FrontierState.UNAVAILABLE, "identity_refresh_unavailable"),
        (
            RuntimeError("provider unavailable"),
            FrontierState.UNAVAILABLE,
            "identity_refresh_failed",
        ),
    ],
)
async def test_identity_refresh_availability_is_distinct_from_stale_snapshot(
    failed_refresh,
    failure_value,
    expected_state,
    expected_reason,
):
    init_run_details()
    stage_config = config()
    handler = FakeHandler([output()])
    refresh_count = 0

    def current_identity():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == failed_refresh:
            if isinstance(failure_value, Exception):
                raise failure_value
            return failure_value
        return "head-1"

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=current_identity,
    )

    assert result.state is expected_state
    assert result.failure_reason == expected_reason
    assert result.telemetry["state"] == expected_state.value
    assert result.telemetry["failure_reason"] == expected_reason
    assert handler.calls == ([] if failed_refresh == 1 else ["frontier-primary"])


@pytest.mark.asyncio
async def test_success_latency_includes_final_identity_refresh():
    init_run_details()
    stage_config = config(stage_timeout=0.5)
    refresh_count = 0

    def delayed_final_identity_refresh():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 2:
            time.sleep(0.03)
        return "head-1"

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()]),
        current_identity=delayed_final_identity_refresh,
    )

    assert result.state is FrontierState.CONFIRMED
    assert refresh_count == 2
    assert result.telemetry["latency_seconds"] >= 0.02


@pytest.mark.asyncio
async def test_provider_failure_fails_unavailable():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([RuntimeError("provider down")]),
        current_identity=lambda: "head-1",
    )
    assert result.state is FrontierState.PROVIDER_FAILURE
    assert result.failure_reason == "provider_failure"


@pytest.mark.asyncio
async def test_availability_fallback_remains_distinct_and_fails_closed_without_attempt_accounting():
    init_run_details()
    stage_config = config(fallback=True)
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([RuntimeError("primary down"), output()]),
        current_identity=lambda: "head-1",
    )
    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.failure_reason == "telemetry_incomplete"
    assert result.telemetry["fallback_used"] is True
    assert result.telemetry["route_attempts"] == 2
    assert result.telemetry["retries"]["model"] == {
        "status": "complete",
        "configured_attempts_per_model": 1,
        "attempts": 2,
        "retry_attempts": 0,
    }
    assert result.telemetry["retries"]["provider"] == {
        "status": "complete",
        "configured_retries_per_model_attempt": 0,
        "attempts": 2,
        "retry_attempts": 0,
        "unavailable_reason": None,
    }
    assert result.telemetry["model"] == "frontier-fallback"
    assert result.telemetry["provider"] == "provider-fallback"


@pytest.mark.asyncio
async def test_model_retry_attempts_are_distinct_from_route_attempts():
    init_run_details()
    stage_config = config(model_attempts_per_model=2)
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], model_attempts_per_call=2),
        current_identity=lambda: "head-1",
    )

    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.failure_reason == "telemetry_incomplete"
    assert result.telemetry["route_attempts"] == 1
    assert result.telemetry["retries"]["model"] == {
        "status": "complete",
        "configured_attempts_per_model": 2,
        "attempts": 2,
        "retry_attempts": 1,
    }


@pytest.mark.asyncio
async def test_observed_provider_reinvocation_is_counted_and_fails_closed_when_unmetered():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], provider_attempts_per_call=2),
        current_identity=lambda: "head-1",
    )

    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.failure_reason == "telemetry_incomplete"
    assert result.telemetry["retries"]["provider"] == {
        "status": "complete",
        "configured_retries_per_model_attempt": 0,
        "attempts": 2,
        "retry_attempts": 1,
        "unavailable_reason": None,
    }
    assert result.telemetry["usage"]["ai_calls"] == 1


@pytest.mark.asyncio
async def test_missing_model_attempt_observation_fails_telemetry_closed():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], telemetry_gap="model_attempts"),
        current_identity=lambda: "head-1",
    )

    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "telemetry_incomplete"


@pytest.mark.asyncio
async def test_missing_provider_attempt_observation_fails_telemetry_closed():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], telemetry_gap="provider_attempts"),
        current_identity=lambda: "head-1",
    )

    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "telemetry_incomplete"


@pytest.mark.asyncio
async def test_distinct_routes_may_share_pinned_completion_identity():
    init_run_details()
    shared_provider = "provider-shared"
    shared_revision = "revision-shared"
    stage_config = FrontierAdjudicationConfig(
        enabled=True,
        route=AIModelRoute(
            models=("frontier-primary", "frontier-fallback"),
            deployments=("deployment-primary", "deployment-fallback"),
            timeout_seconds=0.5,
            model_retries=1,
            provider_retries=0,
            max_output_tokens=256,
            collect_cost=True,
        ),
        model_identities=(
            FrontierModelIdentity(
                model="frontier-primary",
                provider=shared_provider,
                revision=shared_revision,
                deployment="deployment-primary",
            ),
            FrontierModelIdentity(
                model="frontier-fallback",
                provider=shared_provider,
                revision=shared_revision,
                deployment="deployment-fallback",
            ),
        ),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        stage_timeout_seconds=1,
    )

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler(
            [RuntimeError("primary down"), output()],
            provider_override=shared_provider,
            revision_override=shared_revision,
            account_failed_attempts=True,
        ),
        current_identity=lambda: "head-1",
    )

    assert result.decision is FrontierDecision.CONFIRM
    assert result.telemetry["model"] == "frontier-fallback"
    assert result.telemetry["deployment"] == "deployment-fallback"
    assert result.telemetry["provider"] == shared_provider
    assert result.telemetry["model_revision"] == shared_revision
    assert result.telemetry["usage"]["ai_calls"] == 2
    assert result.telemetry["cost"]["total_usd"] == "0.02"


@pytest.mark.asyncio
async def test_stale_snapshot_before_call_fails_without_provider():
    init_run_details()
    stage_config = config()
    handler = FakeHandler([output()])
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=lambda: "head-2",
    )
    assert result.state is FrontierState.STALE
    assert result.to_telemetry_dict()["stable_finding_id"] == "sha256:finding"
    assert handler.calls == []


@pytest.mark.asyncio
async def test_snapshot_that_changes_during_call_discards_confirmation():
    init_run_details()
    stage_config = config()
    identities = iter(("head-1", "head-2"))
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()]),
        current_identity=lambda: next(identities),
    )
    assert result.state is FrontierState.STALE
    assert result.normalized_finding is None


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_refresh", [1, 2])
async def test_snapshot_refreshes_are_covered_by_stage_deadline(
    monkeypatch,
    blocked_refresh,
):
    init_run_details()
    refresh_count = 0

    async def delayed_to_thread(callback, *args, **kwargs):
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == blocked_refresh:
            await asyncio.sleep(0.02)
        return callback(*args, **kwargs)

    monkeypatch.setattr(
        "pr_agent.algo.frontier_adjudication.asyncio.to_thread",
        delayed_to_thread,
    )
    stage_config = config(stage_timeout=0.005)
    handler = FakeHandler([output()])

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=lambda: "head-1",
    )

    assert result.state is FrontierState.TIMEOUT
    assert result.failure_reason == "timeout"
    assert handler.calls == ([] if blocked_refresh == 1 else ["frontier-primary"])


@pytest.mark.asyncio
@pytest.mark.parametrize("telemetry_gap", ["usage", "cost"])
async def test_missing_usage_or_cost_fails_unavailable(telemetry_gap):
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], telemetry_gap=telemetry_gap),
        current_identity=lambda: "head-1",
    )
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "telemetry_incomplete"
    expected_usage = "partial" if telemetry_gap == "usage" else "complete"
    expected_cost = "unavailable" if telemetry_gap == "cost" else "complete"
    assert result.telemetry["usage"]["status"] == expected_usage
    assert result.telemetry["cost"]["status"] == expected_cost


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_gap", ["provider", "revision"])
async def test_missing_completion_identity_fails_unavailable(identity_gap):
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], identity_gap=identity_gap),
        current_identity=lambda: "head-1",
    )
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "completion_identity_unverified"


@pytest.mark.asyncio
async def test_mismatched_completion_identity_fails_unavailable():
    init_run_details()
    stage_config = config()
    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()], revision_override="rolling-alias"),
        current_identity=lambda: "head-1",
    )
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "completion_identity_unverified"


def test_frontier_route_requires_attributed_cost_collection():
    with pytest.raises(ValueError, match="collect attributed cost"):
        config(collect_cost=False)


def test_enabled_frontier_route_requires_zero_provider_retries():
    with pytest.raises(ValueError, match="provider retries to be disabled"):
        config(provider_retries=1)


def test_frontier_route_rejects_duplicate_model_deployment_identity():
    identity = FrontierModelIdentity(
        model="frontier-primary",
        provider="provider-primary",
        revision="revision-primary",
    )
    with pytest.raises(ValueError, match="route identities must be unique"):
        FrontierAdjudicationConfig(
            enabled=True,
            route=AIModelRoute(
                models=("frontier-primary", "frontier-primary"),
                deployments=(None, None),
                model_retries=1,
                provider_retries=0,
                collect_cost=True,
            ),
            model_identities=(identity, identity),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
        )


def test_loader_requires_exact_fallback_identities():
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        "frontier_adjudication_fallback_models": ["frontier-fallback"],
        "frontier_adjudication_fallback_providers": [],
        "frontier_adjudication_fallback_revisions": [],
    }
    with pytest.raises(ValueError, match="fallback providers"):
        load_frontier_adjudication_config(section, {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
            "prompt_version": "frontier-adjudication-prompt-v1",
            "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
            "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
        })


@pytest.mark.parametrize(
    ("timeout_key", "timeout_value", "timeout_name"),
    [
        ("frontier_adjudication_timeout_seconds", "inf", "stage"),
        ("frontier_adjudication_timeout_seconds", "nan", "stage"),
        ("frontier_adjudication_model_timeout_seconds", "inf", "model"),
        ("frontier_adjudication_model_timeout_seconds", "nan", "model"),
    ],
)
def test_loader_rejects_non_finite_frontier_timeouts(
    timeout_key,
    timeout_value,
    timeout_name,
):
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        timeout_key: timeout_value,
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }

    with pytest.raises(
        ValueError,
        match=rf"frontier {timeout_name} timeout must be finite and positive",
    ):
        load_frontier_adjudication_config(section, prompt)


@pytest.mark.parametrize(
    ("setting", "boolean_value"),
    [
        ("frontier_adjudication_timeout_seconds", True),
        ("frontier_adjudication_timeout_seconds", False),
        ("frontier_adjudication_model_timeout_seconds", True),
        ("frontier_adjudication_model_timeout_seconds", False),
        ("frontier_adjudication_minimum_confidence", True),
        ("frontier_adjudication_minimum_confidence", False),
    ],
)
def test_loader_rejects_boolean_frontier_number_settings(setting, boolean_value):
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        setting: boolean_value,
    }

    with pytest.raises(ValueError, match=rf"{setting} must be a non-boolean number"):
        load_frontier_adjudication_config(section, {})


def test_loader_accepts_numeric_string_frontier_number_settings():
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        "frontier_adjudication_timeout_seconds": "90.5",
        "frontier_adjudication_model_timeout_seconds": "30",
        "frontier_adjudication_minimum_confidence": "0.75",
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }

    loaded = load_frontier_adjudication_config(section, prompt)

    assert loaded.stage_timeout_seconds == 90.5
    assert loaded.route.timeout_seconds == 30.0
    assert loaded.minimum_confidence == 0.75


@pytest.mark.parametrize("minimum_confidence", ["inf", "nan"])
def test_loader_rejects_non_finite_minimum_confidence(minimum_confidence):
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        "frontier_adjudication_minimum_confidence": minimum_confidence,
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }

    with pytest.raises(ValueError, match="minimum confidence must be between 0 and 1"):
        load_frontier_adjudication_config(section, prompt)


@pytest.mark.parametrize(
    ("setting", "invalid_value"),
    [
        ("frontier_adjudication_model_retries", True),
        ("frontier_adjudication_model_retries", 1.5),
        ("frontier_adjudication_provider_retries", False),
        ("frontier_adjudication_provider_retries", 0.5),
        ("frontier_adjudication_max_output_tokens", True),
        ("frontier_adjudication_max_output_tokens", 1.5),
        ("frontier_adjudication_max_calls", False),
        ("frontier_adjudication_max_calls", 1.5),
    ],
)
def test_loader_rejects_non_integral_frontier_settings(setting, invalid_value):
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        setting: invalid_value,
    }

    with pytest.raises(ValueError, match=rf"{setting} must be a non-boolean integer"):
        load_frontier_adjudication_config(section, {})


def test_loader_accepts_integral_string_frontier_settings():
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
        "frontier_adjudication_model_retries": "2",
        "frontier_adjudication_provider_retries": "0",
        "frontier_adjudication_max_output_tokens": "512",
        "frontier_adjudication_max_calls": "2",
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }

    loaded = load_frontier_adjudication_config(section, prompt)

    assert loaded.route.model_retries == 2
    assert loaded.route.provider_retries == 0
    assert loaded.route.max_output_tokens == 512
    assert loaded.max_calls == 2


@pytest.mark.parametrize(
    "candidate_verification_config",
    [{}, {"enable_candidate_verification": False}, {"enable_candidate_verification": "false"}],
)
def test_loader_requires_candidate_verification_when_frontier_is_enabled(
    candidate_verification_config,
):
    section = {
        "enable_frontier_adjudication": True,
        **candidate_verification_config,
    }

    with pytest.raises(
        ValueError,
        match="frontier adjudication requires candidate verification to be enabled",
    ):
        load_frontier_adjudication_config(section, {})


def test_candidate_verification_evidence_is_bounded_to_candidate():
    result = build_frontier_evidence("candidate-1", [
        {
            "candidate_id": "candidate-1",
            "evidence_id": "evidence-1",
            "source": "changed_patch",
            "path": "src/auth.py",
            "side": "new",
            "start_line": 10,
            "end_line": 12,
            "content": "return object_by_id(value)",
        },
        {
            "candidate_id": "candidate-2",
            "evidence_id": "evidence-2",
            "source": "changed_patch",
            "path": "src/billing.py",
            "side": "new",
            "start_line": 1,
            "end_line": 1,
            "content": "charge()",
        },
    ])
    assert [item.evidence_id for item in result] == ["evidence-1"]
    assert result[0].content_sha256.startswith("sha256:")


def test_contract_versions_are_explicit():
    assert FRONTIER_INPUT_SCHEMA_VERSION == "frontier-adjudication-input-v1"
    assert FRONTIER_OUTPUT_SCHEMA_VERSION == "frontier-adjudication-output-v1"
