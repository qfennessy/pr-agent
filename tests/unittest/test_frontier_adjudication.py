import asyncio
import json
import threading
import time
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.checkpoint_cost_authority import CheckpointCostAuthorityError
from pr_agent.algo.frontier_adjudication import (
    FRONTIER_INPUT_SCHEMA_VERSION,
    FRONTIER_OUTPUT_SCHEMA_VERSION,
    FrontierAdjudicationConfig,
    FrontierAdjudicationRequest,
    FrontierCandidate,
    FrontierContractError,
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


@pytest.mark.parametrize(
    "prompt",
    [True, 7, "not-a-table", ["not", "a", "table"]],
    ids=["boolean", "number", "string", "list"],
)
def test_loader_rejects_non_mapping_prompt_section(prompt):
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
    }

    with pytest.raises(
        ValueError,
        match="frontier adjudication prompt must be a mapping",
    ):
        load_frontier_adjudication_config(section, prompt)


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


def test_frontier_configuration_round_trips_exact_route_prompts_and_hashes():
    stage_config = config(fallback=True)
    deployments = ("primary-deployment", "fallback-deployment")
    stage_config = replace(
        stage_config,
        route=replace(
            stage_config.route,
            deployments=deployments,
            attribution="frontier-adjudication",
        ),
        model_identities=tuple(
            replace(identity, deployment=deployment)
            for identity, deployment in zip(stage_config.model_identities, deployments, strict=True)
        ),
        system_prompt="  Preserve exact system text.\n",
        user_prompt="Evidence: {{ adjudication_input_json }}\n",
        policy_version="frontier-policy-pinned-v2",
    )
    payload = json.loads(json.dumps(stage_config.to_dict()))

    restored = FrontierAdjudicationConfig.from_dict(payload)

    assert restored == stage_config
    assert restored.configuration_hash == stage_config.configuration_hash
    assert restored.prompt_hash == stage_config.prompt_hash
    assert restored.system_prompt == "  Preserve exact system text.\n"
    assert restored.route.deployments == deployments
    assert restored.route.attribution == "frontier-adjudication"
    assert restored.to_dict() == payload


def test_frontier_configuration_rejects_unknown_missing_and_malformed_fields():
    payload = config().to_dict()

    with_unknown = dict(payload)
    with_unknown["unexpected"] = True
    with pytest.raises(FrontierContractError, match="invalid frontier adjudication configuration fields"):
        FrontierAdjudicationConfig.from_dict(with_unknown)

    with_missing = dict(payload)
    with_missing.pop("policy_version")
    with pytest.raises(FrontierContractError, match="invalid frontier adjudication configuration fields"):
        FrontierAdjudicationConfig.from_dict(with_missing)

    malformed = json.loads(json.dumps(payload))
    malformed["route"]["collect_cost"] = "true"
    with pytest.raises(FrontierContractError, match="collect_cost must be a boolean"):
        FrontierAdjudicationConfig.from_dict(malformed)

    unknown_route = json.loads(json.dumps(payload))
    unknown_route["route"]["api_key"] = "must-not-be-accepted"
    with pytest.raises(FrontierContractError, match="invalid frontier adjudication route fields"):
        FrontierAdjudicationConfig.from_dict(unknown_route)


def test_frontier_configuration_rejects_prompt_and_configuration_tampering():
    prompt_tamper = config().to_dict()
    prompt_tamper["system_prompt"] = "changed after hashing"
    with pytest.raises(FrontierContractError, match="prompt hash mismatch"):
        FrontierAdjudicationConfig.from_dict(prompt_tamper)

    configuration_tamper = config().to_dict()
    configuration_tamper["max_calls"] += 1
    with pytest.raises(FrontierContractError, match="configuration hash mismatch"):
        FrontierAdjudicationConfig.from_dict(configuration_tamper)

    attribution_tamper = config().to_dict()
    attribution_tamper["route"]["attribution"] = "changed-after-hashing"
    with pytest.raises(FrontierContractError, match="configuration hash mismatch"):
        FrontierAdjudicationConfig.from_dict(attribution_tamper)


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
    assert result.telemetry["state"] == "unavailable"
    assert result.telemetry["failure_reason"] == "handler_telemetry_unsupported"
    assert result.telemetry["prompt_version"] == "frontier-adjudication-prompt-v1"
    assert result.telemetry["input_schema_version"] == FRONTIER_INPUT_SCHEMA_VERSION
    assert result.telemetry["schema_version"] == FRONTIER_OUTPUT_SCHEMA_VERSION
    assert result.telemetry["latency_seconds"] >= 0
    assert result.telemetry["retries"]["model"]["configured_attempts_per_model"] == 1
    assert result.telemetry["retries"]["provider"]["configured_retries_per_model_attempt"] == 0
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
@pytest.mark.parametrize(
    "raw_output",
    [
        (
            f'{{"schema_version":"{FRONTIER_OUTPUT_SCHEMA_VERSION}",'
            '"decision":"confirm","decision":"reject",'
            '"normalized_severity":null,"confidence":0.9,'
            '"evidence_citations":["evidence-1"],"unresolved_questions":[]}'
        ),
        (
            f'{{"schema_version":"{FRONTIER_OUTPUT_SCHEMA_VERSION}",'
            '"decision":"confirm","normalized_severity":"high","confidence":0.9,'
            '"evidence_citations":["evidence-1"],'
            '"unresolved_questions":[{"decision":"confirm","decision":"reject"}]}'
        ),
    ],
    ids=["conflicting-top-level-decision", "nested-duplicate"],
)
async def test_duplicate_json_object_keys_are_rejected_as_malformed(raw_output):
    from pr_agent.algo import frontier_adjudication as frontier_module

    init_run_details()
    stage_config = config()
    adjudication_request = request(stage_config)
    with pytest.raises(ValueError, match="duplicate object keys"):
        frontier_module._validate_output(
            raw_output,
            adjudication_request,
            stage_config,
        )

    result = await run_frontier_adjudication(
        adjudication_request,
        stage_config,
        FakeHandler([raw_output]),
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
async def test_synchronous_identity_refresh_capacity_is_bounded_and_isolated(monkeypatch):
    from pr_agent.algo import frontier_adjudication as frontier_module

    init_run_details()
    assert frontier_module._SYNC_IDENTITY_REFRESH_WORKER_LIMIT == 4
    worker_limit = 2
    monkeypatch.setattr(
        frontier_module,
        "_SYNC_IDENTITY_REFRESH_SLOTS",
        threading.BoundedSemaphore(worker_limit),
    )
    loop = asyncio.get_running_loop()
    default_executor_calls = []

    def reject_default_executor(*args, **kwargs):
        default_executor_calls.append((args, kwargs))
        raise AssertionError("identity refresh used the event-loop executor")

    monkeypatch.setattr(loop, "run_in_executor", reject_default_executor)
    stage_config = config(stage_timeout=0.1)
    release_refresh = threading.Event()
    state_lock = threading.Lock()
    active_workers = 0
    peak_workers = 0
    worker_names = []

    def never_returning_identity_refresh():
        nonlocal active_workers, peak_workers
        with state_lock:
            active_workers += 1
            peak_workers = max(peak_workers, active_workers)
            worker_names.append(threading.current_thread().name)
        try:
            release_refresh.wait(timeout=1)
            return "head-1"
        finally:
            with state_lock:
                active_workers -= 1

    blocking_handlers = [FakeHandler([output()]) for _ in range(worker_limit)]
    try:
        blocking_results = await asyncio.gather(*(
            run_frontier_adjudication(
                request(stage_config),
                stage_config,
                handler,
                current_identity=never_returning_identity_refresh,
            )
            for handler in blocking_handlers
        ))
        with state_lock:
            assert active_workers == worker_limit
            assert peak_workers == worker_limit

        later_handlers = [FakeHandler([output()]) for _ in range(worker_limit * 3)]
        later_started_at = time.monotonic()
        later_results = await asyncio.gather(*(
            run_frontier_adjudication(
                request(stage_config),
                stage_config,
                handler,
                current_identity=never_returning_identity_refresh,
            )
            for handler in later_handlers
        ))
        later_elapsed = time.monotonic() - later_started_at
    finally:
        release_refresh.set()

    for _ in range(100):
        with state_lock:
            if active_workers == 0:
                break
        await asyncio.sleep(0.01)

    assert all(result.state is FrontierState.TIMEOUT for result in blocking_results)
    assert all(result.failure_reason == "timeout" for result in blocking_results)
    assert all(handler.calls == [] for handler in blocking_handlers)
    assert later_elapsed < stage_config.stage_timeout_seconds
    assert all(result.state is FrontierState.UNAVAILABLE for result in later_results)
    assert all(
        result.failure_reason == "identity_refresh_capacity_exhausted"
        for result in later_results
    )
    assert all(handler.calls == [] for handler in later_handlers)
    assert set(worker_names) == {"frontier-identity-refresh"}
    assert active_workers == 0
    assert default_executor_calls == []
    serialized = json.dumps([result.to_telemetry_dict() for result in later_results])
    assert "src/auth.py" not in serialized
    assert "object_by_id" not in serialized


@pytest.mark.asyncio
async def test_async_identity_refresh_remains_supported_without_worker_dispatch(monkeypatch):
    init_run_details()
    stage_config = config()
    refresh_count = 0
    loop = asyncio.get_running_loop()
    executor_calls = []

    def reject_executor_dispatch(*args, **kwargs):
        executor_calls.append((args, kwargs))
        raise AssertionError("async identity refresh dispatched a worker")

    monkeypatch.setattr(loop, "run_in_executor", reject_executor_dispatch)

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
    assert executor_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_refresh", [1, 2], ids=["pre", "post"])
async def test_unavailable_identity_refresh_is_distinct_from_stale_snapshot(failed_refresh):
    init_run_details()
    stage_config = config()
    handler = FakeHandler([output()])
    refresh_count = 0

    def current_identity():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == failed_refresh:
            return None
        return "head-1"

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=current_identity,
    )

    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "identity_refresh_unavailable"
    assert result.telemetry["state"] == "unavailable"
    assert result.telemetry["failure_reason"] == "identity_refresh_unavailable"
    assert handler.calls == ([] if failed_refresh == 1 else ["frontier-primary"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_refresh", [1, 2], ids=["pre", "post"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_identity_refresh_exception_is_source_free_provider_unavailable(
    failed_refresh,
    asynchronous,
):
    init_run_details()
    stage_config = config()
    handler = FakeHandler([output()])
    refresh_count = 0
    private_error = "provider failed for src/auth.py object_by_id"

    def failing_identity_refresh():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == failed_refresh:
            raise RuntimeError(private_error)
        return "head-1"

    async def failing_async_identity_refresh():
        await asyncio.sleep(0)
        return failing_identity_refresh()

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=(
            failing_async_identity_refresh if asynchronous else failing_identity_refresh
        ),
    )

    assert refresh_count == failed_refresh
    assert handler.calls == ([] if failed_refresh == 1 else ["frontier-primary"])
    assert result.decision is FrontierDecision.UNAVAILABLE
    assert result.state is FrontierState.UNAVAILABLE
    assert result.failure_reason == "identity_refresh_failed"
    assert result.normalized_finding is None
    assert result.telemetry["state"] == "unavailable"
    assert result.telemetry["failure_reason"] == "identity_refresh_failed"
    serialized = json.dumps(result.to_telemetry_dict())
    assert private_error not in serialized
    assert "src/auth.py" not in serialized
    assert "object_by_id" not in serialized
    assert "stale_snapshot" not in serialized


@pytest.mark.asyncio
async def test_success_latency_includes_final_identity_refresh(monkeypatch):
    init_run_details()
    stage_config = config(stage_timeout=1)
    refresh_count = 0
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "pr_agent.algo.frontier_adjudication.time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )

    async def delayed_final_identity_refresh():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 2:
            clock["now"] += 0.75
        return "head-1"

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        FakeHandler([output()]),
        current_identity=delayed_final_identity_refresh,
    )

    assert result.state is FrontierState.CONFIRMED
    assert refresh_count == 2
    assert result.telemetry["latency_seconds"] == pytest.approx(0.75)
    assert (
        get_run_details().adjudication_runs["sha256:finding"].latency_seconds
        == pytest.approx(0.75)
    )


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
async def test_cost_authority_denial_escapes_frontier_failure_handling():
    init_run_details()
    stage_config = config()
    denial = CheckpointCostAuthorityError("gateway route is not authorized")
    handler = FakeHandler([denial])

    with pytest.raises(CheckpointCostAuthorityError) as exc_info:
        await run_frontier_adjudication(
            request(stage_config),
            stage_config,
            handler,
            current_identity=lambda: "head-1",
        )

    assert exc_info.value is denial
    assert handler.calls == [stage_config.route.models[0]]


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
async def test_snapshot_refreshes_are_covered_by_stage_deadline(blocked_refresh):
    init_run_details()
    refresh_count = 0
    release_refresh = threading.Event()

    def delayed_identity_refresh():
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == blocked_refresh:
            release_refresh.wait(timeout=1)
        return "head-1"

    stage_config = config(stage_timeout=0.005)
    handler = FakeHandler([output()])

    try:
        result = await run_frontier_adjudication(
            request(stage_config),
            stage_config,
            handler,
            current_identity=delayed_identity_refresh,
        )
    finally:
        release_refresh.set()

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_field",
    ["system", "user", "prompt_version", "input_schema_version", "schema_version"],
)
@pytest.mark.parametrize(
    "invalid_value",
    [True, 7, {"unexpected": "mapping"}],
    ids=["boolean", "number", "non-string"],
)
async def test_loader_rejects_non_string_prompt_contract_before_model_call(
    prompt_field,
    invalid_value,
):
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
        prompt_field: invalid_value,
    }
    handler = FakeHandler([output()])

    with pytest.raises(
        ValueError,
        match=rf"frontier prompt {prompt_field} must be a non-blank string",
    ):
        stage_config = load_frontier_adjudication_config(section, prompt)
        await run_frontier_adjudication(
            request(stage_config),
            stage_config,
            handler,
            current_identity=lambda: "head-1",
        )
    assert handler.calls == []


@pytest.mark.asyncio
async def test_loader_accepts_string_prompt_contract_fields():
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": "frontier-primary",
        "frontier_adjudication_provider": "provider-primary",
        "frontier_adjudication_revision": "revision-primary",
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }
    stage_config = load_frontier_adjudication_config(section, prompt)
    handler = FakeHandler([output()])
    init_run_details()

    result = await run_frontier_adjudication(
        request(stage_config),
        stage_config,
        handler,
        current_identity=lambda: "head-1",
    )

    assert stage_config.system_prompt == SYSTEM_PROMPT
    assert stage_config.user_prompt == USER_PROMPT
    assert stage_config.prompt_version == "frontier-adjudication-prompt-v1"
    assert stage_config.input_schema_version == FRONTIER_INPUT_SCHEMA_VERSION
    assert stage_config.output_schema_version == FRONTIER_OUTPUT_SCHEMA_VERSION
    assert result.state is FrontierState.CONFIRMED
    assert handler.calls == ["frontier-primary"]


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
    "setting",
    [
        "frontier_adjudication_model",
        "frontier_adjudication_provider",
        "frontier_adjudication_revision",
        "frontier_adjudication_deployment",
        "frontier_adjudication_fallback_models",
        "frontier_adjudication_fallback_providers",
        "frontier_adjudication_fallback_revisions",
        "frontier_adjudication_fallback_deployments",
    ],
)
@pytest.mark.parametrize("invalid_value", [[True], [7]], ids=["boolean", "number"])
@pytest.mark.asyncio
async def test_loader_rejects_non_string_route_identity_entries_before_model_call(
    setting,
    invalid_value,
):
    handler = FakeHandler([output()])
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": ["frontier-primary"],
        "frontier_adjudication_provider": ["provider-primary"],
        "frontier_adjudication_revision": ["revision-primary"],
        "frontier_adjudication_deployment": ["deployment-primary"],
        "frontier_adjudication_fallback_models": ["frontier-fallback"],
        "frontier_adjudication_fallback_providers": ["provider-fallback"],
        "frontier_adjudication_fallback_revisions": ["revision-fallback"],
        "frontier_adjudication_fallback_deployments": ["deployment-fallback"],
        setting: invalid_value,
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }

    with pytest.raises(ValueError, match=rf"{setting} must be a string list"):
        stage_config = load_frontier_adjudication_config(section, prompt)
        await run_frontier_adjudication(
            request(stage_config),
            stage_config,
            handler,
            current_identity=lambda: "head-1",
        )
    assert handler.calls == []


def test_loader_retains_valid_string_array_route_identities():
    section = {
        "enable_frontier_adjudication": True,
        "enable_candidate_verification": True,
        "frontier_adjudication_model": [" frontier-primary "],
        "frontier_adjudication_provider": [" provider-primary "],
        "frontier_adjudication_revision": [" revision-primary "],
        "frontier_adjudication_deployment": [" deployment-primary "],
        "frontier_adjudication_fallback_models": [" frontier-fallback "],
        "frontier_adjudication_fallback_providers": [" provider-fallback "],
        "frontier_adjudication_fallback_revisions": [" revision-fallback "],
        "frontier_adjudication_fallback_deployments": [" deployment-fallback "],
    }
    prompt = {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
        "prompt_version": "frontier-adjudication-prompt-v1",
        "input_schema_version": FRONTIER_INPUT_SCHEMA_VERSION,
        "schema_version": FRONTIER_OUTPUT_SCHEMA_VERSION,
    }

    loaded = load_frontier_adjudication_config(section, prompt)

    assert loaded.route.models == ("frontier-primary", "frontier-fallback")
    assert loaded.route.deployments == ("deployment-primary", "deployment-fallback")
    assert [identity.provider for identity in loaded.model_identities] == [
        "provider-primary",
        "provider-fallback",
    ]
    assert [identity.revision for identity in loaded.model_identities] == [
        "revision-primary",
        "revision-fallback",
    ]


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
