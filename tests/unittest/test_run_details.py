import asyncio
from decimal import Decimal

import pytest

from pr_agent.algo.ai_request_context import AIRequestOptions, use_ai_request_options
from pr_agent.algo.run_details import (
    AdjudicationRunDetails,
    RunDetails,
    SpecialistRunDetails,
    add_token_usage,
    deserialize_run_details_for_evaluation,
    get_run_details,
    init_run_details,
    record_ai_call,
    record_model_request_attempt,
    record_model_used,
    record_review_profile,
    record_review_route,
    record_specialist_model_attempt,
    record_specialist_result,
    serialize_run_details_for_evaluation,
    specialist_runs_to_dict,
)


def test_evaluation_round_trip_retains_source_free_stage_telemetry_only():
    details = RunDetails(
        model_used="main-model",
        review_profile="bugs_only",
        route_attempts=2,
        model_retry_attempts=1,
        num_ai_calls=1,
        total_cost_usd=Decimal("0.01"),
        known_cost_call_count=1,
        model_costs_usd={"main-model": Decimal("0.01")},
        specialist_runs={
            "candidate_verification": SpecialistRunDetails(
                role="candidate_verification",
                model_used="verifier-model",
                route_attempts=3,
                model_retry_attempts=2,
                prompt_version="prompt-v1",
                input_schema_version="input-v1",
                schema_version="output-v1",
                state="success",
                latency_seconds=0.4,
                num_ai_calls=1,
                output={"source": "must not cross the boundary"},
            ),
        },
        adjudication_runs={
            "sha256:" + "a" * 64: AdjudicationRunDetails(
                finding_id="sha256:" + "a" * 64,
                model_used="frontier-model",
                provider="openai",
                model_revision="revision-1",
                prompt_version="prompt-v1",
                input_schema_version="input-v1",
                schema_version="output-v1",
                state="confirmed",
                latency_seconds=0.2,
                cache_state="not_requested",
            ),
        },
        start_time=0.0,
        finish_time=1.5,
    )

    payload = serialize_run_details_for_evaluation(details)
    restored = deserialize_run_details_for_evaluation(payload)

    assert "source" not in repr(payload)
    assert "output" not in payload["specialist_runs"]["candidate_verification"]
    assert restored.specialist_runs["candidate_verification"].output is None
    assert restored.specialist_runs["candidate_verification"].model_used == "verifier-model"
    assert restored.route_attempts == 2
    assert restored.model_retry_attempts == 1
    assert restored.specialist_runs["candidate_verification"].route_attempts == 3
    assert restored.specialist_runs["candidate_verification"].model_retry_attempts == 2
    assert restored.adjudication_runs["sha256:" + "a" * 64].provider == "openai"
    assert restored.duration_seconds == 1.5


def test_evaluation_run_details_reject_unknown_nested_fields():
    payload = serialize_run_details_for_evaluation(RunDetails(start_time=0.0, finish_time=1.0))
    payload["specialist_runs"] = {
        "candidate_verification": {
            "role": "candidate_verification",
            "source": "repository text",
        },
    }

    with pytest.raises(ValueError, match="specialist run"):
        deserialize_run_details_for_evaluation(payload)


class _Usage:
    """Stand-in for litellm's usage object (attribute access)."""

    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def test_init_returns_fresh_instance_with_zeroed_counters():
    details = init_run_details()

    assert isinstance(details, RunDetails)
    assert details.model_used is None
    assert details.review_profile is None
    assert details.fallback_used is False
    assert details.route_attempts == 0
    assert details.model_retry_attempts == 0
    assert details.prompt_tokens == 0
    assert details.completion_tokens == 0
    assert details.total_tokens == 0
    assert details.num_ai_calls == 0
    assert details.total_cost_usd == Decimal("0")
    assert details.known_cost_call_count == 0
    assert details.model_costs_usd == {}
    assert details.cost_status == "unavailable"
    assert details.has_token_usage is False
    assert details.duration_seconds >= 0


def test_init_replaces_previous_instance():
    first = init_run_details()
    record_model_used("model-a", is_fallback=True)

    second = init_run_details()

    assert second is not first
    assert get_run_details() is second
    assert second.model_used is None
    assert second.fallback_used is False


def test_route_attempts_are_attributed_to_main_and_specialist_runs():
    details = init_run_details()

    record_specialist_model_attempt(
        "main-model",
        attribution=None,
        deployment_id=None,
        is_fallback=False,
    )
    record_model_request_attempt()
    record_specialist_model_attempt(
        "specialist-model",
        attribution="candidate_verification",
        deployment_id="specialist-deployment",
        is_fallback=True,
    )
    record_model_request_attempt("candidate_verification")

    assert details.route_attempts == 1
    assert details.model_retry_attempts == 1
    specialist = details.specialist_runs["candidate_verification"]
    assert specialist.route_attempts == 1
    assert specialist.model_retry_attempts == 1
    assert specialist.model_used == "specialist-model"
    assert specialist.deployment_id == "specialist-deployment"
    assert specialist.fallback_used is True


def test_freeze_duration_stops_elapsed_time(monkeypatch):
    timestamps = iter((12.5, 99.0))
    monkeypatch.setattr("pr_agent.algo.run_details.time.monotonic", lambda: next(timestamps))
    details = RunDetails(start_time=10.0)

    details.freeze_duration()

    assert details.duration_seconds == 2.5


def test_record_model_used_tracks_model_and_fallback_flag():
    init_run_details()

    record_model_used("openai/gpt-5.4", is_fallback=True)

    details = get_run_details()
    assert details.model_used == "openai/gpt-5.4"
    assert details.fallback_used is True


def test_record_review_profile_tracks_selected_reviewer_mode():
    init_run_details()

    record_review_profile("bugs_only")

    assert get_run_details().review_profile == "bugs_only"


def test_record_review_route_keeps_an_isolated_snapshot():
    init_run_details()
    route = {"applied_depth": "deep", "reasons": [{"code": "sensitive"}]}

    record_review_route(route)
    route["reasons"][0]["code"] = "mutated"

    assert get_run_details().review_route == {
        "applied_depth": "deep",
        "reasons": [{"code": "sensitive"}],
    }


def test_fallback_flag_is_sticky_once_a_fallback_was_used():
    init_run_details()

    record_model_used("fallback-model", is_fallback=True)
    record_model_used("primary-model", is_fallback=False)

    details = get_run_details()
    # last successful model wins, but the fallback flag must not be cleared
    assert details.model_used == "primary-model"
    assert details.fallback_used is True


def test_add_token_usage_accumulates_across_calls():
    init_run_details()

    add_token_usage(_Usage(100, 10, 110))
    add_token_usage(
        {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    )

    details = get_run_details()
    assert details.prompt_tokens == 105
    assert details.completion_tokens == 11
    assert details.total_tokens == 116
    assert details.has_token_usage is True


def test_add_token_usage_derives_total_when_missing():
    init_run_details()

    add_token_usage({"prompt_tokens": 20, "completion_tokens": 5})

    assert get_run_details().total_tokens == 25


def test_add_token_usage_ignores_none_and_partial_objects():
    init_run_details()

    add_token_usage(None)
    add_token_usage(object())

    details = get_run_details()
    assert details.total_tokens == 0
    assert details.has_token_usage is False


def test_record_ai_call_counts_calls_even_without_usage():
    init_run_details()

    record_ai_call(_Usage(10, 2, 12))
    record_ai_call(None)

    details = get_run_details()
    assert details.num_ai_calls == 2
    assert details.total_tokens == 12


def test_record_ai_call_aggregates_decimal_costs_by_model():
    init_run_details()
    record_model_used("model-b", is_fallback=True)

    record_ai_call(_Usage(10, 2, 12), model="model-a", cost_usd=Decimal("0.0710"))
    record_ai_call(_Usage(5, 1, 6), model="model-b", cost_usd="0.0132")
    record_ai_call(_Usage(1, 1, 2), model="model-a", cost_usd=0.00001)

    details = get_run_details()
    assert details.total_cost_usd == Decimal("0.08421")
    assert details.known_cost_call_count == 3
    assert details.cost_status == "complete"
    assert details.fallback_used is True
    assert details.model_costs_usd == {
        "model-a": Decimal("0.07101"),
        "model-b": Decimal("0.0132"),
    }


def test_record_ai_call_treats_zero_cost_as_unpriced():
    """litellm.completion_cost returns 0.0 for unpriced models and empty usage;
    recording it would render a false '$0.00' with cost status complete."""
    init_run_details()

    record_ai_call(_Usage(10, 2, 12), model="zero-priced", cost_usd=0.0)
    record_ai_call(_Usage(10, 2, 12), model="zero-priced", cost_usd=Decimal("0"))

    details = get_run_details()
    assert details.total_cost_usd == Decimal("0")
    assert details.known_cost_call_count == 0
    assert details.cost_status == "unavailable"
    assert details.model_costs_usd == {}


def test_record_ai_call_marks_partial_and_unavailable_cost_without_fabricating_zero():
    init_run_details()

    record_ai_call(_Usage(10, 2, 12), model="known", cost_usd=Decimal("0.0042"))
    record_ai_call(None, model="unknown", cost_usd=None)

    details = get_run_details()
    assert details.total_cost_usd == Decimal("0.0042")
    assert details.known_cost_call_count == 1
    assert details.cost_status == "partial"

    init_run_details()
    record_ai_call(None, model="unknown", cost_usd=None)

    details = get_run_details()
    assert details.total_cost_usd == Decimal("0")
    assert details.known_cost_call_count == 0
    assert details.cost_status == "unavailable"
    assert details.model_costs_usd == {}


def test_specialist_usage_is_attributed_without_changing_primary_totals():
    init_run_details()
    record_model_used("main-model", is_fallback=False)
    record_ai_call(_Usage(100, 10, 110), model="main-model", cost_usd="0.01")

    with use_ai_request_options(
        AIRequestOptions(deployment_id="risk-deployment", attribution="risk_recommendation")
    ):
        record_ai_call(_Usage(20, 3, 23), model="risk-model", cost_usd="0.002")
    record_model_used(
        "risk-model",
        is_fallback=True,
        attribution="risk_recommendation",
        deployment_id="risk-deployment",
    )
    record_specialist_result(
        "risk_recommendation",
        prompt_version="risk-prompt-v1",
        input_schema_version="risk-input-v1",
        schema_version="risk-output-v1",
        state="success",
        latency_seconds=0.4,
        confidence=0.8,
        input_token_reservation=20,
        output_token_reservation=100,
        output={"recommendation": "escalate"},
    )

    details = get_run_details()
    assert details.model_used == "main-model"
    assert details.fallback_used is False
    assert details.num_ai_calls == 1
    assert details.total_tokens == 110
    record = specialist_runs_to_dict()["risk_recommendation"]
    assert record["model"] == "risk-model"
    assert record["deployment"] == "risk-deployment"
    assert record["fallback_used"] is True
    assert record["input_schema_version"] == "risk-input-v1"
    assert record["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 3,
        "total_tokens": 23,
        "ai_calls": 1,
    }
    assert record["cost"]["total_usd"] == "0.002"


@pytest.mark.asyncio
async def test_concurrent_child_tasks_accumulate_into_parent_collector():
    init_run_details()

    async def record(prompt_tokens, completion_tokens):
        await asyncio.sleep(0)
        record_ai_call(_Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens))

    await asyncio.gather(
        record(10, 1),
        record(20, 2),
        record(30, 3),
    )

    details = get_run_details()
    assert details.num_ai_calls == 3
    assert details.prompt_tokens == 60
    assert details.completion_tokens == 6
    assert details.total_tokens == 66


@pytest.mark.asyncio
async def test_concurrent_runs_keep_collectors_isolated():
    async def run_with_usage(prompt_tokens, completion_tokens):
        init_run_details()
        await asyncio.sleep(0)
        record_ai_call(_Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens))
        return get_run_details()

    first, second = await asyncio.gather(
        run_with_usage(10, 1),
        run_with_usage(20, 2),
    )

    assert first.num_ai_calls == 1
    assert first.prompt_tokens == 10
    assert first.completion_tokens == 1
    assert first.total_tokens == 11

    assert second.num_ai_calls == 1
    assert second.prompt_tokens == 20
    assert second.completion_tokens == 2
    assert second.total_tokens == 22


def test_helpers_are_noops_when_not_initialized():
    from pr_agent.algo import run_details

    token = run_details._run_details.set(None)
    try:
        assert get_run_details() is None
        record_model_used("m", is_fallback=False)  # must not raise
        record_ai_call(_Usage(1, 1, 2))  # must not raise
        add_token_usage({"total_tokens": 5})  # must not raise
    finally:
        run_details._run_details.reset(token)
