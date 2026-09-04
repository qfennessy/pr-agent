import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from pr_agent.algo.ai_handlers import litellm_ai_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.ai_request_context import AIRequestOptions, use_ai_request_options
from pr_agent.algo.checkpoint_cost_authority import (
    CHECKPOINT_GATEWAY_ROUTE_HEADER,
    CheckpointCostAuthorityError,
    CostAuthorityLedger,
    FrozenCostAuthority,
    ProviderMaximumCharge,
    checkpoint_cost_stage,
    gateway_api_base_identity_hash,
    reserve_checkpoint_provider_attempt,
    use_checkpoint_cost_authority,
    validate_cost_authorities,
    validate_cost_authority_for_pair,
)
from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    EvaluationArm,
    EvaluationArmKind,
    EvaluationCohort,
    EvaluationManifest,
    EvaluationModelIdentity,
)
from pr_agent.algo.checkpoint_evaluation_execution import PaidExecutionRequest, PaidPlanItemBudget
from pr_agent.algo.review_snapshot import ReviewEvent


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


_GATEWAY_API_BASE = "https://checkpoint-gateway.example/v1"


def _contracts():
    case = CheckpointCase(
        case_id="case-one",
        snapshot_id=_hash("snapshot"),
        snapshot_artifact_hash=_hash("snapshot-artifact"),
        event=ReviewEvent.PRE_COMMIT,
        cohort=EvaluationCohort.HOLDOUT,
    )
    arm = EvaluationArm(
        arm_id="arm-general",
        kind=EvaluationArmKind.GENERAL_REVIEW,
        configuration_hash=_hash("arm-configuration"),
        prompt_hash=_hash("prompt"),
        model_id="openai/model-primary",
        provider_id="openai",
        model_revision="model-primary-2026-09-01",
        fallback_models=(
            EvaluationModelIdentity(
                model_id="openai/model-fallback",
                provider_id="openai",
                model_revision="model-fallback-2026-09-01",
            ),
        ),
    )
    manifest = EvaluationManifest(
        name="cost-boundary",
        corpus_hash=_hash("corpus"),
        policy_hash=_hash("policy"),
        configuration_hash=_hash("manifest-configuration"),
        cases=(case,),
        arms=(arm,),
    )
    request = PaidExecutionRequest(
        manifest_id=manifest.manifest_id,
        cost_cap_usd=0.03,
        plan_item_budgets=(PaidPlanItemBudget(case.case_id, arm.arm_id, 0.03, 1),),
        credential_present_by_provider={"openai": True},
    )
    quotes = tuple(
        ProviderMaximumCharge(
            stage="general_review",
            model_id=model_id,
            provider_id=provider_id,
            model_revision=model_revision,
            deployment_id_hash=None,
            gateway_api_base_hash=gateway_api_base_identity_hash(_GATEWAY_API_BASE),
            gateway_route_binding_id=_hash(f"gateway-route:{provider_id}:{model_revision}"),
            max_output_tokens=128,
            maximum_charge_usd=Decimal("0.01"),
        )
        for model_id, provider_id, model_revision in arm.aggregate_model_identities()
    )
    authority = FrozenCostAuthority(
        manifest_id=manifest.manifest_id,
        paid_request_id=request.request_id,
        case_id=case.case_id,
        arm_id=arm.arm_id,
        snapshot_id=case.snapshot_id,
        arm_configuration_hash=arm.configuration_hash,
        review_configuration_hash=_hash("review-configuration"),
        hard_cost_cap_usd=Decimal("0.03"),
        authority_name="gateway-budget-service",
        authority_revision="rate-card-2026-09-01",
        authority_reference_hash=_hash("signed-gateway-contract"),
        expires_at="2099-01-01T00:00:00Z",
        quotes=quotes,
    )
    return manifest, request, case, arm, authority


def test_cost_authority_round_trip_is_exact_and_source_free():
    _manifest, _request, _case, _arm, authority = _contracts()

    restored = FrozenCostAuthority.from_dict(authority.to_dict())

    assert restored == authority
    assert restored.authority_id.startswith("sha256:")
    assert "prompt" not in restored.to_dict()
    assert "diff" not in restored.to_dict()
    assert _GATEWAY_API_BASE not in str(restored.to_dict())
    assert all(quote.gateway_route_binding_id.startswith("sha256:") for quote in restored.quotes)
    tampered = authority.to_dict()
    tampered["hard_cost_cap_usd"] = "0.04"
    with pytest.raises(CheckpointCostAuthorityError, match="identity"):
        FrozenCostAuthority.from_dict(tampered)


def test_cost_authority_must_cover_the_exact_frozen_routes():
    manifest, request, case, arm, authority = _contracts()

    validate_cost_authority_for_pair(manifest, request, case, arm, authority)
    validated = validate_cost_authorities(manifest, request, (authority,))

    assert validated[(case.case_id, arm.arm_id)] == authority
    incomplete = replace(authority, quotes=authority.quotes[:1])
    with pytest.raises(CheckpointCostAuthorityError, match="every exact arm route"):
        validate_cost_authority_for_pair(manifest, request, case, arm, incomplete)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"provider_max_retries": 1}, "retries"),
        ({"max_output_tokens": None}, "bounded output"),
        ({"max_output_tokens": 129}, "quoted output"),
        ({"model_id": "openai/unpriced-model"}, "no authoritative"),
        ({"gateway_api_base": "https://different-gateway.example/v1"}, "no authoritative"),
    ),
)
def test_ledger_rejects_unbounded_or_unpriced_provider_requests(kwargs, message):
    _manifest, _request, _case, _arm, authority = _contracts()
    arguments = {
        "stage": "general_review",
        "model_id": "openai/model-primary",
        "deployment_id": None,
        "gateway_api_base": _GATEWAY_API_BASE,
        "max_output_tokens": 128,
        "provider_max_retries": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(CheckpointCostAuthorityError, match=message):
        CostAuthorityLedger(authority).reserve(**arguments)


def test_ledger_atomically_consumes_worst_case_charge_across_concurrency():
    _manifest, _request, _case, _arm, authority = _contracts()
    quote = replace(authority.quotes[0], maximum_charge_usd=Decimal("0.01"))
    authority = replace(authority, hard_cost_cap_usd=Decimal("0.05"), quotes=(quote,))
    ledger = CostAuthorityLedger(authority)

    def reserve():
        try:
            return ledger.reserve(
                stage="general_review",
                model_id=quote.model_id,
                deployment_id=None,
                gateway_api_base=_GATEWAY_API_BASE,
                max_output_tokens=128,
                provider_max_retries=0,
            )
        except CheckpointCostAuthorityError:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        reservations = tuple(executor.map(lambda _index: reserve(), range(20)))

    accepted = tuple(item for item in reservations if item is not None)
    assert len(accepted) == 5
    assert ledger.reserved_usd == Decimal("0.05")
    assert sorted(item.sequence for item in accepted) == [1, 2, 3, 4, 5]
    assert len({item.reservation_id for item in accepted}) == 5


def test_context_reservation_defaults_to_general_review_stage():
    _manifest, _request, _case, _arm, authority = _contracts()

    with use_checkpoint_cost_authority(authority) as ledger:
        reservation = reserve_checkpoint_provider_attempt(
            model_id="openai/model-primary",
            deployment_id=None,
            gateway_api_base=_GATEWAY_API_BASE,
            max_output_tokens=64,
            provider_max_retries=0,
            attribution=None,
        )

    assert reservation is not None
    assert reservation.stage == "general_review"
    assert ledger.reserved_usd == Decimal("0.01")


def test_context_ledger_is_process_wide_for_worker_threads():
    _manifest, _request, _case, _arm, authority = _contracts()

    def reserve_once():
        return reserve_checkpoint_provider_attempt(
            model_id="openai/model-primary",
            deployment_id=None,
            gateway_api_base=_GATEWAY_API_BASE,
            max_output_tokens=64,
            provider_max_retries=0,
            attribution="general_review",
        )

    with use_checkpoint_cost_authority(authority) as ledger:
        with ThreadPoolExecutor(max_workers=3) as executor:
            reservations = tuple(executor.map(lambda _index: reserve_once(), range(3)))

    assert all(item is not None for item in reservations)
    assert ledger.reserved_usd == Decimal("0.03")


@pytest.mark.asyncio
async def test_litellm_provider_boundary_reserves_before_each_actual_call(monkeypatch):
    _manifest, _request, _case, _arm, authority = _contracts()
    authority = replace(authority, hard_cost_cap_usd=Decimal("0.01"))
    completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    })
    monkeypatch.setattr(litellm_ai_handler, "acompletion", completion)
    handler = object.__new__(LiteLLMAIHandler)
    handler.streaming_required_models = []
    handler.force_streaming_provider = ""
    handler.force_streaming_api_base_substrings = []

    options = AIRequestOptions(
        provider_retries=0,
        max_output_tokens=64,
        attribution="general_review",
    )
    kwargs = {
        "model": "openai/model-primary",
        "deployment_id": None,
        "messages": [],
        "api_base": _GATEWAY_API_BASE,
        "max_tokens": 64,
        "max_retries": 0,
    }
    with use_checkpoint_cost_authority(authority), use_ai_request_options(options):
        response = await handler._get_completion(display_model="openai/model-primary", **kwargs)
        with pytest.raises(CheckpointCostAuthorityError, match="hard cost cap"):
            await handler._get_completion(display_model="openai/model-primary", **kwargs)

    assert response[:2] == ("ok", "stop")
    completion.assert_awaited_once()
    sent_headers = completion.await_args.kwargs["extra_headers"]
    assert sent_headers[CHECKPOINT_GATEWAY_ROUTE_HEADER] == authority.quotes[0].gateway_route_binding_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"api_base": None}, "explicit enforcing gateway"),
        ({"api_base": "https://different-gateway.example/v1"}, "no authoritative"),
        (
            {
                "extra_headers": {
                    CHECKPOINT_GATEWAY_ROUTE_HEADER.lower(): _hash("conflicting-gateway-route")
                }
            },
            "route binding conflicts",
        ),
    ),
)
async def test_litellm_gateway_binding_denials_stop_before_provider_call(monkeypatch, overrides, message):
    _manifest, _request, _case, _arm, authority = _contracts()
    completion = AsyncMock()
    monkeypatch.setattr(litellm_ai_handler, "acompletion", completion)
    handler = object.__new__(LiteLLMAIHandler)
    handler.streaming_required_models = []
    handler.force_streaming_provider = ""
    handler.force_streaming_api_base_substrings = []
    options = AIRequestOptions(
        provider_retries=0,
        max_output_tokens=64,
        attribution="general_review",
    )
    kwargs = {
        "model": "openai/model-primary",
        "deployment_id": None,
        "messages": [],
        "api_base": _GATEWAY_API_BASE,
        "max_tokens": 64,
        "max_retries": 0,
        **overrides,
    }

    with use_checkpoint_cost_authority(authority), use_ai_request_options(options):
        with pytest.raises(CheckpointCostAuthorityError, match=message):
            await handler._get_completion(display_model="openai/model-primary", **kwargs)

    completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_ledger_is_shared_by_concurrent_child_tasks():
    _manifest, _request, _case, _arm, authority = _contracts()

    async def reserve_once():
        await asyncio.sleep(0)
        return reserve_checkpoint_provider_attempt(
            model_id="openai/model-primary",
            deployment_id=None,
            gateway_api_base=_GATEWAY_API_BASE,
            max_output_tokens=64,
            provider_max_retries=0,
            attribution="general_review",
        )

    with use_checkpoint_cost_authority(authority) as ledger:
        reservations = await asyncio.gather(*(reserve_once() for _index in range(3)))

    assert all(item is not None for item in reservations)
    assert ledger.reserved_usd == Decimal("0.03")


def test_dynamic_frontier_attribution_uses_the_fixed_quoted_stage():
    assert checkpoint_cost_stage("frontier_adjudication:finding-123") == "frontier_adjudication"


def test_context_reservation_rejects_missing_enforcing_gateway_binding():
    _manifest, _request, _case, _arm, authority = _contracts()

    with use_checkpoint_cost_authority(authority):
        with pytest.raises(CheckpointCostAuthorityError, match="explicit enforcing gateway"):
            reserve_checkpoint_provider_attempt(
                model_id="openai/model-primary",
                deployment_id=None,
                gateway_api_base=None,
                max_output_tokens=64,
                provider_max_retries=0,
                attribution="general_review",
            )
