import asyncio
from unittest.mock import MagicMock

import pytest

from pr_agent.algo.ai_handlers.langchain_ai_handler import LangChainOpenAIHandler
from pr_agent.algo.ai_handlers.openai_ai_handler import OpenAIHandler
from pr_agent.algo.ai_request_context import (
    AIModelRoute,
    AIRequestOptions,
    get_ai_request_options,
    use_ai_request_options,
)
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.algo.run_details import get_run_details, init_run_details
from pr_agent.algo.utils import ModelType
from pr_agent.config_loader import get_settings
from tests.unittest._settings_helpers import SENTINEL, restore_settings, snapshot_settings

_TRACKED_KEYS = (
    "config.model",
    "config.model_weak",
    "config.model_reasoning",
    "config.fallback_models",
    "config.last_used_model",
    "openai.deployment_id",
    "openai.fallback_deployments",
)


def _snapshot_settings():
    return snapshot_settings(_TRACKED_KEYS)


def _restore_settings(snapshot):
    restore_settings(snapshot)


def test_primary_model_success_invoked_once_and_returns_value():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1", "fallback-2"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        calls = []

        async def fake_f(model):
            calls.append(model)
            return "primary-result"

        result = asyncio.run(retry_with_fallback_models(fake_f))

        assert result == "primary-result"
        assert calls == ["primary-model"]
    finally:
        _restore_settings(snapshot)


def test_empty_explicit_route_raises_instead_of_returning_implicitly():
    class EmptyRoute:
        models = ()
        deployments = ()

    async def fake_f(_model):
        raise AssertionError("an empty route must not invoke the model callback")

    with pytest.raises(RuntimeError, match="Model route exhausted"):
        asyncio.run(retry_with_fallback_models(fake_f, model_route=EmptyRoute()))


def test_request_local_deployment_is_visible_to_all_legacy_handler_properties():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("openai.deployment_id", "global-deployment")
        handlers = (
            OpenAIHandler.__new__(OpenAIHandler),
            LangChainOpenAIHandler.__new__(LangChainOpenAIHandler),
        )

        with use_ai_request_options(AIRequestOptions(deployment_id="role-deployment")):
            assert all(handler.deployment_id == "role-deployment" for handler in handlers)

        assert all(handler.deployment_id == "global-deployment" for handler in handlers)
    finally:
        _restore_settings(snapshot)


def test_primary_fails_fallback_succeeds():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1", "fallback-2"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        calls = []

        async def fake_f(model):
            calls.append(model)
            if model == "primary-model":
                raise RuntimeError("primary failed")
            return f"ok:{model}"

        result = asyncio.run(retry_with_fallback_models(fake_f))

        assert result == "ok:fallback-1"
        assert calls == ["primary-model", "fallback-1"]
    finally:
        _restore_settings(snapshot)


def test_all_models_fail_raises_with_aggregate_message_and_cause():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        last_error = ValueError("last failure")
        attempted = []

        async def fake_f(model):
            attempted.append(model)
            if model == "fallback-1":
                raise last_error
            raise RuntimeError("primary failure")

        with pytest.raises(Exception) as exc_info:
            asyncio.run(retry_with_fallback_models(fake_f))

        assert attempted == ["primary-model", "fallback-1"]
        assert "Failed to generate prediction with any model" in str(exc_info.value)
        # Production code uses `raise ... from e`, so the last failure should be chained.
        assert exc_info.value.__cause__ is last_error
    finally:
        _restore_settings(snapshot)


def test_deployment_id_updated_per_attempt():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1", "fallback-2"])
        get_settings().set("openai.deployment_id", "deployment-primary")
        get_settings().set(
            "openai.fallback_deployments",
            ["deployment-fb1", "deployment-fb2"],
        )

        observed = []

        async def fake_f(model):
            observed.append(
                (model, get_settings().get("openai.deployment_id", None))
            )
            if model != "fallback-1":
                raise RuntimeError(f"fail for {model}")
            return "fallback-ok"

        result = asyncio.run(retry_with_fallback_models(fake_f))

        assert result == "fallback-ok"
        assert observed == [
            ("primary-model", "deployment-primary"),
            ("fallback-1", "deployment-fb1"),
        ]
        assert get_settings().get("openai.deployment_id", None) == "deployment-primary"
    finally:
        _restore_settings(snapshot)


def test_concurrent_explicit_routes_do_not_leak_deployments_or_controls():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("openai.deployment_id", "shared-deployment")
        observed = []

        async def run(role, deployment, delay):
            route = AIModelRoute(
                models=(f"model-{role}",),
                deployments=(deployment,),
                timeout_seconds=delay + 1,
                model_retries=1,
                provider_retries=0,
                max_output_tokens=100 + len(role),
                attribution=role,
            )

            async def fake_f(model):
                await asyncio.sleep(delay)
                options = get_ai_request_options()
                observed.append(
                    (
                        role,
                        model,
                        options.deployment_id,
                        options.timeout_seconds,
                        options.max_output_tokens,
                        get_settings().get("openai.deployment_id", None),
                    )
                )
                return role

            return await retry_with_fallback_models(fake_f, model_route=route)

        async def run_both():
            return await asyncio.gather(
                run("classification", "deployment-a", 0.02),
                run("risk", "deployment-b", 0.01),
            )

        results = asyncio.run(run_both())

        assert results == ["classification", "risk"]
        assert set(observed) == {
            ("classification", "model-classification", "deployment-a", 1.02, 114, "shared-deployment"),
            ("risk", "model-risk", "deployment-b", 1.01, 104, "shared-deployment"),
        }
        assert get_settings().get("openai.deployment_id", None) == "shared-deployment"
    finally:
        _restore_settings(snapshot)


def test_fallback_deployment_does_not_poison_the_next_retry():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1"])
        get_settings().set("openai.deployment_id", "deployment-primary")
        get_settings().set("openai.fallback_deployments", ["deployment-fallback"])

        observed = []

        async def fake_f(model):
            observed.append((model, get_settings().get("openai.deployment_id", None)))
            if model == "primary-model":
                raise RuntimeError("primary failed")
            return "fallback-ok"

        assert asyncio.run(retry_with_fallback_models(fake_f)) == "fallback-ok"
        assert get_settings().get("openai.deployment_id") == "deployment-primary"
        assert asyncio.run(retry_with_fallback_models(fake_f)) == "fallback-ok"

        assert observed == [
            ("primary-model", "deployment-primary"),
            ("fallback-1", "deployment-fallback"),
            ("primary-model", "deployment-primary"),
            ("fallback-1", "deployment-fallback"),
        ]
    finally:
        _restore_settings(snapshot)


def test_attributed_route_failure_log_omits_provider_exception_text(monkeypatch):
    secret = "PRIVATE_REPOSITORY_EXCERPT_FROM_PROVIDER_ERROR"
    logger = MagicMock()
    monkeypatch.setattr("pr_agent.algo.pr_processing.get_logger", lambda: logger)
    route = AIModelRoute(
        models=("verifier-model",),
        deployments=(None,),
        attribution="candidate_verification",
    )

    async def fail(_model):
        raise RuntimeError(secret)

    with pytest.raises(Exception, match="Failed to generate prediction with any model"):
        asyncio.run(retry_with_fallback_models(fail, model_route=route))

    warning_call = logger.warning.call_args
    assert secret not in str(warning_call)
    assert "error" not in warning_call.kwargs["artifact"]
    assert warning_call.kwargs["artifact"]["error_class"] == "RuntimeError"


def test_deployment_id_is_restored_when_retry_is_cancelled():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1"])
        get_settings().set("openai.deployment_id", "deployment-primary")
        get_settings().set("openai.fallback_deployments", ["deployment-fallback"])

        async def fake_f(model):
            if model == "primary-model":
                raise RuntimeError("primary failed")
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(retry_with_fallback_models(fake_f))

        assert get_settings().get("openai.deployment_id") == "deployment-primary"
    finally:
        _restore_settings(snapshot)


def test_weak_model_type_uses_weak_setting_and_forwards_identifier():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "regular-model")
        get_settings().set("config.model_weak", "weak-model-id")
        get_settings().set("config.fallback_models", [])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        calls = []

        async def fake_f(model):
            calls.append(model)
            return model

        result = asyncio.run(
            retry_with_fallback_models(fake_f, model_type=ModelType.WEAK)
        )

        assert result == "weak-model-id"
        assert calls == ["weak-model-id"]
    finally:
        _restore_settings(snapshot)


def test_reasoning_model_type_uses_reasoning_setting():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "regular-model")
        get_settings().set("config.model_reasoning", "reasoning-model-id")
        get_settings().set("config.fallback_models", [])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        calls = []

        async def fake_f(model):
            calls.append(model)
            return model

        result = asyncio.run(
            retry_with_fallback_models(fake_f, model_type=ModelType.REASONING)
        )

        assert result == "reasoning-model-id"
        assert calls == ["reasoning-model-id"]
    finally:
        _restore_settings(snapshot)


def test_restore_settings_truly_removes_originally_missing_dotted_keys():
    """Regression: SENTINEL-snapshotted dotted leaves must be removed, not left behind."""
    settings = get_settings()
    key = "openai.fallback_deployments"
    # Ensure key is absent on entry; if a previous test leaked it, clean it.
    if settings.get(key, SENTINEL) is not SENTINEL:
        _restore_settings({key: SENTINEL})
    assert settings.get(key, SENTINEL) is SENTINEL

    snapshot = _snapshot_settings()
    try:
        settings.set(key, ["leaked-deployment"])
        assert settings.get(key) == ["leaked-deployment"]
    finally:
        _restore_settings(snapshot)

    assert settings.get(key, SENTINEL) is SENTINEL


def test_records_primary_model_without_fallback_flag():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])
        init_run_details()

        async def fake_f(model):
            return "ok"

        asyncio.run(retry_with_fallback_models(fake_f))

        details = get_run_details()
        assert details.model_used == "primary-model"
        assert details.fallback_used is False
    finally:
        _restore_settings(snapshot)


def test_records_fallback_model_with_fallback_flag():
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])
        init_run_details()

        async def fake_f(model):
            if model == "primary-model":
                raise RuntimeError("primary failed")
            return "ok"

        asyncio.run(retry_with_fallback_models(fake_f))

        details = get_run_details()
        assert details.model_used == "fallback-1"
        assert details.fallback_used is True
        assert get_settings().config.get("last_used_model") == "fallback-1"
    finally:
        _restore_settings(snapshot)


def test_fallback_flag_set_even_when_fallback_repeats_primary_model_name():
    """`fallback_models` may repeat the primary model; the flag is index-based."""
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "same-model")
        get_settings().set("config.fallback_models", ["same-model"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])
        init_run_details()

        attempts = []

        async def fake_f(model):
            attempts.append(model)
            if len(attempts) == 1:
                raise RuntimeError("first attempt failed")
            return "ok"

        asyncio.run(retry_with_fallback_models(fake_f))

        details = get_run_details()
        assert details.model_used == "same-model"
        assert details.fallback_used is True
    finally:
        _restore_settings(snapshot)


def test_recording_successful_model_does_not_trigger_fallback_retry(monkeypatch):
    snapshot = _snapshot_settings()
    try:
        get_settings().set("config.model", "primary-model")
        get_settings().set("config.fallback_models", ["fallback-1"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])
        init_run_details()

        calls = []

        async def fake_f(model):
            calls.append(model)
            return "ok"

        def boom(*_args, **_kwargs):
            raise RuntimeError("telemetry failed")

        monkeypatch.setattr("pr_agent.algo.pr_processing.record_model_used", boom)

        with pytest.raises(RuntimeError, match="telemetry failed"):
            asyncio.run(retry_with_fallback_models(fake_f))

        assert calls == ["primary-model"]
    finally:
        _restore_settings(snapshot)


def test_all_models_failed_writes_ci_summary_with_attempt_evidence(tmp_path, monkeypatch):
    """A provider outage must be diagnosable without downloading the raw log archive."""
    snapshot = _snapshot_settings()
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    try:
        get_settings().set("config.model", "openai/primary")
        get_settings().set("config.fallback_models", ["openai/fallback"])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        async def fake_f(model):
            raise TimeoutError("Request timed out - timeout value=90.0, api_key=sk-secret123456789")

        with pytest.raises(Exception):
            asyncio.run(retry_with_fallback_models(fake_f))

        written = summary.read_text(encoding="utf-8")
        assert "no model produced a response" in written
        assert "openai/primary" in written and "openai/fallback" in written
        assert "1/2" in written and "2/2" in written
        assert "TimeoutError" in written
        assert "timeout value=90.0" in written
        assert "sk-secret123456789" not in written  # credential-shaped text is redacted
    finally:
        _restore_settings(snapshot)


@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"])
def test_all_models_failed_redacts_github_token_shapes(tmp_path, monkeypatch, prefix):
    snapshot = _snapshot_settings()
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    fake_token = prefix + "x" * 20
    try:
        get_settings().set("config.model", "openai/primary")
        get_settings().set("config.fallback_models", [])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        async def fake_f(model):
            raise RuntimeError(f"provider rejected credential {fake_token}")

        with pytest.raises(Exception):
            asyncio.run(retry_with_fallback_models(fake_f))

        written = summary.read_text(encoding="utf-8")
        assert fake_token not in written
        assert "[redacted]" in written
    finally:
        _restore_settings(snapshot)


def test_no_ci_summary_written_outside_ci(tmp_path, monkeypatch):
    snapshot = _snapshot_settings()
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    try:
        get_settings().set("config.model", "openai/primary")
        get_settings().set("config.fallback_models", [])
        get_settings().set("openai.deployment_id", None)
        get_settings().set("openai.fallback_deployments", [])

        async def fake_f(model):
            raise TimeoutError("boom")

        with pytest.raises(Exception):
            asyncio.run(retry_with_fallback_models(fake_f))
    finally:
        _restore_settings(snapshot)
