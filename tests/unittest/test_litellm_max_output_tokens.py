"""
Tests for config.max_output_tokens in LiteLLMAIHandler.chat_completion: a positive
value is sent as `max_tokens` for every provider, 0 (default) sends nothing, and a
limit set by the extended-thinking path stays authoritative.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import openai
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_request_context import AIModelRoute, AIRequestOptions, use_ai_request_options
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.config_loader import get_settings
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

# Environment variables that LiteLLMAIHandler.__init__ reads or mutates: the AWS
# credential path (entered when AWS_USE_IMDS is set) writes the AWS_* variables,
# and OPENAI_API_KEY influences the litellm.api_key fallback.
_HANDLER_ENV_VARS = (
    "AWS_USE_IMDS",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION_NAME",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _restore_litellm_globals():
    """LiteLLMAIHandler.__init__ mutates global litellm/openai state and, when
    AWS_USE_IMDS is set, os.environ; snapshot and restore both, and drop
    AWS_USE_IMDS so the AWS credential path never runs in these tests."""
    saved = (litellm.api_key, getattr(litellm, "openai_key", None), openai.api_key)
    saved_env = {name: os.environ.get(name) for name in _HANDLER_ENV_VARS}
    os.environ.pop("AWS_USE_IMDS", None)
    try:
        yield
    finally:
        litellm.api_key = saved[0]
        litellm.openai_key = saved[1]
        openai.api_key = saved[2]
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make_settings(config_values=None, settings_values=None):
    """Minimal settings whose `config.get(key, ...)` serves the given dict."""
    config_values = config_values or {}
    settings_values = settings_values or {}

    class Config:
        reasoning_effort = None
        ai_timeout = 30
        custom_reasoning_model = False
        max_model_tokens = 32000
        verbosity_level = 0
        seed = -1

        def get(self, key, default=None):
            return config_values.get(key, default)

    return type("Settings", (), {
        "config": Config(),
        "litellm": type("LiteLLM", (), {
            "get": lambda self, key, default=None: default,
        })(),
        "get": lambda self, key, default=None: settings_values.get(key, default),
    })()


def _mock_response():
    mock = MagicMock()
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    mock.__getitem__.side_effect = response.__getitem__
    mock.dict.return_value = response
    return mock


async def _run(monkeypatch, model, config_values):
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: _make_settings(config_values))
    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
               new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()
        await handler.chat_completion(model=model, system="sys", user="usr")
    return mock_call.call_args[1]


class TestMaxOutputTokens:

    def test_effective_cap_ignores_explicitly_disabled_openrouter_reasoning(
        self,
        monkeypatch,
    ):
        settings = _make_settings(settings_values={
            "openrouter": {
                "max_tokens": 1_500,
                "reasoning_effort": "none",
                "reasoning_max_tokens": 16_000,
            },
        })
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)

        cap = litellm_handler.get_effective_litellm_output_token_cap(
            "openrouter/anthropic/claude-sonnet-4",
            require_bounded_reasoning=True,
        )

        assert cap == 1_500

    @pytest.mark.parametrize("reasoning_effort", ["", "medium", "invalid"])
    def test_effective_cap_requires_headroom_when_openrouter_reasoning_is_not_disabled(
        self,
        monkeypatch,
        reasoning_effort,
    ):
        settings = _make_settings(settings_values={
            "openrouter": {
                "max_tokens": 1_500,
                "reasoning_effort": reasoning_effort,
                "reasoning_max_tokens": 16_000,
            },
        })
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)

        with pytest.raises(ValueError, match="response headroom"):
            litellm_handler.get_effective_litellm_output_token_cap(
                "openrouter/anthropic/claude-sonnet-4",
                require_bounded_reasoning=True,
            )

    @pytest.mark.asyncio
    async def test_default_sends_no_max_tokens(self, monkeypatch):
        kwargs = await _run(monkeypatch, "bedrock/anthropic.claude-sonnet-5-v1:0", {})
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        "bedrock/anthropic.claude-sonnet-5-v1:0",
        "gpt-4o",
    ])
    async def test_positive_value_sent_as_max_tokens(self, monkeypatch, model):
        kwargs = await _run(monkeypatch, model, {"max_output_tokens": 16000})
        assert kwargs["max_tokens"] == 16000

    @pytest.mark.asyncio
    async def test_extended_thinking_limit_stays_authoritative(self, monkeypatch):
        kwargs = await _run(monkeypatch, "claude-3-7-sonnet-20250219", {
            "max_output_tokens": 16000,
            "enable_claude_extended_thinking": True,
            "extended_thinking_budget_tokens": 2048,
            "extended_thinking_max_output_tokens": 4096,
        })
        assert kwargs["max_tokens"] == 4096
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    @pytest.mark.asyncio
    async def test_extended_thinking_string_false_keeps_generic_cap(self, monkeypatch):
        kwargs = await _run(monkeypatch, "claude-3-7-sonnet-20250219", {
            "max_output_tokens": 12000,
            "enable_claude_extended_thinking": "false",
            "extended_thinking_budget_tokens": "invalid",
            "extended_thinking_max_output_tokens": -1,
        })

        assert kwargs["max_tokens"] == 12000
        assert "thinking" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("output_cap", "thinking_enabled"),
        [(2_047, False), (2_048, False), (2_049, True)],
    )
    async def test_request_local_cap_requires_strict_thinking_headroom(
        self, monkeypatch, output_cap, thinking_enabled
    ):
        with use_ai_request_options(AIRequestOptions(max_output_tokens=output_cap)):
            kwargs = await _run(monkeypatch, "claude-3-7-sonnet-20250219", {
                "enable_claude_extended_thinking": True,
                "extended_thinking_budget_tokens": 2048,
                "extended_thinking_max_output_tokens": 4096,
            })

        assert kwargs["max_tokens"] == output_cap
        if thinking_enabled:
            assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        else:
            assert "thinking" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("output_cap", "thinking_enabled"),
        [(2_047, False), (2_048, False), (2_049, True)],
    )
    async def test_global_cap_requires_strict_thinking_headroom(
        self, monkeypatch, output_cap, thinking_enabled
    ):
        kwargs = await _run(monkeypatch, "claude-3-7-sonnet-20250219", {
            "max_output_tokens": output_cap,
            "enable_claude_extended_thinking": True,
            "extended_thinking_budget_tokens": 2048,
            "extended_thinking_max_output_tokens": 4096,
        })

        assert kwargs["max_tokens"] == output_cap
        if thinking_enabled:
            assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        else:
            assert "thinking" not in kwargs

    @pytest.mark.asyncio
    async def test_equal_cap_is_applied_to_claude_fallback_without_thinking(self, monkeypatch):
        tracked_keys = (
            "config.enable_claude_extended_thinking",
            "config.extended_thinking_budget_tokens",
            "config.extended_thinking_max_output_tokens",
            "config.last_used_model",
        )
        snapshot = snapshot_settings(tracked_keys)
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["model"] == "gpt-4o":
                raise RuntimeError("primary unavailable")
            return _mock_response()

        try:
            settings = get_settings()
            settings.set("config.enable_claude_extended_thinking", True)
            settings.set("config.extended_thinking_budget_tokens", 2_048)
            settings.set("config.extended_thinking_max_output_tokens", 4_096)
            monkeypatch.setattr(litellm_handler, "acompletion", fake_completion)
            handler = litellm_handler.LiteLLMAIHandler()
            route = AIModelRoute(
                models=("gpt-4o", "claude-3-7-sonnet-20250219"),
                deployments=(None, None),
                model_retries=1,
                max_output_tokens=2_048,
            )

            result = await retry_with_fallback_models(
                lambda model: handler.chat_completion(model=model, system="sys", user="usr"),
                model_route=route,
            )
        finally:
            restore_settings(snapshot)

        assert result[0] == "ok"
        assert [call["model"] for call in calls] == ["gpt-4o", "claude-3-7-sonnet-20250219"]
        assert calls[1]["max_tokens"] == 2_048
        assert "thinking" not in calls[1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("request_cap", "thinking_enabled"),
        [(2047, False), (2048, False), (2049, True)],
    )
    async def test_request_local_cap_matches_thinking_headroom_boundary(
        self,
        monkeypatch,
        request_cap,
        thinking_enabled,
    ):
        config_values = {
            "enable_claude_extended_thinking": True,
            "extended_thinking_budget_tokens": 2048,
            "extended_thinking_max_output_tokens": 4096,
        }
        with use_ai_request_options(AIRequestOptions(max_output_tokens=request_cap)):
            kwargs = await _run(
                monkeypatch,
                "claude-3-7-sonnet-20250219",
                config_values,
            )

        assert kwargs["max_tokens"] == request_cap
        assert ("thinking" in kwargs) is thinking_enabled
        if thinking_enabled:
            assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    @pytest.mark.asyncio
    async def test_string_override_is_coerced(self, monkeypatch):
        # Dynaconf/env overrides can arrive as strings.
        kwargs = await _run(monkeypatch, "gpt-4o", {"max_output_tokens": "16000"})
        assert kwargs["max_tokens"] == 16000

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["16k", None, 0, -1])
    async def test_unset_or_invalid_values_send_nothing(self, monkeypatch, value):
        kwargs = await _run(monkeypatch, "gpt-4o", {"max_output_tokens": value})
        assert "max_tokens" not in kwargs
