"""
Tests for deepseek.reasoning_effort in LiteLLMAIHandler.chat_completion.

LiteLLM reduces a top-level reasoning_effort for DeepSeek to thinking={"type": "enabled"},
so the handler sends the effort in extra_body instead. Only native "deepseek/..." models
are affected, and an unset value leaves the request unchanged.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import openai
import pytest
from litellm.utils import get_optional_params

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.config_loader import get_settings

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
    """LiteLLMAIHandler.__init__ mutates global litellm/openai state; snapshot and restore it."""
    saved = (litellm.api_key, getattr(litellm, "openai_key", None), openai.api_key, litellm.drop_params)
    saved_env = {name: os.environ.get(name) for name in _HANDLER_ENV_VARS}
    os.environ.pop("AWS_USE_IMDS", None)
    litellm.drop_params = False
    try:
        yield
    finally:
        litellm.api_key, litellm.openai_key, openai.api_key, litellm.drop_params = saved
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make_settings(deepseek=None, custom_llm_provider=""):
    """Minimal settings whose `.get("deepseek", ...)` returns the given dict."""
    deepseek = deepseek or {}
    return type("Settings", (), {
        "config": type("Config", (), {
            "reasoning_effort": "high",
            "ai_timeout": 30,
            "custom_reasoning_model": False,
            "max_model_tokens": 32000,
            "verbosity_level": 0,
            "seed": -1,
            "get": lambda self, key, default=None: default,
        })(),
        "litellm": type("LiteLLM", (), {
            "custom_llm_provider": custom_llm_provider,
            "get": lambda self, key, default=None: default,
        })(),
        "get": lambda self, key, default=None: (deepseek if key == "deepseek" else default),
    })()


def _mock_response():
    mock = MagicMock()
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    mock.__getitem__.side_effect = response.__getitem__
    mock.dict.return_value = response
    return mock


async def _run(monkeypatch, model, deepseek, custom_llm_provider=""):
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: _make_settings(deepseek, custom_llm_provider),
    )
    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion",
               new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()
        await handler.chat_completion(model=model, system="sys", user="usr")
    return mock_call.call_args[1]


class TestDeepSeekReasoningEffort:

    @pytest.mark.asyncio
    async def test_low_is_sent_in_extra_body(self, monkeypatch):
        kwargs = await _run(monkeypatch, "deepseek/deepseek-flash", {"reasoning_effort": "low"})
        assert kwargs["extra_body"] == {"reasoning_effort": "low"}
        assert "reasoning_effort" not in kwargs
        assert "thinking" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("configured", "sent"),
        [("minimal", "low"), ("low", "low"), ("medium", "high"), ("high", "high"), ("xhigh", "max"),
         ("max", "max"), (" LOW ", "low")],
    )
    async def test_effort_maps_to_deepseek_levels(self, monkeypatch, configured, sent):
        kwargs = await _run(monkeypatch, "deepseek/deepseek-flash", {"reasoning_effort": configured})
        assert kwargs["extra_body"] == {"reasoning_effort": sent}

    @pytest.mark.asyncio
    async def test_none_disables_thinking(self, monkeypatch):
        kwargs = await _run(monkeypatch, "deepseek/deepseek-flash", {"reasoning_effort": "none"})
        assert kwargs["thinking"] == {"type": "disabled"}
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("deepseek", [{}, {"reasoning_effort": ""}, {"reasoning_effort": "extreme"}])
    async def test_unset_or_invalid_leaves_request_unchanged(self, monkeypatch, deepseek):
        kwargs = await _run(monkeypatch, "deepseek/deepseek-flash", deepseek)
        assert "extra_body" not in kwargs
        assert "thinking" not in kwargs
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_other_provider_models_are_unaffected(self, monkeypatch):
        kwargs = await _run(monkeypatch, "openai/gpt-5.6-luna", {"reasoning_effort": "low"})
        assert kwargs["reasoning_effort"] == "high"
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    async def test_custom_llm_provider_override_is_unaffected(self, monkeypatch):
        """A raw "deepseek/..." id sent to another provider is not DeepSeek's native API."""
        kwargs = await _run(
            monkeypatch, "deepseek/deepseek-flash", {"reasoning_effort": "low"}, custom_llm_provider="openai",
        )
        assert "extra_body" not in kwargs

    def test_caller_extra_body_is_merged_not_mutated(self, monkeypatch):
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: _make_settings({"reasoning_effort": "low"}))
        caller_body = {"service_tier": "flex"}
        kwargs = litellm_handler.LiteLLMAIHandler._configure_deepseek_reasoning_effort(
            "deepseek/deepseek-flash", {"extra_body": caller_body},
        )
        assert kwargs["extra_body"] == {"service_tier": "flex", "reasoning_effort": "low"}
        assert caller_body == {"service_tier": "flex"}

    def test_litellm_keeps_extra_body_effort_but_strips_top_level(self):
        """Pin the LiteLLM behavior this handler relies on."""
        via_extra_body = get_optional_params(
            model="deepseek-flash", custom_llm_provider="deepseek", extra_body={"reasoning_effort": "low"},
        )
        assert via_extra_body["extra_body"] == {"reasoning_effort": "low"}
        top_level = get_optional_params(
            model="deepseek-flash", custom_llm_provider="deepseek", reasoning_effort="low",
        )
        assert top_level["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in top_level["extra_body"]

    def test_default_configuration_leaves_effort_unset(self):
        assert get_settings().get("deepseek", {}).get("reasoning_effort") == ""
