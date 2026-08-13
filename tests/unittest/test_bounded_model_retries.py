"""Bounded-retry behaviour: attempts per model, and the provider client's own retries.

The failure these guard against is wall clock, not correctness. Attempts multiply -
tenacity attempts x fallback models x the provider client's internal retries - so an
unresponsive endpoint can consume an entire CI job window and publish nothing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler


class FakeBox:
    def __init__(self, values=None, **attrs):
        self._values = values or {}
        for key, value in attrs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return self._values.get(key, default)


class FakeSettings:
    def __init__(self, config_values=None):
        self.config = FakeBox(
            config_values or {},
            reasoning_effort=None,
            ai_timeout=30,
            custom_reasoning_model=False,
            max_model_tokens=32000,
            verbosity_level=0,
            model="gpt-4o",
        )
        self.litellm = FakeBox()

    def get(self, key, default=None):
        return default


def _mock_response():
    mock = MagicMock()
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    mock.__getitem__.side_effect = response.__getitem__
    mock.dict.return_value = response
    return mock


class _RetryState:
    def __init__(self, attempt_number):
        self.attempt_number = attempt_number


@pytest.mark.parametrize(
    "configured, attempt_number, should_stop",
    [
        (None, 1, False),   # default budget is 2 attempts
        (None, 2, True),
        (1, 1, True),       # single attempt per model
        (3, 2, False),
        (3, 3, True),
        (0, 1, True),       # a zero/negative budget still allows one attempt
        (-5, 1, True),
        ("not-a-number", 1, False),  # invalid config falls back to the default
        ("not-a-number", 2, True),
    ],
)
def test_model_retries_stop_reads_config_at_call_time(monkeypatch, configured, attempt_number, should_stop):
    config_values = {} if configured is None else {"model_retries": configured}
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: FakeSettings(config_values=config_values))

    assert litellm_handler._model_retries_stop(_RetryState(attempt_number)) is should_stop


@pytest.mark.asyncio
async def test_chat_completion_retries_are_bounded_by_config(monkeypatch):
    """One attempt per model when configured, instead of the default two."""
    monkeypatch.setattr(
        litellm_handler, "get_settings", lambda: FakeSettings(config_values={"model_retries": 1})
    )

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = openai.APIError("boom", request=MagicMock(), body=None)
        handler = litellm_handler.LiteLLMAIHandler()

        with pytest.raises(openai.APIError):
            await handler.chat_completion(model="gpt-4o", system="sys", user="usr")

    assert mock_call.call_count == 1


@pytest.mark.asyncio
async def test_chat_completion_default_retry_budget_is_two_attempts(monkeypatch):
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: FakeSettings())

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = openai.APIError("boom", request=MagicMock(), body=None)
        handler = litellm_handler.LiteLLMAIHandler()

        with pytest.raises(openai.APIError):
            await handler.chat_completion(model="gpt-4o", system="sys", user="usr")

    assert mock_call.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("configured, expected", [(0, 0), (1, 1), ("2", 2), (-3, 0)])
async def test_provider_max_retries_is_forwarded(monkeypatch, configured, expected):
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: FakeSettings(config_values={"ai_provider_max_retries": configured}),
    )

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model="gpt-4o", system="sys", user="usr")

    assert mock_call.call_args.kwargs["max_retries"] == expected


@pytest.mark.asyncio
async def test_provider_max_retries_unset_leaves_provider_default(monkeypatch):
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: FakeSettings())

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model="gpt-4o", system="sys", user="usr")

    assert "max_retries" not in mock_call.call_args.kwargs


@pytest.mark.asyncio
async def test_invalid_provider_max_retries_is_ignored(monkeypatch):
    monkeypatch.setattr(
        litellm_handler,
        "get_settings",
        lambda: FakeSettings(config_values={"ai_provider_max_retries": "many"}),
    )

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()

        await handler.chat_completion(model="gpt-4o", system="sys", user="usr")

    assert "max_retries" not in mock_call.call_args.kwargs
