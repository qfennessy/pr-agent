"""Independent model calls share inputs, never settings or persistent comments."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette_context import request_cycle_context

from pr_agent.config_loader import get_settings
from pr_agent.git_providers.git_provider import IncrementalPR
from pr_agent.tools.pr_reviewer import PRReviewer
from tests.unittest.test_persistent_comment_identity import FakeProvider


@pytest.fixture
def settings():
    get_settings().as_dict()
    with request_cycle_context({"settings": copy.deepcopy(get_settings())}):
        current = get_settings()
        current.set("pr_reviewer.review_models", ["provider/a", "provider/b"])
        current.set("config.model", "original")
        current.set("config.fallback_models", ["fallback"])
        current.set("config.persistent_comment_id", "original-id")
        current.set("config.publish_output", True)
        current.set("pr_reviewer.enable_help_text", False)
        current.set("config.output_run_details", False)
        current.set("config.ai_timeout", 1)
        yield current


def make_reviewer(monkeypatch, handler):
    provider = FakeProvider()
    provider.pr = SimpleNamespace(title="Example")
    provider.get_files = lambda: ["example.py"]
    provider.publish_persistent_comment = provider.publish_persistent_comment_full
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.vars = {"title": "Example", "related_tickets": []}
    reviewer.main_language = "Python"
    reviewer.incremental = IncrementalPR(False)
    reviewer._ai_handler_factory = handler
    reviewer.remaining_files_list = []
    reviewer.deleted_files_list = []
    reviewer.prediction = None
    reviewer.patches_diff = "the shared diff"
    reviewer.pr_url = "https://example.test/pr/1"
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.TokenHandler", MagicMock())
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.get_max_tokens", lambda model: 10000)
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(PRReviewer, "_prepare_review_diff", MagicMock())
    return reviewer, provider


async def test_parallel_identical_prompts_isolated_settings_and_update_in_place(monkeypatch, settings):
    started = set()
    all_started = asyncio.Event()
    prompts = []
    instances = []

    class Handler:
        def __init__(self):
            instances.append(self)

        async def chat_completion(self, model, system, user, **kwargs):
            started.add(model)
            prompts.append((system, user))
            if len(started) == 2:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), .5)
            assert get_settings().config.model == model
            assert get_settings().config.persistent_comment_id == model
            get_settings().set("config.test_mutation", model)
            await asyncio.sleep(0)
            assert get_settings().config.test_mutation == model
            return "review:\n  estimated_effort_to_review_[1-5]: 2\n", "stop"

    reviewer, provider = make_reviewer(monkeypatch, Handler)
    settings.set("pr_review_prompt.system", "Review {{ title }}")
    settings.set("pr_review_prompt.user", "{{ diff }}")
    await reviewer.run()
    assert len(instances) == 2
    assert prompts == [("Review Example", "the shared diff")] * 2
    assert len(provider.existing) == 2
    for model in settings.pr_reviewer.review_models:
        comment = next(c for c in provider.existing if f"PR Reviewer Guide ({model})" in c.body)
        assert f"PR Reviewer Guide ({model})" in comment.body
        assert f"> Reviewed by `{model}`" in comment.body
        assert "fallback model" not in comment.body
        assert comment.body.endswith(f"<!-- pr-agent-persistent-id: {model} -->")
        assert "Review failed" not in comment.body
    assert settings.config.model == "original"
    assert settings.config.persistent_comment_id == "original-id"
    assert settings.config.fallback_models == ["fallback"]
    assert settings.get("config.test_mutation") is None
    assert reviewer.prediction is None
    settings.set("pr_reviewer.review_models", ["provider/b", "provider/a"])
    await reviewer.run()
    assert len(provider.existing) == 2
    assert len(provider.edited) == 2


@pytest.mark.parametrize("failure", ["timeout", "provider", "invalid", "empty"])
async def test_one_failure_does_not_cancel_other_model(monkeypatch, settings, failure):
    class Handler:
        async def chat_completion(self, model, **kwargs):
            if model == "provider/a":
                if failure == "timeout":
                    await asyncio.sleep(10)
                if failure == "provider":
                    raise RuntimeError("secret provider response")
                return ("nonsense" if failure == "invalid" else ""), "stop"
            return "review:\n  estimated_effort_to_review_[1-5]: 2\n", "stop"

    settings.set("config.ai_timeout", .05)
    settings.set("pr_review_prompt.system", "review")
    settings.set("pr_review_prompt.user", "{{ diff }}")
    reviewer, provider = make_reviewer(monkeypatch, Handler)
    await reviewer.run()
    bodies = {comment.body.split("\n")[0]: comment.body for comment in provider.existing}
    assert len(bodies) == 2
    assert "Review failed:" in bodies["## PR Reviewer Guide (provider/a) 🔍"]
    assert "Review failed:" not in bodies["## PR Reviewer Guide (provider/b) 🔍"]
    assert "secret provider response" not in "".join(bodies.values())


async def test_no_publish_duplicate_models_and_bugs_only_empty_result(monkeypatch, settings):
    handler = MagicMock()
    handler.chat_completion = AsyncMock(return_value=("review:\n  key_issues_to_review: []\n", "stop"))
    reviewer, provider = make_reviewer(monkeypatch, lambda: handler)
    reviewer.review_profile = "bugs_only"
    settings.set("pr_review_prompt.system", "review")
    settings.set("pr_review_prompt.user", "{{ diff }}")
    settings.set("config.publish_output", False)
    settings.set("pr_reviewer.review_models", ["provider/a", "provider/a"])
    await reviewer.run()
    assert handler.chat_completion.await_count == 1
    assert provider.existing == []
    assert "No major issues detected" in settings.data.reviews["provider/a"]


async def test_empty_list_uses_existing_flow(monkeypatch, settings):
    settings.set("pr_reviewer.review_models", [])
    reviewer, provider = make_reviewer(monkeypatch, MagicMock)
    provider.get_files = lambda: []
    monkeypatch.setattr(reviewer, "_prepare_route_after_empty_review_inventory", MagicMock())
    parallel = AsyncMock()
    monkeypatch.setattr(reviewer, "_run_review_models", parallel)
    await reviewer.run()
    parallel.assert_not_awaited()


async def test_unknown_model_and_publication_failure_are_isolated(monkeypatch, settings):
    handler = MagicMock()
    handler.chat_completion = AsyncMock(return_value=("review:\n  key_issues_to_review: []\n", "stop"))
    reviewer, provider = make_reviewer(monkeypatch, lambda: handler)
    settings.set("pr_review_prompt.system", "review")
    settings.set("pr_review_prompt.user", "{{ diff }}")

    def budget(model):
        if model == "provider/a":
            raise ValueError("unknown model")
        return 10000

    monkeypatch.setattr("pr_agent.tools.pr_reviewer.get_max_tokens", budget)
    await reviewer.run()
    assert handler.chat_completion.await_count == 1
    assert len(provider.existing) == 2
    assert "model context window unavailable" in provider.existing[0].body
    publish = provider.publish_persistent_comment

    def fail_one(body, **kwargs):
        if get_settings().config.model == "provider/a":
            raise RuntimeError("publication unavailable")
        return publish(body, **kwargs)

    provider.publish_persistent_comment = fail_one
    await reviewer.run()
    assert len(provider.edited) == 1
    assert "provider/b" in settings.data.reviews
