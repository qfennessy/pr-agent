"""Independent review models share inputs while retaining separate comment ownership."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette_context import request_cycle_context

from pr_agent.config_loader import get_settings
from pr_agent.tools.pr_reviewer import PRReviewer


@pytest.fixture
def settings():
    get_settings().as_dict()
    with request_cycle_context({"settings": copy.deepcopy(get_settings())}):
        current = get_settings()
        current.set("pr_reviewer.review_models", ["provider/a", "provider/b"])
        current.set("config.model", "original")
        current.set("config.fallback_models", ["fallback"])
        current.set("config.publish_output", True)
        current.set("config.ai_timeout", 1)
        current.set("pr_review_prompt.system", "Review {{ title }}")
        current.set("pr_review_prompt.user", "{{ diff }}")
        yield current


def make_reviewer(monkeypatch, handler):
    published = []
    provider = SimpleNamespace(
        pr=SimpleNamespace(title="Example"),
        get_files=lambda: ["example.py"],
        get_diff_files=lambda: [],
        is_supported=lambda _capability: False,
        publish_persistent_comment=lambda body, **kwargs: published.append((body, kwargs)),
    )
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.vars = {"title": "Example", "related_tickets": []}
    reviewer.main_language = "Python"
    reviewer._ai_handler_factory = handler
    reviewer.patches_diff = None
    reviewer.remaining_files_list = []
    reviewer.pr_url = "https://example.test/pr/1"
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(
        "pr_agent.tools.pr_reviewer.fit_related_tickets_to_prompt_budget",
        lambda _pr, variables, *_args: (variables, MagicMock()),
    )
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.get_pr_diff", lambda *_args, **_kwargs: ("shared diff", []))
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.get_max_tokens", lambda _model: 10000)
    return reviewer, published


async def test_models_run_concurrently_with_identical_prompts_and_separate_comments(monkeypatch, settings):
    started = set()
    ready = asyncio.Event()
    prompts = []

    class Handler:
        async def chat_completion(self, model, system, user, **_kwargs):
            started.add(model)
            prompts.append((system, user))
            if len(started) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), .5)
            assert get_settings().config.model == model
            assert get_settings().config.persistent_comment_id == model
            return "review:\n  estimated_effort_to_review_[1-5]: 2\n", "stop"

    reviewer, published = make_reviewer(monkeypatch, Handler)
    await reviewer._run_review_models(settings.pr_reviewer.review_models)

    assert prompts == [("Review Example", "shared diff")] * 2
    assert len(published) == 2
    for model in settings.pr_reviewer.review_models:
        body, kwargs = next(item for item in published if f"({model})" in item[0])
        assert f"PR Reviewer Guide ({model})" in body
        assert kwargs["initial_header"] == f"## PR Reviewer Guide ({model}) 🔍"


@pytest.mark.parametrize("failure", ["timeout", "provider", "invalid", "empty"])
async def test_one_failed_model_does_not_cancel_the_other(monkeypatch, settings, failure):
    class Handler:
        async def chat_completion(self, model, **_kwargs):
            if model == "provider/a":
                if failure == "timeout":
                    await asyncio.sleep(10)
                if failure == "provider":
                    raise RuntimeError("provider response must not be published")
                return ("not yaml" if failure == "invalid" else ""), "stop"
            return "review:\n  estimated_effort_to_review_[1-5]: 2\n", "stop"

    settings.set("config.ai_timeout", .05)
    reviewer, published = make_reviewer(monkeypatch, Handler)
    await reviewer._run_review_models(settings.pr_reviewer.review_models)

    bodies = {kwargs["initial_header"]: body for body, kwargs in published}
    assert "Review failed:" in bodies["## PR Reviewer Guide (provider/a) 🔍"]
    assert "Review failed:" not in bodies["## PR Reviewer Guide (provider/b) 🔍"]
    assert "provider response must not be published" not in "".join(bodies.values())


async def test_duplicate_and_unknown_models_are_isolated(monkeypatch, settings):
    handler = MagicMock()
    handler.chat_completion = AsyncMock(return_value=("review:\n  key_issues_to_review: []\n", "stop"))
    reviewer, published = make_reviewer(monkeypatch, lambda: handler)
    settings.set("pr_reviewer.review_models", ["provider/a", "provider/a", "unknown"])
    monkeypatch.setattr(
        "pr_agent.tools.pr_reviewer.get_max_tokens",
        lambda model: 10000 if model == "provider/a" else (_ for _ in ()).throw(ValueError("unknown")),
    )
    await reviewer._run_review_models(settings.pr_reviewer.review_models)

    assert handler.chat_completion.await_count == 1
    assert len(published) == 2
    assert any("model context window unavailable" in body for body, _kwargs in published)


def test_prompt_rendering_preserves_source_characters(settings):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.vars = {"title": "Check <tag> & condition"}
    reviewer.patches_diff = 'if x < 2 and y > 1: print("hello")'
    settings.set("pr_review_prompt.system", "{{ title }}")
    settings.set("pr_review_prompt.user", "{{ diff }}")
    assert reviewer._render_review_prompts() == (
        reviewer.vars["title"], reviewer.patches_diff,
    )
