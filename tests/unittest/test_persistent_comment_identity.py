"""Several PR-Agent runs, one persistent comment each.

Persistent comments are found by their visible header, which is identical for every run.
Running two reviews on one PR (e.g. a different model per CI job) therefore made the
second run overwrite the first run's comment. config.persistent_comment_id gives each run
its own hidden marker, so each finds and updates only its own comment.
"""

import pytest

from pr_agent.config_loader import get_settings
from pr_agent.git_providers.git_provider import (PERSISTENT_COMMENT_ID_MARKER,
                                                 GitProvider,
                                                 attach_persistent_comment_id,
                                                 get_persistent_comment_id,
                                                 is_own_persistent_comment)

HEADER = "## PR Reviewer Guide 🔍"


class FakeComment:
    def __init__(self, body):
        self.body = body


class FakeProvider(GitProvider):
    """Minimal provider exercising only the persistent-comment path."""

    def __init__(self, existing=None):
        self.existing = [FakeComment(b) for b in (existing or [])]
        self.published = []
        self.edited = []

    # --- the surface publish_persistent_comment_full actually uses ---
    def get_issue_comments(self):
        return self.existing

    def get_latest_commit_url(self):
        return "https://example.test/commit/abc123"

    def get_comment_url(self, comment):
        return "https://example.test/comment/1"

    def edit_comment(self, comment, body):
        self.edited.append((comment, body))
        comment.body = body

    def publish_comment(self, pr_comment: str, is_temporary: bool = False):
        self.published.append(pr_comment)
        comment = FakeComment(pr_comment)
        self.existing.append(comment)
        return comment

    # --- abstract members we do not exercise ---
    def is_supported(self, capability: str) -> bool:
        return False

    def get_files(self):
        return []

    def get_diff_files(self):
        return []

    def publish_description(self, pr_title: str, pr_body: str):
        raise NotImplementedError

    def publish_inline_comment(self, body: str, relevant_file: str, relevant_line_in_file: str,
                               original_suggestion=None):
        raise NotImplementedError

    def publish_inline_comments(self, comments: list):
        raise NotImplementedError

    def remove_initial_comment(self):
        pass

    def remove_comment(self, comment):
        pass

    def get_languages(self):
        return {}

    def get_pr_branch(self):
        return "main"

    def get_user_id(self):
        return "tester"

    def get_pr_description_full(self):
        return ""

    def get_issue_url(self):
        return ""

    def get_repo_settings(self):
        return ""

    def add_eyes_reaction(self, issue_comment_id: int, disable_eyes: bool = False):
        return None

    def remove_reaction(self, issue_comment_id: int, reaction_id: int) -> bool:
        return True

    def get_commit_messages(self):
        return ""

    def get_pr_labels(self, update=False):
        return []

    def publish_labels(self, labels):
        pass

    def publish_code_suggestions(self, code_suggestions: list) -> bool:
        return True


@pytest.fixture
def comment_id():
    """Set config.persistent_comment_id for one test and restore it afterwards."""
    original = get_settings().config.get("persistent_comment_id", None)

    def _set(value):
        get_settings().set("config.persistent_comment_id", value)
        return value

    yield _set
    get_settings().set("config.persistent_comment_id", original if original is not None else "")


def test_no_id_configured_keeps_the_historical_behaviour(comment_id):
    comment_id("")
    body = f"{HEADER}\nreview text"

    assert attach_persistent_comment_id(body) == body
    assert is_own_persistent_comment(body, HEADER) is True
    assert get_persistent_comment_id() == ""


def test_marker_is_attached_once_and_names_the_answering_model(comment_id):
    comment_id("kimi-k3")
    get_settings().set("config.last_used_model", "openai/glm-5")

    marked = attach_persistent_comment_id(f"{HEADER}\nreview text")

    assert f"{PERSISTENT_COMMENT_ID_MARKER} kimi-k3 -->" in marked
    assert "answered by `openai/glm-5`" in marked  # honest attribution when a fallback answered
    assert attach_persistent_comment_id(marked) == marked  # idempotent across updates


def test_each_reviewer_updates_only_its_own_comment(comment_id):
    """The regression this exists for: two reviewers, two comments, no clobbering."""
    provider = FakeProvider()

    comment_id("deepseek")
    provider.publish_persistent_comment_full(f"{HEADER}\nfrom deepseek", initial_header=HEADER,
                                             final_update_message=False)
    comment_id("kimi")
    provider.publish_persistent_comment_full(f"{HEADER}\nfrom kimi", initial_header=HEADER,
                                             final_update_message=False)

    assert len(provider.existing) == 2
    assert provider.edited == []

    # Second push: each run updates its own comment rather than publishing another.
    comment_id("deepseek")
    provider.publish_persistent_comment_full(f"{HEADER}\nfrom deepseek v2", initial_header=HEADER,
                                             final_update_message=False)

    assert len(provider.existing) == 2
    assert len(provider.edited) == 1
    assert "from deepseek v2" in provider.existing[0].body
    assert "from kimi" in provider.existing[1].body


def test_unidentified_run_does_not_adopt_an_identified_comment(comment_id):
    """An un-ided run must not hijack a reviewer's comment - it publishes its own."""
    comment_id("kimi")
    marked = attach_persistent_comment_id(f"{HEADER}\nfrom kimi")

    comment_id("")

    assert is_own_persistent_comment(marked, HEADER) is False
    provider = FakeProvider(existing=[marked])
    provider.publish_persistent_comment_full(f"{HEADER}\nfrom the default run", initial_header=HEADER,
                                             final_update_message=False)
    assert len(provider.existing) == 2
    assert provider.edited == []


def test_identified_run_ignores_a_legacy_unmarked_comment(comment_id):
    comment_id("deepseek")

    assert is_own_persistent_comment(f"{HEADER}\nlegacy", HEADER) is False
