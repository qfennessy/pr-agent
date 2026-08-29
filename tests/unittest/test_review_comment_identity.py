from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pr_agent.algo.utils import (PRReviewIdentity, add_pr_review_identity,
                                 comment_matches_identity,
                                 convert_to_markdown_v2,
                                 format_pr_review_header,
                                 get_pr_review_comment_identifiers)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.azuredevops_provider import AzureDevopsProvider
from pr_agent.git_providers.gitea_provider import GiteaProvider
from pr_agent.git_providers.git_provider import GitProvider
from pr_agent.git_providers.github_provider import GithubProvider
from tests.unittest._settings_helpers import (restore_settings,
                                              snapshot_settings)


def _review_data():
    return {"review": {"security_concerns": "No"}}


def test_custom_review_heading_changes_presentation_only():
    snapshot = snapshot_settings(["pr_reviewer.review_heading"])
    try:
        get_settings().set("pr_reviewer.review_heading", "Guideline Compliance Check")

        full = convert_to_markdown_v2(_review_data())
        incremental = convert_to_markdown_v2(
            _review_data(),
            incremental_review="Starting from commit abc123",
        )
    finally:
        restore_settings(snapshot)

    assert full.startswith("## Guideline Compliance Check 🔍\n\n")
    assert incremental.startswith("## Incremental Guideline Compliance Check 🔍\n\n")
    assert "<!-- pr-agent:review" not in full
    assert "<!-- pr-agent:review" not in incremental


@pytest.mark.parametrize(
    "invalid_heading",
    [None, "", "  ", "first\nsecond", "first\u2028second", 42],
)
def test_invalid_review_heading_falls_back_to_default(invalid_heading):
    snapshot = snapshot_settings(["pr_reviewer.review_heading"])
    try:
        get_settings().set("pr_reviewer.review_heading", invalid_heading)
        header = format_pr_review_header()
    finally:
        restore_settings(snapshot)

    assert header == "## PR Reviewer Guide 🔍"


def test_identity_is_inserted_after_visible_heading_and_is_idempotent():
    review = "## Team Review 🔍\n\n<table>review</table>"

    marked = add_pr_review_identity(review, PRReviewIdentity.REGULAR.value)

    assert marked.startswith(
        "## Team Review 🔍\n\n<!-- pr-agent:review:full -->\n\n"
    )
    assert add_pr_review_identity(marked, PRReviewIdentity.REGULAR.value) == marked


def test_hidden_identity_only_matches_as_a_bounded_standalone_line():
    marker = PRReviewIdentity.REGULAR.value
    valid = f"## Team Review 🔍\n\n{marker}\n\nbody"
    quoted = f"## Human comment\n\n> {marker}\n\nbody"
    late = "## Human comment\n\na\nb\nc\nd\n" + marker

    assert comment_matches_identity(valid, marker)
    assert not comment_matches_identity(quoted, marker)
    assert not comment_matches_identity(late, marker)


def test_full_and_incremental_review_identities_remain_distinct():
    full_identifiers = get_pr_review_comment_identifiers(full=True, incremental=False)
    incremental_identifiers = get_pr_review_comment_identifiers(full=False, incremental=True)
    incremental = add_pr_review_identity(
        "## Incremental Team Review 🔍\n\nbody",
        PRReviewIdentity.INCREMENTAL.value,
    )

    assert PRReviewIdentity.REGULAR.value in full_identifiers
    assert PRReviewIdentity.INCREMENTAL.value not in full_identifiers
    assert PRReviewIdentity.BUGS_ONLY.value not in full_identifiers
    assert PRReviewIdentity.INCREMENTAL.value in incremental_identifiers
    assert PRReviewIdentity.BUGS_ONLY.value in incremental_identifiers
    assert not any(comment_matches_identity(incremental, item) for item in full_identifiers)


@pytest.mark.parametrize(
    ("body", "full", "incremental"),
    [
        ("## PR Reviewer Guide 🔍\n\nlegacy", True, False),
        ("## Incremental PR Reviewer Guide 🔍\n\nlegacy", False, True),
        (
            "## Team Review 🔍\n\n<!-- pr-agent:review:full -->\n\nmarked",
            True,
            False,
        ),
        (
            "## Incremental Team Review 🔍\n\n<!-- pr-agent:review:incremental -->\n\nmarked",
            False,
            True,
        ),
        (
            "## Team Review 🔍\n\n<!-- pr-agent:review:bugs-only -->\n\nmarked",
            True,
            True,
        ),
    ],
)
def test_github_previous_review_accepts_markers_and_legacy_headers(body, full, incremental):
    provider = GithubProvider.__new__(GithubProvider)
    provider.pr = MagicMock()
    expected = SimpleNamespace(body=body)
    provider.pr.get_issue_comments.return_value = [expected]

    result = provider.get_previous_review(full=full, incremental=incremental)

    assert result is expected


def test_github_full_lookup_does_not_adopt_incremental_marker():
    provider = GithubProvider.__new__(GithubProvider)
    provider.pr = MagicMock()
    incremental = SimpleNamespace(
        body=(
            "## Incremental Team Review 🔍\n\n"
            "<!-- pr-agent:review:incremental -->\n\nbody"
        )
    )
    provider.pr.get_issue_comments.return_value = [incremental]

    assert provider.get_previous_review(full=True, incremental=False) is None


def test_clear_persistent_bugs_only_review_preserves_full_review():
    provider = MagicMock()
    full_review = SimpleNamespace(
        body="## Team Review 🔍\n\n<!-- pr-agent:review:full -->\n\nfull findings"
    )
    bugs_only_review = SimpleNamespace(
        body="## Team Review 🔍\n\n<!-- pr-agent:review:bugs-only -->\n\ndefect"
    )
    provider.get_issue_comments.return_value = [full_review, bugs_only_review]

    result = GitProvider.clear_persistent_review(
        provider,
        PRReviewIdentity.BUGS_ONLY.value,
        "bugs-only review",
    )

    assert result is True
    provider.remove_comment.assert_called_once_with(bugs_only_review)


def test_persistent_review_updates_dictionary_shaped_comment_by_identity():
    provider = MagicMock()
    existing = {
        "comment_id": 42,
        "comment": "## Team Review 🔍\n\n<!-- pr-agent:review:bugs-only -->\n\nold defect",
    }
    provider.get_issue_comments.return_value = [existing]
    provider.get_latest_commit_url.return_value = "https://example.test/commit/deadbeef"
    provider.get_comment_url.return_value = ""

    GitProvider.publish_persistent_comment_full(
        provider,
        "## Team Review 🔍\n\nnew defect",
        initial_header="## Team Review 🔍",
        final_update_message=False,
        identity_marker=PRReviewIdentity.BUGS_ONLY.value,
    )

    provider.edit_comment.assert_called_once()
    assert provider.edit_comment.call_args.args[0] is existing
    assert PRReviewIdentity.BUGS_ONLY.value in provider.edit_comment.call_args.args[1]
    provider.publish_comment.assert_not_called()


def test_github_check_run_receives_presentation_without_comment_identity():
    snapshot = snapshot_settings(["github.publish_as_check_run"])
    provider = GithubProvider.__new__(GithubProvider)
    provider._publish_check_run = MagicMock(return_value=True)
    provider.publish_persistent_comment_full = MagicMock()
    review = "## Team Review 🔍\n\nbody"
    try:
        get_settings().set("github.publish_as_check_run", True)

        provider.publish_persistent_comment(
            review,
            initial_header="## Team Review 🔍",
            identity_marker=PRReviewIdentity.REGULAR.value,
        )
    finally:
        restore_settings(snapshot)

    provider._publish_check_run.assert_called_once_with(review, "review")
    provider.publish_persistent_comment_full.assert_not_called()


def test_github_comment_path_forwards_review_identity():
    snapshot = snapshot_settings(["github.publish_as_check_run"])
    provider = GithubProvider.__new__(GithubProvider)
    provider.publish_persistent_comment_full = MagicMock()
    review = "## Team Review 🔍\n\nbody"
    legacy_header = "## PR Reviewer Guide 🔍"
    try:
        get_settings().set("github.publish_as_check_run", False)

        provider.publish_persistent_comment(
            review,
            initial_header="## Team Review 🔍",
            identity_marker=PRReviewIdentity.REGULAR.value,
            legacy_initial_header=legacy_header,
        )
    finally:
        restore_settings(snapshot)

    assert provider.publish_persistent_comment_full.call_args.kwargs == {
        "identity_marker": PRReviewIdentity.REGULAR.value,
        "legacy_initial_header": legacy_header,
    }


def test_azure_comment_path_forwards_review_identity():
    provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
    provider.publish_persistent_comment_full = MagicMock()
    review = "## Team Review 🔍\n\nbody"
    legacy_header = "## PR Reviewer Guide 🔍"

    provider.publish_persistent_comment(
        review,
        initial_header="## Team Review 🔍",
        identity_marker=PRReviewIdentity.REGULAR.value,
        legacy_initial_header=legacy_header,
    )

    assert provider.publish_persistent_comment_full.call_args.kwargs == {
        "identity_marker": PRReviewIdentity.REGULAR.value,
        "legacy_initial_header": legacy_header,
    }


def test_gitea_comment_path_forwards_review_identity():
    provider = GiteaProvider.__new__(GiteaProvider)
    provider.publish_persistent_comment_full = MagicMock()
    review = "## Team Review 🔍\n\nbody"

    provider.publish_persistent_comment(
        review,
        initial_header="## Team Review 🔍",
        identity_marker=PRReviewIdentity.REGULAR.value,
        legacy_initial_header="## PR Reviewer Guide 🔍",
    )

    assert provider.publish_persistent_comment_full.call_args.kwargs == {
        "identity_marker": PRReviewIdentity.REGULAR.value,
        "legacy_initial_header": "## PR Reviewer Guide 🔍",
    }
