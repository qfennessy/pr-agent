import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pr_agent.algo.inline_comment_dedup import (
    body_with_finding_identity_marker,
    build_summary_fallback_marker,
)
from pr_agent.algo.review_thread_reconciler import (
    ReviewThreadActionKind,
    ReviewThreadActionOutcome,
    ReviewThreadActionState,
    ReviewThreadAnchor,
    ReviewThreadCommentSnapshot,
    ReviewThreadFailureKind,
    ReviewThreadSnapshot,
    finding_identities_from_verified_findings,
)
from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
from pr_agent.tools.pr_reviewer import PRReviewer


class _Section(dict):
    __getattr__ = dict.__getitem__


class _Settings:
    def __init__(
        self,
        *,
        enabled=True,
        obsolete_policy="keep",
        provider="github",
        persistent_comment=True,
        num_max_findings=3,
    ):
        self.config = _Section(
            git_provider=provider,
            publish_output=True,
            output_relevant_configurations=False,
            output_run_details=False,
        )
        self.pr_reviewer = _Section(
            inline_key_issues=True,
            enable_review_coverage_footer=False,
            enable_help_text=False,
            persistent_comment=persistent_comment,
            num_max_findings=num_max_findings,
        )
        self.review_thread_lifecycle = _Section(
            enabled=enabled,
            obsolete_thread_policy=obsolete_policy,
        )

    def get(self, key, default=None):
        if key == "config":
            return self.config
        current = self
        for part in key.split("."):
            if isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
            else:
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)
        return current


def _finding(**overrides):
    finding = {
        "relevant_file": "src/app.py",
        "issue_header": "Bug",
        "issue_content": "The unchecked value can crash this request.",
        "start_line": 2,
        "end_line": 2,
        "side": "new",
        "root_cause_id": f"sha256:{'a' * 64}",
        "root_cause_id_schema": "verified-root-cause-v2",
        "trusted_stable_key": f"sha256:{'b' * 64}",
        "_trusted_anchor_shape_id": f"sha256:{'c' * 64}",
        "_trusted_anchor_shape_occurrence_count": 1,
        "_trusted_same_anchor_candidate_count": 1,
        "_trusted_patch_is_complete": True,
    }
    finding.update(overrides)
    return finding


def _identity(finding=None):
    return finding_identities_from_verified_findings(
        [finding or _finding()],
        repository="owner/repo",
        pull_request_number=7,
    )[0]


def _snapshot(
    finding=None,
    *,
    body=None,
    line=2,
    replies=(),
    resolved=False,
    viewer_can_resolve=True,
):
    finding = finding or _finding()
    identity = _identity(finding)
    root_body = body_with_finding_identity_marker(
        body or "**Bug**\n\nThe unchecked value can crash this request.",
        identity.finding_id,
    )
    root = ReviewThreadCommentSnapshot(
        node_id="comment-1",
        database_id=101,
        author_login="pr-agent[bot]",
        author_id="BOT-1",
        author_type="Bot",
        body=root_body,
    )
    return ReviewThreadSnapshot(
        thread_id="thread-1",
        finding_id=identity.finding_id,
        anchor=ReviewThreadAnchor("src/app.py", line),
        original_anchor=ReviewThreadAnchor("src/app.py", line),
        is_resolved=resolved,
        is_outdated=False,
        bot_owned=True,
        has_replies=bool(replies),
        reviewed_head_sha="head-1",
        comments=(root, *replies),
        viewer_can_resolve=viewer_can_resolve,
        resolved_by_viewer_bot=resolved,
    )


class _Provider:
    repo = "owner/repo"
    pr_num = 7

    def __init__(
        self,
        inventory=(),
        *,
        refresh_heads=("head-1", "head-1"),
        outcomes=(),
        summary_bodies=(),
        supports_lifecycle=True,
    ):
        self.inventory = tuple(inventory)
        self.refresh_heads = list(refresh_heads)
        self.outcomes = list(outcomes)
        self.summary_bodies = tuple(summary_bodies)
        self.supports_lifecycle = supports_lifecycle
        self.calls = []
        self.structured = None

    def get_diff_files(self):
        return [FilePatchInfo(
            "old\nvalue\nthird\nfourth\n",
            "new\nvalue\nthird\nfourth\n",
            "@@ -1 +1 @@",
            "src/app.py",
        )]

    def supports_review_thread_lifecycle(self):
        return self.supports_lifecycle

    def get_pr_head_sha(self, refresh=False):
        if not refresh:
            return "head-1"
        return self.refresh_heads.pop(0)

    def get_review_thread_snapshots(self, *, require_viewer_bot=False):
        self.calls.append(("inventory", require_viewer_bot))
        return self.inventory

    def get_bot_owned_review_summary_bodies(self):
        self.calls.append(("summary_inventory",))
        return self.summary_bodies

    def _outcome(self, kind, expected_head_sha, **kwargs):
        if self.outcomes:
            return self.outcomes.pop(0)
        return ReviewThreadActionOutcome(
            kind=kind,
            state=ReviewThreadActionState.APPLIED,
            expected_head_sha=expected_head_sha,
            current_head_sha=expected_head_sha,
            mutation_attempted=True,
            **kwargs,
        )

    def create_review_thread(self, comment, expected_head_sha, expected_threads=()):
        self.calls.append(("create", comment, expected_threads))
        return self._outcome(ReviewThreadActionKind.CREATE, expected_head_sha)

    def update_review_thread(
        self,
        comment_id,
        body,
        expected_head_sha,
        expected_thread,
        expected_finding_threads=None,
    ):
        self.calls.append(("update", comment_id, body, expected_thread, expected_finding_threads))
        return self._outcome(
            ReviewThreadActionKind.UPDATE,
            expected_head_sha,
            comment_id=comment_id,
        )

    def resolve_review_thread(self, thread_id, expected_head_sha, expected_thread):
        self.calls.append(("resolve", thread_id, expected_thread))
        return self._outcome(
            ReviewThreadActionKind.RESOLVE,
            expected_head_sha,
            thread_id=thread_id,
        )

    def publish_structured_review(self, data):
        self.structured = data


def _reviewer(provider, *, artifact=None, incremental=False, remaining_files=()):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.review_profile = "full"
    reviewer.incremental = SimpleNamespace(is_incremental=incremental)
    reviewer.remaining_files_list = list(remaining_files)
    reviewer.deleted_files_list = []
    reviewer.review_route_decision = None
    reviewer._review_shadow_only = False
    reviewer.candidate_verification_artifact = artifact or {
        "status": "complete",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": 1,
        "verified_count": 1,
        "finding_limit_dropped": 0,
    }
    reviewer.review_thread_reconciliation_artifact = None
    reviewer._review_thread_summary_fallbacks = ()
    reviewer._review_thread_lifecycle_notice = None
    reviewer._review_thread_lifecycle_blocks_summary = False
    return reviewer


def _apply(reviewer, findings, *, settings=None):
    settings = settings or _Settings()
    data = {"review": {"key_issues_to_review": list(findings)}}
    with patch("pr_agent.tools.pr_reviewer.get_settings", return_value=settings):
        return reviewer._apply_review_thread_lifecycle(data)


def _mutation_calls(provider):
    return [call for call in provider.calls if call[0] in {"create", "update", "resolve"}]


def test_review_thread_lifecycle_defaults_are_disabled_and_mirrored():
    repository = tomllib.loads(Path(".pr_agent.toml").read_text(encoding="utf-8"))["review_thread_lifecycle"]
    defaults = tomllib.loads(
        Path("pr_agent/settings/configuration.toml").read_text(encoding="utf-8")
    )["review_thread_lifecycle"]

    assert repository == defaults == {
        "enabled": False,
        "obsolete_thread_policy": "keep",
    }


def test_new_verified_finding_creates_one_thread_and_removes_it_from_the_summary_body():
    provider = _Provider()
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert _mutation_calls(provider)[0][0] == "create"
    assert "key_issues_to_review" not in result["review"]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "complete"
    assert reviewer.review_thread_reconciliation_artifact["results"]["created"] == 1
    assert provider.calls[0] == ("inventory", True)


def test_deleted_code_finding_creates_a_left_side_thread():
    provider = _Provider()
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding(side="old")])

    create = _mutation_calls(provider)[0]
    assert create[1]["side"] == "LEFT"
    assert create[1]["line"] == 2
    assert "key_issues_to_review" not in result["review"]


def test_renamed_file_old_side_finding_uses_old_path_for_lookup_and_new_path_for_thread():
    provider = _Provider()
    provider.get_diff_files = MagicMock(return_value=[FilePatchInfo(
        "old\nvalue\nthird\nfourth\n",
        "new\nvalue\nthird\nfourth\n",
        "@@ -1 +1 @@",
        "src/new-app.py",
        edit_type=EDIT_TYPE.RENAMED,
        old_filename="src/old-app.py",
    )])
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding(relevant_file="src/old-app.py", side="old")])

    create = _mutation_calls(provider)[0]
    assert create[1]["path"] == "src/new-app.py"
    assert create[1]["side"] == "LEFT"
    assert "key_issues_to_review" not in result["review"]


def test_identical_rerun_reuses_the_same_thread_without_a_mutation():
    provider = _Provider((_snapshot(),))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert _mutation_calls(provider) == []
    assert "key_issues_to_review" not in result["review"]
    assert reviewer.review_thread_reconciliation_artifact["results"]["unchanged"] == 1


def test_reworded_finding_updates_the_same_anchor():
    finding = _finding(issue_content="The request now fails for a missing value.")
    provider = _Provider((_snapshot(finding, body="**Bug**\n\nEarlier wording."),))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [finding])

    assert _mutation_calls(provider)[0][0] == "update"
    assert "key_issues_to_review" not in result["review"]
    assert reviewer.review_thread_reconciliation_artifact["results"]["updated"] == 1


def test_moved_finding_creates_the_replacement_before_resolving_the_old_thread():
    finding = _finding(start_line=3, end_line=3)
    provider = _Provider((_snapshot(finding, line=2),))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [finding])

    assert [call[0] for call in _mutation_calls(provider)] == ["create", "resolve"]
    assert "key_issues_to_review" not in result["review"]
    assert reviewer.review_thread_reconciliation_artifact["results"]["resolved"] == 1


def test_same_anchor_human_reply_is_preserved_and_finding_stays_in_summary():
    reply = ReviewThreadCommentSnapshot(
        node_id="reply-1",
        database_id=102,
        author_login="maintainer",
        author_type="User",
        body="I disagree.",
    )
    provider = _Provider((_snapshot(replies=(reply,)),))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert _mutation_calls(provider) == []
    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "protected_discussion"


def test_invalid_anchor_becomes_one_visible_deduplicated_summary_fallback():
    finding = _finding(relevant_file="missing.py")
    provider = _Provider()
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [finding])
    rendered = reviewer._append_review_thread_lifecycle_summary("review")

    assert _mutation_calls(provider) == []
    assert "key_issues_to_review" not in result["review"]
    assert rendered.count("PR-Agent inline fallback") == 1
    assert rendered.count(build_summary_fallback_marker(_identity(finding).finding_id)) == 1
    assert reviewer.review_thread_reconciliation_artifact["status"] == "complete_with_fallback"


def test_append_only_rerun_reuses_an_existing_bot_owned_summary_fallback():
    finding = _finding(relevant_file="missing.py")
    marker = build_summary_fallback_marker(_identity(finding).finding_id)
    provider = _Provider(summary_bodies=(f"Earlier fallback\n\n{marker}",))
    reviewer = _reviewer(provider)

    result = _apply(
        reviewer,
        [finding],
        settings=_Settings(persistent_comment=False),
    )

    assert "key_issues_to_review" not in result["review"]
    assert reviewer._review_thread_summary_fallbacks == ()
    assert reviewer.review_thread_reconciliation_artifact["status"] == "complete_with_fallback"
    assert reviewer.review_thread_reconciliation_artifact["summary_fallback_count"] == 0
    assert reviewer.review_thread_reconciliation_artifact["reused_summary_fallback_count"] == 1
    assert ("summary_inventory",) in provider.calls


def test_permission_failure_is_visible_and_counted_without_discarding_the_finding():
    failed = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.PERMISSION_DENIED,
        reason="create_failed: forbidden",
        mutation_attempted=True,
    )
    provider = _Provider(outcomes=(failed,))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])
    rendered = reviewer._append_review_thread_lifecycle_summary("")

    assert "key_issues_to_review" not in result["review"]
    assert "permission denied" in rendered
    assert reviewer.review_thread_reconciliation_artifact["failure_kinds"] == {"permission_denied": 1}
    assert reviewer.review_thread_reconciliation_artifact["results"]["failed"] == 1


def test_head_change_during_inventory_mutates_nothing_and_retains_the_finding():
    provider = _Provider(refresh_heads=("head-1", "head-2"))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert _mutation_calls(provider) == []
    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "stale_head"
    assert "changed while threads were inventoried" in reviewer._review_thread_lifecycle_notice


def test_head_change_before_inventory_mutates_nothing_and_retains_the_finding():
    provider = _Provider(refresh_heads=("head-2",))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert provider.calls == []
    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "stale_head"
    assert reviewer._review_thread_lifecycle_blocks_summary is True


def test_non_github_provider_keeps_the_verified_finding_in_the_summary():
    provider = _Provider(supports_lifecycle=False)
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()], settings=_Settings(provider="gitlab"))

    assert provider.calls == []
    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "unsupported_provider"


def test_missing_verification_artifact_never_mutates_threads():
    provider = _Provider()
    reviewer = _reviewer(provider)
    reviewer.candidate_verification_artifact = None

    result = _apply(reviewer, [_finding()])

    assert provider.calls == []
    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "configuration_invalid"


def test_malformed_identity_never_reads_or_mutates_threads():
    provider = _Provider()
    reviewer = _reviewer(provider)
    finding = _finding(root_cause_id="model-supplied-prose")

    result = _apply(reviewer, [finding])

    assert provider.calls == []
    assert result["review"]["key_issues_to_review"] == [finding]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "finding_identity_invalid"


def test_inventory_failure_is_visible_and_retains_all_findings():
    provider = _Provider()
    provider.get_review_thread_snapshots = MagicMock(side_effect=RuntimeError("forbidden"))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert _mutation_calls(provider) == []
    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer.review_thread_reconciliation_artifact["status"] == "inventory_unavailable"
    assert reviewer.review_thread_reconciliation_artifact["results"]["failed"] == 1
    assert set(reviewer.review_thread_reconciliation_artifact["metrics"]["action_states"]) == {
        f"{kind}.{state}"
        for kind in ("create", "update", "resolve", "unchanged", "skip", "summary_fallback")
        for state in (
            "applied",
            "already_applied",
            "stale_head",
            "stale_inventory",
            "failed",
            "not_executed",
            "skipped",
            "fallback_required",
            "applied_requires_refresh",
        )
    }
    assert "could not be read" in reviewer._review_thread_lifecycle_notice


def test_ambiguous_create_blocks_later_summary_publication_until_a_fresh_inventory():
    failed = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
        reason="response lost after send",
        mutation_attempted=True,
        mutation_result_ambiguous=True,
    )
    provider = _Provider(outcomes=(failed,))
    reviewer = _reviewer(provider)

    result = _apply(reviewer, [_finding()])

    assert result["review"]["key_issues_to_review"] == [_finding()]
    assert reviewer._review_thread_lifecycle_blocks_summary is True
    assert reviewer.review_thread_reconciliation_artifact["requires_fresh_inventory"] is True


def test_complete_full_clean_run_may_resolve_an_obsolete_bot_owned_thread():
    provider = _Provider((_snapshot(),))
    reviewer = _reviewer(provider, artifact={
        "status": "no_candidates",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": 0,
        "verified_count": 0,
        "finding_limit_dropped": 0,
    })

    _apply(reviewer, [], settings=_Settings(obsolete_policy="resolve"))

    assert [call[0] for call in _mutation_calls(provider)] == ["resolve"]
    assert reviewer.review_thread_reconciliation_artifact["authoritative_absence"] is True


def test_incremental_clean_run_cannot_resolve_an_obsolete_thread():
    provider = _Provider((_snapshot(),))
    reviewer = _reviewer(provider, incremental=True, artifact={
        "status": "no_candidates",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": 0,
        "verified_count": 0,
        "finding_limit_dropped": 0,
    })

    _apply(reviewer, [], settings=_Settings(obsolete_policy="resolve"))

    assert _mutation_calls(provider) == []
    assert reviewer.review_thread_reconciliation_artifact["authoritative_absence"] is False


def test_omitted_or_budgeted_findings_make_absence_non_authoritative():
    artifact = {
        "status": "complete",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": 1,
        "verified_count": 1,
        "finding_limit_dropped": 0,
    }
    reviewer = _reviewer(_Provider(), artifact=artifact, remaining_files=("omitted.py",))
    assert reviewer._review_thread_absence_is_authoritative(1) is False

    reviewer.remaining_files_list = []
    assert reviewer._review_thread_absence_is_authoritative(0) is False


@pytest.mark.parametrize(
    ("proposed_count", "generation_cap", "published_count", "expected"),
    [
        (2, 3, 2, True),
        (3, 3, 3, False),
        (4, 3, 4, False),
        (None, 3, 0, False),
        (True, 3, 0, False),
        (-1, 3, 0, False),
        (0, 0, 0, False),
        (0, True, 0, False),
    ],
)
def test_generation_cap_saturation_and_malformed_counts_make_absence_non_authoritative(
    proposed_count, generation_cap, published_count, expected
):
    reviewer = _reviewer(_Provider(), artifact={
        "status": "complete",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": proposed_count,
        "verified_count": published_count,
        "finding_limit_dropped": 0,
    })

    with patch(
        "pr_agent.tools.pr_reviewer.get_settings",
        return_value=_Settings(num_max_findings=generation_cap),
    ):
        assert reviewer._review_thread_absence_is_authoritative(published_count) is expected


@pytest.mark.parametrize("generation_complete", [False, None])
def test_incomplete_or_missing_first_pass_completion_makes_absence_non_authoritative(
    generation_complete,
):
    artifact = {
        "status": "complete",
        "publication_safe": True,
        "proposed_candidate_count": 1,
        "verified_count": 1,
        "finding_limit_dropped": 0,
    }
    if generation_complete is not None:
        artifact["first_pass_generation_complete"] = generation_complete
    reviewer = _reviewer(_Provider(), artifact=artifact)

    with patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings()):
        assert reviewer._review_thread_absence_is_authoritative(1) is False


def test_applied_route_generation_cap_controls_authoritative_absence():
    reviewer = _reviewer(_Provider(), artifact={
        "status": "complete",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": 2,
        "verified_count": 2,
        "finding_limit_dropped": 0,
    })
    reviewer._review_max_findings = 2

    with patch(
        "pr_agent.tools.pr_reviewer.get_settings",
        return_value=_Settings(num_max_findings=3),
    ):
        assert reviewer._review_thread_absence_is_authoritative(2) is False


def test_saturated_run_reconciles_current_finding_without_resolving_absent_thread():
    provider = _Provider((_snapshot(),))
    current_finding = _finding(
        start_line=3,
        end_line=3,
        root_cause_id=f"sha256:{'d' * 64}",
        trusted_stable_key=f"sha256:{'e' * 64}",
        _trusted_anchor_shape_id=f"sha256:{'f' * 64}",
    )
    reviewer = _reviewer(provider, artifact={
        "status": "complete",
        "publication_safe": True,
        "first_pass_generation_complete": True,
        "proposed_candidate_count": 1,
        "verified_count": 1,
        "finding_limit_dropped": 0,
    })

    _apply(
        reviewer,
        [current_finding],
        settings=_Settings(obsolete_policy="resolve", num_max_findings=1),
    )

    assert [call[0] for call in _mutation_calls(provider)] == ["create"]
    assert reviewer.review_thread_reconciliation_artifact["authoritative_absence"] is False


def test_structured_output_contains_bounded_reconciliation_metrics():
    provider = _Provider()
    reviewer = _reviewer(provider)
    reviewer._force_no_publish = False
    reviewer.frontier_adjudication_artifact = None
    reviewer.specialist_shadow_result = None

    with patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings()):
        reviewer._apply_review_thread_lifecycle({"review": {"key_issues_to_review": [_finding()]}})
        reviewer._publish_structured_review_data({"review": {"key_issues_to_review": [_finding()]}})

    lifecycle = provider.structured["review_thread_lifecycle"]
    assert lifecycle["status"] == "complete"
    assert lifecycle["results"] == {
        "created": 1,
        "updated": 0,
        "unchanged": 0,
        "resolved": 0,
        "failed": 0,
    }
    assert "create.applied" in lifecycle["metrics"]["action_states"]


def test_prepare_review_uses_lifecycle_instead_of_legacy_inline_publication():
    reviewer = _reviewer(_Provider())
    reviewer.verified_review_data = {"review": {"key_issues_to_review": [_finding()]}}
    reviewer.prediction = "review: {}"
    reviewer._candidate_verification_blocks_publication = MagicMock(return_value=False)
    reviewer._apply_review_thread_lifecycle = MagicMock(return_value={"review": {}})
    reviewer._publish_structured_review_data = MagicMock()
    reviewer._publish_key_issues_as_inline_comments = MagicMock()
    reviewer._provider_mutations_allowed = MagicMock(return_value=True)
    reviewer.set_review_labels = MagicMock()
    reviewer.git_provider.is_supported = MagicMock(return_value=False)
    action_output = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings(enabled=True)),
        patch("pr_agent.tools.pr_reviewer.github_action_output", action_output),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", return_value="review"),
    ):
        rendered = reviewer._prepare_pr_review()

    assert rendered == "review"
    reviewer._apply_review_thread_lifecycle.assert_called_once()
    reviewer._publish_key_issues_as_inline_comments.assert_not_called()
    action_output.assert_called_once_with(
        {"review": {"key_issues_to_review": [_finding()]}},
        "review",
    )
    assert reviewer._prepared_push_output_payload == {
        "key_issues_to_review": [_finding()],
    }


def test_threaded_bugs_only_finding_clears_stale_comment_without_rewriting_the_check():
    provider = _Provider()
    provider.clear_persistent_review = MagicMock()
    provider.clear_persistent_review_comment = MagicMock()
    reviewer = _reviewer(provider)
    reviewer.review_profile = "bugs_only"
    reviewer.verified_review_data = {"review": {"key_issues_to_review": [_finding()]}}
    reviewer.prediction = "review: {}"
    reviewer._normalize_bugs_only_review = MagicMock(side_effect=lambda data: data)
    reviewer._candidate_verification_blocks_publication = MagicMock(return_value=False)
    reviewer._publish_structured_review_data = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings(enabled=True)),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
    ):
        rendered = reviewer._prepare_pr_review()
        reviewer._clear_stale_persistent_bugs_only_review()

    assert rendered == ""
    assert reviewer._review_thread_lifecycle_threaded_findings is True
    provider.clear_persistent_review.assert_not_called()
    provider.clear_persistent_review_comment.assert_called_once_with(
        identity_marker="<!-- pr-agent:review:bugs-only -->",
        name="bugs-only review",
    )


def test_non_authoritative_clean_lifecycle_run_preserves_the_persistent_review():
    provider = _Provider()
    provider.clear_persistent_review = MagicMock()
    provider.clear_persistent_review_comment = MagicMock()
    reviewer = _reviewer(provider, artifact={
        "status": "no_candidates",
        "publication_safe": True,
        "first_pass_generation_complete": False,
        "proposed_candidate_count": 0,
        "verified_count": 0,
        "finding_limit_dropped": 0,
    })
    reviewer.review_profile = "bugs_only"
    reviewer.verified_review_data = {"review": {"key_issues_to_review": []}}
    reviewer.prediction = "review: {}"
    reviewer._publish_structured_review_data = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings(enabled=True)),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
    ):
        rendered = reviewer._prepare_pr_review()
        reviewer._clear_stale_persistent_bugs_only_review()

    assert rendered == ""
    assert reviewer.review_thread_reconciliation_artifact["authoritative_absence"] is False
    provider.clear_persistent_review.assert_not_called()
    provider.clear_persistent_review_comment.assert_not_called()


def test_disabled_setting_preserves_legacy_inline_path():
    reviewer = _reviewer(_Provider())
    reviewer.verified_review_data = {"review": {"key_issues_to_review": [_finding()]}}
    reviewer.prediction = "review: {}"
    reviewer._candidate_verification_blocks_publication = MagicMock(return_value=False)
    reviewer._apply_review_thread_lifecycle = MagicMock()
    reviewer._publish_structured_review_data = MagicMock()
    reviewer._publish_key_issues_as_inline_comments = MagicMock(return_value={"review": {}})
    reviewer._provider_mutations_allowed = MagicMock(return_value=True)
    reviewer.set_review_labels = MagicMock()
    reviewer.git_provider.is_supported = MagicMock(return_value=False)

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings(enabled=False)),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", return_value="review"),
    ):
        reviewer._prepare_pr_review()

    reviewer._apply_review_thread_lifecycle.assert_not_called()
    reviewer._publish_key_issues_as_inline_comments.assert_called_once()


def test_active_non_github_provider_preserves_legacy_inline_path_despite_github_setting():
    reviewer = _reviewer(_Provider(supports_lifecycle=False))
    reviewer.verified_review_data = {"review": {"key_issues_to_review": [_finding()]}}
    reviewer.prediction = "review: {}"
    reviewer._candidate_verification_blocks_publication = MagicMock(return_value=False)
    reviewer._apply_review_thread_lifecycle = MagicMock()
    reviewer._publish_structured_review_data = MagicMock()
    reviewer._publish_key_issues_as_inline_comments = MagicMock(return_value={"review": {}})
    reviewer._provider_mutations_allowed = MagicMock(return_value=True)
    reviewer.set_review_labels = MagicMock()
    reviewer.git_provider.is_supported = MagicMock(return_value=False)

    with (
        patch(
            "pr_agent.tools.pr_reviewer.get_settings",
            return_value=_Settings(enabled=True, provider="github"),
        ),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", return_value="review"),
    ):
        reviewer._prepare_pr_review()

    reviewer._apply_review_thread_lifecycle.assert_not_called()
    reviewer._publish_key_issues_as_inline_comments.assert_called_once()
