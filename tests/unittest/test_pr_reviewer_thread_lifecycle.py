from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from pr_agent.algo.types import FilePatchInfo
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
    ):
        self.inventory = tuple(inventory)
        self.refresh_heads = list(refresh_heads)
        self.outcomes = list(outcomes)
        self.summary_bodies = tuple(summary_bodies)
        self.calls = []
        self.structured = None

    def get_diff_files(self):
        return [FilePatchInfo(
            "old\nvalue\nthird\nfourth\n",
            "new\nvalue\nthird\nfourth\n",
            "@@ -1 +1 @@",
            "src/app.py",
        )]

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
    provider = _Provider()
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
        "verified_count": 1,
        "finding_limit_dropped": 0,
    }
    reviewer = _reviewer(_Provider(), artifact=artifact, remaining_files=("omitted.py",))
    assert reviewer._review_thread_absence_is_authoritative(1) is False

    reviewer.remaining_files_list = []
    assert reviewer._review_thread_absence_is_authoritative(0) is False


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

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_Settings(enabled=True)),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", return_value="review"),
    ):
        rendered = reviewer._prepare_pr_review()

    assert rendered == "review"
    reviewer._apply_review_thread_lifecycle.assert_called_once()
    reviewer._publish_key_issues_as_inline_comments.assert_not_called()


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
