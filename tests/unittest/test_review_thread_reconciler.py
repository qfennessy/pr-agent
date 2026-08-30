import pytest

from pr_agent.algo.inline_comment_dedup import (
    body_fingerprint,
    body_with_finding_identity_marker,
    build_summary_fallback_marker,
    finding_identity_markers,
    has_marker,
    marker_fingerprints,
)
from pr_agent.algo.review_thread_reconciler import (
    FIXED_THREAD_NOTICE,
    DesiredReviewThread,
    FindingIdentity,
    ReviewThreadActionKind,
    ReviewThreadActionOutcome,
    ReviewThreadActionState,
    ReviewThreadAnchor,
    ReviewThreadCommentSnapshot,
    ReviewThreadFailureKind,
    ReviewThreadReconciliationOutcome,
    ReviewThreadSnapshot,
    SummaryFallbackReason,
    execute_review_thread_action_plan,
    plan_review_thread_actions,
)


def _identity(root_cause_id="cause-1", path="src/app.py", symbol="run", trusted_stable_key=None):
    return FindingIdentity(
        repository="Owner/Repo",
        pull_request_number=7,
        root_cause_id=root_cause_id,
        path=path,
        symbol=symbol,
        trusted_stable_key=trusted_stable_key,
    )


def _comment(identity, body="old wording", author="pr-agent[bot]", database_id=10):
    is_bot = author.endswith("[bot]")
    return ReviewThreadCommentSnapshot(
        node_id=f"comment-{database_id}",
        database_id=database_id,
        author_id="BOT-1" if is_bot else "USER-1",
        author_login=author,
        author_type="Bot" if is_bot else "User",
        body=body_with_finding_identity_marker(body, identity.finding_id),
    )


def _snapshot(
    identity,
    *,
    line=10,
    body="old wording",
    resolved=False,
    outdated=False,
    bot_owned=True,
    replies=False,
    anchor=True,
    viewer_can_resolve=True,
):
    comments = [_comment(identity, body=body)]
    if replies:
        comments.append(
            ReviewThreadCommentSnapshot(
                node_id="reply-1",
                database_id=11,
                author_id="USER-1",
                author_login="human",
                author_type="User",
                body="Please keep this open.",
            )
        )
    thread_anchor = ReviewThreadAnchor(path=identity.path, line=line) if anchor else None
    return ReviewThreadSnapshot(
        thread_id="thread-1",
        finding_id=identity.finding_id,
        anchor=thread_anchor,
        original_anchor=ReviewThreadAnchor(path=identity.path, line=line),
        is_resolved=resolved,
        is_outdated=outdated,
        bot_owned=bot_owned,
        has_replies=replies,
        reviewed_head_sha="head-old",
        comments=tuple(comments),
        subject_type="LINE",
        viewer_can_resolve=viewer_can_resolve,
    )


class _MutationProvider:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.calls = []

    def _outcome(self, kind):
        if self.outcomes:
            return self.outcomes.pop(0)
        return ReviewThreadActionOutcome(
            kind=kind,
            state=ReviewThreadActionState.APPLIED,
            expected_head_sha="head-1",
            current_head_sha="head-1",
        )

    def create_review_thread(self, comment, expected_head_sha):
        self.calls.append(("create", comment, expected_head_sha))
        return self._outcome(ReviewThreadActionKind.CREATE)

    def update_review_thread(self, comment_id, body, expected_head_sha):
        self.calls.append(("update", comment_id, body, expected_head_sha))
        return self._outcome(ReviewThreadActionKind.UPDATE)

    def resolve_review_thread(self, thread_id, expected_head_sha):
        self.calls.append(("resolve", thread_id, expected_head_sha))
        return self._outcome(ReviewThreadActionKind.RESOLVE)


def test_finding_identity_is_stable_across_cosmetic_input_and_wording_changes():
    first = _identity()
    second = FindingIdentity(
        repository=" owner/repo/ ",
        pull_request_number=7,
        root_cause_id=" cause-1 ",
        path="/src/app.py",
        symbol="  run ",
    )

    assert first.finding_id == second.finding_id
    assert first.finding_id.startswith("sha256:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_cause_id", "cause-2"),
        ("path", "src/other.py"),
        ("symbol", "other"),
    ],
)
def test_finding_identity_changes_for_logical_scope_without_stable_key(field, value):
    kwargs = {"root_cause_id": "cause-1", "path": "src/app.py", "symbol": "run"}
    original = _identity(**kwargs)
    kwargs[field] = value

    assert _identity(**kwargs).finding_id != original.finding_id


def test_trusted_stable_key_preserves_identity_across_file_move():
    before = _identity(path="src/old.py", symbol="old_name", trusted_stable_key="symbol:core.run")
    after = _identity(path="src/new.py", symbol="new_name", trusted_stable_key=" symbol:core.run ")
    unrelated = _identity(path="src/new.py", symbol="new_name", trusted_stable_key="symbol:core.other")

    assert before.finding_id == after.finding_id
    assert unrelated.finding_id != before.finding_id


def test_anchor_canonicalizes_multiline_location_and_preserves_both_sides():
    anchor = ReviewThreadAnchor.from_github("`\\src\\app.py`", "12", "10", "right", "left")

    assert anchor == ReviewThreadAnchor("src/app.py", 12, 10, "RIGHT", "LEFT")
    assert anchor.to_github_comment("finding") == {
        "body": "finding",
        "path": "src/app.py",
        "line": 12,
        "side": "RIGHT",
        "start_line": 10,
        "start_side": "LEFT",
    }


def test_anchor_collapses_single_line_range_and_rejects_invalid_locations():
    assert ReviewThreadAnchor.from_github("src/app.py", 12, 12) == ReviewThreadAnchor("src/app.py", 12)
    assert ReviewThreadAnchor.from_github("src/app.py", 10, 12) is None
    assert ReviewThreadAnchor.from_github("src/app.py", None) is None
    assert ReviewThreadAnchor.from_github("", 10) is None


def test_versioned_finding_marker_keeps_legacy_markers_readable():
    identity = _identity()
    legacy = body_fingerprint("src/app.py", 10, "old wording")
    body = body_with_finding_identity_marker(f"old wording\n\n<!-- pr-agent-dedup: {legacy} -->", identity.finding_id)

    assert has_marker(body)
    assert marker_fingerprints(body) == {legacy}
    assert finding_identity_markers(body) == (("v1", identity.finding_id),)


def test_same_anchor_changed_wording_updates_root_comment():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "new wording")

    plan = plan_review_thread_actions((desired,), (_snapshot(identity),), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.UPDATE]
    assert plan.actions[0].root_comment_id == 10
    assert identity.finding_id in plan.actions[0].body


def test_visible_body_comparison_ignores_legacy_and_lifecycle_markers():
    identity = _identity()
    legacy = body_fingerprint(identity.path, 10, "old wording")
    existing = _snapshot(identity, body=f"old wording\n\n<!-- pr-agent-dedup: {legacy} -->")
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "old wording")

    plan = plan_review_thread_actions((desired,), (existing,), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.UNCHANGED]


@pytest.mark.parametrize(
    "existing",
    [
        lambda identity: _snapshot(identity, line=10),
        lambda identity: _snapshot(identity, line=20, outdated=True, anchor=False),
    ],
)
def test_moved_or_outdated_finding_creates_before_resolving_old_thread(existing):
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "old wording")

    plan = plan_review_thread_actions((desired,), (existing(identity),), "head-1")

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.CREATE,
        ReviewThreadActionKind.RESOLVE,
    ]
    assert plan.actions[1].depends_on_action_id == plan.actions[0].action_id


def test_invalid_desired_anchor_uses_summary_fallback_instead_of_inline_mutation():
    identity = _identity()
    desired = DesiredReviewThread(identity, None, "finding on deleted code")

    plan = plan_review_thread_actions((desired,), (_snapshot(identity, outdated=True, anchor=False),), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SUMMARY_FALLBACK]
    assert plan.actions[0].reason == "invalid_inline_location"


def test_moved_finding_is_not_duplicated_when_old_thread_cannot_be_resolved():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")

    plan = plan_review_thread_actions(
        (desired,),
        (_snapshot(identity, viewer_can_resolve=False),),
        "head-1",
    )

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SKIP]
    assert plan.actions[0].reason == "thread_cannot_be_resolved_safely"


def test_resolved_history_does_not_hide_one_active_thread():
    identity = _identity()
    resolved = _snapshot(identity, line=5, resolved=True)
    active = ReviewThreadSnapshot(
        thread_id="thread-2",
        finding_id=identity.finding_id,
        anchor=ReviewThreadAnchor(identity.path, 10),
        original_anchor=ReviewThreadAnchor(identity.path, 10),
        is_resolved=False,
        is_outdated=False,
        bot_owned=True,
        has_replies=False,
        reviewed_head_sha="head-1",
        comments=(_comment(identity, database_id=20),),
        subject_type="LINE",
        viewer_can_resolve=True,
    )
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "new wording")

    plan = plan_review_thread_actions((desired,), (resolved, active), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.UPDATE]
    assert plan.actions[0].thread_id == "thread-2"


@pytest.mark.parametrize(
    "snapshot",
    [
        lambda identity: _snapshot(identity, resolved=True),
        lambda identity: _snapshot(identity, bot_owned=False),
        lambda identity: _snapshot(identity, replies=True),
    ],
)
def test_human_controlled_or_resolved_thread_is_never_mutated(snapshot):
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "new wording")

    plan = plan_review_thread_actions((desired,), (snapshot(identity),), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SKIP]


def test_obsolete_mutation_requires_authoritative_absence():
    identity = _identity()
    existing = (_snapshot(identity),)

    kept = plan_review_thread_actions((), existing, "head-1")
    blocked = plan_review_thread_actions((), existing, "head-1", obsolete_policy="resolve")
    resolved = plan_review_thread_actions((), existing, "head-1", obsolete_policy="resolve", authoritative_absence=True)

    assert kept.actions[0].reason == "obsolete_thread_preserved"
    assert blocked.actions[0].reason == "absence_not_authoritative"
    assert resolved.actions[0].kind == ReviewThreadActionKind.RESOLVE


def test_mark_fixed_policy_is_visible_then_resolves_with_dependency():
    identity = _identity()

    plan = plan_review_thread_actions(
        (),
        (_snapshot(identity),),
        "head-1",
        obsolete_policy="mark_fixed",
        authoritative_absence=True,
    )

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.UPDATE,
        ReviewThreadActionKind.RESOLVE,
    ]
    assert FIXED_THREAD_NOTICE in plan.actions[0].body
    assert plan.actions[1].depends_on_action_id == plan.actions[0].action_id


def test_duplicate_desired_identity_is_rejected():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "wording")

    with pytest.raises(ValueError, match="unique identities"):
        plan_review_thread_actions((desired, desired), (), "head-1")


def test_executor_enforces_create_before_resolve_and_emits_deduplicated_fallback():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")
    plan = plan_review_thread_actions((desired,), (_snapshot(identity),), "head-1")
    failed = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.INVALID_INLINE_LOCATION,
        reason="create_failed: 422 invalid line",
    )
    provider = _MutationProvider([failed])

    outcome = execute_review_thread_action_plan(plan, provider)

    assert [result.state for result in outcome.action_outcomes] == [
        ReviewThreadActionState.FAILED,
        ReviewThreadActionState.NOT_EXECUTED,
    ]
    assert [call[0] for call in provider.calls] == ["create"]
    assert len(outcome.summary_fallbacks) == 1
    assert outcome.summary_fallbacks[0].reason == SummaryFallbackReason.INLINE_REJECTED

    existing_body = f"already reported\n\n{build_summary_fallback_marker(identity.finding_id)}"
    repeated = execute_review_thread_action_plan(
        plan,
        _MutationProvider([failed]),
        existing_summary_bodies=(existing_body,),
    )
    assert repeated.summary_fallbacks == ()


def test_executor_returns_invalid_location_fallback_contract_without_provider_call():
    identity = _identity()
    plan = plan_review_thread_actions((DesiredReviewThread(identity, None, "finding"),), (), "head-1")
    provider = _MutationProvider()

    outcome = execute_review_thread_action_plan(plan, provider)

    assert provider.calls == []
    assert outcome.action_outcomes[0].state == ReviewThreadActionState.FALLBACK_REQUIRED
    assert outcome.summary_fallbacks[0].reason == SummaryFallbackReason.INVALID_INLINE_LOCATION
    assert build_summary_fallback_marker(identity.finding_id) in outcome.summary_fallbacks[0].rendered_body


def test_executor_exposes_permission_failure_and_action_state_metrics():
    identity = _identity()
    plan = plan_review_thread_actions(
        (),
        (_snapshot(identity),),
        "head-1",
        obsolete_policy="resolve",
        authoritative_absence=True,
    )
    failed = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.RESOLVE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.PERMISSION_DENIED,
        reason="resolve_failed: forbidden",
    )

    outcome = execute_review_thread_action_plan(plan, _MutationProvider([failed]))

    assert outcome.complete is False
    assert outcome.summary_fallbacks[0].reason == SummaryFallbackReason.PERMISSION_DENIED
    assert outcome.metrics["actions"]["resolve"] == 1
    assert outcome.metrics["action_states"]["resolve.failed"] == 1
    assert outcome.metrics["states"]["failed"] == 1


def test_structured_outcome_reports_partial_failure():
    applied = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.APPLIED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
    )
    failed = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.RESOLVE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        reason="permission denied",
    )
    outcome = ReviewThreadReconciliationOutcome("head-1", "head-1", (applied, failed))

    assert outcome.complete is False
    assert outcome.counts["applied"] == 1
    assert outcome.counts["failed"] == 1
