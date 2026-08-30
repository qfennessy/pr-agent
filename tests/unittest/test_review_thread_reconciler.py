import pytest

from pr_agent.algo.inline_comment_dedup import (
    body_fingerprint,
    body_with_finding_identity_marker,
    finding_identity_markers,
    has_marker,
    marker_fingerprints,
)
from pr_agent.algo.review_thread_reconciler import (
    DesiredReviewThread,
    FindingIdentity,
    ReviewThreadActionKind,
    ReviewThreadActionOutcome,
    ReviewThreadActionState,
    ReviewThreadAnchor,
    ReviewThreadCommentSnapshot,
    ReviewThreadReconciliationOutcome,
    ReviewThreadSnapshot,
    plan_review_thread_actions,
)


def _identity(root_cause_id="cause-1", path="src/app.py", symbol="run"):
    return FindingIdentity(
        repository="Owner/Repo",
        pull_request_number=7,
        root_cause_id=root_cause_id,
        path=path,
        symbol=symbol,
    )


def _comment(identity, body="old wording", author="pr-agent[bot]", database_id=10):
    return ReviewThreadCommentSnapshot(
        node_id=f"comment-{database_id}",
        database_id=database_id,
        author_login=author,
        body=body_with_finding_identity_marker(body, identity.finding_id),
    )


def _snapshot(identity, *, line=10, body="old wording", resolved=False, bot_owned=True, replies=False):
    comments = [_comment(identity, body=body)]
    if replies:
        comments.append(ReviewThreadCommentSnapshot(
            node_id="reply-1", database_id=11, author_login="human", body="Please keep this open."
        ))
    return ReviewThreadSnapshot(
        thread_id="thread-1",
        finding_id=identity.finding_id,
        anchor=ReviewThreadAnchor(path=identity.path, line=line),
        is_resolved=resolved,
        is_outdated=False,
        bot_owned=bot_owned,
        has_replies=replies,
        reviewed_head_sha="head-old",
        comments=tuple(comments),
    )


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


@pytest.mark.parametrize("field,value", [
    ("root_cause_id", "cause-2"),
    ("path", "src/other.py"),
    ("symbol", "other"),
])
def test_finding_identity_changes_for_logical_scope(field, value):
    kwargs = {"root_cause_id": "cause-1", "path": "src/app.py", "symbol": "run"}
    original = _identity(**kwargs)
    kwargs[field] = value

    assert _identity(**kwargs).finding_id != original.finding_id


def test_versioned_finding_marker_keeps_legacy_markers_readable():
    identity = _identity()
    legacy = body_fingerprint("src/app.py", 10, "old wording")
    body = body_with_finding_identity_marker(
        f"old wording\n\n<!-- pr-agent-dedup: {legacy} -->", identity.finding_id
    )

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


def test_same_finding_at_new_anchor_creates_before_resolving_old_thread():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "old wording")

    plan = plan_review_thread_actions((desired,), (_snapshot(identity),), "head-1")

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.CREATE,
        ReviewThreadActionKind.RESOLVE,
    ]
    assert plan.actions[1].depends_on_action_id == plan.actions[0].action_id


def test_resolved_history_does_not_hide_one_active_thread():
    identity = _identity()
    resolved = _snapshot(identity, line=5, resolved=True)
    active = ReviewThreadSnapshot(
        thread_id="thread-2",
        finding_id=identity.finding_id,
        anchor=ReviewThreadAnchor(identity.path, 10),
        is_resolved=False,
        is_outdated=False,
        bot_owned=True,
        has_replies=False,
        reviewed_head_sha="head-1",
        comments=(_comment(identity, database_id=20),),
    )
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "new wording")

    plan = plan_review_thread_actions((desired,), (resolved, active), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.UPDATE]
    assert plan.actions[0].thread_id == "thread-2"


@pytest.mark.parametrize("snapshot", [
    lambda identity: _snapshot(identity, resolved=True),
    lambda identity: _snapshot(identity, bot_owned=False),
    lambda identity: _snapshot(identity, replies=True),
])
def test_human_controlled_or_resolved_thread_is_never_mutated(snapshot):
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "new wording")

    plan = plan_review_thread_actions((desired,), (snapshot(identity),), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SKIP]


def test_obsolete_policy_is_explicit_and_defaults_to_keep():
    identity = _identity()
    existing = (_snapshot(identity),)

    kept = plan_review_thread_actions((), existing, "head-1")
    resolved = plan_review_thread_actions((), existing, "head-1", obsolete_policy="resolve")

    assert kept.actions[0].kind == ReviewThreadActionKind.SKIP
    assert resolved.actions[0].kind == ReviewThreadActionKind.RESOLVE


def test_duplicate_desired_identity_is_rejected():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "wording")

    with pytest.raises(ValueError, match="unique identities"):
        plan_review_thread_actions((desired, desired), (), "head-1")


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
