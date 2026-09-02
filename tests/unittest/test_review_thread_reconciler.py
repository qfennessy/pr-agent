import hashlib

import pytest

from pr_agent.algo.candidate_verification import _changed_anchor_identity_details, apply_verification_decisions
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
    FIXED_THREAD_STATE_MARKER,
    VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION,
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
    finding_identities_from_verified_findings,
    plan_review_thread_actions,
)


def _identity(root_cause_id="cause-1", path="src/app.py", symbol="run", trusted_stable_key=None):
    root_cause_id = root_cause_id.strip()
    if not root_cause_id.startswith("sha256:"):
        root_cause_id = f"sha256:{hashlib.sha256(root_cause_id.encode()).hexdigest()}"
    if trusted_stable_key:
        trusted_stable_key = trusted_stable_key.strip()
        if not trusted_stable_key.startswith("sha256:"):
            trusted_stable_key = f"sha256:{hashlib.sha256(trusted_stable_key.encode()).hexdigest()}"
    return FindingIdentity(
        repository="Owner/Repo",
        pull_request_number=7,
        root_cause_id=root_cause_id,
        path=path,
        symbol=symbol,
        trusted_stable_key=trusted_stable_key,
        root_cause_id_schema=VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION,
    )


def _comment(
    identity,
    body="old wording",
    author="pr-agent[bot]",
    database_id=10,
    created_at="2026-08-30T12:00:00Z",
):
    is_bot = author.endswith("[bot]")
    return ReviewThreadCommentSnapshot(
        node_id=f"comment-{database_id}",
        database_id=database_id,
        author_id="BOT-1" if is_bot else "USER-1",
        author_login=author,
        author_type="Bot" if is_bot else "User",
        body=body_with_finding_identity_marker(body, identity.finding_id),
        created_at=created_at,
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
    thread_id="thread-1",
    database_id=10,
    reviewed_head_sha="head-old",
    created_at="2026-08-30T12:00:00Z",
    resolved_by_viewer_bot=False,
    resolved_by_other_actor=False,
):
    comments = [_comment(identity, body=body, database_id=database_id, created_at=created_at)]
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
        thread_id=thread_id,
        finding_id=identity.finding_id,
        anchor=thread_anchor,
        original_anchor=ReviewThreadAnchor(path=identity.path, line=line),
        is_resolved=resolved,
        is_outdated=outdated,
        bot_owned=bot_owned,
        has_replies=replies,
        reviewed_head_sha=reviewed_head_sha,
        comments=tuple(comments),
        subject_type="LINE",
        viewer_can_resolve=viewer_can_resolve,
        resolved_by_viewer_bot=resolved_by_viewer_bot,
        resolved_by_other_actor=resolved_by_other_actor,
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

    def create_review_thread(self, comment, expected_head_sha, expected_threads=()):
        self.calls.append(("create", comment, expected_head_sha, expected_threads))
        return self._outcome(ReviewThreadActionKind.CREATE)

    def update_review_thread(
        self,
        comment_id,
        body,
        expected_head_sha,
        expected_thread,
        expected_finding_threads=None,
    ):
        self.calls.append(
            ("update", comment_id, body, expected_head_sha, expected_thread, expected_finding_threads)
        )
        return self._outcome(ReviewThreadActionKind.UPDATE)

    def resolve_review_thread(self, thread_id, expected_head_sha, expected_thread):
        self.calls.append(("resolve", thread_id, expected_head_sha, expected_thread))
        return self._outcome(ReviewThreadActionKind.RESOLVE)


def test_finding_identity_is_stable_across_cosmetic_input_and_wording_changes():
    first = _identity()
    second = FindingIdentity(
        repository=" owner/repo/ ",
        pull_request_number=7,
        root_cause_id=f" sha256:{hashlib.sha256(b'cause-1').hexdigest()} ",
        path="/src/app.py",
        symbol="  run ",
        root_cause_id_schema=VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION,
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


def test_verified_finding_identity_consumes_exact_upstream_hashes_without_deriving_a_substitute():
    root_cause_id = f"sha256:{'a' * 64}"
    trusted_stable_key = f"sha256:{'b' * 64}"

    identity = finding_identities_from_verified_findings([{
        "root_cause_id": root_cause_id,
        "relevant_file": "src/app.py",
        "trusted_stable_key": trusted_stable_key,
        "issue_content": "Mutable wording is not identity input.",
        "_trusted_anchor_shape_id": f"sha256:{'c' * 64}",
        "_trusted_anchor_shape_occurrence_count": 1,
        "_trusted_same_anchor_candidate_count": 1,
        "_trusted_patch_is_complete": True,
        "side": "new",
        "start_line": 12,
    }], repository="owner/repo", pull_request_number=7)[0]

    assert identity.root_cause_id == root_cause_id
    assert identity.trusted_stable_key == trusted_stable_key
    assert identity.root_cause_id_schema == VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION


@pytest.mark.parametrize(
    "replacement,error",
    [
        ({"root_cause_id": "model prose"}, "sha256 identity"),
        ({"root_cause_id": None}, "root_cause_id"),
        ({"trusted_stable_key": "candidate-1"}, "trusted_stable_key"),
        ({"trusted_stable_key": None}, "trusted_stable_key"),
        ({"relevant_file": None}, "publication anchor"),
        ({"_trusted_anchor_shape_id": None}, "trusted anchor shape id"),
        ({"_trusted_anchor_shape_occurrence_count": None}, "occurrence count"),
        ({"_trusted_anchor_shape_occurrence_count": True}, "occurrence count"),
        ({"_trusted_same_anchor_candidate_count": None}, "same-anchor candidate count"),
        ({"_trusted_same_anchor_candidate_count": True}, "same-anchor candidate count"),
        ({"_trusted_patch_is_complete": False}, "complete patch"),
    ],
)
def test_verified_finding_identity_fails_closed_on_untrusted_or_malformed_identity(replacement, error):
    finding = {
        "root_cause_id": f"sha256:{'a' * 64}",
        "relevant_file": "src/app.py",
        "trusted_stable_key": f"sha256:{'b' * 64}",
        "_trusted_anchor_shape_id": f"sha256:{'c' * 64}",
        "_trusted_anchor_shape_occurrence_count": 1,
        "_trusted_same_anchor_candidate_count": 1,
        "_trusted_patch_is_complete": True,
        "side": "new",
        "start_line": 12,
    }
    finding.update(replacement)

    with pytest.raises(ValueError, match=error):
        finding_identities_from_verified_findings(
            [finding], repository="owner/repo", pull_request_number=7
        )


def test_verified_finding_identity_accepts_actual_verification_output():
    candidate = {
        "candidate_id": "candidate-1",
        "relevant_file": "src/service.py",
        "issue_header": "Potential bug",
        "issue_content": "Original candidate wording.",
        "start_line": 12,
        "end_line": 12,
        "side": "new",
        "trigger": "The changed branch receives an empty value.",
        "impact": "The request fails instead of using the fallback.",
        "_changed_line_ranges": [(12, 12)],
        "_changed_anchor_shape": "return <id> ( <id> )",
        "_changed_anchor_ordinal": 1,
        "_changed_anchor_occurrence_count": 1,
        "_trusted_defect_ordinal": 1,
        "_trusted_same_anchor_candidate_count": 1,
        "_trusted_patch_is_complete": True,
        "_trusted_lineage_key": "file:src/service.py",
        "_trusted_side_line_count": 20,
    }
    evidence = [{
        "candidate_id": "candidate-1",
        "source": "changed_head",
        "path": "src/service.py",
        "content": "return fallback(value)",
        "start_line": 12,
        "end_line": 12,
        "side": "new",
    }]
    verification = {"verification": {"decisions": [{
        "candidate_id": "candidate-1",
        "verdict": "verified",
        "root_cause_id": "model-invented-id",
        "trusted_stable_key": "model-invented-key",
        "issue_content": "Verified wording can change.",
        "trigger": "An empty value reaches this branch.",
        "impact": "The request fails.",
        "relevant_file": "src/service.py",
        "start_line": 12,
        "end_line": 12,
        "evidence_paths": ["src/service.py"],
    }]}}

    findings, decisions = apply_verification_decisions([candidate], evidence, verification)
    identity = finding_identities_from_verified_findings(
        tuple(findings), repository="owner/repo", pull_request_number=7
    )[0]

    assert decisions[0]["verdict"] == "verified"
    assert findings[0]["root_cause_id"].startswith("sha256:")
    assert findings[0]["trusted_stable_key"].startswith("sha256:")
    assert "model-invented" not in identity.root_cause_id
    assert identity.trusted_stable_key == findings[0]["trusted_stable_key"]
    assert identity.path == findings[0]["relevant_file"]
    assert identity.root_cause_id_schema == "verified-root-cause-v2"

    incomplete_candidate = {**candidate, "_trusted_patch_is_complete": False}
    incomplete_findings, _ = apply_verification_decisions(
        [incomplete_candidate], evidence, verification
    )
    assert incomplete_findings[0]["_trusted_patch_is_complete"] is False
    with pytest.raises(ValueError, match="complete patch"):
        finding_identities_from_verified_findings(
            incomplete_findings, repository="owner/repo", pull_request_number=7
        )


def test_same_anchor_v2_identity_reordering_fails_closed_without_swapping_thread_mapping():
    def verified_findings(labels, verified_labels=None):
        verified_labels = set(labels if verified_labels is None else verified_labels)
        candidates = []
        evidence = []
        decisions = []
        for ordinal, label in enumerate(labels, start=1):
            candidate_id = f"candidate-{ordinal}"
            candidates.append({
                "candidate_id": candidate_id,
                "relevant_file": "src/service.py",
                "issue_header": label,
                "issue_content": f"{label} candidate",
                "start_line": 12,
                "end_line": 12,
                "side": "new",
                "trigger": f"{label} trigger",
                "impact": f"{label} impact",
                "_changed_line_ranges": [(12, 12)],
                "_changed_anchor_shape": "return <id> ( <id> )",
                "_changed_anchor_ordinal": 1,
                "_changed_anchor_occurrence_count": 1,
                "_trusted_defect_ordinal": ordinal,
                "_trusted_same_anchor_candidate_count": len(labels),
                "_trusted_patch_is_complete": True,
                "_trusted_lineage_key": "file:src/service.py",
                "_trusted_side_line_count": 20,
            })
            evidence.append({
                "candidate_id": candidate_id,
                "source": "changed_head",
                "path": "src/service.py",
                "content": "return fallback(value)",
                "start_line": 12,
                "end_line": 12,
                "side": "new",
            })
            decisions.append({
                "candidate_id": candidate_id,
                "verdict": "verified" if label in verified_labels else "rejected",
                "issue_header": label,
                "issue_content": f"Verified {label}",
                "trigger": f"{label} trigger",
                "impact": f"{label} impact",
                "relevant_file": "src/service.py",
                "start_line": 12,
                "end_line": 12,
                "evidence_paths": ["src/service.py"],
            })
        return apply_verification_decisions(
            candidates,
            evidence,
            {"verification": {"decisions": decisions}},
        )[0]

    forward = verified_findings(("authentication bypass", "unbounded retry"))
    reversed_order = verified_findings(("unbounded retry", "authentication bypass"))
    single_survivor = verified_findings(
        ("authentication bypass", "unbounded retry"),
        verified_labels={"unbounded retry"},
    )
    forward_ids = {finding["issue_header"]: finding["root_cause_id"] for finding in forward}
    reversed_ids = {finding["issue_header"]: finding["root_cause_id"] for finding in reversed_order}
    persisted_mapping = dict(forward_ids)

    assert forward_ids["authentication bypass"] == reversed_ids["unbounded retry"]
    assert len(single_survivor) == 1
    assert single_survivor[0]["_trusted_same_anchor_candidate_count"] == 2
    for findings in (forward, reversed_order, single_survivor):
        with pytest.raises(ValueError, match="ambiguous same-anchor candidate set"):
            finding_identities_from_verified_findings(
                tuple(findings), repository="owner/repo", pull_request_number=7
            )
    assert persisted_mapping == forward_ids


def test_equal_shape_v2_occurrence_shift_after_deletion_fails_closed_before_mapping():
    def verified_findings(specs):
        candidates = []
        evidence = []
        decisions = []
        for label, line, anchor_ordinal, anchor_shape in specs:
            candidate_id = f"candidate-{label}"
            candidates.append({
                "candidate_id": candidate_id,
                "relevant_file": "src/service.py",
                "issue_header": label,
                "issue_content": f"{label} candidate",
                "start_line": line,
                "end_line": line,
                "side": "new",
                "trigger": f"{label} trigger",
                "impact": f"{label} impact",
                "_changed_line_ranges": [(line, line)],
                "_changed_anchor_shape": anchor_shape,
                "_changed_anchor_ordinal": anchor_ordinal,
                "_changed_anchor_occurrence_count": len(specs),
                "_trusted_defect_ordinal": 1,
                "_trusted_same_anchor_candidate_count": 1,
                "_trusted_patch_is_complete": True,
                "_trusted_lineage_key": "file:src/service.py",
                "_trusted_side_line_count": 30,
            })
            evidence.append({
                "candidate_id": candidate_id,
                "source": "changed_head",
                "path": "src/service.py",
                "content": f"return {label}(value)",
                "start_line": line,
                "end_line": line,
                "side": "new",
            })
            decisions.append({
                "candidate_id": candidate_id,
                "verdict": "verified",
                "issue_header": label,
                "issue_content": f"Verified {label}",
                "trigger": f"{label} trigger",
                "impact": f"{label} impact",
                "relevant_file": "src/service.py",
                "start_line": line,
                "end_line": line,
                "evidence_paths": ["src/service.py"],
            })
        return apply_verification_decisions(
            candidates,
            evidence,
            {"verification": {"decisions": decisions}},
        )[0]

    repeated_shape = "return <id> ( <id> )"
    before = verified_findings((
        ("first", 12, 1, repeated_shape),
        ("second", 20, 2, repeated_shape),
    ))
    after_deletion = verified_findings((("second", 12, 1, repeated_shape),))
    before_by_header = {finding["issue_header"]: finding for finding in before}

    assert before_by_header["first"]["root_cause_id"] == after_deletion[0]["root_cause_id"]
    assert before_by_header["second"]["root_cause_id"] != after_deletion[0]["root_cause_id"]
    assert len({finding["_trusted_anchor_shape_id"] for finding in before}) == 1
    with pytest.raises(ValueError, match="anchor shape is not unique in the patch"):
        finding_identities_from_verified_findings(
            tuple(before), repository="owner/repo", pull_request_number=7
        )


def test_distinct_trusted_anchor_shapes_in_one_file_remain_eligible():
    findings = []
    for index, shape_id in enumerate(("a", "b"), start=1):
        findings.append({
            "root_cause_id": f"sha256:{str(index) * 64}",
            "relevant_file": "src/service.py",
            "trusted_stable_key": f"sha256:{str(index + 2) * 64}",
            "_trusted_anchor_shape_id": f"sha256:{shape_id * 64}",
            "_trusted_anchor_shape_occurrence_count": 1,
            "_trusted_same_anchor_candidate_count": 1,
            "_trusted_patch_is_complete": True,
            "side": "new",
            "start_line": index * 10,
        })

    identities = finding_identities_from_verified_findings(
        findings, repository="owner/repo", pull_request_number=7
    )

    assert len(identities) == 2


def test_single_verified_finding_with_a_patch_repeated_anchor_shape_fails_closed():
    patch_text = (
        "@@ -0,0 +10,3 @@\n"
        "+return first(value)\n"
        "+record(event)\n"
        "+return second(value)"
    )
    anchor_shape, anchor_ordinal, anchor_occurrence_count = (
        _changed_anchor_identity_details(patch_text, 12, 12)
    )
    candidate = {
        "candidate_id": "candidate-1",
        "relevant_file": "src/service.py",
        "issue_header": "Incorrect fallback",
        "issue_content": "The second return skips the required fallback.",
        "start_line": 12,
        "end_line": 12,
        "side": "new",
        "trigger": "The primary lookup returns no result.",
        "impact": "The request returns an invalid response.",
        "_changed_line_ranges": [(10, 12)],
        "_changed_anchor_shape": anchor_shape,
        "_changed_anchor_ordinal": anchor_ordinal,
        "_changed_anchor_occurrence_count": anchor_occurrence_count,
        "_trusted_defect_ordinal": 1,
        "_trusted_same_anchor_candidate_count": 1,
        "_trusted_patch_is_complete": True,
        "_trusted_lineage_key": "file:src/service.py",
        "_trusted_side_line_count": 20,
    }
    evidence = [{
        "candidate_id": "candidate-1",
        "source": "changed_patch",
        "path": "src/service.py",
        "content": "return second(value)",
        "start_line": 12,
        "end_line": 12,
        "side": "new",
    }]
    verification = {"verification": {"decisions": [{
        "candidate_id": "candidate-1",
        "verdict": "verified",
        "evidence_paths": ["src/service.py"],
    }]}}

    findings, decisions = apply_verification_decisions([candidate], evidence, verification)

    assert decisions[0]["verdict"] == "verified"
    assert findings[0]["_trusted_anchor_shape_occurrence_count"] == 2
    with pytest.raises(ValueError, match="anchor shape is not unique in the patch"):
        finding_identities_from_verified_findings(
            findings, repository="owner/repo", pull_request_number=7
        )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"root_cause_id": "model prose"}, "sha256 identity"),
        ({"root_cause_id_schema": None}, "root_cause_id_schema"),
        ({"root_cause_id_schema": "verified-root-cause-v1"}, "root_cause_id_schema"),
        ({"trusted_stable_key": "candidate-1"}, "trusted_stable_key"),
        ({"schema_version": "untrusted-lifecycle-v2"}, "schema_version"),
        ({"pull_request_number": True}, "pull_request_number"),
        ({"pull_request_number": 1.5}, "pull_request_number"),
    ],
)
def test_direct_finding_identity_constructor_fails_closed_on_unverified_values(overrides, error):
    kwargs = {
        "repository": "owner/repo",
        "pull_request_number": 7,
        "root_cause_id": f"sha256:{'a' * 64}",
        "path": "src/app.py",
        "root_cause_id_schema": VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=error):
        FindingIdentity(**kwargs)


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
    existing = _snapshot(identity)
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 10), "new wording")

    plan = plan_review_thread_actions((desired,), (existing,), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.UPDATE]
    assert plan.actions[0].root_comment_id == 10
    assert identity.finding_id in plan.actions[0].body
    assert plan.actions[0].expected_threads == (existing,)


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


@pytest.mark.parametrize("reverse_inventory", [False, True])
def test_partial_move_recovery_keeps_canonical_replacement_and_resolves_old_thread(reverse_inventory):
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")
    old = _snapshot(identity, thread_id="thread-old", line=10, outdated=True, anchor=False)
    replacement = _snapshot(
        identity,
        thread_id="thread-replacement",
        database_id=20,
        line=20,
        body="finding",
        reviewed_head_sha="head-1",
        created_at="2026-08-30T12:01:00Z",
    )
    inventory = (old, replacement)
    if reverse_inventory:
        inventory = tuple(reversed(inventory))

    plan = plan_review_thread_actions((desired,), inventory, "head-1")

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.UNCHANGED,
        ReviewThreadActionKind.RESOLVE,
    ]
    assert plan.actions[0].thread_id == "thread-replacement"
    assert plan.actions[1].thread_id == "thread-old"
    assert plan.actions[1].depends_on_action_id == plan.actions[0].action_id

    repaired = plan_review_thread_actions(
        (desired,),
        (replacement, _snapshot(identity, thread_id="thread-old", resolved=True, outdated=True, anchor=False)),
        "head-1",
    )
    assert [action.kind for action in repaired.actions] == [ReviewThreadActionKind.UNCHANGED]


def test_create_succeeds_resolve_goes_stale_then_next_plan_recovers_partial_move():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")
    old = _snapshot(identity, thread_id="thread-old", line=10)
    first_plan = plan_review_thread_actions((desired,), (old,), "head-1")
    created = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.APPLIED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        comment_id=20,
    )
    stale_resolve = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.RESOLVE,
        state=ReviewThreadActionState.STALE_HEAD,
        expected_head_sha="head-1",
        current_head_sha="head-2",
        thread_id="thread-old",
    )

    first_outcome = execute_review_thread_action_plan(first_plan, _MutationProvider([created, stale_resolve]))

    assert [item.state for item in first_outcome.action_outcomes] == [
        ReviewThreadActionState.APPLIED,
        ReviewThreadActionState.STALE_HEAD,
    ]
    assert first_outcome.requires_fresh_inventory is True

    replacement = _snapshot(
        identity,
        thread_id="thread-replacement",
        database_id=20,
        line=20,
        body="finding",
        reviewed_head_sha="head-1",
    )
    recovery = plan_review_thread_actions((desired,), (old, replacement), "head-2")
    assert [action.kind for action in recovery.actions] == [
        ReviewThreadActionKind.UNCHANGED,
        ReviewThreadActionKind.RESOLVE,
    ]
    assert recovery.actions[1].thread_id == "thread-old"


def test_two_outdated_safe_copies_create_once_then_resolve_both_dependencies():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 30), "finding")
    old_threads = (
        _snapshot(identity, thread_id="thread-old-1", outdated=True, anchor=False),
        _snapshot(identity, thread_id="thread-old-2", database_id=20, outdated=True, anchor=False),
    )

    plan = plan_review_thread_actions((desired,), old_threads, "head-1")

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.CREATE,
        ReviewThreadActionKind.RESOLVE,
        ReviewThreadActionKind.RESOLVE,
    ]
    assert plan.actions[0].expected_threads == old_threads
    assert {action.thread_id for action in plan.actions[1:]} == {"thread-old-1", "thread-old-2"}
    assert all(action.depends_on_action_id == plan.actions[0].action_id for action in plan.actions[1:])


def test_create_failure_blocks_every_dependent_resolve():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 30), "finding")
    plan = plan_review_thread_actions(
        (desired,),
        (
            _snapshot(identity, thread_id="thread-old-1", outdated=True, anchor=False),
            _snapshot(identity, thread_id="thread-old-2", database_id=20, outdated=True, anchor=False),
        ),
        "head-1",
    )
    failed_create = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
    )
    provider = _MutationProvider([failed_create])

    outcome = execute_review_thread_action_plan(plan, provider)

    assert [item.state for item in outcome.action_outcomes] == [
        ReviewThreadActionState.FAILED,
        ReviewThreadActionState.NOT_EXECUTED,
        ReviewThreadActionState.NOT_EXECUTED,
    ]
    assert [call[0] for call in provider.calls] == ["create"]


def test_two_current_same_anchor_copies_fail_closed():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")

    plan = plan_review_thread_actions(
        (desired,),
        (
            _snapshot(identity, thread_id="thread-current-1", line=20, body="finding"),
            _snapshot(identity, thread_id="thread-current-2", database_id=20, line=20, body="finding"),
        ),
        "head-1",
    )

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.SKIP,
        ReviewThreadActionKind.SKIP,
    ]
    assert {action.reason for action in plan.actions} == {"duplicate_current_anchor_requires_manual_audit"}


def test_human_reply_among_moved_duplicates_is_preserved_while_safe_copy_is_cleaned_up():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 30), "finding")

    plan = plan_review_thread_actions(
        (desired,),
        (
            _snapshot(identity, thread_id="thread-safe", outdated=True, anchor=False),
            _snapshot(identity, thread_id="thread-replied", database_id=20, outdated=True, anchor=False, replies=True),
        ),
        "head-1",
    )

    assert [action.kind for action in plan.actions] == [
        ReviewThreadActionKind.CREATE,
        ReviewThreadActionKind.RESOLVE,
        ReviewThreadActionKind.SKIP,
    ]
    assert plan.actions[1].thread_id == "thread-safe"
    assert plan.actions[2].thread_id == "thread-replied"
    assert plan.actions[2].reason == "thread_with_human_replies_preserved"


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


def test_bot_fixed_marker_history_allows_exactly_one_recurrence():
    identity = _identity()
    resolved = _snapshot(
        identity,
        resolved=True,
        body=f"old wording\n\n{FIXED_THREAD_STATE_MARKER}",
        resolved_by_viewer_bot=True,
    )
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "returned finding")

    plan = plan_review_thread_actions((desired,), (resolved,), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.CREATE]
    assert plan.actions[0].reason == "finding_reintroduced_after_fixed_marker"


def test_authoritative_bot_resolved_reply_free_history_can_recur_without_fixed_marker():
    identity = _identity()
    resolved = _snapshot(identity, resolved=True, resolved_by_viewer_bot=True)
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "returned finding")

    plan = plan_review_thread_actions((desired,), (resolved,), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.CREATE]
    assert plan.actions[0].reason == "finding_reintroduced_after_bot_resolution"
    assert plan.actions[0].expected_threads == (resolved,)


def test_bot_resolved_history_with_human_reply_does_not_recur():
    identity = _identity()
    resolved = _snapshot(identity, resolved=True, resolved_by_viewer_bot=True, replies=True)
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "returned finding")

    plan = plan_review_thread_actions((desired,), (resolved,), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SKIP]


@pytest.mark.parametrize(
    "resolver_kwargs",
    [
        {},
        {"resolved_by_other_actor": True},
        {"resolved_by_other_actor": True, "body": f"old wording\n\n{FIXED_THREAD_STATE_MARKER}"},
    ],
)
def test_human_or_unknown_resolution_stays_untouched(resolver_kwargs):
    identity = _identity()
    resolved = _snapshot(identity, resolved=True, **resolver_kwargs)
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "returned finding")

    plan = plan_review_thread_actions((desired,), (resolved,), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SKIP]
    assert plan.actions[0].reason == "human_or_unknown_resolution_preserved"


def test_any_human_resolved_history_blocks_automatic_recurrence():
    identity = _identity()
    bot_history = _snapshot(
        identity,
        thread_id="thread-bot-history",
        resolved=True,
        resolved_by_viewer_bot=True,
        created_at="2026-08-30T12:01:00Z",
    )
    human_history = _snapshot(
        identity,
        thread_id="thread-human-history",
        database_id=20,
        resolved=True,
        resolved_by_other_actor=True,
        created_at="2026-08-30T12:00:00Z",
    )
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "returned finding")

    plan = plan_review_thread_actions((desired,), (human_history, bot_history), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.SKIP]


def test_active_recurrence_wins_over_resolved_history_and_does_not_create_again():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "returned finding")
    resolved = _snapshot(
        identity,
        thread_id="thread-history",
        resolved=True,
        resolved_by_viewer_bot=True,
    )
    recurrence = _snapshot(
        identity,
        thread_id="thread-recurrence",
        database_id=20,
        line=20,
        body="returned finding",
        reviewed_head_sha="head-1",
    )

    plan = plan_review_thread_actions((desired,), (resolved, recurrence), "head-1")

    assert [action.kind for action in plan.actions] == [ReviewThreadActionKind.UNCHANGED]
    assert plan.actions[0].thread_id == "thread-recurrence"


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

    future_body = build_summary_fallback_marker(identity.finding_id, marker_version="v2")
    future_version = execute_review_thread_action_plan(
        plan,
        _MutationProvider([failed]),
        existing_summary_bodies=(future_body,),
    )
    assert len(future_version.summary_fallbacks) == 1

    mixed_versions = execute_review_thread_action_plan(
        plan,
        _MutationProvider([failed]),
        existing_summary_bodies=(f"{existing_body}\n{future_body}",),
    )
    assert len(mixed_versions.summary_fallbacks) == 1


def test_post_create_head_change_blocks_cleanup_and_requires_fresh_inventory():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")
    plan = plan_review_thread_actions((desired,), (_snapshot(identity),), "head-1")
    created_on_stale_inventory = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
        expected_head_sha="head-1",
        current_head_sha="head-2",
        comment_id=77,
        reason="pull_request_head_changed_after_mutation",
        mutation_attempted=True,
        mutation_result_ambiguous=True,
    )
    provider = _MutationProvider([created_on_stale_inventory])

    outcome = execute_review_thread_action_plan(plan, provider)

    assert outcome.complete is False
    assert outcome.requires_fresh_inventory is True
    assert [item.state for item in outcome.action_outcomes] == [
        ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
        ReviewThreadActionState.NOT_EXECUTED,
    ]
    assert outcome.action_outcomes[1].reason == "fresh_inventory_required"
    assert [call[0] for call in provider.calls] == ["create"]


def test_rate_limited_create_requires_inventory_refresh_without_duplicate_fallback_or_cleanup():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")
    plan = plan_review_thread_actions((desired,), (_snapshot(identity),), "head-1")
    rate_limited = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.RATE_LIMITED,
        retry_after_seconds=60,
        retry_source="retry-after",
        mutation_attempted=True,
        mutation_result_ambiguous=True,
    )
    provider = _MutationProvider([rate_limited])

    outcome = execute_review_thread_action_plan(plan, provider)

    assert outcome.requires_fresh_inventory is True
    assert outcome.action_outcomes[0].retryable is True
    assert outcome.action_outcomes[0].retry_requires_fresh_inventory is True
    assert outcome.summary_fallbacks == ()
    assert outcome.action_outcomes[1].state == ReviewThreadActionState.NOT_EXECUTED
    assert outcome.action_outcomes[1].reason == "fresh_inventory_required"
    assert [call[0] for call in provider.calls] == ["create"]


def test_ambiguous_provider_failure_create_requires_inventory_before_fallback_or_cleanup():
    identity = _identity()
    desired = DesiredReviewThread(identity, ReviewThreadAnchor(identity.path, 20), "finding")
    plan = plan_review_thread_actions((desired,), (_snapshot(identity),), "head-1")
    ambiguous_failure = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.FAILED,
        expected_head_sha="head-1",
        current_head_sha="head-1",
        failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
        reason="create_failed: response lost after send",
        mutation_attempted=True,
        mutation_result_ambiguous=True,
    )

    outcome = execute_review_thread_action_plan(plan, _MutationProvider([ambiguous_failure]))

    assert outcome.requires_fresh_inventory is True
    assert outcome.summary_fallbacks == ()
    assert outcome.action_outcomes[1].state == ReviewThreadActionState.NOT_EXECUTED
    assert outcome.action_outcomes[1].reason == "fresh_inventory_required"


def test_applied_rate_limit_signal_requires_refresh_but_is_not_retryable():
    outcome = ReviewThreadActionOutcome(
        kind=ReviewThreadActionKind.CREATE,
        state=ReviewThreadActionState.APPLIED_REQUIRES_REFRESH,
        expected_head_sha="head-1",
        current_head_sha=None,
        failure_kind=ReviewThreadFailureKind.RATE_LIMITED,
        mutation_attempted=True,
        mutation_result_ambiguous=True,
    )

    assert outcome.requires_fresh_inventory is True
    assert outcome.retryable is False
    assert outcome.retry_requires_fresh_inventory is False


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
