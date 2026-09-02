import json
import threading
from types import SimpleNamespace

import pytest

from pr_agent.algo.inline_comment_dedup import \
    body_with_finding_identity_marker
from pr_agent.algo.review_thread_reconciler import (
    FIXED_THREAD_STATE_MARKER, VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION,
    FindingIdentity, ReviewThreadActionKind, ReviewThreadActionState,
    ReviewThreadAnchor, ReviewThreadCommentSnapshot, ReviewThreadFailureKind,
    ReviewThreadSnapshot, execute_review_thread_action_plan,
    plan_review_thread_actions)
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.servers.utils import RateLimitExceeded


def _graphql(data, errors=None, *, status=200, headers=None):
    body = {"data": data}
    if errors:
        body["errors"] = errors
    return status, headers or {}, json.dumps(body)


class _Requester:
    def __init__(self, *, graphql=(), rest=()):
        self.graphql = list(graphql)
        self.rest = list(rest)
        self.calls = []

    def requestJson(self, method, url, input=None):
        self.calls.append(("graphql", method, url, input))
        return self.graphql.pop(0)

    def requestJsonAndCheck(self, method, url, input=None):
        self.calls.append(("rest", method, url, input))
        return {}, self.rest.pop(0)


def _provider(requester):
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "owner/repo"
    provider.pr_num = 42
    provider.base_url = "https://api.github.com"
    provider.pr = SimpleNamespace(_requester=requester)
    provider.github_client = SimpleNamespace(_Github__requester=requester)
    return provider


def _identity():
    return FindingIdentity(
        "owner/repo",
        42,
        f"sha256:{'a' * 64}",
        "src/app.py",
        "run",
        root_cause_id_schema=VERIFIED_ROOT_CAUSE_ID_SCHEMA_VERSION,
    )


def _create_comment(body="finding"):
    return {
        "body": body_with_finding_identity_marker(body, _identity().finding_id),
        "path": "src/app.py",
        "line": 10,
        "side": "RIGHT",
    }


def _comment(
    body,
    *,
    node_id="comment-1",
    database_id=101,
    author="pr-agent[bot]",
    author_id=None,
    author_type=None,
    commit="head-1",
):
    author_type = author_type or ("Bot" if author.endswith("[bot]") else "User")
    author_id = author_id or ("BOT-1" if author_type == "Bot" else "USER-1")
    return {
        "id": node_id,
        "databaseId": database_id,
        "body": body,
        "createdAt": "2026-08-30T12:00:00Z",
        "url": f"https://github.test/{node_id}",
        "author": {"id": author_id, "login": author, "__typename": author_type},
        "pullRequestReview": {"commit": {"oid": commit}},
    }


def _thread(
    thread_id,
    comments,
    *,
    resolved=False,
    outdated=False,
    line=10,
    start_line=None,
    original_line=10,
    original_start_line=None,
    subject_type="LINE",
    viewer_can_resolve=True,
    resolved_by=None,
    page_info=None,
):
    return {
        "id": thread_id,
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": "src/app.py",
        "line": line,
        "startLine": start_line,
        "diffSide": "RIGHT",
        "startDiffSide": None,
        "originalLine": original_line,
        "originalStartLine": original_start_line,
        "subjectType": subject_type,
        "viewerCanResolve": viewer_can_resolve,
        "resolvedBy": resolved_by,
        "comments": {
            "pageInfo": page_info or {"hasNextPage": False, "endCursor": None},
            "nodes": comments,
        },
    }


def _inventory_page(
    threads,
    *,
    has_next=False,
    cursor=None,
    viewer="pr-agent[bot]",
    viewer_id="BOT-1",
    viewer_type="Bot",
):
    return _graphql(
        {
            "viewer": {"id": viewer_id, "login": viewer, "__typename": viewer_type},
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                        "nodes": threads,
                    }
                }
            },
        }
    )


def _owned_thread_state(body=None, *, replies=(), resolved=False, viewer_can_resolve=True):
    identity = _identity()
    body = body or body_with_finding_identity_marker("finding", identity.finding_id)
    raw_comments = [_comment(body, database_id=77), *replies]
    comments = tuple(ReviewThreadCommentSnapshot(
        node_id=comment["id"],
        database_id=comment.get("databaseId"),
        author_id=comment["author"].get("id"),
        author_login=comment["author"].get("login"),
        author_type=comment["author"].get("__typename"),
        body=comment.get("body") or "",
        created_at=comment.get("createdAt"),
        url=comment.get("url"),
    ) for comment in raw_comments)
    snapshot = ReviewThreadSnapshot(
        thread_id="thread-1",
        finding_id=identity.finding_id,
        anchor=ReviewThreadAnchor("src/app.py", 10),
        original_anchor=ReviewThreadAnchor("src/app.py", 10),
        is_resolved=resolved,
        is_outdated=False,
        bot_owned=True,
        has_replies=bool(replies),
        reviewed_head_sha="head-1",
        comments=comments,
        subject_type="LINE",
        viewer_can_resolve=viewer_can_resolve,
    )
    raw_thread = _thread(
        "thread-1",
        raw_comments,
        resolved=resolved,
        viewer_can_resolve=viewer_can_resolve,
    )
    return snapshot, _inventory_page([raw_thread])


def test_inventory_parses_identity_ownership_anchor_and_reviewed_head():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    requester = _Requester(
        graphql=[
            _inventory_page(
                [
                    _thread("thread-1", [_comment(body)], outdated=True, line=None, original_line=10),
                ]
            )
        ]
    )

    snapshots = _provider(requester).get_review_thread_snapshots()

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.finding_id == identity.finding_id
    assert snapshot.bot_owned is True
    assert snapshot.has_replies is False
    assert snapshot.is_outdated is True
    assert snapshot.anchor is None
    assert snapshot.original_anchor.path == "src/app.py"
    assert snapshot.original_anchor.line == 10
    assert snapshot.viewer_can_resolve is True
    assert snapshot.reviewed_head_sha == "head-1"


def test_inventory_leaves_future_identity_marker_versions_unowned():
    identity = _identity()
    body = body_with_finding_identity_marker("future finding", identity.finding_id, marker_version="v2")
    requester = _Requester(graphql=[_inventory_page([_thread("thread-1", [_comment(body)])])])

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert snapshot.finding_id is None
    assert snapshot.bot_owned is False


def test_inventory_leaves_mixed_supported_and_future_marker_versions_unowned():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    body = body_with_finding_identity_marker(body, f"sha256:{'b' * 64}", marker_version="v2")
    requester = _Requester(graphql=[_inventory_page([_thread("thread-1", [_comment(body)])])])

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert snapshot.finding_id is None
    assert snapshot.bot_owned is False


def test_inventory_paginates_threads_and_comments():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    first_thread = _thread(
        "thread-1",
        [_comment(body)],
        page_info={
            "hasNextPage": True,
            "endCursor": "comment-cursor",
        },
    )
    second_thread = _thread("thread-2", [_comment(body, node_id="comment-2", database_id=102)])
    extra_comments = _graphql(
        {
            "node": {
                "comments": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [_comment("human reply", node_id="reply-1", database_id=103, author="human")],
                }
            },
        }
    )
    requester = _Requester(
        graphql=[
            _inventory_page([first_thread], has_next=True, cursor="thread-cursor"),
            _inventory_page([second_thread]),
            extra_comments,
        ]
    )

    snapshots = _provider(requester).get_review_thread_snapshots()

    assert len(snapshots) == 2
    assert snapshots[0].has_replies is True
    assert requester.calls[1][3]["variables"]["after"] == "thread-cursor"
    assert requester.calls[2][3]["variables"] == {
        "threadId": "thread-1",
        "after": "comment-cursor",
    }


def test_inventory_fails_closed_when_review_thread_cursor_repeats():
    requester = _Requester(graphql=[
        _inventory_page([], has_next=True, cursor="repeated-cursor"),
        _inventory_page([], has_next=True, cursor="repeated-cursor"),
    ])

    with pytest.raises(RuntimeError, match="repeated a review-thread cursor"):
        _provider(requester).get_review_thread_snapshots()


def test_inventory_fails_closed_when_comment_cursor_repeats():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    thread = _thread(
        "thread-1",
        [_comment(body)],
        page_info={"hasNextPage": True, "endCursor": "repeated-cursor"},
    )
    repeated_page = _graphql({
        "node": {
            "comments": {
                "pageInfo": {"hasNextPage": True, "endCursor": "repeated-cursor"},
                "nodes": [],
            }
        }
    })
    requester = _Requester(graphql=[_inventory_page([thread]), repeated_page])

    with pytest.raises(RuntimeError, match="repeated a comment cursor"):
        _provider(requester).get_review_thread_snapshots()


def test_inventory_keeps_current_and_original_multiline_anchors_distinct():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    requester = _Requester(
        graphql=[
            _inventory_page(
                [
                    _thread(
                        "thread-1",
                        [_comment(body)],
                        line=20,
                        start_line=18,
                        original_line=12,
                        original_start_line=10,
                    )
                ]
            )
        ]
    )

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert (snapshot.anchor.start_line, snapshot.anchor.line) == (18, 20)
    assert (snapshot.original_anchor.start_line, snapshot.original_anchor.line) == (10, 12)


def test_inventory_marks_human_thread_unsafe_even_if_body_mentions_agent():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    requester = _Requester(
        graphql=[
            _inventory_page(
                [
                    _thread("thread-1", [_comment(body, author="human")]),
                ]
            )
        ]
    )

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert snapshot.finding_id == identity.finding_id
    assert snapshot.bot_owned is False


def test_inventory_never_marks_pat_user_comment_as_bot_owned():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    requester = _Requester(
        graphql=[
            _inventory_page(
                [_thread("thread-1", [_comment(body, author="qfennessy", author_type="User")])],
                viewer="qfennessy",
                viewer_id="USER-1",
                viewer_type="User",
            )
        ]
    )

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert snapshot.finding_id == identity.finding_id
    assert snapshot.bot_owned is False


def test_inventory_never_marks_another_bot_comment_as_owned():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    requester = _Requester(
        graphql=[
            _inventory_page(
                [
                    _thread(
                        "thread-1",
                        [_comment(body, author="other[bot]", author_id="BOT-2", author_type="Bot")],
                    ),
                ]
            )
        ]
    )

    assert _provider(requester).get_review_thread_snapshots()[0].bot_owned is False


def test_inventory_accepts_exact_github_app_bot_identity_when_graphql_reports_user_type():
    identity = _identity()
    body = body_with_finding_identity_marker("finding", identity.finding_id)
    bot_actor = {"id": "BOT-1", "login": "pr-agent[bot]", "__typename": "User"}
    requester = _Requester(
        graphql=[
            _inventory_page(
                [
                    _thread(
                        "thread-1",
                        [
                            _comment(
                                body,
                                author="pr-agent[bot]",
                                author_id="BOT-1",
                                author_type="User",
                            )
                        ],
                        resolved=True,
                        resolved_by=bot_actor,
                    )
                ],
                viewer="pr-agent[bot]",
                viewer_id="BOT-1",
                viewer_type="User",
            )
        ]
    )

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert snapshot.bot_owned is True
    assert snapshot.resolved_by_viewer_bot is True


@pytest.mark.parametrize(
    "resolved_by,expected_viewer_bot,expected_other_actor",
    [
        ({"id": "BOT-1", "login": "pr-agent[bot]", "__typename": "Bot"}, True, False),
        ({"id": "USER-1", "login": "human", "__typename": "User"}, False, True),
        ({"id": "BOT-2", "login": "other[bot]", "__typename": "Bot"}, False, True),
        (None, False, False),
    ],
)
def test_inventory_attributes_resolution_only_to_exact_authenticated_bot(
    resolved_by,
    expected_viewer_bot,
    expected_other_actor,
):
    identity = _identity()
    body = body_with_finding_identity_marker(
        f"finding\n\n{FIXED_THREAD_STATE_MARKER}", identity.finding_id
    )
    requester = _Requester(
        graphql=[_inventory_page([_thread("thread-1", [_comment(body)], resolved=True, resolved_by=resolved_by)])]
    )

    snapshot = _provider(requester).get_review_thread_snapshots()[0]

    assert snapshot.resolved_by_viewer_bot is expected_viewer_bot
    assert snapshot.resolved_by_other_actor is expected_other_actor
    assert snapshot.has_fixed_state_marker is True


def test_inventory_failure_is_distinct_from_empty_inventory():
    requester = _Requester(graphql=[_graphql({}, errors=[{"message": "permission denied"}])])

    with pytest.raises(RuntimeError, match="permission denied"):
        _provider(requester).get_review_thread_snapshots()


def test_create_review_thread_uses_exact_head_and_returns_structured_outcome():
    comment = _create_comment()
    requester = _Requester(
        graphql=[
            _inventory_page([]),
            _inventory_page([_thread("thread-1", [_comment(comment["body"], database_id=77)])]),
        ],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
            {"head": {"sha": "head-1"}},
        ]
    )
    provider = _provider(requester)

    outcome = provider.create_review_thread(
        comment,
        "head-1",
    )

    assert outcome.kind == ReviewThreadActionKind.CREATE
    assert outcome.state == ReviewThreadActionState.APPLIED
    assert outcome.comment_id == 77
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is False
    assert requester.calls[3][1:3] == (
        "POST",
        "https://api.github.com/repos/owner/repo/pulls/42/comments",
    )
    assert requester.calls[3][3]["commit_id"] == "head-1"


def test_create_review_thread_is_idempotent_when_same_finding_appears_after_planning():
    comment = _create_comment()
    _, inventory = _owned_thread_state(body=comment["body"])
    requester = _Requester(
        graphql=[inventory],
        rest=[{"head": {"sha": "head-1"}}],
    )

    outcome = _provider(requester).create_review_thread(comment, "head-1")

    assert outcome.state == ReviewThreadActionState.ALREADY_APPLIED
    assert outcome.thread_id == "thread-1"
    assert outcome.comment_id == 77
    assert not any(call[0] == "rest" and call[1] == "POST" for call in requester.calls)


def test_concurrent_create_review_thread_calls_publish_once_for_one_finding():
    comment = _create_comment()
    _, created_inventory = _owned_thread_state(body=comment["body"])
    requester = _Requester(
        graphql=[_inventory_page([]), created_inventory, created_inventory],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
        ],
    )
    barrier = threading.Barrier(3)
    outcomes = []

    def create():
        barrier.wait()
        outcomes.append(_provider(requester).create_review_thread(comment, "head-1"))

    workers = [threading.Thread(target=create) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)

    assert not any(worker.is_alive() for worker in workers)
    assert {outcome.state for outcome in outcomes} == {
        ReviewThreadActionState.APPLIED,
        ReviewThreadActionState.ALREADY_APPLIED,
    }
    assert len([call for call in requester.calls if call[0] == "rest" and call[1] == "POST"]) == 1


def test_create_review_thread_converges_safe_concurrent_duplicates_to_oldest_thread():
    comment = _create_comment()
    first_thread = _thread("thread-1", [_comment(comment["body"], database_id=77)])
    second_thread = _thread(
        "thread-2",
        [_comment(comment["body"], node_id="comment-2", database_id=78)],
    )
    duplicate_inventory = _inventory_page([first_thread, second_thread])
    requester = _Requester(
        graphql=[
            _inventory_page([]),
            duplicate_inventory,
            duplicate_inventory,
            _graphql({"resolveReviewThread": {"thread": {"id": "thread-2", "isResolved": True}}}),
        ],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 78, "node_id": "comment-2"},
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
        ],
    )

    outcome = _provider(requester).create_review_thread(comment, "head-1")

    assert outcome.state == ReviewThreadActionState.APPLIED
    assert outcome.thread_id == "thread-1"
    assert outcome.comment_id == 77
    resolve_calls = [call for call in requester.calls if "resolveReviewThread" in str(call[3])]
    assert len(resolve_calls) == 1
    assert resolve_calls[0][3]["variables"] == {"threadId": "thread-2"}


def test_create_review_thread_preserves_concurrent_duplicate_with_different_content():
    comment = _create_comment()
    first_thread = _thread("thread-1", [_comment(comment["body"], database_id=77)])
    different_body = body_with_finding_identity_marker("changed finding", _identity().finding_id)
    second_thread = _thread(
        "thread-2",
        [_comment(different_body, node_id="comment-2", database_id=78)],
    )
    requester = _Requester(
        graphql=[_inventory_page([]), _inventory_page([first_thread, second_thread])],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 78, "node_id": "comment-2"},
            {"head": {"sha": "head-1"}},
        ],
    )

    outcome = _provider(requester).create_review_thread(comment, "head-1")

    assert outcome.state == ReviewThreadActionState.APPLIED_REQUIRES_REFRESH
    assert outcome.reason == "concurrent_finding_thread_not_safe_to_converge"
    assert not any("resolveReviewThread" in str(call[3]) for call in requester.calls)


@pytest.mark.parametrize(
    "created_comment",
    [
        {"node_id": "comment-node-77"},
        {"id": 77},
        {},
    ],
)
def test_create_with_incomplete_returned_identity_forces_inventory_refresh(created_comment):
    requester = _Requester(
        graphql=[_inventory_page([])],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            created_comment,
            {"head": {"sha": "head-1"}},
        ]
    )

    outcome = _provider(requester).create_review_thread(_create_comment(), "head-1")

    assert outcome.state == ReviewThreadActionState.APPLIED_REQUIRES_REFRESH
    assert outcome.reason == "created_comment_identity_incomplete"
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is True
    assert outcome.requires_fresh_inventory is True
    assert outcome.retryable is False


def test_update_review_thread_uses_review_comment_endpoint():
    expected_thread, inventory = _owned_thread_state()
    requester = _Requester(
        graphql=[inventory],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
            {"head": {"sha": "head-1"}},
        ]
    )
    provider = _provider(requester)

    outcome = provider.update_review_thread(77, "new body", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.APPLIED
    assert requester.calls[3][1:4] == (
        "PATCH",
        "https://api.github.com/repos/owner/repo/pulls/comments/77",
        {"body": "new body"},
    )


def test_resolve_review_thread_uses_thread_node_id():
    expected_thread, inventory = _owned_thread_state()
    requester = _Requester(
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
        ],
        graphql=[
            inventory,
            _graphql(
                {
                    "resolveReviewThread": {"thread": {"id": "thread-1", "isResolved": True}},
                }
            )
        ],
    )
    provider = _provider(requester)

    outcome = provider.resolve_review_thread("thread-1", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.APPLIED
    assert outcome.thread_id == "thread-1"
    assert requester.calls[3][3]["variables"] == {"threadId": "thread-1"}


@pytest.mark.parametrize("operation", ["update", "resolve"])
def test_destructive_mutation_aborts_when_human_reply_arrives_after_planning(operation):
    expected_thread, _ = _owned_thread_state()
    human_reply = _comment("do not mutate", node_id="reply-1", database_id=78, author="human")
    _, changed_inventory = _owned_thread_state(replies=(human_reply,))
    requester = _Requester(rest=[{"head": {"sha": "head-1"}}], graphql=[changed_inventory])
    provider = _provider(requester)

    if operation == "update":
        outcome = provider.update_review_thread(77, "new body", "head-1", expected_thread)
    else:
        outcome = provider.resolve_review_thread("thread-1", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.STALE_INVENTORY
    assert outcome.reason == "review_thread_changed_since_inventory"
    assert outcome.mutation_attempted is False
    assert outcome.requires_fresh_inventory is True
    assert not any(
        (call[0] == "rest" and call[1] in {"PATCH", "POST"})
        or (call[0] == "graphql" and "mutation(" in call[3]["query"])
        for call in requester.calls
    )


@pytest.mark.parametrize(
    "changed_body,resolved",
    [
        ("marker removed", False),
        (None, True),
    ],
)
def test_update_aborts_when_marker_or_resolution_state_changes(changed_body, resolved):
    expected_thread, _ = _owned_thread_state()
    _, changed_inventory = _owned_thread_state(body=changed_body, resolved=resolved)
    requester = _Requester(rest=[{"head": {"sha": "head-1"}}], graphql=[changed_inventory])

    outcome = _provider(requester).update_review_thread(77, "new body", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.STALE_INVENTORY
    assert outcome.mutation_attempted is False


def test_mark_fixed_revalidates_again_before_resolve_and_preserves_new_human_reply():
    expected_thread, first_inventory = _owned_thread_state()
    plan = plan_review_thread_actions(
        (),
        (expected_thread,),
        "head-1",
        obsolete_policy="mark_fixed",
        authoritative_absence=True,
    )
    marked_body = plan.actions[0].body
    human_reply = _comment("I am investigating", node_id="reply-1", database_id=78, author="human")
    _, second_inventory = _owned_thread_state(body=marked_body, replies=(human_reply,))
    requester = _Requester(
        graphql=[first_inventory, second_inventory],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-1"},
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
        ],
    )

    outcome = execute_review_thread_action_plan(plan, _provider(requester))

    assert [item.state for item in outcome.action_outcomes] == [
        ReviewThreadActionState.APPLIED,
        ReviewThreadActionState.STALE_INVENTORY,
    ]
    assert len([call for call in requester.calls if "resolveReviewThread" in str(call[3])]) == 0


@pytest.mark.parametrize("operation", ["create", "update", "resolve"])
def test_stale_head_aborts_before_mutation(operation):
    expected_thread, _ = _owned_thread_state()
    requester = _Requester(rest=[{"head": {"sha": "head-2"}}])
    provider = _provider(requester)

    if operation == "create":
        outcome = provider.create_review_thread(_create_comment("x"), "head-1")
    elif operation == "update":
        outcome = provider.update_review_thread(77, "x", "head-1", expected_thread)
    else:
        outcome = provider.resolve_review_thread("thread-1", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.STALE_HEAD
    assert outcome.current_head_sha == "head-2"
    assert len(requester.calls) == 1


@pytest.mark.parametrize("operation", ["create", "update", "resolve"])
def test_head_change_after_mutation_requires_fresh_inventory(operation):
    expected_thread, inventory = _owned_thread_state()
    comment = _create_comment("x")
    rest = [{"head": {"sha": "head-1"}}]
    graphql = []
    if operation == "create":
        rest.extend([
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
            {"head": {"sha": "head-2"}},
        ])
        graphql.extend([
            _inventory_page([]),
            _inventory_page([_thread("thread-1", [_comment(comment["body"], database_id=77)])]),
        ])
    elif operation == "update":
        rest.extend([
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
            {"head": {"sha": "head-2"}},
        ])
        graphql.append(inventory)
    else:
        rest.extend([{"head": {"sha": "head-1"}}, {"head": {"sha": "head-2"}}])
        graphql.extend([
            inventory,
            _graphql({"resolveReviewThread": {"thread": {"id": "thread-1", "isResolved": True}}}),
        ])
    requester = _Requester(rest=rest, graphql=graphql)
    provider = _provider(requester)

    if operation == "create":
        outcome = provider.create_review_thread(comment, "head-1")
    elif operation == "update":
        outcome = provider.update_review_thread(77, "x", "head-1", expected_thread)
    else:
        outcome = provider.resolve_review_thread("thread-1", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.APPLIED_REQUIRES_REFRESH
    assert outcome.current_head_sha == "head-2"
    assert outcome.requires_fresh_inventory is True
    assert outcome.reason == "pull_request_head_changed_after_mutation"
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is True
    assert outcome.retryable is False


def test_post_mutation_head_check_failure_records_applied_but_requires_refresh():
    class _PostCheckFailureRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "GET" and len([call for call in self.calls if call[1] == "GET"]) == 2:
                self.calls.append(("rest", method, url, input))
                raise RuntimeError("head unavailable")
            return super().requestJsonAndCheck(method, url, input=input)

    comment = _create_comment("x")
    requester = _PostCheckFailureRequester(
        graphql=[
            _inventory_page([]),
            _inventory_page([_thread("thread-1", [_comment(comment["body"], database_id=77)])]),
        ],
        rest=[
            {"head": {"sha": "head-1"}},
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
        ],
    )

    outcome = _provider(requester).create_review_thread(comment, "head-1")

    assert outcome.state == ReviewThreadActionState.APPLIED_REQUIRES_REFRESH
    assert outcome.current_head_sha is None
    assert outcome.comment_id == 77
    assert outcome.failure_kind == ReviewThreadFailureKind.PROVIDER_FAILURE
    assert outcome.requires_fresh_inventory is True
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is True
    assert outcome.retryable is False


def test_permission_failure_is_reported_not_raised_or_counted_as_success():
    class _ForbiddenRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "PATCH":
                raise RuntimeError("permission denied")
            return super().requestJsonAndCheck(method, url, input=input)

    expected_thread, inventory = _owned_thread_state()
    requester = _ForbiddenRequester(
        graphql=[inventory],
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
    )

    outcome = _provider(requester).update_review_thread(77, "new body", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == ReviewThreadFailureKind.PERMISSION_DENIED
    assert "permission denied" in outcome.reason


@pytest.mark.parametrize(
    "failure,expected_retry_after,expected_reset,expected_source",
    [
        (
            type(
                "PrimaryRateLimit",
                (RuntimeError,),
                {
                    "status": 403,
                    "headers": {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1790000000"},
                    "data": {"message": "API rate limit exceeded"},
                },
            )("API rate limit exceeded"),
            None,
            1790000000,
            "x-ratelimit-reset",
        ),
        (
            type(
                "SecondaryRateLimit",
                (RuntimeError,),
                {"status": 403, "headers": {"Retry-After": "120"}, "data": {}},
            )("secondary rate limit"),
            120.0,
            None,
            "retry-after",
        ),
        (
            type(
                "TooManyRequests",
                (RuntimeError,),
                {"status": 429, "headers": {"Retry-After": "30"}, "data": {}},
            )("too many requests"),
            30.0,
            None,
            "retry-after",
        ),
        (RateLimitExceeded("Rate limit exceeded"), None, None, "provider-signal"),
    ],
)
def test_rate_limit_failures_are_distinct_and_keep_retry_evidence(
    failure,
    expected_retry_after,
    expected_reset,
    expected_source,
):
    class _RateLimitedRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "POST":
                raise failure
            return super().requestJsonAndCheck(method, url, input=input)

    requester = _RateLimitedRequester(
        graphql=[_inventory_page([])],
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
    )

    outcome = _provider(requester).create_review_thread(_create_comment(), "head-1")

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == ReviewThreadFailureKind.RATE_LIMITED
    assert outcome.retryable is True
    assert outcome.retry_after_seconds == expected_retry_after
    assert outcome.rate_limit_reset_at == expected_reset
    assert outcome.retry_source == expected_source
    assert outcome.requires_fresh_inventory is True
    assert outcome.retry_requires_fresh_inventory is True
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is True


def test_ambiguous_create_transport_failure_requires_inventory_before_retry():
    class _AmbiguousRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "POST":
                raise RuntimeError("connection closed after request was sent")
            return super().requestJsonAndCheck(method, url, input=input)

    outcome = _provider(_AmbiguousRequester(
        graphql=[_inventory_page([])],
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
    )).create_review_thread(
        _create_comment(), "head-1"
    )

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == ReviewThreadFailureKind.PROVIDER_FAILURE
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is True
    assert outcome.requires_fresh_inventory is True
    assert outcome.retryable is False


def test_ordinary_403_remains_permission_denied_without_retry_evidence():
    class _PermissionFailure(RuntimeError):
        status = 403
        headers = {}
        data = {"message": "Resource not accessible by integration"}

    class _ForbiddenRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "PATCH":
                raise _PermissionFailure("forbidden")
            return super().requestJsonAndCheck(method, url, input=input)

    expected_thread, inventory = _owned_thread_state()
    requester = _ForbiddenRequester(
        graphql=[inventory],
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
    )
    outcome = _provider(requester).update_review_thread(
        77, "new body", "head-1", expected_thread
    )

    assert outcome.failure_kind == ReviewThreadFailureKind.PERMISSION_DENIED
    assert outcome.retryable is False
    assert outcome.retry_source is None


def test_graphql_rate_limit_preserves_headers_for_retry_evidence():
    expected_thread, inventory = _owned_thread_state()
    requester = _Requester(
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
        graphql=[
            inventory,
            _graphql(
                {},
                errors=[{"message": "secondary rate limit"}],
                status=403,
                headers={"Retry-After": "45"},
            )
        ],
    )

    outcome = _provider(requester).resolve_review_thread("thread-1", "head-1", expected_thread)

    assert outcome.failure_kind == ReviewThreadFailureKind.RATE_LIMITED
    assert outcome.retry_after_seconds == 45.0
    assert outcome.retry_source == "retry-after"


@pytest.mark.parametrize(
    "response,expected_kind,expected_retry_after",
    [
        (
            (429, {"Retry-After": "20"}, json.dumps({"message": "too many requests"})),
            ReviewThreadFailureKind.RATE_LIMITED,
            20.0,
        ),
        (
            (403, {}, "upstream proxy denied the request"),
            ReviewThreadFailureKind.PERMISSION_DENIED,
            None,
        ),
    ],
)
def test_graphql_http_and_malformed_body_failures_preserve_response_evidence(
    response,
    expected_kind,
    expected_retry_after,
):
    expected_thread, inventory = _owned_thread_state()
    requester = _Requester(
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
        graphql=[inventory, response],
    )

    outcome = _provider(requester).resolve_review_thread("thread-1", "head-1", expected_thread)

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == expected_kind
    assert outcome.retry_after_seconds == expected_retry_after


def test_create_422_is_classified_as_invalid_inline_location_with_details():
    class _ValidationFailure(RuntimeError):
        status = 422
        data = {"message": "Validation Failed", "errors": [{"field": "line"}]}

    class _InvalidLocationRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "POST":
                raise _ValidationFailure("invalid review comment location")
            return super().requestJsonAndCheck(method, url, input=input)

    requester = _InvalidLocationRequester(
        graphql=[_inventory_page([])],
        rest=[{"head": {"sha": "head-1"}}, {"head": {"sha": "head-1"}}],
    )

    outcome = _provider(requester).create_review_thread(
        _create_comment(),
        "head-1",
    )

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == ReviewThreadFailureKind.INVALID_INLINE_LOCATION
    assert "invalid review comment location" in outcome.reason
    assert outcome.mutation_attempted is True
    assert outcome.mutation_result_ambiguous is False
    assert outcome.requires_fresh_inventory is False
