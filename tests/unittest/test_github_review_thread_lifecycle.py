import json
from types import SimpleNamespace

import pytest

from pr_agent.algo.inline_comment_dedup import body_with_finding_identity_marker
from pr_agent.algo.review_thread_reconciler import (
    FindingIdentity,
    ReviewThreadActionKind,
    ReviewThreadActionState,
    ReviewThreadFailureKind,
)
from pr_agent.git_providers.github_provider import GithubProvider


def _graphql(data, errors=None):
    body = {"data": data}
    if errors:
        body["errors"] = errors
    return 200, {}, json.dumps(body)


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
    return FindingIdentity("owner/repo", 42, "root-cause-1", "src/app.py", "run")


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


def test_inventory_failure_is_distinct_from_empty_inventory():
    requester = _Requester(graphql=[_graphql({}, errors=[{"message": "permission denied"}])])

    with pytest.raises(RuntimeError, match="permission denied"):
        _provider(requester).get_review_thread_snapshots()


def test_create_review_thread_uses_exact_head_and_returns_structured_outcome():
    requester = _Requester(
        rest=[
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
        ]
    )
    provider = _provider(requester)

    outcome = provider.create_review_thread(
        {"body": "finding", "path": "src/app.py", "line": 10, "side": "RIGHT"},
        "head-1",
    )

    assert outcome.kind == ReviewThreadActionKind.CREATE
    assert outcome.state == ReviewThreadActionState.APPLIED
    assert outcome.comment_id == 77
    assert requester.calls[1][1:3] == (
        "POST",
        "https://api.github.com/repos/owner/repo/pulls/42/comments",
    )
    assert requester.calls[1][3]["commit_id"] == "head-1"


def test_update_review_thread_uses_review_comment_endpoint():
    requester = _Requester(
        rest=[
            {"head": {"sha": "head-1"}},
            {"id": 77, "node_id": "comment-node-77"},
        ]
    )
    provider = _provider(requester)

    outcome = provider.update_review_thread(77, "new body", "head-1")

    assert outcome.state == ReviewThreadActionState.APPLIED
    assert requester.calls[1][1:4] == (
        "PATCH",
        "https://api.github.com/repos/owner/repo/pulls/comments/77",
        {"body": "new body"},
    )


def test_resolve_review_thread_uses_thread_node_id():
    requester = _Requester(
        rest=[{"head": {"sha": "head-1"}}],
        graphql=[
            _graphql(
                {
                    "resolveReviewThread": {"thread": {"id": "thread-1", "isResolved": True}},
                }
            )
        ],
    )
    provider = _provider(requester)

    outcome = provider.resolve_review_thread("thread-1", "head-1")

    assert outcome.state == ReviewThreadActionState.APPLIED
    assert outcome.thread_id == "thread-1"
    assert requester.calls[1][3]["variables"] == {"threadId": "thread-1"}


@pytest.mark.parametrize("operation", ["create", "update", "resolve"])
def test_stale_head_aborts_before_mutation(operation):
    requester = _Requester(rest=[{"head": {"sha": "head-2"}}])
    provider = _provider(requester)

    if operation == "create":
        outcome = provider.create_review_thread({"body": "x"}, "head-1")
    elif operation == "update":
        outcome = provider.update_review_thread(77, "x", "head-1")
    else:
        outcome = provider.resolve_review_thread("thread-1", "head-1")

    assert outcome.state == ReviewThreadActionState.STALE_HEAD
    assert outcome.current_head_sha == "head-2"
    assert len(requester.calls) == 1


def test_permission_failure_is_reported_not_raised_or_counted_as_success():
    class _ForbiddenRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "PATCH":
                raise RuntimeError("permission denied")
            return super().requestJsonAndCheck(method, url, input=input)

    requester = _ForbiddenRequester(rest=[{"head": {"sha": "head-1"}}])

    outcome = _provider(requester).update_review_thread(77, "new body", "head-1")

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == ReviewThreadFailureKind.PERMISSION_DENIED
    assert "permission denied" in outcome.reason


def test_create_422_is_classified_as_invalid_inline_location_with_details():
    class _ValidationFailure(RuntimeError):
        status = 422
        data = {"message": "Validation Failed", "errors": [{"field": "line"}]}

    class _InvalidLocationRequester(_Requester):
        def requestJsonAndCheck(self, method, url, input=None):
            if method == "POST":
                raise _ValidationFailure("invalid review comment location")
            return super().requestJsonAndCheck(method, url, input=input)

    requester = _InvalidLocationRequester(rest=[{"head": {"sha": "head-1"}}])

    outcome = _provider(requester).create_review_thread(
        {"body": "finding", "path": "src/app.py", "line": 10, "side": "RIGHT"},
        "head-1",
    )

    assert outcome.state == ReviewThreadActionState.FAILED
    assert outcome.failure_kind == ReviewThreadFailureKind.INVALID_INLINE_LOCATION
    assert "invalid review comment location" in outcome.reason
