"""Pure contracts and planning for stable review-thread lifecycle management.

This module deliberately has no provider calls. It turns verified findings and a
snapshot of existing review threads into an explicit action plan; a later, gated
integration can execute that plan after re-checking the pull-request head.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from pr_agent.algo.inline_comment_dedup import (
    body_with_finding_identity_marker,
    build_summary_fallback_marker,
    strip_inline_comment_markers,
    summary_fallback_markers,
)

REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION = "review-thread-lifecycle-v1"
FIXED_THREAD_NOTICE = "> ✅ **PR-Agent status:** Fixed or obsolete in the latest revision."
FIXED_THREAD_STATE_MARKER = "<!-- pr-agent-thread-state:v1 state=fixed -->"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalise_text(value: str) -> str:
    return " ".join((value or "").split())


@dataclass(frozen=True)
class FindingIdentity:
    """Stable logical identity supplied by the verified-finding pipeline.

    ``root_cause_id`` is intentionally an upstream contract rather than prose
    derived here. A model may reword a finding without changing its root cause,
    and the reconciler must not guess semantic equivalence from mutable wording.
    """

    repository: str
    pull_request_number: int
    root_cause_id: str
    path: str
    symbol: Optional[str] = None
    trusted_stable_key: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION
    finding_id: str = field(init=False)

    def __post_init__(self) -> None:
        repository = (self.repository or "").strip().strip("/").casefold()
        path = (self.path or "").strip().lstrip("/")
        root_cause_id = _normalise_text(self.root_cause_id)
        symbol = _normalise_text(self.symbol) if self.symbol else None
        trusted_stable_key = _normalise_text(self.trusted_stable_key) if self.trusted_stable_key else None
        if not repository or "/" not in repository:
            raise ValueError("repository must be in owner/name form")
        if self.pull_request_number < 1:
            raise ValueError("pull_request_number must be positive")
        if not root_cause_id:
            raise ValueError("root_cause_id is required")
        if not path:
            raise ValueError("path is required")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "root_cause_id", root_cause_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "trusted_stable_key", trusted_stable_key)
        scope = {"trusted_stable_key": trusted_stable_key} if trusted_stable_key else {"path": path, "symbol": symbol}
        identity = {
            "schema_version": self.schema_version,
            "repository": repository,
            "pull_request_number": self.pull_request_number,
            "root_cause_id": root_cause_id,
            "scope": scope,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        object.__setattr__(self, "finding_id", f"sha256:{digest}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewThreadAnchor:
    path: str
    line: int
    start_line: Optional[int] = None
    side: str = "RIGHT"
    start_side: Optional[str] = None

    @classmethod
    def from_github(
        cls,
        path: object,
        line: object,
        start_line: object = None,
        side: object = "RIGHT",
        start_side: object = None,
    ) -> Optional["ReviewThreadAnchor"]:
        """Canonicalize a GitHub anchor, returning ``None`` for unusable locations."""
        canonical_path = str(path or "").strip().strip("`").replace("\\", "/").lstrip("/")
        try:
            canonical_line = int(line)
            canonical_start = int(start_line) if start_line is not None else None
        except (TypeError, ValueError, OverflowError):
            return None
        if canonical_line < 1 or (
            canonical_start is not None and (canonical_start < 1 or canonical_start > canonical_line)
        ):
            return None
        canonical_side = str(side or "RIGHT").upper()
        canonical_start_side = str(start_side or canonical_side).upper() if canonical_start else None
        if canonical_side not in {"LEFT", "RIGHT"} or canonical_start_side not in {None, "LEFT", "RIGHT"}:
            return None
        if canonical_start == canonical_line:
            canonical_start = None
            canonical_start_side = None
        try:
            return cls(
                path=canonical_path,
                line=canonical_line,
                start_line=canonical_start,
                side=canonical_side,
                start_side=canonical_start_side,
            )
        except ValueError:
            return None

    def __post_init__(self) -> None:
        path = (self.path or "").strip().strip("`").replace("\\", "/").lstrip("/")
        if not path:
            raise ValueError("path is required")
        if self.line < 1:
            raise ValueError("line must be positive")
        if self.start_line is not None and (self.start_line < 1 or self.start_line > self.line):
            raise ValueError("start_line must be positive and no greater than line")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "side", self.side.upper())
        start_side = self.start_side.upper() if self.start_side else None
        if self.side not in {"LEFT", "RIGHT"} or start_side not in {None, "LEFT", "RIGHT"}:
            raise ValueError("side values must be LEFT or RIGHT")
        if self.start_line == self.line:
            object.__setattr__(self, "start_line", None)
            start_side = None
        elif self.start_line is not None and start_side is None:
            start_side = self.side
        object.__setattr__(self, "start_side", start_side)

    def to_github_comment(self, body: str) -> dict[str, Any]:
        comment = {"body": body, "path": self.path, "line": self.line, "side": self.side}
        if self.start_line is not None and self.start_line != self.line:
            comment["start_line"] = self.start_line
            comment["start_side"] = self.start_side or self.side
        return comment


@dataclass(frozen=True)
class ReviewThreadCommentSnapshot:
    node_id: str
    database_id: Optional[int]
    author_login: Optional[str]
    body: str
    author_id: Optional[str] = None
    author_type: Optional[str] = None
    created_at: Optional[str] = None
    url: Optional[str] = None


@dataclass(frozen=True)
class ReviewThreadSnapshot:
    thread_id: str
    finding_id: Optional[str]
    anchor: Optional[ReviewThreadAnchor]
    is_resolved: bool
    is_outdated: bool
    bot_owned: bool
    has_replies: bool
    reviewed_head_sha: Optional[str]
    comments: tuple[ReviewThreadCommentSnapshot, ...]
    original_anchor: Optional[ReviewThreadAnchor] = None
    subject_type: Optional[str] = None
    viewer_can_resolve: bool = False
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def root_comment(self) -> Optional[ReviewThreadCommentSnapshot]:
        return self.comments[0] if self.comments else None


@dataclass(frozen=True)
class DesiredReviewThread:
    identity: FindingIdentity
    anchor: Optional[ReviewThreadAnchor]
    body: str

    @property
    def marked_body(self) -> str:
        return body_with_finding_identity_marker(self.body, self.identity.finding_id)


class ReviewThreadActionKind(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    RESOLVE = "resolve"
    UNCHANGED = "unchanged"
    SKIP = "skip"
    SUMMARY_FALLBACK = "summary_fallback"


@dataclass(frozen=True)
class ReviewThreadAction:
    action_id: str
    kind: ReviewThreadActionKind
    finding_id: str
    expected_head_sha: str
    reason: str
    thread_id: Optional[str] = None
    root_comment_id: Optional[int] = None
    anchor: Optional[ReviewThreadAnchor] = None
    body: Optional[str] = None
    depends_on_action_id: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION


@dataclass(frozen=True)
class ReviewThreadActionPlan:
    expected_head_sha: str
    actions: tuple[ReviewThreadAction, ...]
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.expected_head_sha:
            raise ValueError("expected_head_sha is required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for action in data["actions"]:
            action["kind"] = action["kind"].value
        return data


class ReviewThreadActionState(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    STALE_HEAD = "stale_head"
    FAILED = "failed"
    NOT_EXECUTED = "not_executed"
    SKIPPED = "skipped"
    FALLBACK_REQUIRED = "fallback_required"


class ReviewThreadFailureKind(str, Enum):
    INVALID_INLINE_LOCATION = "invalid_inline_location"
    PERMISSION_DENIED = "permission_denied"
    PROVIDER_FAILURE = "provider_failure"


@dataclass(frozen=True)
class ReviewThreadActionOutcome:
    kind: ReviewThreadActionKind
    state: ReviewThreadActionState
    expected_head_sha: str
    current_head_sha: Optional[str]
    thread_id: Optional[str] = None
    comment_id: Optional[int] = None
    comment_node_id: Optional[str] = None
    failure_kind: Optional[ReviewThreadFailureKind] = None
    reason: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        return self.state in {
            ReviewThreadActionState.APPLIED,
            ReviewThreadActionState.ALREADY_APPLIED,
        }


class SummaryFallbackReason(str, Enum):
    INVALID_INLINE_LOCATION = "invalid_inline_location"
    INLINE_REJECTED = "inline_rejected"
    PERMISSION_DENIED = "permission_denied"
    PROVIDER_FAILURE = "provider_failure"


@dataclass(frozen=True)
class SummaryFallbackEntry:
    finding_id: str
    body: str
    reason: SummaryFallbackReason
    details: Optional[str] = None
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def marker(self) -> str:
        return build_summary_fallback_marker(self.finding_id)

    @property
    def rendered_body(self) -> str:
        reason = self.reason.value.replace("_", " ")
        return f"{self.body}\n\n> **PR-Agent inline fallback:** {reason}.\n\n{self.marker}"


@dataclass(frozen=True)
class ReviewThreadReconciliationOutcome:
    expected_head_sha: str
    current_head_sha: Optional[str]
    action_outcomes: tuple[ReviewThreadActionOutcome, ...]
    summary_fallbacks: tuple[SummaryFallbackEntry, ...] = field(default_factory=tuple)
    schema_version: str = REVIEW_THREAD_LIFECYCLE_SCHEMA_VERSION

    @property
    def counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in ReviewThreadActionState}
        for outcome in self.action_outcomes:
            counts[outcome.state.value] += 1
        return counts

    @property
    def action_counts(self) -> dict[str, int]:
        counts = {kind.value: 0 for kind in ReviewThreadActionKind}
        for outcome in self.action_outcomes:
            counts[outcome.kind.value] += 1
        return counts

    @property
    def metrics(self) -> dict[str, dict[str, int]]:
        return {
            "actions": self.action_counts,
            "action_states": {
                f"{kind.value}.{state.value}": sum(
                    1 for outcome in self.action_outcomes if outcome.kind == kind and outcome.state == state
                )
                for kind in ReviewThreadActionKind
                for state in ReviewThreadActionState
            },
            "states": self.counts,
            "summary_fallbacks": {
                reason.value: sum(1 for entry in self.summary_fallbacks if entry.reason == reason)
                for reason in SummaryFallbackReason
            },
        }

    @property
    def complete(self) -> bool:
        return all(outcome.succeeded for outcome in self.action_outcomes)


class ReviewThreadMutationProvider(Protocol):
    """Provider surface required by the disabled lifecycle executor."""

    def create_review_thread(self, comment: dict, expected_head_sha: str) -> ReviewThreadActionOutcome: ...

    def update_review_thread(
        self,
        comment_id: int,
        body: str,
        expected_head_sha: str,
    ) -> ReviewThreadActionOutcome: ...

    def resolve_review_thread(self, thread_id: str, expected_head_sha: str) -> ReviewThreadActionOutcome: ...


def body_with_fixed_thread_notice(body: str) -> str:
    """Add the visible fixed state once while preserving identity markers."""
    body = (body or "").rstrip()
    if FIXED_THREAD_STATE_MARKER in body:
        return body
    return f"{body}\n\n{FIXED_THREAD_NOTICE}\n\n{FIXED_THREAD_STATE_MARKER}"


def deduplicate_summary_fallbacks(
    entries: tuple[SummaryFallbackEntry, ...],
    existing_summary_bodies: tuple[str, ...] = (),
) -> tuple[SummaryFallbackEntry, ...]:
    """Return one not-yet-published fallback entry per logical finding."""
    seen = {finding_id for body in existing_summary_bodies for _, finding_id in summary_fallback_markers(body)}
    pending = []
    for entry in entries:
        if entry.finding_id in seen:
            continue
        pending.append(entry)
        seen.add(entry.finding_id)
    return tuple(pending)


def _action_id(kind: ReviewThreadActionKind, finding_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{kind.value}|{finding_id}|{ordinal}".encode("utf-8")).hexdigest()[:16]
    return f"thread-action:{digest}"


def plan_review_thread_actions(
    desired_threads: tuple[DesiredReviewThread, ...],
    existing_threads: tuple[ReviewThreadSnapshot, ...],
    expected_head_sha: str,
    *,
    obsolete_policy: str = "keep",
    authoritative_absence: bool = False,
) -> ReviewThreadActionPlan:
    """Plan lifecycle actions without calling a provider.

    Existing resolved threads and any thread with replies are never mutated. For
    moved findings, creation is planned before resolution; the resolve action is
    explicitly dependent on successful creation so execution cannot silently
    discard the finding.
    """
    if obsolete_policy not in {"keep", "resolve", "mark_fixed"}:
        raise ValueError("obsolete_policy must be 'keep', 'resolve', or 'mark_fixed'")
    desired_by_id = {desired.identity.finding_id: desired for desired in desired_threads}
    if len(desired_by_id) != len(desired_threads):
        raise ValueError("desired findings must have unique identities")
    existing_by_id: dict[str, list[ReviewThreadSnapshot]] = {}
    for thread in existing_threads:
        if thread.finding_id:
            existing_by_id.setdefault(thread.finding_id, []).append(thread)

    actions: list[ReviewThreadAction] = []

    def add(kind: ReviewThreadActionKind, finding_id: str, reason: str, **kwargs) -> ReviewThreadAction:
        action = ReviewThreadAction(
            action_id=_action_id(kind, finding_id, len(actions)),
            kind=kind,
            finding_id=finding_id,
            expected_head_sha=expected_head_sha,
            reason=reason,
            **kwargs,
        )
        actions.append(action)
        return action

    for finding_id, desired in desired_by_id.items():
        matches = existing_by_id.get(finding_id, [])
        if desired.anchor is None:
            add(
                ReviewThreadActionKind.SUMMARY_FALLBACK,
                finding_id,
                "invalid_inline_location",
                body=desired.marked_body,
            )
            continue
        if not matches:
            add(
                ReviewThreadActionKind.CREATE,
                finding_id,
                "new_finding",
                anchor=desired.anchor,
                body=desired.marked_body,
            )
            continue
        active_matches = [thread for thread in matches if not thread.is_resolved]
        if not active_matches:
            add(ReviewThreadActionKind.SKIP, finding_id, "existing_thread_resolved", thread_id=matches[-1].thread_id)
            continue
        if len(active_matches) > 1:
            for duplicate in active_matches:
                add(
                    ReviewThreadActionKind.SKIP,
                    finding_id,
                    "duplicate_existing_identity_requires_manual_audit",
                    thread_id=duplicate.thread_id,
                )
            continue
        current = active_matches[0]
        if not current.bot_owned or current.has_replies:
            add(ReviewThreadActionKind.SKIP, finding_id, "thread_not_safe_to_mutate", thread_id=current.thread_id)
            continue
        root = current.root_comment
        if not current.is_outdated and current.anchor == desired.anchor:
            old_body = strip_inline_comment_markers(root.body if root else "").strip()
            new_body = strip_inline_comment_markers(desired.body).strip()
            if old_body == new_body:
                add(
                    ReviewThreadActionKind.UNCHANGED,
                    finding_id,
                    "same_anchor_and_body",
                    thread_id=current.thread_id,
                    root_comment_id=root.database_id if root else None,
                )
            else:
                add(
                    ReviewThreadActionKind.UPDATE,
                    finding_id,
                    "same_anchor_changed_body",
                    thread_id=current.thread_id,
                    root_comment_id=root.database_id if root else None,
                    body=desired.marked_body,
                )
        else:
            if not current.viewer_can_resolve:
                add(
                    ReviewThreadActionKind.SKIP,
                    finding_id,
                    "thread_cannot_be_resolved_safely",
                    thread_id=current.thread_id,
                )
                continue
            move_reason = (
                "outdated_or_deleted_anchor" if current.is_outdated or current.anchor is None else "finding_moved"
            )
            create = add(
                ReviewThreadActionKind.CREATE, finding_id, move_reason, anchor=desired.anchor, body=desired.marked_body
            )
            add(
                ReviewThreadActionKind.RESOLVE,
                finding_id,
                "superseded_by_replacement_finding",
                thread_id=current.thread_id,
                root_comment_id=root.database_id if root else None,
                depends_on_action_id=create.action_id,
            )
    for finding_id, matches in existing_by_id.items():
        if finding_id in desired_by_id:
            continue
        for thread in matches:
            root = thread.root_comment
            safe_to_mutate = (
                thread.bot_owned and not thread.has_replies and not thread.is_resolved and thread.viewer_can_resolve
            )
            if obsolete_policy != "keep" and not authoritative_absence:
                add(
                    ReviewThreadActionKind.SKIP,
                    finding_id,
                    "absence_not_authoritative",
                    thread_id=thread.thread_id,
                )
            elif obsolete_policy == "resolve" and safe_to_mutate:
                add(
                    ReviewThreadActionKind.RESOLVE,
                    finding_id,
                    "finding_no_longer_present",
                    thread_id=thread.thread_id,
                    root_comment_id=root.database_id if root else None,
                )
            elif obsolete_policy == "mark_fixed" and safe_to_mutate and root and root.database_id:
                marked_body = body_with_fixed_thread_notice(root.body)
                if marked_body == root.body.rstrip():
                    add(
                        ReviewThreadActionKind.RESOLVE,
                        finding_id,
                        "visible_fixed_state_already_present",
                        thread_id=thread.thread_id,
                        root_comment_id=root.database_id,
                    )
                else:
                    update = add(
                        ReviewThreadActionKind.UPDATE,
                        finding_id,
                        "mark_finding_fixed_or_obsolete",
                        thread_id=thread.thread_id,
                        root_comment_id=root.database_id,
                        body=marked_body,
                    )
                    add(
                        ReviewThreadActionKind.RESOLVE,
                        finding_id,
                        "resolve_after_visible_fixed_state",
                        thread_id=thread.thread_id,
                        root_comment_id=root.database_id,
                        depends_on_action_id=update.action_id,
                    )
            else:
                add(ReviewThreadActionKind.SKIP, finding_id, "obsolete_thread_preserved", thread_id=thread.thread_id)

    return ReviewThreadActionPlan(expected_head_sha=expected_head_sha, actions=tuple(actions))


def _fallback_reason(failure_kind: Optional[ReviewThreadFailureKind]) -> SummaryFallbackReason:
    if failure_kind == ReviewThreadFailureKind.INVALID_INLINE_LOCATION:
        return SummaryFallbackReason.INLINE_REJECTED
    if failure_kind == ReviewThreadFailureKind.PERMISSION_DENIED:
        return SummaryFallbackReason.PERMISSION_DENIED
    return SummaryFallbackReason.PROVIDER_FAILURE


def execute_review_thread_action_plan(
    plan: ReviewThreadActionPlan,
    provider: ReviewThreadMutationProvider,
    *,
    existing_summary_bodies: tuple[str, ...] = (),
) -> ReviewThreadReconciliationOutcome:
    """Execute a precomputed plan without publishing its summary fallbacks.

    Callers remain responsible for publishing returned ``summary_fallbacks`` as
    one de-duplicated PR summary. Production publication intentionally remains
    disconnected until issues #9 and #27 provide their upstream contracts.
    """
    outcomes: list[ReviewThreadActionOutcome] = []
    outcomes_by_action_id: dict[str, ReviewThreadActionOutcome] = {}
    fallbacks: list[SummaryFallbackEntry] = []
    current_head_sha: Optional[str] = plan.expected_head_sha

    def local_outcome(
        action: ReviewThreadAction,
        state: ReviewThreadActionState,
        reason: str,
        *,
        failure_kind: Optional[ReviewThreadFailureKind] = None,
    ) -> ReviewThreadActionOutcome:
        return ReviewThreadActionOutcome(
            kind=action.kind,
            state=state,
            expected_head_sha=action.expected_head_sha,
            current_head_sha=current_head_sha,
            thread_id=action.thread_id,
            comment_id=action.root_comment_id,
            failure_kind=failure_kind,
            reason=reason,
        )

    def add_fallback(
        action: ReviewThreadAction,
        reason: SummaryFallbackReason,
        details: Optional[str] = None,
    ) -> None:
        body = action.body or (
            "PR-Agent could not update the existing finding thread. "
            "The thread was left unchanged for a maintainer to review."
        )
        fallbacks.append(SummaryFallbackEntry(action.finding_id, body, reason, details))

    for action in plan.actions:
        dependency = outcomes_by_action_id.get(action.depends_on_action_id) if action.depends_on_action_id else None
        if action.depends_on_action_id and (dependency is None or not dependency.succeeded):
            outcome = local_outcome(action, ReviewThreadActionState.NOT_EXECUTED, "dependency_not_satisfied")
        elif action.kind == ReviewThreadActionKind.UNCHANGED:
            outcome = local_outcome(action, ReviewThreadActionState.ALREADY_APPLIED, action.reason)
        elif action.kind == ReviewThreadActionKind.SKIP:
            outcome = local_outcome(action, ReviewThreadActionState.SKIPPED, action.reason)
        elif action.kind == ReviewThreadActionKind.SUMMARY_FALLBACK:
            outcome = local_outcome(action, ReviewThreadActionState.FALLBACK_REQUIRED, action.reason)
            add_fallback(action, SummaryFallbackReason.INVALID_INLINE_LOCATION)
        elif action.kind == ReviewThreadActionKind.CREATE and action.anchor and action.body:
            outcome = provider.create_review_thread(
                action.anchor.to_github_comment(action.body), action.expected_head_sha
            )
        elif action.kind == ReviewThreadActionKind.UPDATE and action.root_comment_id and action.body:
            outcome = provider.update_review_thread(action.root_comment_id, action.body, action.expected_head_sha)
        elif action.kind == ReviewThreadActionKind.RESOLVE and action.thread_id:
            outcome = provider.resolve_review_thread(action.thread_id, action.expected_head_sha)
        else:
            outcome = local_outcome(
                action,
                ReviewThreadActionState.FAILED,
                "malformed_review_thread_action",
                failure_kind=ReviewThreadFailureKind.PROVIDER_FAILURE,
            )

        if outcome.current_head_sha is not None:
            current_head_sha = outcome.current_head_sha
        if outcome.state == ReviewThreadActionState.FAILED:
            add_fallback(action, _fallback_reason(outcome.failure_kind), outcome.reason)
        outcomes.append(outcome)
        outcomes_by_action_id[action.action_id] = outcome

    return ReviewThreadReconciliationOutcome(
        expected_head_sha=plan.expected_head_sha,
        current_head_sha=current_head_sha,
        action_outcomes=tuple(outcomes),
        summary_fallbacks=deduplicate_summary_fallbacks(tuple(fallbacks), existing_summary_bodies),
    )
