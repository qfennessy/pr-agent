from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ReviewDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class RequestedReviewDepth(str, Enum):
    AUTO = "auto"
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ReviewOutputProfile(str, Enum):
    FULL = "full"
    BUGS_ONLY = "bugs_only"


class ChangeKind(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ReviewBudgetPolicy:
    """Profile-specific limits; ``None`` preserves the current PR-Agent setting."""

    context_tokens: int | None = None
    max_findings: int | None = None
    max_verification_candidates: int | None = None
    model_route: str | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    max_output_tokens: int | None = None
    max_published_findings: int | None = None
    publication_threshold: str | None = None
    shadow_only: bool | None = None


@dataclass(frozen=True, slots=True)
class SensitiveCategory:
    name: str
    path_patterns: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_patterns", _freeze_tuple(self.path_patterns))
        object.__setattr__(self, "labels", _freeze_tuple(self.labels))


DEFAULT_DOC_PATTERNS = (
    "*.md",
    "*.mdx",
    "**/*.md",
    "**/*.mdx",
    "docs/**",
    "**/docs/**",
    "README*",
    "**/README*",
    "CHANGELOG*",
    "**/CHANGELOG*",
    "LICENSE*",
    "**/LICENSE*",
)

DEFAULT_TEST_PATTERNS = (
    "test/**",
    "tests/**",
    "**/test/**",
    "**/tests/**",
    "test_*.py",
    "**/test_*.py",
    "*_test.*",
    "**/*_test.*",
    "*.test.*",
    "**/*.test.*",
    "*.spec.*",
    "**/*.spec.*",
)

DEFAULT_GENERATED_PATTERNS = (
    "generated/**",
    "**/generated/**",
    "*.generated.*",
    "**/*.generated.*",
    "*.min.js",
    "**/*.min.js",
    "*.map",
    "**/*.map",
    "dist/**",
    "build/**",
)

DEFAULT_DEPENDENCY_PATTERNS = (
    "requirements*.txt",
    "**/requirements*.txt",
    "pyproject.toml",
    "**/pyproject.toml",
    "poetry.lock",
    "**/poetry.lock",
    "uv.lock",
    "**/uv.lock",
    "package.json",
    "**/package.json",
    "package-lock.json",
    "**/package-lock.json",
    "pnpm-lock.yaml",
    "**/pnpm-lock.yaml",
    "yarn.lock",
    "**/yarn.lock",
    "go.mod",
    "**/go.mod",
    "go.sum",
    "**/go.sum",
    "Cargo.toml",
    "**/Cargo.toml",
    "Cargo.lock",
    "**/Cargo.lock",
    "Gemfile",
    "**/Gemfile",
    "Gemfile.lock",
    "**/Gemfile.lock",
    "composer.json",
    "**/composer.json",
    "composer.lock",
    "**/composer.lock",
)


@dataclass(frozen=True, slots=True)
class ReviewRouterPolicy:
    version: str = "review-router-v1"
    quick: ReviewBudgetPolicy = ReviewBudgetPolicy()
    standard: ReviewBudgetPolicy = ReviewBudgetPolicy()
    deep: ReviewBudgetPolicy = ReviewBudgetPolicy()
    sensitive_categories: tuple[SensitiveCategory, ...] = ()
    large_change_files: int = 25
    large_change_lines: int = 1_000
    docs_patterns: tuple[str, ...] = DEFAULT_DOC_PATTERNS
    test_patterns: tuple[str, ...] = DEFAULT_TEST_PATTERNS
    generated_patterns: tuple[str, ...] = DEFAULT_GENERATED_PATTERNS
    dependency_patterns: tuple[str, ...] = DEFAULT_DEPENDENCY_PATTERNS

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensitive_categories", _freeze_tuple(self.sensitive_categories))
        object.__setattr__(self, "docs_patterns", _freeze_tuple(self.docs_patterns))
        object.__setattr__(self, "test_patterns", _freeze_tuple(self.test_patterns))
        object.__setattr__(self, "generated_patterns", _freeze_tuple(self.generated_patterns))
        object.__setattr__(self, "dependency_patterns", _freeze_tuple(self.dependency_patterns))


@dataclass(frozen=True, slots=True)
class ChangedFile:
    new_path: str | None
    old_path: str | None = None
    kind: ChangeKind | str = ChangeKind.MODIFIED
    additions: int | None = None
    deletions: int | None = None
    generated: bool | None = None


@dataclass(frozen=True, slots=True)
class ReviewDepthEscalation:
    """An immutable minimum-depth input; consumers cannot use it to down-route."""

    source: str
    minimum_depth: ReviewDepth | str | None
    reasons: tuple[str, ...]
    available: bool = True
    uncertain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", _freeze_tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class ReviewRouteRequest:
    files: tuple[ChangedFile, ...]
    requested_depth: RequestedReviewDepth | str = RequestedReviewDepth.AUTO
    review_profile: ReviewOutputProfile | str = ReviewOutputProfile.FULL
    labels: tuple[str, ...] | None = None
    escalation: ReviewDepthEscalation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", _freeze_tuple(self.files))
        if self.labels is not None:
            object.__setattr__(self, "labels", _freeze_tuple(self.labels))


@dataclass(frozen=True, slots=True)
class RoutingReason:
    code: str
    message: str
    minimum_depth: ReviewDepth
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _freeze_tuple(self.evidence))


@dataclass(frozen=True, slots=True)
class ReviewRouteDecision:
    requested_depth: str
    applied_depth: ReviewDepth
    review_profile: str
    requested_budget: ReviewBudgetPolicy | None
    applied_budget: ReviewBudgetPolicy
    reasons: tuple[RoutingReason, ...]
    matched_sensitive_categories: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    policy_version: str
    policy_valid: bool
    policy_errors: tuple[str, ...]
    routing_enabled: bool
    escalation_applied: bool


_DEPTH_RANK = {
    ReviewDepth.QUICK: 0,
    ReviewDepth.STANDARD: 1,
    ReviewDepth.DEEP: 2,
}

_DEFAULT_INHERITED_BUDGET = ReviewBudgetPolicy()


def route_review(request: ReviewRouteRequest, policy: ReviewRouterPolicy | None) -> ReviewRouteDecision:
    """Select a review depth from immutable, provider-neutral evidence."""

    if not isinstance(request, ReviewRouteRequest):
        return _invalid_request_decision(request, policy)

    requested_depth, requested_error = _requested_depth(request.requested_depth)
    review_profile, review_profile_error = _review_profile_value(request.review_profile)

    if policy is None:
        return _route_without_policy(
            request,
            requested_depth,
            requested_error,
            review_profile,
            review_profile_error,
        )

    policy_errors = _validate_policy(policy)
    if policy_errors:
        reasons = []
        if requested_error:
            reasons.append(_invalid_requested_depth_reason(requested_error))
        if review_profile_error:
            reasons.append(_invalid_review_profile_reason(review_profile_error))
        reasons.append(RoutingReason(
            code="policy_invalid",
            message="Routing policy is invalid; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=policy_errors,
        ))
        return ReviewRouteDecision(
            requested_depth=_raw_value(request.requested_depth),
            applied_depth=ReviewDepth.DEEP,
            review_profile=review_profile,
            requested_budget=None,
            applied_budget=_DEFAULT_INHERITED_BUDGET,
            reasons=tuple(reasons),
            matched_sensitive_categories=(),
            missing_inputs=(),
            policy_version=(
                policy.version
                if isinstance(policy, ReviewRouterPolicy) and isinstance(policy.version, str)
                else "invalid"
            ),
            policy_valid=False,
            policy_errors=policy_errors,
            routing_enabled=True,
            escalation_applied=False,
        )

    reasons: list[RoutingReason] = []
    requested_budget = None
    if requested_error:
        depth = ReviewDepth.DEEP
        reasons.append(_invalid_requested_depth_reason(requested_error))
    elif requested_depth == RequestedReviewDepth.AUTO:
        depth = ReviewDepth.QUICK
        reasons.append(RoutingReason(
            code="requested_auto",
            message="Automatic deterministic routing was requested.",
            minimum_depth=ReviewDepth.QUICK,
        ))
    else:
        depth = ReviewDepth(requested_depth.value)
        requested_budget = _budget_for_depth(policy, depth)
        reasons.append(RoutingReason(
            code=f"requested_{depth.value}",
            message=f"The caller explicitly requested {depth.value} review.",
            minimum_depth=depth,
        ))

    if review_profile_error:
        depth = ReviewDepth.DEEP
        reasons.append(_invalid_review_profile_reason(review_profile_error))

    evidence = _collect_file_evidence(request, policy)
    if evidence.input_errors:
        depth = _deeper(depth, ReviewDepth.DEEP)
        reasons.append(RoutingReason(
            code="input_invalid",
            message="Malformed routing evidence cannot be classified safely; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=evidence.input_errors,
        ))

    matched_categories = []
    for category, category_evidence in evidence.sensitive_matches:
        depth = ReviewDepth.DEEP
        matched_categories.append(category)
        reasons.append(RoutingReason(
            code=f"sensitive_category:{category}",
            message=f"Sensitive category '{category}' forces deep review.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=category_evidence,
        ))

    if evidence.file_count >= policy.large_change_files:
        depth = ReviewDepth.DEEP
        reasons.append(RoutingReason(
            code="large_change:files",
            message="The changed-file count reached the configured deep-review threshold.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=(f"files={evidence.file_count}", f"threshold={policy.large_change_files}"),
        ))

    if evidence.total_lines >= policy.large_change_lines:
        depth = ReviewDepth.DEEP
        reasons.append(RoutingReason(
            code="large_change:lines",
            message="The changed-line count reached the configured deep-review threshold.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=(f"lines={evidence.total_lines}", f"threshold={policy.large_change_lines}"),
        ))

    depth, escalation_reason, escalation_applied, escalation_missing = _apply_escalation(
        depth, request.escalation
    )
    if escalation_reason:
        reasons.append(escalation_reason)

    if evidence.dependency_paths:
        depth = _deeper(depth, ReviewDepth.STANDARD)
        reasons.append(RoutingReason(
            code="dependency_change",
            message="Dependency metadata changes require at least standard review.",
            minimum_depth=ReviewDepth.STANDARD,
            evidence=evidence.dependency_paths,
        ))

    missing_inputs = list(evidence.missing_inputs)
    if escalation_missing:
        missing_inputs.append(escalation_missing)
    missing_inputs = list(_ordered_unique(missing_inputs))
    if missing_inputs:
        depth = _deeper(depth, ReviewDepth.STANDARD)
        reasons.append(RoutingReason(
            code="inputs_missing",
            message="Incomplete metadata prevents a low-risk route; at least standard review was applied.",
            minimum_depth=ReviewDepth.STANDARD,
            evidence=tuple(missing_inputs),
        ))

    low_risk_reasons = []
    if evidence.docs_only:
        low_risk_reasons.append(RoutingReason(
            code="docs_only",
            message="Every changed path is documentation.",
            minimum_depth=ReviewDepth.QUICK,
            evidence=evidence.all_paths,
        ))
    if evidence.tests_only:
        low_risk_reasons.append(RoutingReason(
            code="tests_only",
            message="Every changed path is a test.",
            minimum_depth=ReviewDepth.QUICK,
            evidence=evidence.all_paths,
        ))
    if evidence.generated_only:
        low_risk_reasons.append(RoutingReason(
            code="generated_only",
            message="Every changed file is marked or classified as generated.",
            minimum_depth=ReviewDepth.QUICK,
            evidence=evidence.all_paths,
        ))
    reasons.extend(low_risk_reasons)

    if (
        requested_depth == RequestedReviewDepth.AUTO
        and not requested_error
        and not low_risk_reasons
        and not evidence.sensitive_matches
        and evidence.file_count < policy.large_change_files
        and evidence.total_lines < policy.large_change_lines
    ):
        depth = _deeper(depth, ReviewDepth.STANDARD)
        reasons.append(RoutingReason(
            code="default_standard",
            message="No complete low-risk-only signal matched; standard review was applied.",
            minimum_depth=ReviewDepth.STANDARD,
        ))

    applied_budget = _budget_for_depth(policy, depth)
    return ReviewRouteDecision(
        requested_depth=_raw_value(request.requested_depth),
        applied_depth=depth,
        review_profile=review_profile,
        requested_budget=requested_budget,
        applied_budget=applied_budget,
        reasons=tuple(reasons),
        matched_sensitive_categories=tuple(matched_categories),
        missing_inputs=tuple(missing_inputs),
        policy_version=policy.version,
        policy_valid=True,
        policy_errors=(),
        routing_enabled=True,
        escalation_applied=escalation_applied,
    )


def _invalid_request_decision(request: Any, policy: Any) -> ReviewRouteDecision:
    reasons = [RoutingReason(
        code="request_invalid",
        message="Routing input is malformed; deep review was applied.",
        minimum_depth=ReviewDepth.DEEP,
        evidence=(f"type:{type(request).__name__}",),
    )]
    missing_inputs = ["request"]
    routing_enabled = policy is not None
    policy_errors = () if policy is None else _validate_policy(policy)
    if policy is None:
        policy_version = "unconfigured"
        applied_budget = _DEFAULT_INHERITED_BUDGET
        missing_inputs.append("routing_policy")
        reasons.append(RoutingReason(
            code="policy_missing",
            message="Depth routing is not configured; inherited review budgets were preserved.",
            minimum_depth=ReviewDepth.STANDARD,
            evidence=("routing_policy",),
        ))
    elif policy_errors:
        policy_version = (
            policy.version
            if isinstance(policy, ReviewRouterPolicy) and isinstance(policy.version, str)
            else "invalid"
        )
        applied_budget = _DEFAULT_INHERITED_BUDGET
        reasons.append(RoutingReason(
            code="policy_invalid",
            message="Routing policy is also invalid; inherited review budgets were preserved.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=policy_errors,
        ))
    else:
        policy_version = policy.version
        applied_budget = policy.deep

    return ReviewRouteDecision(
        requested_depth="invalid",
        applied_depth=ReviewDepth.DEEP,
        review_profile=ReviewOutputProfile.FULL.value,
        requested_budget=None,
        applied_budget=applied_budget,
        reasons=tuple(reasons),
        matched_sensitive_categories=(),
        missing_inputs=tuple(missing_inputs),
        policy_version=policy_version,
        policy_valid=not policy_errors,
        policy_errors=policy_errors,
        routing_enabled=routing_enabled,
        escalation_applied=False,
    )


@dataclass(frozen=True, slots=True)
class _FileEvidence:
    file_count: int
    total_lines: int
    all_paths: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    sensitive_matches: tuple[tuple[str, tuple[str, ...]], ...]
    missing_inputs: tuple[str, ...]
    input_errors: tuple[str, ...]
    docs_only: bool
    tests_only: bool
    generated_only: bool


def _collect_file_evidence(request: ReviewRouteRequest, policy: ReviewRouterPolicy) -> _FileEvidence:
    missing_inputs = []
    input_errors = []
    paths_by_file: list[tuple[str, ...]] = []
    all_paths = []
    total_lines = 0
    generated_files = []

    if not isinstance(request.files, tuple):
        input_errors.append("files")
        files = ()
    else:
        files = request.files

    if not files:
        missing_inputs.append("changed_files")

    for index, changed_file in enumerate(files):
        if not isinstance(changed_file, ChangedFile):
            input_errors.append(f"files[{index}]")
            paths_by_file.append(())
            generated_files.append(False)
            continue

        kind = _change_kind(changed_file.kind)
        if kind is None:
            input_errors.append(f"files[{index}].kind")
            kind = ChangeKind.UNKNOWN
        elif kind == ChangeKind.UNKNOWN:
            missing_inputs.append(f"files[{index}].kind")

        old_path, old_error = _normalize_path(changed_file.old_path)
        new_path, new_error = _normalize_path(changed_file.new_path)
        if old_error:
            input_errors.append(f"files[{index}].old_path")
        if new_error:
            input_errors.append(f"files[{index}].new_path")

        if kind == ChangeKind.ADDED and new_path is None:
            missing_inputs.append(f"files[{index}].new_path")
        elif kind == ChangeKind.DELETED and old_path is None:
            missing_inputs.append(f"files[{index}].old_path")
        elif kind == ChangeKind.RENAMED:
            if old_path is None:
                missing_inputs.append(f"files[{index}].old_path")
            if new_path is None:
                missing_inputs.append(f"files[{index}].new_path")
        elif kind in {ChangeKind.MODIFIED, ChangeKind.UNKNOWN} and new_path is None and old_path is None:
            missing_inputs.append(f"files[{index}].path")

        file_paths = _ordered_unique(path for path in (old_path, new_path) if path)
        paths_by_file.append(file_paths)
        all_paths.extend(file_paths)

        line_count = 0
        counts_valid = True
        for field_name, count in (("additions", changed_file.additions), ("deletions", changed_file.deletions)):
            if count is None:
                missing_inputs.append(f"files[{index}].line_counts")
                counts_valid = False
                break
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                input_errors.append(f"files[{index}].{field_name}")
                counts_valid = False
                break
            line_count += count
        if counts_valid:
            total_lines += line_count

        if changed_file.generated is not None and not isinstance(changed_file.generated, bool):
            input_errors.append(f"files[{index}].generated")
            generated = False
        else:
            generated = changed_file.generated is True or _all_paths_match(file_paths, policy.generated_patterns)
        generated_files.append(generated)

    if request.labels is None:
        labels = ()
        missing_inputs.append("labels")
    elif not isinstance(request.labels, tuple):
        labels = ()
        input_errors.append("labels")
    else:
        labels = []
        for index, label in enumerate(request.labels):
            if not isinstance(label, str) or not label.strip():
                input_errors.append(f"labels[{index}]")
                continue
            labels.append(label.strip().casefold())
        labels = _ordered_unique(labels)

    sensitive_matches = []
    for category in policy.sensitive_categories:
        category_evidence = []
        for path in _ordered_unique(all_paths):
            if _path_matches_any(path, category.path_patterns):
                category_evidence.append(f"path:{path}")
        configured_labels = {label.casefold() for label in category.labels}
        for label in labels:
            if label in configured_labels:
                category_evidence.append(f"label:{label}")
        if category_evidence:
            sensitive_matches.append((category.name, tuple(category_evidence)))

    unique_paths = _ordered_unique(all_paths)
    dependency_paths = tuple(path for path in unique_paths if _path_matches_any(path, policy.dependency_patterns))
    has_files_with_paths = bool(paths_by_file) and all(paths_by_file)

    return _FileEvidence(
        file_count=len(files),
        total_lines=total_lines,
        all_paths=unique_paths,
        dependency_paths=dependency_paths,
        sensitive_matches=tuple(sensitive_matches),
        missing_inputs=_ordered_unique(missing_inputs),
        input_errors=_ordered_unique(input_errors),
        docs_only=(
            has_files_with_paths
            and all(_all_paths_match(paths, policy.docs_patterns) for paths in paths_by_file)
        ),
        tests_only=(
            has_files_with_paths
            and all(_all_paths_match(paths, policy.test_patterns) for paths in paths_by_file)
        ),
        generated_only=bool(generated_files) and all(generated_files),
    )


def _route_without_policy(
    request: ReviewRouteRequest,
    requested_depth: RequestedReviewDepth | None,
    requested_error: str | None,
    review_profile: str,
    review_profile_error: str | None,
) -> ReviewRouteDecision:
    reasons = []
    if requested_error:
        depth = ReviewDepth.DEEP
        reasons.append(_invalid_requested_depth_reason(requested_error))
    elif requested_depth == RequestedReviewDepth.DEEP:
        depth = ReviewDepth.DEEP
        reasons.append(RoutingReason(
            code="requested_deep",
            message="The caller explicitly requested deep review.",
            minimum_depth=ReviewDepth.DEEP,
        ))
    else:
        depth = ReviewDepth.STANDARD

    if review_profile_error:
        depth = ReviewDepth.DEEP
        reasons.append(_invalid_review_profile_reason(review_profile_error))

    reasons.append(RoutingReason(
        code="policy_missing",
        message="Depth routing is not configured; the existing standard review behavior was preserved.",
        minimum_depth=ReviewDepth.STANDARD,
        evidence=("routing_policy",),
    ))

    depth, escalation_reason, escalation_applied, escalation_missing = _apply_escalation(depth, request.escalation)
    if escalation_reason:
        reasons.append(escalation_reason)
    missing_inputs = ["routing_policy"]
    if escalation_missing:
        missing_inputs.append(escalation_missing)

    return ReviewRouteDecision(
        requested_depth=_raw_value(request.requested_depth),
        applied_depth=depth,
        review_profile=review_profile,
        requested_budget=None,
        applied_budget=_DEFAULT_INHERITED_BUDGET,
        reasons=tuple(reasons),
        matched_sensitive_categories=(),
        missing_inputs=tuple(missing_inputs),
        policy_version="unconfigured",
        policy_valid=True,
        policy_errors=(),
        routing_enabled=False,
        escalation_applied=escalation_applied,
    )


def _apply_escalation(
    current_depth: ReviewDepth,
    escalation: ReviewDepthEscalation | None,
) -> tuple[ReviewDepth, RoutingReason | None, bool, str | None]:
    if escalation is None:
        return current_depth, None, False, None
    if not isinstance(escalation, ReviewDepthEscalation):
        return ReviewDepth.DEEP, RoutingReason(
            code="escalation_invalid",
            message="Malformed escalation evidence cannot be used safely; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
        ), True, None

    source = escalation.source.strip() if isinstance(escalation.source, str) else ""
    if not source:
        return ReviewDepth.DEEP, RoutingReason(
            code="escalation_invalid",
            message="Escalation evidence has no source; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
        ), True, None
    if not isinstance(escalation.available, bool) or not isinstance(escalation.uncertain, bool):
        return ReviewDepth.DEEP, RoutingReason(
            code="escalation_invalid",
            message="Escalation availability metadata is malformed; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=(f"source:{source}",),
        ), True, None
    if not escalation.available:
        depth = _deeper(current_depth, ReviewDepth.STANDARD)
        return depth, RoutingReason(
            code="escalation_unavailable",
            message="Requested escalation evidence is unavailable; at least standard review was applied.",
            minimum_depth=ReviewDepth.STANDARD,
            evidence=(f"source:{source}",),
        ), depth != current_depth, f"escalation:{source}"

    if not isinstance(escalation.reasons, tuple):
        return ReviewDepth.DEEP, RoutingReason(
            code="escalation_invalid",
            message="Escalation reasons are malformed; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=(f"source:{source}", "reasons"),
        ), True, None

    minimum_depth = _review_depth(escalation.minimum_depth)
    if minimum_depth is None:
        return ReviewDepth.DEEP, RoutingReason(
            code="escalation_invalid",
            message="Escalation evidence names an unknown depth; deep review was applied.",
            minimum_depth=ReviewDepth.DEEP,
            evidence=(f"source:{source}",),
        ), True, None

    escalation_reasons = []
    for index, reason in enumerate(escalation.reasons):
        if not isinstance(reason, str) or not reason.strip():
            return ReviewDepth.DEEP, RoutingReason(
                code="escalation_invalid",
                message="Escalation reasons are malformed; deep review was applied.",
                minimum_depth=ReviewDepth.DEEP,
                evidence=(f"source:{source}", f"reasons[{index}]"),
            ), True, None
        escalation_reasons.append(reason.strip())

    floor = _deeper(minimum_depth, ReviewDepth.STANDARD) if escalation.uncertain else minimum_depth
    depth = _deeper(current_depth, floor)
    return depth, RoutingReason(
        code="external_escalation",
        message="External routing evidence supplied a minimum depth and cannot lower deterministic routing.",
        minimum_depth=floor,
        evidence=(f"source:{source}", *escalation_reasons),
    ), depth != current_depth, None


def _validate_policy(policy: Any) -> tuple[str, ...]:
    if not isinstance(policy, ReviewRouterPolicy):
        return ("policy must be a ReviewRouterPolicy",)

    errors = []
    if not isinstance(policy.version, str) or not policy.version.strip():
        errors.append("version must be a non-empty string")

    for depth, budget in (
        (ReviewDepth.QUICK, policy.quick),
        (ReviewDepth.STANDARD, policy.standard),
        (ReviewDepth.DEEP, policy.deep),
    ):
        errors.extend(_validate_budget(depth, budget))

    for field_name, value in (
        ("large_change_files", policy.large_change_files),
        ("large_change_lines", policy.large_change_lines),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{field_name} must be a positive integer")

    for field_name, patterns in (
        ("docs_patterns", policy.docs_patterns),
        ("test_patterns", policy.test_patterns),
        ("generated_patterns", policy.generated_patterns),
        ("dependency_patterns", policy.dependency_patterns),
    ):
        errors.extend(_validate_patterns(field_name, patterns))

    if not isinstance(policy.sensitive_categories, tuple):
        errors.append("sensitive_categories must be a sequence")
    else:
        seen_names = set()
        for index, category in enumerate(policy.sensitive_categories):
            prefix = f"sensitive_categories[{index}]"
            if not isinstance(category, SensitiveCategory):
                errors.append(f"{prefix} must be a SensitiveCategory")
                continue
            if not isinstance(category.name, str) or not category.name.strip():
                errors.append(f"{prefix}.name must be a non-empty string")
            elif category.name != category.name.strip():
                errors.append(f"{prefix}.name must not have surrounding whitespace")
            else:
                normalized_name = category.name.strip().casefold()
                if normalized_name in seen_names:
                    errors.append(f"{prefix}.name duplicates another category")
                seen_names.add(normalized_name)
            errors.extend(_validate_patterns(f"{prefix}.path_patterns", category.path_patterns))
            if not isinstance(category.labels, tuple):
                errors.append(f"{prefix}.labels must be a sequence")
            else:
                for label_index, label in enumerate(category.labels):
                    if not isinstance(label, str) or not label.strip():
                        errors.append(f"{prefix}.labels[{label_index}] must be a non-empty string")
                    elif label != label.strip():
                        errors.append(f"{prefix}.labels[{label_index}] must not have surrounding whitespace")
            if not category.path_patterns and not category.labels:
                errors.append(f"{prefix} must define at least one path pattern or label")

    return tuple(errors)


def _validate_budget(depth: ReviewDepth, budget: Any) -> list[str]:
    prefix = f"{depth.value} budget"
    if not isinstance(budget, ReviewBudgetPolicy):
        return [f"{prefix} must be a ReviewBudgetPolicy"]

    errors = []
    for field_name, value in (
        ("context_tokens", budget.context_tokens),
        ("max_findings", budget.max_findings),
        ("max_verification_candidates", budget.max_verification_candidates),
        ("max_output_tokens", budget.max_output_tokens),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            errors.append(f"{prefix}.{field_name} must be a positive integer or null")

    if budget.timeout_seconds is not None and (
        isinstance(budget.timeout_seconds, bool)
        or not isinstance(budget.timeout_seconds, (int, float))
        or not math.isfinite(budget.timeout_seconds)
        or budget.timeout_seconds <= 0
    ):
        errors.append(f"{prefix}.timeout_seconds must be a positive finite number or null")

    if budget.max_retries is not None and (
        isinstance(budget.max_retries, bool)
        or not isinstance(budget.max_retries, int)
        or budget.max_retries < 0
    ):
        errors.append(f"{prefix}.max_retries must be a non-negative integer or null")

    if budget.max_published_findings is not None and (
        isinstance(budget.max_published_findings, bool)
        or not isinstance(budget.max_published_findings, int)
        or budget.max_published_findings < 0
    ):
        errors.append(f"{prefix}.max_published_findings must be a non-negative integer or null")

    if budget.model_route is not None and (
        not isinstance(budget.model_route, str) or not budget.model_route.strip()
    ):
        errors.append(f"{prefix}.model_route must be a non-empty string or null")

    valid_thresholds = {"none", "low", "medium", "high", "critical"}
    if budget.publication_threshold is not None and (
        not isinstance(budget.publication_threshold, str)
        or budget.publication_threshold.casefold() not in valid_thresholds
    ):
        errors.append(f"{prefix}.publication_threshold must be a supported severity or null")

    if budget.shadow_only is not None and not isinstance(budget.shadow_only, bool):
        errors.append(f"{prefix}.shadow_only must be a boolean or null")
    return errors


def _validate_patterns(field_name: str, patterns: Any) -> list[str]:
    if not isinstance(patterns, tuple):
        return [f"{field_name} must be a sequence"]
    errors = []
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, str) or not pattern.strip():
            errors.append(f"{field_name}[{index}] must be a non-empty string")
            continue
        normalized = pattern.replace("\\", "/")
        if (
            pattern != pattern.strip()
            or "\x00" in normalized
            or normalized.startswith(("/", "./"))
            or ".." in normalized.split("/")
        ):
            errors.append(f"{field_name}[{index}] must be a repository-relative glob")
    return errors


def _budget_for_depth(policy: ReviewRouterPolicy, depth: ReviewDepth) -> ReviewBudgetPolicy:
    return {
        ReviewDepth.QUICK: policy.quick,
        ReviewDepth.STANDARD: policy.standard,
        ReviewDepth.DEEP: policy.deep,
    }[depth]


def _requested_depth(value: RequestedReviewDepth | str) -> tuple[RequestedReviewDepth | None, str | None]:
    try:
        return RequestedReviewDepth(_raw_value(value).casefold()), None
    except (AttributeError, ValueError):
        return None, f"unknown requested depth: {_raw_value(value)}"


def _review_depth(value: ReviewDepth | str | None) -> ReviewDepth | None:
    try:
        return ReviewDepth(_raw_value(value).casefold())
    except (AttributeError, ValueError):
        return None


def _change_kind(value: ChangeKind | str) -> ChangeKind | None:
    try:
        return ChangeKind(_raw_value(value).casefold())
    except (AttributeError, ValueError):
        return None


def _review_profile_value(value: ReviewOutputProfile | str) -> tuple[str, str | None]:
    try:
        profile = ReviewOutputProfile(_raw_value(value).casefold())
        return profile.value, None
    except (AttributeError, ValueError):
        return ReviewOutputProfile.FULL.value, f"unknown review profile: {_raw_value(value)}"


def _raw_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _invalid_requested_depth_reason(error: str) -> RoutingReason:
    return RoutingReason(
        code="requested_depth_invalid",
        message="The requested review depth is unknown; deep review was applied.",
        minimum_depth=ReviewDepth.DEEP,
        evidence=(error,),
    )


def _invalid_review_profile_reason(error: str) -> RoutingReason:
    return RoutingReason(
        code="review_profile_invalid",
        message="The output profile is unknown; full output and deep review were applied.",
        minimum_depth=ReviewDepth.DEEP,
        evidence=(error,),
    )


def _normalize_path(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if not isinstance(value, str) or not value.strip():
        return None, True
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or "\x00" in path or ".." in path.split("/"):
        return None, True
    return path, False


def _all_paths_match(paths: tuple[str, ...], patterns: tuple[str, ...]) -> bool:
    return bool(paths) and all(_path_matches_any(path, patterns) for path in paths)


def _path_matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        normalized_pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatchcase(path, normalized_pattern):
            return True
        if normalized_pattern.startswith("**/") and fnmatch.fnmatchcase(path, normalized_pattern[3:]):
            return True
    return False


def _deeper(left: ReviewDepth, right: ReviewDepth) -> ReviewDepth:
    return left if _DEPTH_RANK[left] >= _DEPTH_RANK[right] else right


def _ordered_unique(values: Any) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _freeze_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return value
