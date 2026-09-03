from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from pr_agent.algo.review_router import (
    ChangedFile,
    ChangeKind,
    RequestedReviewDepth,
    ReviewBudgetPolicy,
    ReviewDepth,
    ReviewDepthEscalation,
    ReviewOutputProfile,
    ReviewRouteRequest,
    ReviewRouterPolicy,
    SensitiveCategory,
    load_review_routing_configuration,
    review_route_decision_to_dict,
    route_review,
)
from pr_agent.config_loader import get_settings


def _file(
    path: str = "src/service.py",
    *,
    old_path: str | None = None,
    kind: ChangeKind = ChangeKind.MODIFIED,
    additions: int | None = 4,
    deletions: int | None = 2,
    generated: bool | None = False,
) -> ChangedFile:
    return ChangedFile(
        new_path=path,
        old_path=old_path,
        kind=kind,
        additions=additions,
        deletions=deletions,
        generated=generated,
    )


def _request(
    *files: ChangedFile,
    requested_depth: RequestedReviewDepth | str = RequestedReviewDepth.AUTO,
    review_profile: ReviewOutputProfile | str = ReviewOutputProfile.FULL,
    labels: tuple[str, ...] | None = (),
    escalation: ReviewDepthEscalation | None = None,
) -> ReviewRouteRequest:
    return ReviewRouteRequest(
        files=files or (_file(),),
        requested_depth=requested_depth,
        review_profile=review_profile,
        labels=labels,
        escalation=escalation,
    )


def _policy(**overrides) -> ReviewRouterPolicy:
    return replace(ReviewRouterPolicy(), **overrides)


def _codes(decision) -> list[str]:
    return [reason.code for reason in decision.reasons]


def _configuration(**overrides):
    section = {
        "enabled": True,
        "requested_depth": "auto",
        "profiles": {
            "quick": {"context_tokens": 8_000, "model_route": "weak"},
            "standard": {"context_tokens": 24_000, "model_route": "regular"},
            "deep": {"context_tokens": 32_000, "model_route": "reasoning"},
        },
    }
    section.update(overrides)
    return section


def test_runtime_configuration_loads_all_named_profiles_and_sensitive_categories():
    configuration = load_review_routing_configuration(_configuration(
        consume_specialist_escalation=True,
        specialist_escalation_depth="deep",
        sensitive_categories=[{
            "name": "authorization",
            "path_patterns": ["**/auth/**"],
            "labels": ["authorization"],
        }],
    ))

    decision = route_review(
        ReviewRouteRequest(
            files=(_file("services/auth/guard.py"),),
            labels=(),
            requested_depth=configuration.requested_depth,
        ),
        configuration.policy,
    )

    assert configuration.enabled is True
    assert configuration.consume_specialist_escalation is True
    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.applied_budget.context_tokens == 32_000
    assert decision.matched_sensitive_categories == ("authorization",)


@pytest.mark.parametrize(
    "override",
    [
        {"profiles": {"quick": {}, "standard": {}, "deep": {}, "unknown": {}}},
        {"profiles": {"quick": {"model_route": "provider-specific"}, "standard": {}, "deep": {}}},
        {"specialist_escalation_depth": "quick"},
        {"unknown_key": True},
    ],
)
def test_runtime_configuration_errors_fail_closed_with_operational_deep_budget(override):
    configuration = load_review_routing_configuration(_configuration(**override))

    decision = route_review(_request(_file("docs/guide.md")), configuration.policy)

    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.policy_valid is False
    assert decision.applied_budget.model_route == "reasoning"
    assert decision.applied_budget.context_tokens == 32_000
    assert decision.policy_errors


def test_missing_or_disabled_runtime_configuration_preserves_legacy_policy():
    assert load_review_routing_configuration(None).enabled is False
    assert load_review_routing_configuration({"enabled": False}).policy is None


def test_runtime_configuration_normalizes_environment_boolean_strings():
    configuration = load_review_routing_configuration(_configuration(
        enabled="true",
        consume_specialist_escalation="true",
        profiles={
            "quick": {"shadow_only": "false"},
            "standard": {"shadow_only": "false"},
            "deep": {"shadow_only": "true"},
        },
    ))
    decision = route_review(
        _request(_file(), requested_depth=RequestedReviewDepth.DEEP),
        configuration.policy,
    )

    assert configuration.consume_specialist_escalation is True
    assert decision.policy_valid is True
    assert decision.applied_budget.shadow_only is True


def test_repository_default_profiles_form_a_valid_enabled_policy():
    section = deepcopy(dict(get_settings().review_depth))
    section["enabled"] = True

    configuration = load_review_routing_configuration(section)
    decision = route_review(_request(_file("docs/guide.md")), configuration.policy)

    assert configuration.enabled is True
    assert decision.policy_valid is True
    assert decision.applied_depth is ReviewDepth.QUICK


def test_route_serialization_preserves_ordered_reasons_and_all_budget_fields():
    decision = route_review(
        _request(_file("package-lock.json")),
        _policy(standard=ReviewBudgetPolicy(context_tokens=12_000, max_findings=4)),
    )

    serialized = review_route_decision_to_dict(decision)

    assert [reason["code"] for reason in serialized["reasons"]] == _codes(decision)
    assert serialized["applied_depth"] == "standard"
    assert serialized["applied_budget"]["context_tokens"] == 12_000
    assert serialized["applied_budget"]["max_findings"] == 4


def test_router_inputs_and_outputs_are_immutable():
    files = [_file("docs/guide.md")]
    request = ReviewRouteRequest(files=files, labels=["documentation"])
    files.append(_file("src/new.py"))

    decision = route_review(request, _policy())

    assert len(request.files) == 1
    assert request.labels == ("documentation",)
    assert decision.applied_depth == ReviewDepth.QUICK
    with pytest.raises(FrozenInstanceError):
        decision.applied_depth = ReviewDepth.DEEP


@pytest.mark.parametrize(
    ("requested_depth", "expected"),
    [
        (RequestedReviewDepth.AUTO, ReviewDepth.STANDARD),
        (RequestedReviewDepth.QUICK, ReviewDepth.STANDARD),
        (RequestedReviewDepth.STANDARD, ReviewDepth.STANDARD),
        (RequestedReviewDepth.DEEP, ReviewDepth.DEEP),
    ],
)
def test_missing_policy_preserves_standard_behavior_and_never_down_routes(requested_depth, expected):
    decision = route_review(_request(_file("docs/guide.md"), requested_depth=requested_depth), None)

    assert decision.applied_depth == expected
    assert decision.routing_enabled is False
    assert decision.policy_valid is True
    assert decision.missing_inputs == ("routing_policy",)
    assert "policy_missing" in _codes(decision)


def test_profile_budgets_record_requested_and_applied_policy():
    quick = ReviewBudgetPolicy(context_tokens=2_000, max_findings=1, model_route="cheap")
    standard = ReviewBudgetPolicy(context_tokens=8_000, max_findings=3, model_route="default")
    deep = ReviewBudgetPolicy(
        context_tokens=32_000,
        max_findings=8,
        max_verification_candidates=8,
        model_route="frontier",
        timeout_seconds=240,
        max_retries=2,
        max_output_tokens=8_000,
        max_published_findings=3,
        publication_threshold="medium",
        shadow_only=False,
    )
    policy = _policy(
        quick=quick,
        standard=standard,
        deep=deep,
        sensitive_categories=(SensitiveCategory("authorization", ("auth/**",)),),
    )

    decision = route_review(
        _request(_file("auth/guard.py"), requested_depth=RequestedReviewDepth.QUICK),
        policy,
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.requested_budget is quick
    assert decision.applied_budget is deep


@pytest.mark.parametrize(
    ("requested_depth", "expected_depth", "budget_field"),
    [
        (RequestedReviewDepth.QUICK, ReviewDepth.QUICK, 1_000),
        (RequestedReviewDepth.STANDARD, ReviewDepth.STANDARD, 8_000),
        (RequestedReviewDepth.DEEP, ReviewDepth.DEEP, 32_000),
    ],
)
def test_explicit_profiles_select_their_own_budget_policy(requested_depth, expected_depth, budget_field):
    policy = _policy(
        quick=ReviewBudgetPolicy(context_tokens=1_000),
        standard=ReviewBudgetPolicy(context_tokens=8_000),
        deep=ReviewBudgetPolicy(context_tokens=32_000),
    )

    decision = route_review(
        _request(_file("src/service.py"), requested_depth=requested_depth),
        policy,
    )

    assert decision.applied_depth == expected_depth
    assert decision.requested_budget.context_tokens == budget_field
    assert decision.applied_budget.context_tokens == budget_field


@pytest.mark.parametrize(
    ("path", "generated", "reason"),
    [
        ("docs/guide.md", False, "docs_only"),
        ("tests/test_service.py", False, "tests_only"),
        ("src/client.generated.ts", False, "generated_only"),
        ("vendor/bundle.data", True, "generated_only"),
    ],
)
def test_auto_routes_complete_low_risk_only_changes_to_quick(path, generated, reason):
    decision = route_review(_request(_file(path, generated=generated)), _policy())

    assert decision.applied_depth == ReviewDepth.QUICK
    assert reason in _codes(decision)
    assert decision.missing_inputs == ()


def test_auto_routes_ordinary_code_to_standard():
    decision = route_review(_request(_file("src/service.py")), _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert _codes(decision) == ["requested_auto", "default_standard"]


@pytest.mark.parametrize("path", ["requirements.txt", "backend/pyproject.toml", "web/pnpm-lock.yaml"])
def test_dependency_changes_require_at_least_standard(path):
    decision = route_review(_request(_file(path)), _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert "dependency_change" in _codes(decision)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [(99, ReviewDepth.QUICK), (100, ReviewDepth.DEEP), (101, ReviewDepth.DEEP)],
)
def test_large_line_threshold_is_inclusive(lines, expected):
    policy = _policy(large_change_lines=100)
    changed_file = _file("src/service.py", additions=lines, deletions=0)

    decision = route_review(
        _request(changed_file, requested_depth=RequestedReviewDepth.QUICK),
        policy,
    )

    assert decision.applied_depth == expected
    assert ("large_change:lines" in _codes(decision)) is (lines >= 100)


@pytest.mark.parametrize(
    ("file_count", "expected"),
    [(2, ReviewDepth.QUICK), (3, ReviewDepth.DEEP), (4, ReviewDepth.DEEP)],
)
def test_large_file_threshold_is_inclusive(file_count, expected):
    policy = _policy(large_change_files=3)
    files = tuple(_file(f"src/file_{index}.py", additions=1, deletions=0) for index in range(file_count))

    decision = route_review(
        _request(*files, requested_depth=RequestedReviewDepth.QUICK),
        policy,
    )

    assert decision.applied_depth == expected
    assert ("large_change:files" in _codes(decision)) is (file_count >= 3)


@pytest.mark.parametrize(
    ("category", "pattern", "path"),
    [
        ("security", "security/**", "security/policy.py"),
        ("authorization", "auth/**", "auth/permissions.py"),
        ("tenant_isolation", "data/tenant/**", "data/tenant/query.py"),
        ("billing", "billing/**", "billing/refund.py"),
        ("migration", "migrations/**", "migrations/0042_drop_column.sql"),
        ("concurrency", "jobs/**", "jobs/locks.py"),
        ("destructive_operation", "scripts/destructive/**", "scripts/destructive/purge.py"),
        ("deployment_credentials", ".github/workflows/**", ".github/workflows/deploy.yml"),
    ],
)
def test_configured_sensitive_path_categories_force_deep(category, pattern, path):
    policy = _policy(sensitive_categories=(SensitiveCategory(category, (pattern,)),))

    decision = route_review(
        _request(_file(path), requested_depth=RequestedReviewDepth.QUICK),
        policy,
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == (category,)
    assert f"sensitive_category:{category}" in _codes(decision)


def test_configured_sensitive_label_forces_deep_case_insensitively():
    policy = _policy(sensitive_categories=(SensitiveCategory("security", labels=("critical-security",)),))

    decision = route_review(
        _request(_file("docs/guide.md"), labels=("Critical-Security",)),
        policy,
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == ("security",)


@pytest.mark.parametrize(
    "changed_file",
    [
        _file("docs/new.md", old_path="auth/old.py", kind=ChangeKind.RENAMED),
        ChangedFile(None, "billing/refund.py", ChangeKind.DELETED, 0, 4, False),
    ],
)
def test_sensitive_matching_uses_old_paths_for_renames_and_deletions(changed_file):
    policy = _policy(sensitive_categories=(
        SensitiveCategory("authorization", ("auth/**",)),
        SensitiveCategory("billing", ("billing/**",)),
    ))

    decision = route_review(_request(changed_file), policy)

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.matched_sensitive_categories


def test_rename_must_be_low_risk_on_both_old_and_new_paths():
    renamed = _file("docs/guide.md", old_path="src/guide.py", kind=ChangeKind.RENAMED)

    decision = route_review(_request(renamed), _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert "docs_only" not in _codes(decision)


def test_deleted_document_uses_old_path_and_can_route_quick():
    deleted = ChangedFile(None, "docs/old.md", ChangeKind.DELETED, 0, 8, False)

    decision = route_review(_request(deleted), _policy())

    assert decision.applied_depth == ReviewDepth.QUICK
    assert "docs_only" in _codes(decision)
    assert decision.missing_inputs == ()


@pytest.mark.parametrize(
    ("changed_file", "labels", "missing"),
    [
        (_file("docs/guide.md"), None, "labels"),
        (_file("docs/guide.md", additions=None), (), "files[0].line_counts"),
        (_file("docs/guide.md", old_path=None, kind=ChangeKind.RENAMED), (), "files[0].old_path"),
        (ChangedFile(None, None, ChangeKind.DELETED, 0, 1, False), (), "files[0].old_path"),
        (_file("docs/guide.md", kind=ChangeKind.UNKNOWN), (), "files[0].kind"),
    ],
)
def test_missing_metadata_is_recorded_and_prevents_quick(changed_file, labels, missing):
    decision = route_review(_request(changed_file, labels=labels), _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert missing in decision.missing_inputs
    assert "inputs_missing" in _codes(decision)


def test_empty_change_set_is_missing_evidence_not_a_clean_quick_change():
    request = ReviewRouteRequest(files=(), labels=())

    decision = route_review(request, _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert decision.missing_inputs == ("changed_files",)


@pytest.mark.parametrize(
    ("requested_depth", "expected_depth"),
    [
        (RequestedReviewDepth.AUTO, ReviewDepth.STANDARD),
        (RequestedReviewDepth.QUICK, ReviewDepth.QUICK),
        (RequestedReviewDepth.STANDARD, ReviewDepth.STANDARD),
        (RequestedReviewDepth.DEEP, ReviewDepth.DEEP),
    ],
)
def test_authoritative_empty_change_set_selects_requested_profile_without_missing_evidence(
    requested_depth,
    expected_depth,
):
    request = ReviewRouteRequest(
        files=(),
        labels=(),
        requested_depth=requested_depth,
        changed_files_complete=True,
    )

    decision = route_review(request, _policy())

    assert decision.applied_depth is expected_depth
    assert decision.missing_inputs == ()


@pytest.mark.parametrize("path", ["../auth/guard.py", "/auth/guard.py", "auth/\x00guard.py"])
def test_invalid_paths_fail_closed_to_deep(path):
    decision = route_review(_request(_file(path)), _policy())

    assert decision.applied_depth == ReviewDepth.DEEP
    assert "input_invalid" in _codes(decision)


@pytest.mark.parametrize(
    "route_request",
    [
        ReviewRouteRequest(files="src/service.py", labels=()),
        ReviewRouteRequest(files=(_file(),), labels="security"),
        ReviewRouteRequest(files=(), labels=(), changed_files_complete="yes"),
    ],
)
def test_malformed_input_collections_fail_closed_to_deep(route_request):
    decision = route_review(route_request, _policy())

    assert decision.applied_depth == ReviewDepth.DEEP
    assert "input_invalid" in _codes(decision)


@pytest.mark.parametrize(
    "policy",
    [
        _policy(quick=ReviewBudgetPolicy(context_tokens=-1)),
        _policy(standard=ReviewBudgetPolicy(max_findings=True)),
        _policy(standard=ReviewBudgetPolicy(max_verification_candidates=0)),
        _policy(deep=ReviewBudgetPolicy(timeout_seconds=float("nan"))),
        _policy(deep=ReviewBudgetPolicy(max_retries=-1)),
        _policy(deep=ReviewBudgetPolicy(max_output_tokens=0)),
        _policy(deep=ReviewBudgetPolicy(context_tokens=8_192, max_output_tokens=8_192)),
        _policy(deep=ReviewBudgetPolicy(max_published_findings=-1)),
        _policy(deep=ReviewBudgetPolicy(model_route="")),
        _policy(deep=ReviewBudgetPolicy(publication_threshold="urgent")),
        _policy(deep=ReviewBudgetPolicy(shadow_only="false")),
        _policy(quick=None),
    ],
)
def test_malformed_budget_policy_fails_closed_to_deep(policy):
    decision = route_review(_request(_file("docs/guide.md")), policy)

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.policy_valid is False
    assert decision.policy_errors
    assert _codes(decision) == ["policy_invalid"]


@pytest.mark.parametrize(
    "policy",
    [
        _policy(large_change_files=0),
        _policy(large_change_lines=True),
        _policy(sensitive_categories=(SensitiveCategory("", ("auth/**",)),)),
        _policy(sensitive_categories=(SensitiveCategory("security"),)),
        _policy(sensitive_categories=(
            SensitiveCategory("security", ("auth/**",)),
            SensitiveCategory("SECURITY", ("secrets/**",)),
        )),
        _policy(sensitive_categories=(SensitiveCategory("security", ("../auth/**",)),)),
    ],
)
def test_malformed_rule_policy_fails_closed_to_deep(policy):
    decision = route_review(_request(_file("docs/guide.md")), policy)

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.policy_valid is False
    assert decision.policy_errors


def test_wrong_policy_type_fails_closed_instead_of_raising():
    decision = route_review(_request(_file("docs/guide.md")), {"quick": {}})

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.policy_valid is False
    assert decision.policy_version == "invalid"
    assert decision.policy_errors == ("policy must be a ReviewRouterPolicy",)


@pytest.mark.parametrize("route_request", [None, {}, "request"])
def test_wrong_request_type_fails_closed_instead_of_raising(route_request):
    policy = _policy(deep=ReviewBudgetPolicy(context_tokens=32_000))

    decision = route_review(route_request, policy)

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.applied_budget is policy.deep
    assert decision.requested_depth == "invalid"
    assert decision.missing_inputs == ("request",)
    assert _codes(decision) == ["request_invalid"]


@pytest.mark.parametrize(
    "category",
    [
        SensitiveCategory(" security", ("auth/**",)),
        SensitiveCategory("security", (" auth/** ",)),
        SensitiveCategory("security", ("./auth/**",)),
        SensitiveCategory("security", labels=(" security ",)),
    ],
)
def test_noncanonical_sensitive_rules_fail_closed_instead_of_missing_matches(category):
    decision = route_review(
        _request(_file("auth/guard.py"), labels=("security",)),
        _policy(sensitive_categories=(category,)),
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.policy_valid is False
    assert decision.policy_errors


def test_unordered_sensitive_categories_are_rejected_to_preserve_reason_order():
    categories = {
        SensitiveCategory("authorization", ("auth/**",)),
        SensitiveCategory("security", ("auth/**",)),
        SensitiveCategory("tenant_isolation", ("auth/**",)),
    }

    decision = route_review(
        _request(_file("auth/guard.py")),
        _policy(sensitive_categories=categories),
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.policy_valid is False
    assert decision.policy_errors == ("sensitive_categories must be a sequence",)


@pytest.mark.parametrize(
    "route_request",
    [
        ReviewRouteRequest(files={_file()}, labels=()),
        ReviewRouteRequest(files=(_file(),), labels={"security", "billing"}),
        ReviewRouteRequest(
            files=(_file("docs/guide.md"),),
            labels=(),
            escalation=ReviewDepthEscalation(
                "issue-12",
                ReviewDepth.STANDARD,
                {"signal-a", "signal-b"},
            ),
        ),
    ],
)
def test_unordered_request_collections_fail_closed(route_request):
    decision = route_review(route_request, _policy())

    assert decision.applied_depth == ReviewDepth.DEEP


def test_unknown_requested_profile_fails_closed_to_deep():
    decision = route_review(_request(_file("docs/guide.md"), requested_depth="turbo"), _policy())

    assert decision.applied_depth == ReviewDepth.DEEP
    assert _codes(decision)[0] == "requested_depth_invalid"


@pytest.mark.parametrize("review_profile", [None, "turbo", object()])
def test_unknown_output_profile_fails_closed_to_full_output_and_deep_review(review_profile):
    decision = route_review(
        _request(_file("docs/guide.md"), review_profile=review_profile),
        _policy(),
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.review_profile == ReviewOutputProfile.FULL.value
    assert "review_profile_invalid" in _codes(decision)


def test_future_escalation_can_raise_a_quick_route_to_deep():
    escalation = ReviewDepthEscalation(
        source="issue-12-risk-specialist-v1",
        minimum_depth=ReviewDepth.DEEP,
        reasons=("ambiguous authorization boundary",),
    )

    decision = route_review(_request(_file("docs/guide.md"), escalation=escalation), _policy())

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.escalation_applied is True
    assert "external_escalation" in _codes(decision)


def test_future_escalation_cannot_lower_a_deterministic_forced_deep_route():
    policy = _policy(sensitive_categories=(SensitiveCategory("billing", ("billing/**",)),))
    escalation = ReviewDepthEscalation(
        source="issue-12-risk-specialist-v1",
        minimum_depth=ReviewDepth.QUICK,
        reasons=("model classified the change as routine",),
    )

    decision = route_review(_request(_file("billing/refund.py"), escalation=escalation), policy)

    assert decision.applied_depth == ReviewDepth.DEEP
    assert decision.escalation_applied is False
    assert _codes(decision).index("sensitive_category:billing") < _codes(decision).index("external_escalation")


def test_uncertain_future_escalation_requires_at_least_standard_review():
    escalation = ReviewDepthEscalation(
        source="issue-12-classifier-v1",
        minimum_depth=ReviewDepth.QUICK,
        reasons=("classification confidence unavailable",),
        uncertain=True,
    )

    decision = route_review(_request(_file("docs/guide.md"), escalation=escalation), _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert decision.escalation_applied is True


def test_unavailable_future_escalation_records_missing_evidence_and_prevents_quick():
    escalation = ReviewDepthEscalation(
        source="issue-12-classifier-v1",
        minimum_depth=None,
        reasons=(),
        available=False,
    )

    decision = route_review(_request(_file("docs/guide.md"), escalation=escalation), _policy())

    assert decision.applied_depth == ReviewDepth.STANDARD
    assert decision.missing_inputs == ("escalation:issue-12-classifier-v1",)
    assert "escalation_unavailable" in _codes(decision)


@pytest.mark.parametrize(
    "escalation",
    [
        ReviewDepthEscalation("", ReviewDepth.QUICK, ("routine",)),
        ReviewDepthEscalation("issue-12", "unknown", ("routine",)),
        ReviewDepthEscalation("issue-12", ReviewDepth.STANDARD, ("",)),
        ReviewDepthEscalation("issue-12", ReviewDepth.STANDARD, "risk"),
        ReviewDepthEscalation("issue-12", ReviewDepth.STANDARD, None),
        ReviewDepthEscalation("issue-12", ReviewDepth.STANDARD, ("risk",), available="yes"),
    ],
)
def test_malformed_future_escalation_fails_closed_to_deep(escalation):
    decision = route_review(_request(_file("docs/guide.md"), escalation=escalation), _policy())

    assert decision.applied_depth == ReviewDepth.DEEP
    assert "escalation_invalid" in _codes(decision)


@pytest.mark.parametrize(
    "changed_file",
    [
        _file("docs/guide.md"),
        _file("src/service.py"),
        _file("auth/guard.py"),
    ],
)
def test_review_output_profile_is_orthogonal_to_review_depth(changed_file):
    policy = _policy(sensitive_categories=(SensitiveCategory("authorization", ("auth/**",)),))

    full = route_review(_request(changed_file, review_profile=ReviewOutputProfile.FULL), policy)
    bugs_only = route_review(_request(changed_file, review_profile=ReviewOutputProfile.BUGS_ONLY), policy)

    assert full.applied_depth == bugs_only.applied_depth
    assert full.requested_budget == bugs_only.requested_budget
    assert full.applied_budget == bugs_only.applied_budget
    assert full.reasons == bugs_only.reasons
    assert full.review_profile == "full"
    assert bugs_only.review_profile == "bugs_only"


def test_conflicting_signals_have_stable_precedence_and_reason_order():
    policy = _policy(
        sensitive_categories=(SensitiveCategory("deployment", (".github/workflows/**",)),),
        large_change_files=1,
        large_change_lines=5,
    )
    escalation = ReviewDepthEscalation(
        source="issue-12-classifier-v1",
        minimum_depth=ReviewDepth.QUICK,
        reasons=("routine documentation",),
    )
    changed_file = _file(".github/workflows/dependencies.md", additions=5, deletions=0)

    decision = route_review(
        _request(
            changed_file,
            requested_depth=RequestedReviewDepth.QUICK,
            labels=None,
            escalation=escalation,
        ),
        policy,
    )

    assert decision.applied_depth == ReviewDepth.DEEP
    assert _codes(decision) == [
        "requested_quick",
        "sensitive_category:deployment",
        "large_change:files",
        "large_change:lines",
        "external_escalation",
        "inputs_missing",
        "docs_only",
    ]


def test_pattern_matching_accepts_root_paths_for_double_star_patterns():
    policy = _policy(docs_patterns=("**/*.md",))

    decision = route_review(_request(_file("README.md")), policy)

    assert decision.applied_depth == ReviewDepth.QUICK
    assert "docs_only" in _codes(decision)
