from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pr_agent.algo.inline_comment_dedup import (body_with_markers,
                                                get_inline_comment_store,
                                                key_issue_fingerprint)
from pr_agent.algo.ai_request_context import AIModelRoute
from pr_agent.algo.pr_processing import (PRDiffCoverage,
                                         retry_with_fallback_models)
from pr_agent.algo.review_router import (
    ChangedFile,
    ChangeKind,
    ReviewDepth,
    load_review_routing_configuration,
    review_route_decision_to_dict,
)
from pr_agent.algo.review_specialists import (
    RoleExecution,
    SpecialistBatchResult,
    SpecialistRole,
    SpecialistState,
)
from pr_agent.algo.run_details import (get_run_details, init_run_details,
                                       record_model_used)
from pr_agent.algo.types import FilePatchInfo
from pr_agent.algo.utils import (PRReviewHeader, PRReviewIdentity,
                                 show_run_details)
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.azuredevops_provider import AzureDevopsProvider
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.tools.pr_reviewer import PRReviewer
from tests.unittest._settings_helpers import (restore_settings,
                                              snapshot_settings)

# _prepare_prediction now rejects output it cannot parse, so the model's answer can fall
# back to another model instead of failing after every retry is spent.
VALID_PREDICTION = "review:\n  score: 90\nsecurity_concerns: No\n"


def _make_reviewer(git_provider=None):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = git_provider or MagicMock()
    reviewer.pr_url = "https://example/pr/1"
    return reviewer


def _make_prediction_reviewer(git_provider=None):
    reviewer = _make_reviewer(git_provider)
    if isinstance(reviewer.git_provider, MagicMock):
        raw_files = reviewer.git_provider.get_files
        if raw_files.side_effect is None and isinstance(raw_files.return_value, MagicMock):
            # Most unit-test providers model one complete inventory. Tests for raw
            # inventory gaps override this with their explicit failure or mismatch.
            raw_files.return_value = reviewer.git_provider.get_diff_files.return_value
    reviewer.token_handler = MagicMock()
    reviewer.remaining_files_list = []
    reviewer.deleted_files_list = []
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.prediction = None
    return reviewer


def _routing_configuration(
    *, consume=False, requested_depth="auto", sensitive_categories=(), shadow_only=False
):
    return load_review_routing_configuration({
        "enabled": True,
        "requested_depth": requested_depth,
        "consume_specialist_escalation": consume,
        "specialist_escalation_depth": "deep",
        "profiles": {
            "quick": {
                "context_tokens": 8_000,
                "max_findings": 2,
                "max_verification_candidates": 1,
                "model_route": "weak",
                "timeout_seconds": 30,
                "max_retries": 0,
                "max_output_tokens": 2_048,
                "max_published_findings": 2,
                "publication_threshold": "high",
                "shadow_only": shadow_only,
            },
            "standard": {
                "context_tokens": 24_000,
                "max_findings": 3,
                "model_route": "regular",
            },
            "deep": {
                "context_tokens": 32_000,
                "max_findings": 6,
                "model_route": "reasoning",
            },
        },
        "sensitive_categories": list(sensitive_categories),
    })


def _route_file(
    filename="docs/guide.md",
    *,
    edit_type=None,
    old_filename=None,
    additions=2,
    deletions=1,
):
    from pr_agent.algo.types import EDIT_TYPE

    return FilePatchInfo(
        base_file="old\n",
        head_file="new\n",
        patch="@@ -1 +1 @@\n-old\n+new",
        filename=filename,
        edit_type=edit_type or EDIT_TYPE.MODIFIED,
        old_filename=old_filename,
        num_plus_lines=additions,
        num_minus_lines=deletions,
    )


class _MutatingIncrementalProvider:
    """Model GitHub's incremental map changing from file objects to patches."""

    def __init__(self, raw_files, detailed_files=None):
        self.unreviewed_files_map = {file.filename: file for file in raw_files}
        self._detailed_files = tuple(detailed_files or ())
        self.calls = []

    def get_files(self):
        self.calls.append("raw")
        return self.unreviewed_files_map.values()

    def get_diff_files(self):
        self.calls.append("detailed")
        detailed = self._detailed_files or tuple(
            _route_file(file.filename) for file in self.unreviewed_files_map.values()
        )
        for filename, file in tuple(self.unreviewed_files_map.items()):
            self.unreviewed_files_map[filename] = getattr(file, "patch", None) or "@@ patch body"
        return detailed

    def is_supported(self, capability):
        return capability == "get_labels"

    def get_pr_labels(self):
        return []


def _incremental_raw_file(filename, *, status="modified", previous_filename=None):
    return SimpleNamespace(
        filename=filename,
        previous_filename=previous_filename,
        status=status,
        additions=1,
        deletions=1 if status in {"removed", "renamed"} else 0,
        patch="@@ -1 +1 @@\n-old\n+new",
    )


@pytest.mark.asyncio
async def test_prepare_prediction_requests_remaining_files_and_preserves_tuple_result():
    reviewer = _make_prediction_reviewer()
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)

    with patch(
        "pr_agent.tools.pr_reviewer.get_pr_diff",
        return_value=PRDiffCoverage("diff", ["src/one.py", "docs/two.md"], ["deleted.py"]),
    ) as get_pr_diff:
        await reviewer._prepare_prediction("model")

    get_pr_diff.assert_called_once_with(
        reviewer.git_provider,
        reviewer.token_handler,
        "model",
        add_line_numbers_to_hunks=True,
        disable_extra_lines=False,
        return_remaining_files=True,
        return_deleted_files=True,
    )
    assert reviewer.patches_diff == "diff"
    assert reviewer.remaining_files_list == ["src/one.py", "docs/two.md"]
    assert reviewer.deleted_files_list == ["deleted.py"]
    assert reviewer.prediction == VALID_PREDICTION


@pytest.mark.asyncio
async def test_prepare_prediction_accepts_full_diff_string_when_token_budget_is_sufficient():
    reviewer = _make_prediction_reviewer()
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)

    with patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="diff"):
        await reviewer._prepare_prediction("model")

    assert reviewer.patches_diff == "diff"
    assert reviewer.remaining_files_list == []
    assert reviewer.deleted_files_list == []
    assert reviewer.prediction == VALID_PREDICTION


@pytest.mark.asyncio
async def test_routed_prediction_applies_context_budget_and_model_specific_token_handler():
    provider = MagicMock()
    provider.pr = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()
    reviewer._prepare_review_route()
    reviewer._specialists_started = True
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)
    routed_token_handler = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.TokenHandler", return_value=routed_token_handler) as token_handler,
        patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="diff") as get_pr_diff,
    ):
        await reviewer._prepare_prediction("weak-model")

    assert token_handler.call_args.kwargs["model"] == "weak-model"
    get_pr_diff.assert_called_once_with(
        provider,
        routed_token_handler,
        "weak-model",
        add_line_numbers_to_hunks=True,
        disable_extra_lines=False,
        return_remaining_files=True,
        return_deleted_files=True,
        max_context_tokens=8_000,
        max_output_tokens=2_048,
    )


@pytest.mark.asyncio
async def test_routed_prediction_reserves_inherited_global_output_cap():
    reviewer = _make_prediction_reviewer()
    reviewer.review_route_decision = SimpleNamespace(
        routing_enabled=True,
        applied_budget=SimpleNamespace(max_output_tokens=None),
    )
    reviewer._review_context_tokens = 32_000
    reviewer.vars = {}
    reviewer._specialists_started = True
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)
    settings = get_settings()
    previous_cap = settings.config.get("max_output_tokens", 0)

    try:
        settings.set("config.max_output_tokens", "8192")
        with (
            patch("pr_agent.tools.pr_reviewer.TokenHandler", return_value=MagicMock()),
            patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="diff") as get_pr_diff,
        ):
            await reviewer._prepare_prediction("reasoning-model")
    finally:
        settings.set("config.max_output_tokens", previous_cap)

    assert get_pr_diff.call_args.kwargs["max_context_tokens"] == 32_000
    assert get_pr_diff.call_args.kwargs["max_output_tokens"] == 8_192


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "settings_updates", "expected_cap"),
    [
        (
            "claude-3-7-sonnet-20250219",
            {
                "config.enable_claude_extended_thinking": True,
                "config.extended_thinking_budget_tokens": 2_048,
                "config.extended_thinking_max_output_tokens": 4_096,
            },
            4_096,
        ),
        (
            "openrouter/google/gemini-2.5-pro",
            {"openrouter.max_tokens": 3_072},
            3_072,
        ),
    ],
)
async def test_routed_prediction_reserves_inherited_provider_output_cap(
    model, settings_updates, expected_cap
):
    reviewer = _make_prediction_reviewer()
    reviewer.review_route_decision = SimpleNamespace(
        routing_enabled=True,
        applied_budget=SimpleNamespace(max_output_tokens=None),
    )
    reviewer._review_context_tokens = 32_000
    reviewer.vars = {}
    reviewer._specialists_started = True
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)
    settings = get_settings()
    keys = ("config.max_output_tokens", *settings_updates)
    previous = {key: settings.get(key, None) for key in keys}

    try:
        settings.set("config.max_output_tokens", 0)
        for key, value in settings_updates.items():
            settings.set(key, value)
        with (
            patch("pr_agent.tools.pr_reviewer.TokenHandler", return_value=MagicMock()),
            patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="diff") as get_pr_diff,
        ):
            await reviewer._prepare_prediction(model)
    finally:
        for key, value in previous.items():
            settings.set(key, value)

    assert get_pr_diff.call_args.kwargs["max_output_tokens"] == expected_cap


@pytest.mark.asyncio
async def test_routed_prediction_rejects_unbounded_openrouter_reasoning_before_diff():
    reviewer = _make_prediction_reviewer()
    reviewer.review_route_decision = SimpleNamespace(
        routing_enabled=True,
        applied_budget=SimpleNamespace(max_output_tokens=None),
    )
    reviewer._review_context_tokens = 32_000
    reviewer.vars = {}
    reviewer._specialists_started = True
    settings = get_settings()
    keys = (
        "config.max_output_tokens",
        "openrouter.max_tokens",
        "openrouter.reasoning_max_tokens",
    )
    previous = {key: settings.get(key, None) for key in keys}

    try:
        settings.set("config.max_output_tokens", 0)
        settings.set("openrouter.max_tokens", 0)
        settings.set("openrouter.reasoning_max_tokens", 2_048)
        with patch("pr_agent.tools.pr_reviewer.get_pr_diff") as get_pr_diff:
            with pytest.raises(ValueError, match="requires a positive total"):
                await reviewer._prepare_prediction("openrouter/google/gemini-2.5-pro")
    finally:
        for key, value in previous.items():
            settings.set(key, value)

    get_pr_diff.assert_not_called()


@pytest.mark.asyncio
async def test_routed_fallback_rebuilds_diff_for_each_model_context(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.get_diff_files.return_value = []
    provider.get_languages.return_value = {}
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_route_decision = SimpleNamespace(
        routing_enabled=True,
        applied_budget=SimpleNamespace(max_output_tokens=8_192),
    )
    reviewer._review_context_tokens = 32_000
    reviewer._specialists_started = True
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)

    routed_token_handler = MagicMock()
    routed_token_handler.prompt_tokens = 1_000
    routed_token_handler.count_tokens.side_effect = lambda value: len(value.split())
    monkeypatch.setattr(pr_reviewer_module, "TokenHandler", lambda *args, **kwargs: routed_token_handler)
    monkeypatch.setattr(
        "pr_agent.algo.pr_processing.sort_files_by_main_languages",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "pr_agent.algo.pr_processing.pr_generate_extended_diff",
        lambda *args, **kwargs: (["fallback-safe diff"], 2_000, [1_000]),
    )
    monkeypatch.setattr(
        "pr_agent.algo.pr_processing.get_max_tokens",
        lambda model: {"small-window": 8_192, "larger-window": 16_000}[model],
    )
    real_get_pr_diff = pr_reviewer_module.get_pr_diff
    attempts = []

    def recording_get_pr_diff(provider, token_handler, model, **kwargs):
        attempts.append((model, kwargs["max_context_tokens"], kwargs["max_output_tokens"]))
        return real_get_pr_diff(provider, token_handler, model, **kwargs)

    monkeypatch.setattr(pr_reviewer_module, "get_pr_diff", recording_get_pr_diff)
    route = AIModelRoute(
        models=("small-window", "larger-window"),
        deployments=(None, None),
        max_output_tokens=8_192,
    )

    await retry_with_fallback_models(
        reviewer._prepare_prediction,
        model_route=route,
    )

    assert attempts == [
        ("small-window", 32_000, 8_192),
        ("larger-window", 32_000, 8_192),
    ]
    assert reviewer.patches_diff == "fallback-safe diff"
    assert reviewer.prediction == VALID_PREDICTION


def test_profile_model_route_uses_request_local_controls_and_retry_semantics():
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()
    reviewer._prepare_review_route()

    with patch("pr_agent.tools.pr_reviewer.get_model", return_value="weak-model"):
        route = reviewer._review_model_route()

    assert route.models[0] == "weak-model"
    assert route.timeout_seconds == 30
    assert route.model_retries == 1
    assert route.max_output_tokens == 2_048
    assert route.attribution is None


def test_quick_claude_weak_route_disables_thinking_when_cap_equals_budget():
    from pr_agent.algo import CLAUDE_EXTENDED_THINKING_MODELS
    from pr_agent.algo.ai_handlers.litellm_ai_handler import (
        LiteLLMAIHandler,
        get_effective_litellm_output_token_cap,
    )

    provider = MagicMock()
    provider.get_files.return_value = ["docs/guide.md"]
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()
    reviewer._prepare_review_route()
    settings = get_settings()
    snapshot = snapshot_settings((
        "config.enable_claude_extended_thinking",
        "config.extended_thinking_budget_tokens",
        "config.extended_thinking_max_output_tokens",
    ))

    try:
        settings.set("config.enable_claude_extended_thinking", True)
        settings.set("config.extended_thinking_budget_tokens", 2_048)
        settings.set("config.extended_thinking_max_output_tokens", 4_096)
        model = "claude-3-7-sonnet-20250219"
        with patch("pr_agent.tools.pr_reviewer.get_model", return_value=model):
            route = reviewer._review_model_route()
        effective_cap = get_effective_litellm_output_token_cap(
            model,
            route.max_output_tokens,
            claude_extended_thinking_models=CLAUDE_EXTENDED_THINKING_MODELS,
        )
        handler = LiteLLMAIHandler.__new__(LiteLLMAIHandler)
        handler.claude_extended_thinking_models = CLAUDE_EXTENDED_THINKING_MODELS
        kwargs = handler._configure_claude_extended_thinking(
            model,
            {},
            effective_max_output_tokens=effective_cap,
        )
    finally:
        restore_settings(snapshot)

    assert route.models[0] == model
    assert route.max_output_tokens == 2_048
    assert effective_cap == 2_048
    assert "thinking" not in kwargs


def test_publication_budget_clamps_findings_and_shadow_profile_never_publishes_markdown():
    reviewer = _make_prediction_reviewer()
    reviewer._review_max_published_findings = 1
    reviewer._review_shadow_only = True
    data = {
        "review": {
            "key_issues_to_review": [
                {"issue_header": "one"},
                {"issue_header": "two"},
            ]
        }
    }

    limited = reviewer._apply_publication_budget(data)

    assert [issue["issue_header"] for issue in limited["review"]["key_issues_to_review"]] == ["one"]
    assert len(data["review"]["key_issues_to_review"]) == 2
    assert reviewer._should_publish_review_no_suggestions("review") is False


def test_generated_and_publication_finding_budgets_are_independent_and_immutable():
    reviewer = _make_prediction_reviewer()
    reviewer._review_max_findings = 2
    reviewer._review_max_published_findings = 1
    data = {
        "review": {
            "key_issues_to_review": [
                {"issue_header": "one"},
                {"issue_header": "two"},
                {"issue_header": "three"},
            ]
        }
    }

    generated = reviewer._apply_finding_budget(data)
    published = reviewer._apply_publication_budget(generated)

    assert [issue["issue_header"] for issue in generated["review"]["key_issues_to_review"]] == [
        "one", "two",
    ]
    assert [issue["issue_header"] for issue in published["review"]["key_issues_to_review"]] == ["one"]
    assert len(data["review"]["key_issues_to_review"]) == 3


@pytest.mark.asyncio
async def test_disabled_specialists_do_not_construct_or_call_coordinator():
    reviewer = _make_prediction_reviewer()
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)
    settings = get_settings()
    original_enabled = settings.get("specialist_pipeline.enabled", False)
    settings.set("specialist_pipeline.enabled", False)
    try:
        with (
            patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="exact diff"),
            patch(
                "pr_agent.tools.pr_reviewer.load_specialist_pipeline_config",
                side_effect=AssertionError("disabled specialists must not be constructed"),
            ),
            patch("pr_agent.tools.pr_reviewer.run_shadow_specialists", new_callable=AsyncMock) as run_shadow,
        ):
            await reviewer._prepare_prediction("model")
    finally:
        settings.set("specialist_pipeline.enabled", original_enabled)

    run_shadow.assert_not_awaited()
    reviewer._get_prediction.assert_awaited_once_with("model")
    assert reviewer.patches_diff == "exact diff"
    assert reviewer.prediction == VALID_PREDICTION


@pytest.mark.asyncio
async def test_enabled_shadow_specialists_run_at_most_once_across_main_fallback_attempts():
    reviewer = _make_prediction_reviewer()
    reviewer._specialists_started = False
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)
    settings = get_settings()
    original_enabled = settings.get("specialist_pipeline.enabled", False)
    settings.set("specialist_pipeline.enabled", True)

    async def mark_started():
        reviewer._specialists_started = True

    reviewer._run_shadow_specialists_once = AsyncMock(side_effect=mark_started)
    try:
        with patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="exact diff"):
            await reviewer._prepare_prediction("primary")
            await reviewer._prepare_prediction("fallback")
    finally:
        settings.set("specialist_pipeline.enabled", original_enabled)

    reviewer._run_shadow_specialists_once.assert_awaited_once_with()
    assert reviewer._get_prediction.await_args_list == [
        (("primary",), {}),
        (("fallback",), {}),
    ]


@pytest.mark.parametrize(
    ("edit_type", "filename", "old_filename", "expected_kind", "old_path", "new_path"),
    [
        ("ADDED", "new.py", None, "added", None, "new.py"),
        ("DELETED", "old.py", None, "deleted", "old.py", None),
        ("MODIFIED", "same.py", None, "modified", None, "same.py"),
        ("RENAMED", "new.py", "old.py", "renamed", "old.py", "new.py"),
        ("UNKNOWN", "maybe.py", None, "unknown", None, "maybe.py"),
    ],
)
def test_file_patch_info_conversion_preserves_edit_identity(
    edit_type, filename, old_filename, expected_kind, old_path, new_path
):
    from pr_agent.algo.types import EDIT_TYPE

    changed = PRReviewer._changed_file_for_routing(_route_file(
        filename,
        edit_type=getattr(EDIT_TYPE, edit_type),
        old_filename=old_filename,
        additions=-1,
        deletions=-1,
    ))

    assert changed.kind.value == expected_kind
    assert changed.old_path == old_path
    assert changed.new_path == new_path
    assert changed.additions is None
    assert changed.deletions is None


def test_deterministic_route_runs_from_provider_metadata_and_records_structured_decision():
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.QUICK
    assert reviewer.vars["num_max_findings"] == 2
    assert reviewer.vars["publication_threshold"] == "high"
    assert reviewer.vars["max_verification_candidates"] == 1
    assert reviewer._review_context_tokens == 8_000
    assert get_run_details().review_route == review_route_decision_to_dict(decision)


@pytest.mark.parametrize(
    ("raw_configuration", "configuration_name"),
    [(None, "absent"), ({"enabled": False}, "disabled")],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_disabled_or_absent_routing_preserves_run_details_byte_for_byte(
    monkeypatch, raw_configuration, configuration_name
):
    from pr_agent.algo import run_details

    monkeypatch.setattr(run_details.time, "monotonic", lambda: 108.2)
    provider = MagicMock()
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = load_review_routing_configuration(raw_configuration)

    baseline_details = init_run_details()
    baseline_details.start_time = 100.0
    record_model_used("model-a", is_fallback=False)
    baseline = show_run_details(gfm_supported=True)

    routed_details = init_run_details()
    routed_details.start_time = 100.0
    record_model_used("model-a", is_fallback=False)
    decision = reviewer._prepare_review_route()
    routed = show_run_details(gfm_supported=True)

    assert configuration_name in {"absent", "disabled"}
    assert decision.routing_enabled is False
    assert get_run_details().review_route is None
    assert routed == baseline
    provider.get_files.assert_not_called()
    provider.get_diff_files.assert_not_called()
    provider.get_pr_labels.assert_not_called()


def test_enabled_standard_route_is_recorded_and_rendered():
    provider = MagicMock()
    provider.get_files.return_value = ["docs/guide.md"]
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(requested_depth="standard")
    init_run_details()
    record_model_used("model-a", is_fallback=False)

    decision = reviewer._prepare_review_route()

    assert decision.routing_enabled is True
    assert decision.applied_depth is ReviewDepth.STANDARD
    assert get_run_details().review_route == review_route_decision_to_dict(decision)
    assert "Review depth: standard" in show_run_details(gfm_supported=True)


@pytest.mark.asyncio
async def test_disabled_routing_adds_no_provider_inventory_or_model_calls(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.get_files.return_value = ["docs/guide.md"]
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = False
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = load_review_routing_configuration({"enabled": False})

    async def fake_retry(prepare_fn, model_type=None):
        reviewer.prediction = VALID_PREDICTION

    retry = AsyncMock(side_effect=fake_retry)
    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", retry)
    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(pr_reviewer_module, "convert_to_markdown_v2", MagicMock(return_value="legacy review"))

    settings = get_settings()
    snapshot = snapshot_settings((
        "config.publish_output",
        "config.is_auto_command",
        "config.output_run_details",
        "data",
        "pr_reviewer.enable_help_text",
    ))
    try:
        settings.config.publish_output = False
        settings.config.is_auto_command = False
        settings.config.output_run_details = True
        settings.pr_reviewer.enable_help_text = False
        settings.data = {"artifact": ""}

        await reviewer.run()
        artifact = settings.data["artifact"]
    finally:
        restore_settings(snapshot)

    assert artifact == "legacy review"
    assert get_run_details().review_route is None
    assert reviewer._review_model_route() is None
    retry.assert_awaited_once()
    assert provider.get_files.call_count == 1
    assert provider.get_diff_files.call_count == 1
    provider.get_pr_labels.assert_not_called()
    structured = provider.publish_structured_review.call_args.args[0]
    assert "review_route" not in structured["metadata"]


@pytest.mark.parametrize("file_count", [1, 12, 13])
def test_incremental_docs_inventory_is_snapshotted_before_patch_mutation(file_count):
    raw_files = [
        _incremental_raw_file(f"docs/guide-{index}.md")
        for index in range(file_count)
    ]
    provider = _MutatingIncrementalProvider(raw_files)
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.QUICK
    assert provider.calls == ["raw", "detailed"]
    assert all(isinstance(value, str) for value in provider.unreviewed_files_map.values())
    assert len(reviewer.review_route_request.files) == file_count
    assert all(file.kind is not ChangeKind.UNKNOWN for file in reviewer.review_route_request.files)


def test_incremental_sensitive_path_still_forces_deep_after_patch_mutation():
    raw_files = [
        _incremental_raw_file("docs/guide.md"),
        _incremental_raw_file("services/auth/guard.py"),
    ]
    provider = _MutatingIncrementalProvider(raw_files)
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(sensitive_categories=({
        "name": "authorization",
        "path_patterns": ["**/auth/**"],
        "labels": [],
    },))
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == ("authorization",)


def test_incremental_snapshot_preserves_deleted_and_renamed_paths():
    from pr_agent.algo.types import EDIT_TYPE

    raw_files = [
        _incremental_raw_file("docs/deleted.md", status="removed"),
        _incremental_raw_file(
            "docs/new-name.md",
            status="renamed",
            previous_filename="services/auth/old-name.py",
        ),
    ]
    detailed_files = [
        _route_file("docs/deleted.md", edit_type=EDIT_TYPE.DELETED),
        _route_file(
            "docs/new-name.md",
            edit_type=EDIT_TYPE.RENAMED,
            old_filename="services/auth/old-name.py",
        ),
    ]
    reviewer = _make_prediction_reviewer(
        _MutatingIncrementalProvider(raw_files, detailed_files)
    )

    changed_files = reviewer._changed_files_for_routing()

    assert changed_files[0].kind is ChangeKind.DELETED
    assert changed_files[0].old_path == "docs/deleted.md"
    assert changed_files[0].new_path is None
    assert changed_files[1].kind is ChangeKind.RENAMED
    assert changed_files[1].old_path == "services/auth/old-name.py"
    assert changed_files[1].new_path == "docs/new-name.md"


def test_incremental_detailed_raw_mismatch_keeps_fail_safe_unknown_floor():
    raw_files = [_incremental_raw_file("docs/guide.md")]
    detailed_files = [
        _route_file("docs/guide.md"),
        _route_file("src/unmatched.py"),
    ]
    provider = _MutatingIncrementalProvider(raw_files, detailed_files)
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.STANDARD
    assert any(file.kind is ChangeKind.UNKNOWN for file in reviewer.review_route_request.files)
    assert [file.new_path for file in reviewer.review_route_request.files].count("docs/guide.md") == 1


def test_nonincremental_provider_inventory_remains_provider_neutral_and_deduplicated():
    provider = MagicMock()
    raw_file = SimpleNamespace(
        filename="docs/guide.md",
        status="modified",
        additions=1,
        deletions=0,
    )
    provider.get_files.return_value = [raw_file]
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    reviewer = _make_prediction_reviewer(provider)

    changed_files = reviewer._changed_files_for_routing()

    assert changed_files == (ChangedFile(
        new_path="docs/guide.md",
        kind=ChangeKind.MODIFIED,
        additions=2,
        deletions=1,
    ),)
    provider.get_files.assert_called_once_with()
    provider.get_diff_files.assert_called_once_with()


def test_changed_file_metadata_failure_is_recorded_and_prevents_quick():
    provider = MagicMock()
    provider.get_diff_files.side_effect = RuntimeError("metadata unavailable")
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.STANDARD
    assert "files[0].kind" in decision.missing_inputs
    assert "inputs_missing" in [reason.code for reason in decision.reasons]


@pytest.mark.parametrize("labels_supported", [False, True])
def test_unavailable_label_metadata_is_recorded_and_prevents_quick(labels_supported):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = labels_supported
    if labels_supported:
        provider.get_pr_labels.side_effect = RuntimeError("labels unavailable")
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.STANDARD
    assert "labels" in decision.missing_inputs


@pytest.mark.parametrize("provider_type", [GithubProvider, AzureDevopsProvider])
def test_provider_label_failures_remain_unavailable_for_routing(provider_type):
    provider = provider_type.__new__(provider_type)
    if provider_type is GithubProvider:
        exploding_labels = MagicMock()
        type(exploding_labels).labels = property(lambda _: (_ for _ in ()).throw(RuntimeError("labels unavailable")))
        provider.pr = exploding_labels
    else:
        provider.workspace_slug = "workspace"
        provider.repo_slug = "repo"
        provider.pr_num = 1
        provider.azure_devops_client = MagicMock()
        provider.azure_devops_client.get_pull_request_labels.side_effect = RuntimeError("labels unavailable")

    assert provider.get_pr_labels() == []
    assert provider.get_pr_labels_for_routing() is None


@pytest.mark.parametrize("raw_sensitive_file", [
    SimpleNamespace(filename="services/auth/guard.py", status="modified", additions=1, deletions=0),
    SimpleNamespace(filename="services/auth/guard.py", status="removed", additions=0, deletions=1),
    SimpleNamespace(old_path="services/auth/guard.py", status="deleted", additions=0, deletions=1),
    {
        "previous_filename": "services/auth/guard.py",
        "status": "removed",
        "additions": 0,
        "deletions": 1,
    },
    SimpleNamespace(
        filename="docs/old-guard.md",
        previous_filename="services/auth/guard.py",
        status="renamed",
        additions=1,
        deletions=1,
    ),
])
def test_unfiltered_sensitive_path_forces_deep_when_review_diff_ignores_it(raw_sensitive_file):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.get_files.return_value = [
        SimpleNamespace(filename="docs/guide.md", status="modified", additions=1, deletions=0),
        raw_sensitive_file,
    ]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(sensitive_categories=({
        "name": "authorization",
        "path_patterns": ["**/auth/**"],
        "labels": [],
    },))
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == ("authorization",)


def test_unavailable_unfiltered_inventory_prevents_quick():
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.get_files.side_effect = RuntimeError("raw inventory unavailable")
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.STANDARD
    assert "files[1].kind" in decision.missing_inputs


@pytest.mark.parametrize("raw_inventory", [None, []])
def test_empty_unfiltered_inventory_prevents_quick(raw_inventory):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.get_files.return_value = raw_inventory
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.STANDARD
    assert "files[1].kind" in decision.missing_inputs


def test_detailed_inventory_failure_still_uses_raw_sensitive_paths():
    provider = MagicMock()
    provider.get_diff_files.side_effect = RuntimeError("detailed inventory unavailable")
    provider.get_files.return_value = [
        SimpleNamespace(filename="services/auth/guard.py", status="modified", additions=1, deletions=0)
    ]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(sensitive_categories=({
        "name": "authorization",
        "path_patterns": ["**/auth/**"],
        "labels": [],
    },))
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == ("authorization",)
    assert "files[1].kind" in decision.missing_inputs


def test_codecommit_rename_preserves_sensitive_old_path():
    from pr_agent.algo.types import EDIT_TYPE

    provider = MagicMock()
    provider.get_diff_files.side_effect = RuntimeError("detailed inventory unavailable")
    provider.get_files.return_value = [SimpleNamespace(
        a_path="services/auth/guard.py",
        b_path="docs/safe.md",
        filename="docs/safe.md",
        edit_type=EDIT_TYPE.RENAMED,
    )]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(sensitive_categories=({
        "name": "authorization",
        "path_patterns": ["**/auth/**"],
        "labels": [],
    },))
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == ("authorization",)


def test_sensitive_old_rename_path_forces_deep_over_docs_only_signal():
    from pr_agent.algo.types import EDIT_TYPE

    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file(
        "docs/guide.md",
        edit_type=EDIT_TYPE.RENAMED,
        old_filename="services/auth/guard.py",
    )]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(sensitive_categories=({
        "name": "authorization",
        "path_patterns": ["**/auth/**"],
        "labels": [],
    },))
    init_run_details()

    decision = reviewer._prepare_review_route()

    assert decision.applied_depth is ReviewDepth.DEEP
    assert decision.matched_sensitive_categories == ("authorization",)


def _install_specialist_result(
    reviewer,
    *,
    state,
    recommendation="escalate",
    identity="head",
    risk_enabled=True,
    include_risk_record=True,
):
    specialist_input = SimpleNamespace(
        snapshot_id="snapshot",
        head_sha="head",
        input_hash="input-hash",
    )
    prompt = SimpleNamespace(schema_version="risk-recommendation-output-v1")
    pipeline = SimpleNamespace(
        configuration_hash="configuration-hash",
        prompt=lambda role: prompt,
        roles=(SimpleNamespace(role=SpecialistRole.RISK_RECOMMENDATION, enabled=risk_enabled),),
    )
    output = {
        "schema_version": prompt.schema_version,
        "confidence": 0.9,
        "recommendation": recommendation,
        "reasons": [{
            "reason": "untrusted specialist prose must not reach routing",
            "evidence": [{"source": "pull_request", "field": "title"}],
        }],
    }
    reviewer._specialist_input = specialist_input
    reviewer._specialist_pipeline = pipeline
    records = ()
    if include_risk_record:
        records = (RoleExecution(
            role=SpecialistRole.RISK_RECOMMENDATION,
            state=state,
            output=output,
            confidence=0.9,
        ),)
    reviewer.specialist_shadow_result = SpecialistBatchResult(
        snapshot_id="snapshot",
        head_sha=identity,
        input_hash="input-hash",
        configuration_hash="configuration-hash",
        records=records,
        role_records={},
        changed_path_count=1,
        hunk_count=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "recommendation", "expected_depth"),
    [
        (SpecialistState.SUCCESS, "escalate", ReviewDepth.DEEP),
        (SpecialistState.CACHED, "none", ReviewDepth.QUICK),
        (SpecialistState.LOW_CONFIDENCE, "escalate", ReviewDepth.STANDARD),
        (SpecialistState.TIMEOUT, "escalate", ReviewDepth.STANDARD),
        (SpecialistState.MALFORMED_OUTPUT, "escalate", ReviewDepth.DEEP),
    ],
)
async def test_guarded_specialist_consumer_only_escalates_validated_success(
    monkeypatch, state, recommendation, expected_depth
):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(consume=True)
    init_run_details()
    reviewer._prepare_review_route()
    _install_specialist_result(reviewer, state=state, recommendation=recommendation)
    reviewer._run_shadow_specialists_once = AsyncMock()
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.specialists_enabled", lambda: True)

    await reviewer._run_guarded_specialist_escalation()

    assert reviewer.review_route_decision.applied_depth is expected_depth
    serialized_reasons = review_route_decision_to_dict(reviewer.review_route_decision)["reasons"]
    assert "untrusted specialist prose" not in str(serialized_reasons)
    if recommendation == "escalate" and state in {SpecialistState.SUCCESS, SpecialistState.CACHED}:
        assert "pull_request:title" in str(serialized_reasons)


@pytest.mark.asyncio
async def test_guarded_specialist_consumer_treats_disabled_omission_as_unavailable(monkeypatch):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(consume=True)
    init_run_details()
    reviewer._prepare_review_route()
    _install_specialist_result(
        reviewer,
        state=SpecialistState.DISABLED,
        risk_enabled=False,
        include_risk_record=False,
    )
    reviewer._run_shadow_specialists_once = AsyncMock()
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.specialists_enabled", lambda: True)

    await reviewer._run_guarded_specialist_escalation()

    assert reviewer.review_route_decision.applied_depth is ReviewDepth.STANDARD
    reason_codes = [reason.code for reason in reviewer.review_route_decision.reasons]
    assert "escalation_unavailable" in reason_codes
    assert "escalation_invalid" not in reason_codes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    ["disabled_record", "enabled_omission", "duplicate", "wrong_role", "missing_config"],
)
async def test_guarded_specialist_consumer_rejects_contradictory_batches(monkeypatch, corruption):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(consume=True)
    init_run_details()
    reviewer._prepare_review_route()
    _install_specialist_result(
        reviewer,
        state=SpecialistState.SUCCESS,
        risk_enabled=corruption != "disabled_record",
        include_risk_record=corruption not in {"enabled_omission", "wrong_role", "missing_config"},
    )
    batch = reviewer.specialist_shadow_result
    if corruption == "duplicate":
        reviewer.specialist_shadow_result = replace(batch, records=(batch.records[0], batch.records[0]))
    elif corruption == "wrong_role":
        reviewer._specialist_pipeline.roles = (
            *reviewer._specialist_pipeline.roles,
            SimpleNamespace(role=SpecialistRole.CHANGE_CLASSIFICATION, enabled=True),
        )
        reviewer.specialist_shadow_result = replace(
            batch,
            records=(RoleExecution(
                role=SpecialistRole.CHANGE_CLASSIFICATION,
                state=SpecialistState.SUCCESS,
                output={},
                confidence=0.9,
            ),),
        )
    elif corruption == "missing_config":
        reviewer._specialist_pipeline.roles = ()
    reviewer._run_shadow_specialists_once = AsyncMock()
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.specialists_enabled", lambda: True)

    await reviewer._run_guarded_specialist_escalation()

    assert reviewer.review_route_decision.applied_depth is ReviewDepth.DEEP
    assert "escalation_invalid" in [reason.code for reason in reviewer.review_route_decision.reasons]


@pytest.mark.asyncio
async def test_specialist_identity_mismatch_fails_closed_to_deep(monkeypatch):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(consume=True)
    init_run_details()
    reviewer._prepare_review_route()
    _install_specialist_result(
        reviewer,
        state=SpecialistState.SUCCESS,
        identity="different-head",
    )
    reviewer._run_shadow_specialists_once = AsyncMock()
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.specialists_enabled", lambda: True)

    await reviewer._run_guarded_specialist_escalation()

    assert reviewer.review_route_decision.applied_depth is ReviewDepth.DEEP
    assert "escalation_invalid" in [reason.code for reason in reviewer.review_route_decision.reasons]


@pytest.mark.asyncio
async def test_specialist_none_cannot_lower_a_deterministic_forced_deep_route(monkeypatch):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("services/auth/guard.py")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "bugs_only"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(
        consume=True,
        sensitive_categories=({
            "name": "authorization",
            "path_patterns": ["**/auth/**"],
            "labels": [],
        },),
    )
    init_run_details()
    reviewer._prepare_review_route()
    _install_specialist_result(
        reviewer,
        state=SpecialistState.SUCCESS,
        recommendation="none",
    )
    reviewer._run_shadow_specialists_once = AsyncMock()
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.specialists_enabled", lambda: True)

    await reviewer._run_guarded_specialist_escalation()

    assert reviewer.review_route_decision.applied_depth is ReviewDepth.DEEP
    assert reviewer.review_route_decision.review_profile == "bugs_only"
    assert reviewer.review_route_decision.matched_sensitive_categories == ("authorization",)


@pytest.mark.asyncio
async def test_malformed_specialist_record_fails_closed_without_reading_raw_output(monkeypatch):
    provider = MagicMock()
    provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    provider.is_supported.return_value = True
    provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(consume=True)
    init_run_details()
    reviewer._prepare_review_route()
    _install_specialist_result(reviewer, state=SpecialistState.SUCCESS)
    reviewer.specialist_shadow_result = replace(
        reviewer.specialist_shadow_result,
        records=(object(),),
    )
    reviewer._run_shadow_specialists_once = AsyncMock()
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.specialists_enabled", lambda: True)

    await reviewer._run_guarded_specialist_escalation()

    assert reviewer.review_route_decision.applied_depth is ReviewDepth.DEEP
    assert "escalation_invalid" in [reason.code for reason in reviewer.review_route_decision.reasons]


@pytest.mark.asyncio
async def test_run_orders_routing_and_guarded_escalation_before_ticket_and_main_model():
    events = []
    reviewer = _make_prediction_reviewer()
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.git_provider.get_files.return_value = [object()]
    reviewer._prepare_review_route = MagicMock(side_effect=lambda: events.append("route"))
    reviewer._specialist_escalation_consumption_enabled = MagicMock(return_value=True)
    reviewer._run_guarded_specialist_escalation = AsyncMock(
        side_effect=lambda: events.append("specialist")
    )
    reviewer._review_model_route = MagicMock(return_value=None)
    reviewer._prepare_pr_review = MagicMock(return_value="review")
    reviewer._should_publish_review_no_suggestions = MagicMock(return_value=False)
    reviewer._clear_stale_persistent_bugs_only_review = MagicMock()

    async def extract_tickets(*args, **kwargs):
        events.append("ticket")

    async def run_main_model(*args, **kwargs):
        events.append("main")
        reviewer.prediction = VALID_PREDICTION

    settings = get_settings()
    original_publish_output = settings.config.publish_output
    settings.set("config.publish_output", False)
    try:
        with (
            patch(
                "pr_agent.tools.pr_reviewer.extract_and_cache_pr_tickets",
                side_effect=extract_tickets,
            ),
            patch(
                "pr_agent.tools.pr_reviewer.retry_with_fallback_models",
                side_effect=run_main_model,
            ),
        ):
            await reviewer.run()
    finally:
        settings.set("config.publish_output", original_publish_output)

    assert events == ["route", "specialist", "ticket", "main"]


@pytest.mark.asyncio
@pytest.mark.parametrize("head_result", [None, RuntimeError("identity failed")])
async def test_enabled_provider_without_stable_identity_records_unavailable_batch(head_result):
    git_provider = MagicMock()
    if isinstance(head_result, BaseException):
        git_provider.get_pr_head_sha.side_effect = head_result
    else:
        git_provider.get_pr_head_sha.return_value = head_result
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer._specialists_started = False
    reviewer.vars = {"title": "Change behavior"}
    reviewer.pr_description = "Description"
    reviewer.ai_handler = MagicMock()
    pipeline = MagicMock()
    unavailable = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.load_specialist_pipeline_config", return_value=pipeline),
        patch("pr_agent.tools.pr_reviewer.get_specialist_snapshot_context", return_value=None),
        patch("pr_agent.tools.pr_reviewer.unavailable_specialist_batch", return_value=unavailable) as build_unavailable,
        patch("pr_agent.tools.pr_reviewer.run_shadow_specialists", new_callable=AsyncMock) as run_shadow,
    ):
        await reviewer._run_shadow_specialists_once()

    assert reviewer.specialist_shadow_result is unavailable
    build_unavailable.assert_called_once_with(
        pipeline,
        failure_reason="stable_head_identity_unavailable",
    )
    run_shadow.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_prediction_keeps_incremental_review_compatible_with_tuple_result():
    reviewer = _make_prediction_reviewer()
    reviewer.incremental = SimpleNamespace(is_incremental=True)
    reviewer._get_prediction = AsyncMock(return_value=VALID_PREDICTION)

    with patch(
        "pr_agent.tools.pr_reviewer.get_pr_diff",
        return_value=PRDiffCoverage("diff", ["skipped.py"], ["deleted.py"]),
    ):
        await reviewer._prepare_prediction("model")

    assert reviewer.patches_diff == "diff"
    assert reviewer.remaining_files_list == ["skipped.py"]
    assert reviewer.deleted_files_list == ["deleted.py"]
    assert reviewer.prediction == VALID_PREDICTION


def _render_review(reviewer, remaining_files, supports_gfm_markdown=False):
    reviewer.prediction = "review: {}"
    reviewer.remaining_files_list = remaining_files
    reviewer.git_provider.get_diff_files.return_value = []
    reviewer.git_provider.is_supported.return_value = supports_gfm_markdown
    reviewer.set_review_labels = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.load_yaml", return_value={"review": {}}),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", return_value="original review"),
    ):
        return reviewer._prepare_pr_review()


def test_prepare_pr_review_appends_complete_coverage_footer():
    reviewer = _make_prediction_reviewer()
    settings = get_settings()
    original_enable_review_coverage_footer = settings.pr_reviewer.enable_review_coverage_footer

    try:
        settings.pr_reviewer.enable_review_coverage_footer = True
        review = _render_review(reviewer, ["src/one.py", "nested/two.md"])
    finally:
        settings.pr_reviewer.enable_review_coverage_footer = original_enable_review_coverage_footer

    assert review.startswith("original review")
    assert "⚠️ **Review coverage:**" in review
    assert "- `src/one.py`" in review
    assert "- `nested/two.md`" in review
    assert "\n\n<hr>\n\n" in review
    assert "\n\n---\n\n" not in review


def test_prepare_pr_review_hides_coverage_footer_when_disabled():
    reviewer = _make_prediction_reviewer()
    settings = get_settings()
    original_enable_review_coverage_footer = settings.pr_reviewer.enable_review_coverage_footer

    try:
        settings.pr_reviewer.enable_review_coverage_footer = False
        review = _render_review(reviewer, ["skipped.py"])
    finally:
        settings.pr_reviewer.enable_review_coverage_footer = original_enable_review_coverage_footer

    assert review == "original review"
    assert "Review coverage" not in review


def test_prepare_pr_review_places_coverage_footer_before_help_text():
    reviewer = _make_prediction_reviewer()
    settings = get_settings()
    original_enable_review_coverage_footer = settings.pr_reviewer.enable_review_coverage_footer
    original_enable_help_text = settings.pr_reviewer.enable_help_text

    try:
        settings.pr_reviewer.enable_review_coverage_footer = True
        settings.pr_reviewer.enable_help_text = True
        with patch("pr_agent.tools.pr_reviewer.HelpMessage.get_review_usage_guide", return_value="help text"):
            review = _render_review(reviewer, ["skipped.py"], supports_gfm_markdown=True)
    finally:
        settings.pr_reviewer.enable_review_coverage_footer = original_enable_review_coverage_footer
        settings.pr_reviewer.enable_help_text = original_enable_help_text

    assert review.index("⚠️ **Review coverage:**") < review.index("help text")


def test_prepare_pr_review_leaves_original_content_unchanged_without_remaining_files():
    reviewer = _make_prediction_reviewer()

    review = _render_review(reviewer, [])

    assert review == "original review"
    assert "Review coverage" not in review


def test_prepare_pr_review_publishes_omitted_files_in_structured_metadata():
    git_provider = MagicMock()
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer.prediction = "review: {}"
    reviewer.remaining_files_list = ["src/two.py", "src/one.py", "src/two.py"]
    reviewer.deleted_files_list = ["old/two.py", "old/one.py", "old/two.py"]
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    git_provider.get_diff_files.return_value = []
    git_provider.is_supported.return_value = False
    reviewer.set_review_labels = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.load_yaml", return_value={"review": {}}),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", return_value="review"),
    ):
        reviewer._prepare_pr_review()

    structured = git_provider.publish_structured_review.call_args.args[0]
    assert structured["metadata"]["omitted_files"] == ["src/one.py", "src/two.py"]
    assert structured["metadata"]["deleted_files"] == ["old/one.py", "old/two.py"]


def test_prepare_pr_review_limits_coverage_footer_to_50_files():
    reviewer = _make_prediction_reviewer()
    remaining_files = [f"file_{index}.py" for index in range(51)]

    review = _render_review(reviewer, remaining_files)

    assert review.count("- `file_") == 50
    assert "- `file_0.py`" in review
    assert "- `file_49.py`" in review
    assert "- `file_50.py`" not in review


def test_prepare_pr_review_reports_number_of_files_beyond_coverage_limit():
    reviewer = _make_prediction_reviewer()
    remaining_files = [f"file_{index}.py" for index in range(53)]

    review = _render_review(reviewer, remaining_files)

    assert "... and 3 more" in review
    assert "- `file_50.py`" not in review


def _key_issue(**overrides):
    issue = {
        "relevant_file": "app.py",
        "issue_header": "Possible Issue",
        "issue_content": "The new branch never releases the lock.",
        "start_line": 2,
        "end_line": 3,
    }
    issue.update(overrides)
    return issue


def _bugs_only_issue(**overrides):
    issue = {
        "relevant_file": "app.py",
        "issue_header": "Possible Bug",
        "issue_content": "The new branch reuses another tenant's cache entry.",
        "start_line": 2,
        "end_line": 2,
        "finding_type": "bug",
        "trigger": "Two tenants request the same record identifier.",
        "impact": "The second tenant receives the first tenant's data.",
        "root_cause": "The cache key omits the tenant identifier.",
        "duplicates_ci_failure": False,
        "matching_ci_failure": "",
    }
    issue.update(overrides)
    return issue


def _bugs_only_reviewer(*issues):
    git_provider = MagicMock()
    git_provider.get_diff_files.return_value = [
        FilePatchInfo(
            base_file="one\ntwo\nthree\n",
            head_file="one\nchanged\ntwo\nthree\n",
            patch="@@ -1,3 +1,4 @@\n one\n+changed\n two\n three\n",
            filename="app.py",
        )
    ]
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer.review_profile = "bugs_only"
    data = {"review": {"key_issues_to_review": list(issues)}}
    return reviewer, data


@pytest.mark.parametrize("record_separator", ["\r", "\u0085", "\u2028", "\u2029"])
def test_bugs_only_changed_lines_advance_only_on_lf_records(record_separator):
    git_provider = MagicMock()
    git_provider.get_diff_files.return_value = [
        FilePatchInfo(
            base_file="",
            head_file="",
            patch=(
                "@@ -0,0 +1,2 @@\n"
                f"+new{record_separator}+fake\n"
                "+real_second\n"
            ),
            filename="app.py",
        )
    ]
    reviewer = _make_prediction_reviewer(git_provider)

    assert reviewer._changed_lines_by_file()["app.py"] == {1, 2}


def test_bugs_only_keeps_a_complete_defect_and_exposes_only_the_public_finding_shape():
    reviewer, data = _bugs_only_reviewer(_bugs_only_issue())

    result = reviewer._normalize_bugs_only_review(data)

    assert result == {"review": {"key_issues_to_review": [{
        "relevant_file": "app.py",
        "issue_header": "Bug",
        "issue_content": (
            "The new branch reuses another tenant's cache entry.\n\n"
            "**Trigger:** Two tenants request the same record identifier.\n\n"
            "**Impact:** The second tenant receives the first tenant's data."
        ),
        "start_line": 2,
        "end_line": 2,
    }]}}


@pytest.mark.parametrize("issue", [
    _bugs_only_issue(finding_type="style"),
    _bugs_only_issue(start_line=1, end_line=1),
    _bugs_only_issue(trigger=""),
    _bugs_only_issue(impact=""),
])
def test_bugs_only_discards_non_defects_and_unverifiable_findings(issue):
    reviewer, data = _bugs_only_reviewer(issue)

    assert reviewer._normalize_bugs_only_review(data) == {"review": {"key_issues_to_review": []}}


def test_bugs_only_discards_ci_duplicate_only_when_failed_check_evidences_same_defect():
    issue = _bugs_only_issue(duplicates_ci_failure=True, matching_ci_failure="Unit tests")
    reviewer, data = _bugs_only_reviewer(issue)
    reviewer.ci_failure_evidence_by_name = {
        "unit tests": ["test_cache_key: cache key omits tenant identifier"],
    }

    assert reviewer._normalize_bugs_only_review(data) == {"review": {"key_issues_to_review": []}}


@pytest.mark.parametrize("evidence", [
    "Tests failed",
    "test_user_login failed because the session cookie is missing",
])
def test_bugs_only_keeps_claimed_ci_duplicate_without_same_defect_evidence(evidence):
    issue = _bugs_only_issue(duplicates_ci_failure=True, matching_ci_failure="Unit tests")
    reviewer, data = _bugs_only_reviewer(issue)
    reviewer.ci_failure_evidence_by_name = {"unit tests": [evidence]}

    result = reviewer._normalize_bugs_only_review(data)

    assert len(result["review"]["key_issues_to_review"]) == 1


@pytest.mark.parametrize("matching_ci_failure", ["", "Different check"])
def test_bugs_only_keeps_claimed_ci_duplicate_without_matching_evidence(matching_ci_failure):
    issue = _bugs_only_issue(duplicates_ci_failure=True, matching_ci_failure=matching_ci_failure)
    reviewer, data = _bugs_only_reviewer(issue)
    reviewer.ci_failure_evidence_by_name = {
        "unit tests": ["test_cache_key: cache key omits tenant identifier"],
    }

    result = reviewer._normalize_bugs_only_review(data)

    assert len(result["review"]["key_issues_to_review"]) == 1


def test_bugs_only_collapses_multiple_symptoms_with_the_same_root_cause():
    first = _bugs_only_issue()
    second = _bugs_only_issue(
        issue_content="A second endpoint leaks the same cached value.",
        trigger="The same key is requested through the batch endpoint.",
    )
    reviewer, data = _bugs_only_reviewer(first, second)

    result = reviewer._normalize_bugs_only_review(data)

    assert len(result["review"]["key_issues_to_review"]) == 1
    assert "new branch" in result["review"]["key_issues_to_review"][0]["issue_content"]


def test_bugs_only_preserves_an_empty_finding_list():
    reviewer, data = _bugs_only_reviewer()

    assert reviewer._normalize_bugs_only_review(data) == {"review": {"key_issues_to_review": []}}


def _reviewer_with_findings(*issues, head_file="one\ntwo\nthree\nfour\n"):
    git_provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
    git_provider.azure_devops_client = MagicMock()
    git_provider.azure_devops_client.get_threads.return_value = []
    git_provider.repo_slug = "repo"
    git_provider.workspace_slug = "project"
    git_provider.pr_num = 1
    git_provider.get_diff_files = MagicMock()
    git_provider.get_diff_files.return_value = [
        FilePatchInfo(base_file="", head_file=head_file, patch="", filename="app.py")
    ]
    git_provider.publish_code_suggestions = MagicMock(return_value=True)
    git_provider.max_comment_chars = None
    git_provider._inline_comment_store = None
    reviewer = _make_reviewer(git_provider)
    reviewer._published_inline_key_issue_fingerprints = MagicMock(
        side_effect=lambda _store, fingerprints: fingerprints
    )
    return reviewer, {"review": {"key_issues_to_review": list(issues)}}


def _published_comment(git_provider):
    published = git_provider.publish_code_suggestions.call_args_list[0].args[0]
    assert len(published) == 1
    return published[0]


def test_key_issues_are_published_on_their_lines_and_leave_the_summary():
    reviewer, data = _reviewer_with_findings(_key_issue())

    result = reviewer._publish_key_issues_as_inline_comments(data)

    comment = _published_comment(reviewer.git_provider)
    assert comment["relevant_file"] == "app.py"
    assert comment["relevant_lines_start"] == 2
    assert comment["relevant_lines_end"] == 3
    assert "The new branch never releases the lock." in comment["body"]
    assert "```suggestion" not in comment["body"]
    assert "key_issues_to_review" not in result["review"]
    assert len(data["review"]["key_issues_to_review"]) == 1


def test_prepare_pr_review_does_not_publish_key_issues_inline_by_default():
    reviewer = _make_prediction_reviewer()

    review = _render_review(reviewer, [])

    assert review == "original review"
    reviewer.git_provider.publish_code_suggestions.assert_not_called()


def test_prepare_pr_review_publishes_key_issues_inline_when_enabled():
    reviewer = _make_prediction_reviewer()
    settings = get_settings()
    original_inline_key_issues = settings.pr_reviewer.get("inline_key_issues", False)
    reviewer._publish_key_issues_as_inline_comments = MagicMock(return_value={"review": {}})

    try:
        settings.pr_reviewer.inline_key_issues = True
        _render_review(reviewer, [])
    finally:
        settings.pr_reviewer.inline_key_issues = original_inline_key_issues

    reviewer._publish_key_issues_as_inline_comments.assert_called_once()


@pytest.mark.parametrize("issue", [
    _key_issue(relevant_file="not_in_the_diff.py"),
    _key_issue(start_line=0, end_line=0),
    _key_issue(start_line=3, end_line=2),
    _key_issue(start_line=40, end_line=41),
    _key_issue(issue_content=""),
])
def test_unanchorable_key_issue_stays_in_the_summary(issue):
    reviewer, data = _reviewer_with_findings(issue)

    result = reviewer._publish_key_issues_as_inline_comments(data)

    reviewer.git_provider.publish_code_suggestions.assert_not_called()
    assert result["review"]["key_issues_to_review"] == [issue]


def test_key_issue_that_fails_to_publish_stays_in_the_summary():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue)
    reviewer.git_provider.publish_code_suggestions.return_value = False
    reviewer._published_inline_key_issue_fingerprints.side_effect = None
    reviewer._published_inline_key_issue_fingerprints.return_value = set()

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert result["review"]["key_issues_to_review"] == [issue]


def test_key_issue_that_cannot_be_verified_stays_in_the_summary():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue)
    reviewer._published_inline_key_issue_fingerprints.side_effect = None
    reviewer._published_inline_key_issue_fingerprints.return_value = set()

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert result["review"]["key_issues_to_review"] == [issue]


def test_key_issue_without_file_content_stays_in_the_summary():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue, head_file="")

    result = reviewer._publish_key_issues_as_inline_comments(data)

    reviewer.git_provider.publish_code_suggestions.assert_not_called()
    assert result["review"]["key_issues_to_review"] == [issue]


def test_key_issue_is_not_published_when_the_provider_cannot_verify_it():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue)
    reviewer._can_verify_inline_key_issue_publication = MagicMock(return_value=False)

    result = reviewer._publish_key_issues_as_inline_comments(data)

    reviewer.git_provider.publish_code_suggestions.assert_not_called()
    assert result is data


def test_same_key_issue_on_different_lines_is_published_at_each_location():
    first = _key_issue(start_line=1, end_line=1)
    second = _key_issue(start_line=3, end_line=3)
    reviewer, data = _reviewer_with_findings(first, second)

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert reviewer.git_provider.publish_code_suggestions.call_count == 1
    assert len(reviewer.git_provider.publish_code_suggestions.call_args.args[0]) == 2
    assert "key_issues_to_review" not in result["review"]


def test_duplicate_key_issue_is_published_once():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue, issue.copy())

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert len(reviewer.git_provider.publish_code_suggestions.call_args.args[0]) == 1
    assert "key_issues_to_review" not in result["review"]


def test_key_issues_that_diverge_after_eighty_characters_are_both_published():
    prefix = "x" * 100
    first = _key_issue(issue_content=f"{prefix}a")
    second = _key_issue(issue_content=f"{prefix}b")
    reviewer, data = _reviewer_with_findings(first, second)

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert len(reviewer.git_provider.publish_code_suggestions.call_args.args[0]) == 2
    assert "key_issues_to_review" not in result["review"]


def test_unverified_duplicate_key_issue_stays_in_the_summary():
    issue = _key_issue()
    duplicate = issue.copy()
    reviewer, data = _reviewer_with_findings(issue, duplicate)
    reviewer._published_inline_key_issue_fingerprints.side_effect = None
    reviewer._published_inline_key_issue_fingerprints.return_value = set()

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert result is data
    assert result["review"]["key_issues_to_review"] == [issue, duplicate]


def test_batch_publish_failure_keeps_unverified_findings_in_the_summary():
    failing = _key_issue(issue_content="This one raises.")
    working = _key_issue(issue_content="This one publishes.", start_line=1, end_line=1)
    reviewer, data = _reviewer_with_findings(failing, working)
    reviewer.git_provider.publish_code_suggestions.side_effect = RuntimeError("API rejected the comments")
    reviewer._published_inline_key_issue_fingerprints.side_effect = None
    reviewer._published_inline_key_issue_fingerprints.return_value = set()

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert reviewer.git_provider.publish_code_suggestions.call_count == 1
    assert result["review"]["key_issues_to_review"] == [failing, working]


def test_existing_comment_load_failure_skips_inline_publishing():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue)
    reviewer.git_provider.azure_devops_client.get_threads.side_effect = RuntimeError("request failed")

    result = reviewer._publish_key_issues_as_inline_comments(data)

    reviewer.git_provider.publish_code_suggestions.assert_not_called()
    assert result is data


def test_publish_without_a_success_record_keeps_findings_in_the_summary():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue)
    reviewer.git_provider.azure_devops_client.get_threads.return_value = []
    reviewer._published_inline_key_issue_fingerprints = (
        PRReviewer._published_inline_key_issue_fingerprints.__get__(reviewer)
    )

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert result["review"]["key_issues_to_review"] == [issue]


def test_existing_azure_thread_removes_finding_from_summary():
    reviewer, data = _reviewer_with_findings(_key_issue())
    body = "**Possible Issue**\n\nThe new branch never releases the lock."
    fingerprint = key_issue_fingerprint("app.py", body)
    reviewer.git_provider.azure_devops_client.get_threads.return_value = [
        SimpleNamespace(
            thread_context=SimpleNamespace(
                file_path="app.py",
                right_file_start=SimpleNamespace(line=2),
            ),
            comments=[SimpleNamespace(content=body_with_markers(body, fingerprint, None))],
        )
    ]

    result = reviewer._publish_key_issues_as_inline_comments(data)

    reviewer.git_provider.publish_code_suggestions.assert_not_called()
    assert "key_issues_to_review" not in result["review"]


def test_recent_azure_post_is_verified_before_thread_listing_catches_up():
    reviewer, data = _reviewer_with_findings(_key_issue())
    reviewer.git_provider.publish_code_suggestions = (
        AzureDevopsProvider.publish_code_suggestions.__get__(reviewer.git_provider)
    )
    reviewer._published_inline_key_issue_fingerprints = (
        PRReviewer._published_inline_key_issue_fingerprints.__get__(reviewer)
    )

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert reviewer.git_provider.azure_devops_client.create_thread.call_count == 1
    assert "key_issues_to_review" not in result["review"]


def test_recent_azure_post_does_not_reload_threads_for_verification():
    reviewer, data = _reviewer_with_findings(_key_issue())
    reviewer.git_provider.publish_code_suggestions = (
        AzureDevopsProvider.publish_code_suggestions.__get__(reviewer.git_provider)
    )
    reviewer._published_inline_key_issue_fingerprints = (
        PRReviewer._published_inline_key_issue_fingerprints.__get__(reviewer)
    )

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert reviewer.git_provider.azure_devops_client.get_threads.call_count == 1
    assert "key_issues_to_review" not in result["review"]


def test_same_finding_at_failed_location_stays_in_the_summary():
    published = _key_issue(start_line=1, end_line=1)
    failed = _key_issue(start_line=3, end_line=3)
    reviewer, data = _reviewer_with_findings(published, failed)
    reviewer.git_provider.publish_code_suggestions = (
        AzureDevopsProvider.publish_code_suggestions.__get__(reviewer.git_provider)
    )
    reviewer.git_provider.azure_devops_client.create_thread.side_effect = [MagicMock(), RuntimeError("failed")]
    reviewer._published_inline_key_issue_fingerprints = (
        PRReviewer._published_inline_key_issue_fingerprints.__get__(reviewer)
    )

    result = reviewer._publish_key_issues_as_inline_comments(data)

    assert reviewer.git_provider.azure_devops_client.create_thread.call_count == 2
    assert result["review"]["key_issues_to_review"] == [failed]


def test_key_issue_already_anchored_on_the_pr_is_not_published_again():
    issue = _key_issue()
    reviewer, data = _reviewer_with_findings(issue)
    store = get_inline_comment_store(reviewer.git_provider)
    store.add(key_issue_fingerprint(
        "app.py", "**Possible Issue**\n\nThe new branch never releases the lock."
    ))

    result = reviewer._publish_key_issues_as_inline_comments(data)

    reviewer.git_provider.publish_code_suggestions.assert_not_called()
    assert "key_issues_to_review" not in result["review"]


def test_key_issue_path_without_leading_slash_uses_azure_diff_path():
    reviewer, data = _reviewer_with_findings(_key_issue())
    reviewer.git_provider.get_diff_files.return_value[0].filename = "/app.py"

    reviewer._publish_key_issues_as_inline_comments(data)

    assert _published_comment(reviewer.git_provider)["relevant_file"] == "/app.py"


def test_key_issue_suggestion_fence_is_published_as_plain_code():
    issue = _key_issue(issue_content="Use this code:\n```suggestion\nlock.release()\n```")
    reviewer, data = _reviewer_with_findings(issue)

    reviewer._publish_key_issues_as_inline_comments(data)

    body = _published_comment(reviewer.git_provider)["body"]
    assert "```suggestion" not in body
    assert "```text" in body


def test_should_publish_review_no_suggestions_respects_config():
    reviewer = _make_reviewer()
    settings = get_settings()
    original_publish_no_suggestions = settings.pr_reviewer.publish_output_no_suggestions

    try:
        settings.pr_reviewer.publish_output_no_suggestions = False
        assert reviewer._should_publish_review_no_suggestions("No major issues detected") is False
        assert reviewer._should_publish_review_no_suggestions("A major issue was detected") is True

        settings.pr_reviewer.publish_output_no_suggestions = True
        assert reviewer._should_publish_review_no_suggestions("No major issues detected") is True
    finally:
        settings.pr_reviewer.publish_output_no_suggestions = original_publish_no_suggestions


@pytest.mark.asyncio
async def test_run_removes_its_progress_comment_when_quiet_output_suppresses_review(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    git_provider.publish_comment.return_value = progress_comment
    reviewer = _make_reviewer(git_provider)
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None
    reviewer._prepare_pr_review = lambda: "No major issues detected"

    async def fake_retry(prepare_fn, model_type=None):
        reviewer.prediction = "prediction"

    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", fake_retry)

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "publish_output_no_suggestions": settings.pr_reviewer.publish_output_no_suggestions,
        "is_auto_command": settings.config.get("is_auto_command", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.pr_reviewer.publish_output_no_suggestions = False

        await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.pr_reviewer.publish_output_no_suggestions = original["publish_output_no_suggestions"]

    git_provider.publish_comment.assert_called_once_with("Preparing review...", is_temporary=True)
    git_provider.remove_comment.assert_called_once_with(progress_comment)
    git_provider.remove_initial_comment.assert_not_called()
    git_provider.publish_persistent_comment.assert_not_called()


@pytest.mark.asyncio
async def test_clean_bugs_only_rerun_clears_only_bugs_only_persistent_review(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    git_provider.publish_comment.return_value = progress_comment
    reviewer = _make_reviewer(git_provider)
    reviewer.review_profile = "bugs_only"
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None
    reviewer._prepare_pr_review = lambda: ""

    async def fake_retry(prepare_fn, model_type=None):
        reviewer.prediction = "prediction"

    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", fake_retry)

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "persistent_comment": settings.pr_reviewer.persistent_comment,
        "is_auto_command": settings.config.get("is_auto_command", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.pr_reviewer.persistent_comment = True

        await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.pr_reviewer.persistent_comment = original["persistent_comment"]

    git_provider.clear_persistent_review.assert_called_once_with(
        identity_marker=PRReviewIdentity.BUGS_ONLY.value,
        name="bugs-only review",
    )
    git_provider.remove_comment.assert_called_once_with(progress_comment)
    git_provider.publish_persistent_comment.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["local", "health", "mosaico"])
async def test_shadow_only_run_retains_bounded_artifact_without_provider_mutations(
        monkeypatch, consumer):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    git_provider = MagicMock()
    git_provider.get_files.return_value = ["docs/guide.md"]
    git_provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    git_provider.is_supported.return_value = True
    git_provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(shadow_only=True)

    async def fake_retry(prepare_fn, model_type=None, model_route=None):
        reviewer.prediction = """\
review:
  estimated_effort_to_review_[1-5]: '2'
  score: '85'
  key_issues_to_review:
    - issue_header: first bounded finding
    - issue_header: second bounded finding
    - issue_header: third over-budget finding
  security_concerns: 'No'
"""

    def render_bounded_review(data, *_args, **_kwargs):
        headers = [
            issue["issue_header"]
            for issue in data["review"]["key_issues_to_review"]
        ]
        return "## PR Reviewer Guide\n\n" + "\n".join(headers)

    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", fake_retry)
    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(pr_reviewer_module, "convert_to_markdown_v2", render_bounded_review)

    settings = get_settings()
    snapshot = snapshot_settings((
        "config.publish_output",
        "config.is_auto_command",
        "config.output_run_details",
        "config.output_relevant_configurations",
        "data",
        "pr_reviewer.persistent_comment",
        "pr_reviewer.inline_key_issues",
        "pr_reviewer.enable_help_text",
    ))
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.output_run_details = False
        settings.config.output_relevant_configurations = False
        settings.pr_reviewer.persistent_comment = True
        settings.pr_reviewer.inline_key_issues = True
        settings.pr_reviewer.enable_help_text = False
        settings.data = {"artifact": "STALE REVIEW"}

        await reviewer.run()

        if consumer == "local":
            artifact = settings.data["artifact"]
        elif consumer == "health":
            artifact = dict(settings.data)["artifact"]
        else:
            from pr_agent.mosaico.dispatch import _capture_artifact
            artifact = _capture_artifact()
    finally:
        restore_settings(snapshot)

    assert reviewer.review_route_decision.applied_depth is ReviewDepth.QUICK
    assert reviewer._review_shadow_only is True
    assert artifact.startswith("## PR Reviewer Guide")
    assert "first bounded finding" in artifact
    assert "second bounded finding" in artifact
    assert "third over-budget finding" not in artifact
    for method_name in (
        "publish_comment",
        "remove_comment",
        "remove_initial_comment",
        "clear_persistent_review",
        "publish_persistent_comment",
        "publish_structured_review",
        "publish_code_suggestions",
        "publish_inline_comments",
        "publish_labels",
        "set_pr_labels",
    ):
        getattr(git_provider, method_name).assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["unavailable", "failed"])
async def test_shadow_only_unavailable_or_failed_result_clears_stale_artifact_without_mutations(
        monkeypatch, result):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    git_provider = MagicMock()
    git_provider.get_files.return_value = ["docs/guide.md"]
    git_provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    git_provider.is_supported.return_value = True
    git_provider.get_pr_labels.return_value = []
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration(shadow_only=True)

    async def unavailable_retry(prepare_fn, model_type=None, model_route=None):
        return None

    retry = unavailable_retry
    if result == "failed":
        retry = AsyncMock(side_effect=RuntimeError("shadow model failed"))
    render = MagicMock(return_value="must not render")
    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", retry)
    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(pr_reviewer_module, "convert_to_markdown_v2", render)

    settings = get_settings()
    snapshot = snapshot_settings((
        "config.publish_output",
        "config.is_auto_command",
        "config.propagate_tool_errors",
        "data",
    ))
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.propagate_tool_errors = False
        settings.data = {"artifact": "STALE REVIEW"}

        await reviewer.run()
        artifact = settings.data["artifact"]
    finally:
        restore_settings(snapshot)

    assert artifact == ""
    render.assert_not_called()
    for method_name in (
        "publish_comment",
        "remove_comment",
        "remove_initial_comment",
        "clear_persistent_review",
        "publish_persistent_comment",
        "publish_structured_review",
        "publish_code_suggestions",
        "publish_inline_comments",
        "publish_labels",
    ):
        getattr(git_provider, method_name).assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("propagate_tool_errors", [False, True])
async def test_run_removes_its_progress_comment_when_review_generation_fails(
        monkeypatch, propagate_tool_errors):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    git_provider.publish_comment.return_value = progress_comment
    reviewer = _make_reviewer(git_provider)
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None

    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    review_error = RuntimeError("model unavailable")
    monkeypatch.setattr(
        pr_reviewer_module,
        "retry_with_fallback_models",
        AsyncMock(side_effect=review_error),
    )

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "is_auto_command": settings.config.get("is_auto_command", False),
        "propagate_tool_errors": settings.config.get("propagate_tool_errors", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.propagate_tool_errors = propagate_tool_errors

        if propagate_tool_errors:
            with pytest.raises(RuntimeError, match="model unavailable") as exc_info:
                await reviewer.run()
            assert exc_info.value is review_error
        else:
            await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.config.propagate_tool_errors = original["propagate_tool_errors"]

    assert git_provider.publish_comment.call_args_list == [
        (("Preparing review...",), {"is_temporary": True}),
        (("Failed to review PR",), {}),
    ]
    git_provider.remove_comment.assert_called_once_with(progress_comment)
    git_provider.remove_initial_comment.assert_not_called()


@pytest.mark.asyncio
async def test_run_publishes_failure_result_when_progress_comment_has_no_handle(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    git_provider.publish_comment.side_effect = [None, MagicMock()]
    reviewer = _make_reviewer(git_provider)
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None

    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(
        pr_reviewer_module,
        "retry_with_fallback_models",
        AsyncMock(side_effect=RuntimeError("model unavailable")),
    )

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "is_auto_command": settings.config.get("is_auto_command", False),
        "propagate_tool_errors": settings.config.get("propagate_tool_errors", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.propagate_tool_errors = False

        await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.config.propagate_tool_errors = original["propagate_tool_errors"]

    assert git_provider.publish_comment.call_args_list == [
        (("Preparing review...",), {"is_temporary": True}),
        (("Failed to review PR",), {}),
    ]
    git_provider.remove_comment.assert_not_called()
    git_provider.remove_initial_comment.assert_not_called()


@pytest.mark.asyncio
async def test_run_publishes_failure_result_when_review_fails_before_progress_comment():
    review_error = RuntimeError("files unavailable")
    git_provider = MagicMock()
    git_provider.get_files.side_effect = review_error
    reviewer = _make_reviewer(git_provider)

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "is_auto_command": settings.config.get("is_auto_command", False),
        "propagate_tool_errors": settings.config.get("propagate_tool_errors", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.propagate_tool_errors = False

        await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.config.propagate_tool_errors = original["propagate_tool_errors"]

    git_provider.publish_comment.assert_called_once_with("Failed to review PR")
    git_provider.remove_comment.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("publish_output", "is_auto_command", "propagate_tool_errors"),
    [
        (False, False, False),
        (False, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
async def test_run_does_not_publish_failure_result_when_output_disabled_or_auto(
        monkeypatch, publish_output, is_auto_command, propagate_tool_errors):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    reviewer = _make_reviewer(git_provider)
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None

    review_error = RuntimeError("model unavailable")
    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(
        pr_reviewer_module,
        "retry_with_fallback_models",
        AsyncMock(side_effect=review_error),
    )

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "is_auto_command": settings.config.get("is_auto_command", False),
        "propagate_tool_errors": settings.config.get("propagate_tool_errors", False),
    }
    try:
        settings.config.publish_output = publish_output
        settings.config.is_auto_command = is_auto_command
        settings.config.propagate_tool_errors = propagate_tool_errors

        if propagate_tool_errors:
            with pytest.raises(RuntimeError, match="model unavailable") as exc_info:
                await reviewer.run()
            assert exc_info.value is review_error
        else:
            await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.config.propagate_tool_errors = original["propagate_tool_errors"]

    git_provider.publish_comment.assert_not_called()
    git_provider.remove_comment.assert_not_called()
    git_provider.remove_initial_comment.assert_not_called()


@pytest.mark.asyncio
async def test_run_publishes_failure_result_when_progress_cleanup_fails(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    git_provider.publish_comment.return_value = progress_comment
    git_provider.remove_comment.side_effect = RuntimeError("cleanup failed")
    reviewer = _make_reviewer(git_provider)
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None

    review_error = RuntimeError("model unavailable")
    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(
        pr_reviewer_module,
        "retry_with_fallback_models",
        AsyncMock(side_effect=review_error),
    )

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "is_auto_command": settings.config.get("is_auto_command", False),
        "propagate_tool_errors": settings.config.get("propagate_tool_errors", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.propagate_tool_errors = True

        with pytest.raises(RuntimeError, match="model unavailable") as exc_info:
            await reviewer.run()
        assert exc_info.value is review_error
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.config.propagate_tool_errors = original["propagate_tool_errors"]

    assert git_provider.publish_comment.call_args_list == [
        (("Preparing review...",), {"is_temporary": True}),
        (("Failed to review PR",), {}),
    ]
    git_provider.remove_comment.assert_called_once_with(progress_comment)


@pytest.mark.asyncio
@pytest.mark.parametrize("propagate_tool_errors", [False, True])
async def test_run_failure_result_publication_does_not_mask_review_error(
        monkeypatch, propagate_tool_errors):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    publication_error = RuntimeError("comment unavailable")
    git_provider = MagicMock()
    git_provider.get_files.return_value = ["app.py"]
    git_provider.publish_comment.side_effect = [progress_comment, publication_error]
    reviewer = _make_reviewer(git_provider)
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None

    review_error = RuntimeError("model unavailable")
    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", AsyncMock())
    monkeypatch.setattr(
        pr_reviewer_module,
        "retry_with_fallback_models",
        AsyncMock(side_effect=review_error),
    )

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "is_auto_command": settings.config.get("is_auto_command", False),
        "propagate_tool_errors": settings.config.get("propagate_tool_errors", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.config.propagate_tool_errors = propagate_tool_errors

        if propagate_tool_errors:
            with pytest.raises(RuntimeError, match="model unavailable") as exc_info:
                await reviewer.run()
            assert exc_info.value is review_error
        else:
            await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.config.propagate_tool_errors = original["propagate_tool_errors"]

    assert git_provider.publish_comment.call_args_list == [
        (("Preparing review...",), {"is_temporary": True}),
        (("Failed to review PR",), {}),
    ]
    git_provider.remove_comment.assert_called_once_with(progress_comment)


def test_prepare_review_publishes_provider_neutral_structured_data(monkeypatch):
    git_provider = MagicMock()
    git_provider.is_supported.return_value = False
    git_provider.get_diff_files.return_value = []
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer.prediction = """review:
  key_issues_to_review: []
  security_concerns: no
"""
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.set_review_labels = MagicMock()
    monkeypatch.setattr(
        "pr_agent.tools.pr_reviewer.convert_to_markdown_v2",
        lambda *args, **kwargs: "## Review",
    )

    from pr_agent.algo.run_details import add_token_usage, init_run_details

    init_run_details()
    add_token_usage({"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42})

    reviewer._prepare_pr_review()

    git_provider.publish_structured_review.assert_called_once_with({
        "review": {
            "key_issues_to_review": [],
            "security_concerns": False,
        },
        "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        "metadata": {"review_profile": "full", "omitted_files": [], "deleted_files": []},
    })
    # Assert key order to prove the snapshot is isolated: _prepare_pr_review moves
    # key_issues_to_review to the end of its own dict after the hook fires, so an
    # aliased snapshot ends with it while a deep copy keeps the original order.
    # (assert_called_once_with cannot catch this: dict equality ignores key order.)
    published = git_provider.publish_structured_review.call_args[0][0]
    assert list(published["review"].keys()) == ["key_issues_to_review", "security_concerns"]


def test_structured_review_includes_applied_route_without_aliasing_decision(monkeypatch):
    git_provider = MagicMock()
    git_provider.is_supported.return_value = True
    git_provider.get_pr_labels.return_value = []
    git_provider.get_diff_files.return_value = [_route_file("docs/guide.md")]
    reviewer = _make_prediction_reviewer(git_provider)
    reviewer.review_profile = "full"
    reviewer.vars = {}
    reviewer.review_routing_configuration = _routing_configuration()
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.set_review_labels = MagicMock()
    init_run_details()
    decision = reviewer._prepare_review_route()
    reviewer.prediction = """review:
  key_issues_to_review: []
  security_concerns: no
"""
    monkeypatch.setattr(
        "pr_agent.tools.pr_reviewer.convert_to_markdown_v2",
        lambda *args, **kwargs: "## Review",
    )

    reviewer._prepare_pr_review()

    metadata = git_provider.publish_structured_review.call_args[0][0]["metadata"]
    assert metadata["review_route"] == review_route_decision_to_dict(decision)
    metadata["review_route"]["reasons"].append({"code": "mutated"})
    assert "mutated" not in str(review_route_decision_to_dict(decision))


def test_bugs_only_publishes_structured_empty_list_but_no_markdown():
    reviewer, _ = _bugs_only_reviewer()
    reviewer.prediction = "review:\n  key_issues_to_review: []\n"
    reviewer.git_provider.is_supported.return_value = False
    reviewer.git_provider.publish_structured_review = MagicMock()
    reviewer.set_review_labels = MagicMock()

    from pr_agent.algo.run_details import init_run_details

    init_run_details()
    review = reviewer._prepare_pr_review()

    assert review == ""
    reviewer.set_review_labels.assert_not_called()
    reviewer.git_provider.publish_structured_review.assert_called_once_with({
        "review": {"key_issues_to_review": []},
        "usage": {},
        "metadata": {"review_profile": "bugs_only", "omitted_files": [], "deleted_files": []},
    })


def test_can_run_incremental_review_skips_auto_mode_without_new_commit():
    reviewer = _make_reviewer()
    reviewer.is_auto = True
    reviewer.incremental = SimpleNamespace(first_new_commit_sha=None)

    assert reviewer._can_run_incremental_review() is False


def test_set_review_labels_replaces_stale_review_labels_and_keeps_user_labels():
    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "require_estimate_effort_to_review": settings.pr_reviewer.require_estimate_effort_to_review,
        "require_security_review": settings.pr_reviewer.require_security_review,
        "enable_review_labels_effort": settings.pr_reviewer.enable_review_labels_effort,
        "enable_review_labels_security": settings.pr_reviewer.enable_review_labels_security,
    }
    settings.config.publish_output = True
    settings.pr_reviewer.require_estimate_effort_to_review = True
    settings.pr_reviewer.require_security_review = True
    settings.pr_reviewer.enable_review_labels_effort = True
    settings.pr_reviewer.enable_review_labels_security = True
    git_provider = MagicMock()
    git_provider.get_pr_labels.return_value = ["Review effort 1/5", "Possible security concern", "keep-me"]
    reviewer = _make_reviewer(git_provider)
    data = {
        "review": {
            "estimated_effort_to_review_[1-5]": "3, moderate",
            "security_concerns": "yes",
        }
    }

    try:
        reviewer.set_review_labels(data)

        git_provider.publish_labels.assert_called_once_with([
            "Review effort 3/5",
            "Possible security concern",
            "keep-me",
        ])
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.pr_reviewer.require_estimate_effort_to_review = original["require_estimate_effort_to_review"]
        settings.pr_reviewer.require_security_review = original["require_security_review"]
        settings.pr_reviewer.enable_review_labels_effort = original["enable_review_labels_effort"]
        settings.pr_reviewer.enable_review_labels_security = original["enable_review_labels_security"]


def test_get_user_answers_collects_question_and_answer_from_issue_comments():
    git_provider = MagicMock()
    git_provider.get_issue_comments.return_value = [
        SimpleNamespace(body="Unrelated"),
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ]
    reviewer = _make_reviewer(git_provider)
    reviewer.is_answer = True

    question, answer = reviewer._get_user_answers()

    assert question == "Questions to better understand the PR:\n- Why?"
    assert answer == "/answer Because it fixes production."


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("thread_enabled", [True, False])
@pytest.mark.parametrize(
    ("review_profile", "expected_identity", "expected_name", "expected_legacy_header"),
    [
        ("full", PRReviewIdentity.REGULAR.value, "review", f"{PRReviewHeader.REGULAR.value} 🔍"),
        ("bugs_only", PRReviewIdentity.BUGS_ONLY.value, "bugs-only review", None),
    ],
)
async def test_run_threads_only_the_final_review_comment(
        monkeypatch, persistent, thread_enabled, review_profile, expected_identity, expected_name,
        expected_legacy_header):
    """`as_thread` is forwarded to the review's final publish call only when the provider opts in
    (should_publish_review_as_thread), and is omitted entirely otherwise - other providers'
    publish methods don't accept it. Status/progress comments are never threaded.
    """
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    git_provider = MagicMock()
    git_provider.should_publish_review_as_thread.return_value = thread_enabled
    git_provider.supports_review_comment_identity.return_value = False
    git_provider.publish_comment.return_value = progress_comment
    reviewer = _make_reviewer(git_provider)
    reviewer.review_profile = review_profile
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.vars = {}
    reviewer.prediction = None
    review_text = "## PR Reviewer Guide 🔍\n\nsome findings"
    reviewer._prepare_pr_review = lambda: review_text

    async def fake_extract_tickets(git_provider, vars):
        return None

    async def fake_retry(prepare_fn, model_type=None):
        reviewer.prediction = "prediction"

    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", fake_extract_tickets)
    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", fake_retry)

    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "persistent_comment": settings.pr_reviewer.persistent_comment,
        "is_auto_command": settings.config.get("is_auto_command", False),
    }
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.pr_reviewer.persistent_comment = persistent

        await reviewer.run()
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.config.is_auto_command = original["is_auto_command"]
        settings.pr_reviewer.persistent_comment = original["persistent_comment"]

    if persistent:
        publish = git_provider.publish_persistent_comment
        publish.assert_called_once()
        assert publish.call_args.kwargs["name"] == expected_name
        assert publish.call_args.kwargs["identity_marker"] == expected_identity
        assert publish.call_args.kwargs["legacy_initial_header"] == expected_legacy_header
    else:
        publish = git_provider.publish_comment
    assert publish.call_args.args[0] == review_text
    if thread_enabled:
        assert publish.call_args.kwargs.get("as_thread") is True
    else:
        assert "as_thread" not in publish.call_args.kwargs
    # The temporary progress comment is published without as_thread regardless of the flag.
    git_provider.publish_comment.assert_any_call("Preparing review...", is_temporary=True)
    git_provider.remove_comment.assert_called_once_with(progress_comment)
    git_provider.remove_initial_comment.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_incremental", "review_profile", "expected_identity"),
    [
        (False, "full", PRReviewIdentity.REGULAR.value),
        (True, "full", PRReviewIdentity.FULL_INCREMENTAL.value),
        (True, "bugs_only", PRReviewIdentity.BUGS_ONLY_INCREMENTAL.value),
    ],
)
async def test_nonpersistent_review_adds_identity_for_incremental_capable_provider(
    monkeypatch,
    is_incremental,
    review_profile,
    expected_identity,
):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    progress_comment = MagicMock()
    git_provider = MagicMock()
    git_provider.should_publish_review_as_thread.return_value = False
    git_provider.supports_review_comment_identity.return_value = True
    git_provider.publish_comment.return_value = progress_comment
    reviewer = _make_reviewer(git_provider)
    reviewer.review_profile = review_profile
    reviewer.incremental = SimpleNamespace(is_incremental=is_incremental)
    if is_incremental:
        reviewer._can_run_incremental_review = lambda: True
    reviewer.vars = {}
    reviewer.prediction = None
    reviewer._prepare_pr_review = lambda: "## Team Review 🔍\n\nsome findings"

    async def fake_extract_tickets(git_provider, vars):
        return None

    async def fake_retry(prepare_fn, model_type=None):
        reviewer.prediction = "prediction"

    monkeypatch.setattr(pr_reviewer_module, "extract_and_cache_pr_tickets", fake_extract_tickets)
    monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", fake_retry)

    settings = get_settings()
    original_publish_output = settings.config.publish_output
    original_persistent_comment = settings.pr_reviewer.persistent_comment
    original_auto_command = settings.config.get("is_auto_command", False)
    try:
        settings.config.publish_output = True
        settings.config.is_auto_command = False
        settings.pr_reviewer.persistent_comment = False

        await reviewer.run()
    finally:
        settings.config.publish_output = original_publish_output
        settings.config.is_auto_command = original_auto_command
        settings.pr_reviewer.persistent_comment = original_persistent_comment

    published_review = [
        call
        for call in git_provider.publish_comment.call_args_list
        if call.args and call.args[0].startswith("## Team Review")
    ]
    assert len(published_review) == 1
    assert expected_identity in published_review[0].args[0]


def test_init_maps_user_question_and_answer_to_correct_prompt_vars(monkeypatch):
    """Behavioral regression for the swapped-unpacking bug (#2496).

    The bug lived in ``PRReviewer.__init__``: ``_get_user_answers()`` returns
    ``(question, answer)`` but the tuple was unpacked as ``answer, question``,
    so the review prompt rendered the user's answer under ``{{ question_str }}``
    and the question under ``{{ answer_str }}``. This drives the real ``__init__``
    (external collaborators stubbed) and asserts each value lands in ``self.vars``
    under the correct key — so it fails if the unpack is ever swapped again,
    regardless of how the line is formatted.
    """
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_languages.return_value = {}
    provider.get_files.return_value = []
    provider.get_issue_comments.return_value = [
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ]
    provider.get_pr_description.return_value = ("desc", [])

    monkeypatch.setattr(pr_reviewer_module, "get_git_provider_with_context", lambda pr_url: provider)
    monkeypatch.setattr(pr_reviewer_module, "get_main_pr_language", lambda languages, files: "Python")
    monkeypatch.setattr(pr_reviewer_module, "TokenHandler", MagicMock())

    reviewer = PRReviewer(
        "https://example/pr/1",
        is_answer=True,
        ai_handler=lambda: SimpleNamespace(main_pr_language=None),
    )

    assert reviewer.vars["question_str"] == "Questions to better understand the PR:\n- Why?"
    assert reviewer.vars["answer_str"] == "/answer Because it fixes production."


def _build_answer_mode_reviewer(monkeypatch, issue_comments):
    """Drive the real ``PRReviewer.__init__`` in answer mode over ``issue_comments``."""
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_languages.return_value = {}
    provider.get_files.return_value = []
    provider.get_issue_comments.return_value = issue_comments
    provider.get_pr_description.return_value = ("desc", [])

    monkeypatch.setattr(pr_reviewer_module, "get_git_provider_with_context", lambda pr_url: provider)
    monkeypatch.setattr(pr_reviewer_module, "get_main_pr_language", lambda languages, files: "Python")
    monkeypatch.setattr(pr_reviewer_module, "TokenHandler", MagicMock())

    return PRReviewer(
        "https://example/pr/1",
        is_answer=True,
        ai_handler=lambda: SimpleNamespace(main_pr_language=None),
    )


def test_answer_mode_reads_comments_from_a_non_list_iterable(monkeypatch):
    """GitHub hands back a PyGithub ``PaginatedList``, GitLab a plain list.

    Answer mode used to reach for the PyGithub-only ``.reversed`` property, which meant
    it could only ever consume the GitHub shape. Any lazily-paginated iterable must work.
    """

    class _Paginated:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

    reviewer = _build_answer_mode_reviewer(monkeypatch, _Paginated([
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ]))

    assert reviewer.vars["question_str"] == "Questions to better understand the PR:\n- Why?"
    assert reviewer.vars["answer_str"] == "/answer Because it fixes production."


def test_answer_mode_uses_the_lazy_reversed_view_when_the_provider_offers_one(monkeypatch):
    """PyGithub reverses a PaginatedList lazily, walking pages from the end.

    Materialising it instead would page the whole thread just to read the last exchange,
    so the lazy view must win when it exists.
    """

    class _LazyPaginated:
        def __init__(self, items):
            self._items = items

        @property
        def reversed(self):
            return list(reversed(self._items))

        def __iter__(self):
            raise AssertionError("the lazy reversed view should have been used")

    reviewer = _build_answer_mode_reviewer(monkeypatch, _LazyPaginated([
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ]))

    assert reviewer.vars["question_str"] == "Questions to better understand the PR:\n- Why?"
    assert reviewer.vars["answer_str"] == "/answer Because it fixes production."


def test_answer_mode_prefers_the_newest_question_and_answer(monkeypatch):
    """Comments arrive oldest-first, so the walk must run newest-first to pick the latest exchange."""
    reviewer = _build_answer_mode_reviewer(monkeypatch, [
        SimpleNamespace(body="Questions to better understand the PR:\n- Stale question?"),
        SimpleNamespace(body="/answer Stale answer."),
        SimpleNamespace(body="Questions to better understand the PR:\n- Current question?"),
        SimpleNamespace(body="/answer Current answer."),
    ])

    assert reviewer.vars["question_str"] == "Questions to better understand the PR:\n- Current question?"
    assert reviewer.vars["answer_str"] == "/answer Current answer."


def _reviewer_with_prediction(prediction):
    """A bare PRReviewer carrying only the prediction, for the parse guard."""
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.prediction = prediction
    return reviewer


@pytest.mark.parametrize(
    "prediction, expect_error",
    [
        ("review:\n  score: 90\nsecurity_concerns: No\n", False),
        # Seen live: an unquoted colon inside a value derails the YAML parser.
        ("review:\n  summary: note: this breaks\n   - bad indent\n", True),
        ("", True),
        (None, True),
        ("not_a_review:\n  x: 1\n", True),
        ("review: null\nsecurity_concerns: No\n", True),
        ("review: text\nsecurity_concerns: No\n", True),
        ("review: {}\nsecurity_concerns: No\n", True),
    ],
)
def test_unparsable_prediction_is_rejected_so_the_fallback_model_runs(prediction, expect_error):
    reviewer = _reviewer_with_prediction(prediction)
    if expect_error:
        with pytest.raises(ValueError):
            reviewer._reject_unparsable_prediction("openai/some-model")
    else:
        reviewer._reject_unparsable_prediction("openai/some-model")
