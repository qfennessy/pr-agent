import copy
import json
from dataclasses import replace

import pytest

from pr_agent.algo import review_configuration
from pr_agent.algo.review_configuration import (
    ReviewConfigurationBundle,
    materialize_review_configuration,
    replay_review_configuration,
    review_configuration_canonical_bytes,
)
from pr_agent.config_loader import get_settings
from tests.unittest._settings_helpers import restore_settings, snapshot_settings


def _payload(bundle: ReviewConfigurationBundle) -> dict:
    return json.loads(review_configuration_canonical_bytes(bundle))


def test_bundle_round_trip_is_canonical_and_content_addressed():
    bundle = materialize_review_configuration("pinned skills", {"AGENTS.md": "rules"}, prompt_date="")

    restored = ReviewConfigurationBundle.from_dict(_payload(bundle))

    assert restored == bundle
    assert review_configuration_canonical_bytes(restored) == review_configuration_canonical_bytes(bundle)
    assert restored.configuration_hash.startswith("sha256:")


def test_model_visible_setting_change_changes_configuration_hash():
    bundle = materialize_review_configuration("", {})
    changed_settings = dict(bundle.settings)
    changed_settings["config.model"] = "gpt-different-model"

    changed = replace(bundle, settings=changed_settings)

    assert changed.configuration_hash != bundle.configuration_hash


def test_repository_context_order_changes_configuration_hash():
    first = materialize_review_configuration("", {"AGENTS.md": "rules", "CLAUDE.md": "more rules"})
    second = materialize_review_configuration("", {"CLAUDE.md": "more rules", "AGENTS.md": "rules"})

    assert first.configuration_hash != second.configuration_hash
    assert list(first.repo_context_files) == ["AGENTS.md", "CLAUDE.md"]
    assert list(ReviewConfigurationBundle.from_dict(_payload(first)).repo_context_files) == [
        "AGENTS.md",
        "CLAUDE.md",
    ]


def test_repository_context_line_budget_changes_configuration_hash():
    first = materialize_review_configuration(
        "",
        {"AGENTS.md": "first\nsecond"},
        repo_context_max_lines=1,
    )
    second = materialize_review_configuration(
        "",
        {"AGENTS.md": "first\nsecond"},
        repo_context_max_lines=500,
    )

    assert first.configuration_hash != second.configuration_hash
    assert ReviewConfigurationBundle.from_dict(_payload(first)).repo_context_max_lines == 1


def test_materializer_excludes_credentials_callbacks_and_output_sinks():
    protected = snapshot_settings((
        "azure_devops.pat",
        "litellm.success_callback",
        "otel.headers",
        "push_outputs.webhook_url",
    ))
    sentinel = "never-serialize-this-secret"
    settings = get_settings()
    try:
        settings.set("azure_devops.pat", sentinel)
        settings.set("litellm.success_callback", [sentinel])
        settings.set("otel.headers", {"authorization": sentinel})
        settings.set("push_outputs.webhook_url", f"https://example.invalid/{sentinel}")

        encoded = review_configuration_canonical_bytes(materialize_review_configuration("", {}))

        assert sentinel.encode() not in encoded
        assert b"success_callback" not in encoded
        assert b"push_outputs" not in encoded
        assert b"azure_devops" not in encoded
    finally:
        restore_settings(protected)


def test_bundle_rejects_unknown_tampered_and_case_colliding_data():
    bundle = materialize_review_configuration("", {})
    unknown = _payload(bundle)
    unknown["settings"]["github.token"] = "secret"
    unknown["configuration_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="unsupported settings"):
        ReviewConfigurationBundle.from_dict(unknown)

    tampered = _payload(bundle)
    tampered["settings"]["config.model"] = "gpt-tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        ReviewConfigurationBundle.from_dict(tampered)

    collision = dict(bundle.settings)
    collision["review_depth"] = {"enabled": False, "ENABLED": False}
    with pytest.raises(ValueError, match="case-colliding"):
        replace(bundle, settings=collision)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("pr_reviewer.enable_candidate_verification", True),
        ("pr_reviewer.enable_frontier_adjudication", True),
        ("specialist_pipeline.enabled", True),
    ),
)
def test_bundle_rejects_unsupported_production_stages(path, value):
    bundle = materialize_review_configuration("", {})
    changed_settings = dict(bundle.settings)
    changed_settings[path] = value

    with pytest.raises(ValueError, match="unsupported production stage"):
        replace(bundle, settings=changed_settings)


def test_bundle_rejects_credential_bearing_endpoint_and_extra_body():
    bundle = materialize_review_configuration("", {})
    endpoint_settings = dict(bundle.settings)
    endpoint_settings["openai.api_base"] = "https://user:password@example.invalid/v1"
    with pytest.raises(ValueError, match="credential-free HTTP endpoint"):
        replace(bundle, settings=endpoint_settings)

    body_settings = dict(bundle.settings)
    body_settings["litellm.extra_body"] = json.dumps({"authorization": "secret"})
    with pytest.raises(ValueError, match="unsupported keys"):
        replace(bundle, settings=body_settings)


def test_materializer_rejects_ambient_extra_headers():
    protected = snapshot_settings(("litellm.extra_headers",))
    settings = get_settings()
    try:
        settings.set("litellm.extra_headers", {"authorization": "secret"})

        with pytest.raises(ValueError, match="unsupported setting: litellm.extra_headers"):
            materialize_review_configuration("", {})
    finally:
        restore_settings(protected)


def test_bundle_restricts_v1_to_standard_openai_route():
    bundle = materialize_review_configuration("", {})
    provider_settings = dict(bundle.settings)
    provider_settings["config.model"] = "anthropic/claude-sonnet"
    with pytest.raises(ValueError, match="unsupported model provider"):
        replace(bundle, settings=provider_settings)

    endpoint_settings = dict(bundle.settings)
    endpoint_settings["openai.api_base"] = "https://proxy.example.invalid/v1"
    with pytest.raises(ValueError, match="unsupported OpenAI endpoint"):
        replace(
            bundle,
            settings=endpoint_settings,
            missing_settings=tuple(path for path in bundle.missing_settings if path != "openai.api_base"),
        )

    openrouter_endpoint_settings = dict(bundle.settings)
    openrouter_endpoint_settings["openrouter.api_base"] = "https://openrouter.example.invalid/v1"
    with pytest.raises(ValueError, match="unsupported OpenRouter endpoint"):
        replace(
            bundle,
            settings=openrouter_endpoint_settings,
            missing_settings=tuple(path for path in bundle.missing_settings if path != "openrouter.api_base"),
        )


def test_legacy_snapshot_hash_remains_provider_neutral():
    protected = snapshot_settings(("config.model",))
    settings = get_settings()
    try:
        settings.set("config.model", "anthropic/claude-sonnet")

        configuration_hash = review_configuration.snapshot_review_configuration_hash("", {})

        assert configuration_hash.startswith("sha256:")
    finally:
        restore_settings(protected)


def test_bundle_normalizes_supported_dynaconf_string_values():
    bundle = materialize_review_configuration("", {})
    string_settings = dict(bundle.settings)
    string_settings["openrouter.max_tokens"] = "2048"
    string_settings["openrouter.allow_fallbacks"] = "false"
    string_settings["openrouter.provider_order"] = "first, second"

    normalized = replace(bundle, settings=string_settings)

    assert normalized.settings["openrouter.max_tokens"] == 2048
    assert normalized.settings["openrouter.allow_fallbacks"] is False
    assert normalized.settings["openrouter.provider_order"] == ("first", "second")


def test_replay_pins_captured_values_and_restores_ambient_settings():
    protected = snapshot_settings(("config.model", "config.reasoning_effort"))
    settings = get_settings()
    try:
        settings.set("config.model", "gpt-captured-model")
        settings.set("config.reasoning_effort", "low")
        bundle = materialize_review_configuration("", {})
        settings.set("config.model", "gpt-ambient-model")
        settings.set("config.reasoning_effort", "high")

        with replay_review_configuration(bundle):
            assert settings.get("config.model") == "gpt-captured-model"
            assert settings.get("config.reasoning_effort") == "low"

        assert settings.get("config.model") == "gpt-ambient-model"
        assert settings.get("config.reasoning_effort") == "high"
    finally:
        restore_settings(protected)


def test_replay_removes_setting_that_was_missing_at_capture():
    protected = snapshot_settings(("litellm.model_id",))
    settings = get_settings()
    try:
        review_configuration._unset_setting(settings, "litellm.model_id")
        bundle = materialize_review_configuration("", {})
        settings.set("litellm.model_id", "ambient-model-id")

        with replay_review_configuration(bundle):
            assert settings.get("litellm.model_id", None) is None

        assert settings.get("litellm.model_id") == "ambient-model-id"
    finally:
        restore_settings(protected)


def test_bundle_rejects_runtime_artifact_mismatch(monkeypatch):
    bundle = materialize_review_configuration("", {})
    monkeypatch.setattr(review_configuration, "_runtime_artifact_hash", lambda: "sha256:" + "f" * 64)

    with pytest.raises(ValueError, match="runtime mismatch"):
        bundle.require_compatible_runtime()


def test_runtime_artifact_identity_includes_dependency_lock(monkeypatch, tmp_path):
    package = tmp_path / "pr_agent"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "pr-agent"\n', encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(review_configuration, "_PACKAGE_ROOT", tmp_path)
    first = review_configuration._runtime_artifact_hash()

    lock.write_text("version = 2\n", encoding="utf-8")

    assert review_configuration._runtime_artifact_hash() != first


def test_runtime_artifact_identity_requires_dependency_lock(monkeypatch, tmp_path):
    package = tmp_path / "pr_agent"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "pr-agent"\n', encoding="utf-8")
    monkeypatch.setattr(review_configuration, "_PACKAGE_ROOT", tmp_path)

    with pytest.raises(ValueError, match="dependency identity is unavailable"):
        review_configuration._runtime_artifact_hash()


def test_runtime_artifact_identity_includes_installed_dependency_version(monkeypatch, tmp_path):
    package = tmp_path / "pr_agent"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "pr-agent"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "litellm"\nversion = "1"\n', encoding="utf-8")
    installed = {"litellm": "1"}
    monkeypatch.setattr(review_configuration, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(review_configuration, "version", lambda name: installed[name])
    first = review_configuration._runtime_artifact_hash()

    installed["litellm"] = "2"

    assert review_configuration._runtime_artifact_hash() != first


def test_bundle_is_detached_from_mutable_input_objects():
    bundle = materialize_review_configuration("", {})
    settings = copy.deepcopy(_payload(bundle)["settings"])
    settings["review_depth"] = {"enabled": False, "profiles": {"quick": {"max_findings": 2}}}
    candidate = replace(bundle, settings=settings)

    settings["review_depth"]["enabled"] = True

    assert candidate.settings["review_depth"]["enabled"] is False
