import pytest

from pr_agent.cli import run, set_parser
from pr_agent.config_loader import get_settings

# Keys run() mutates on the process-wide settings singleton, directly or via the
# diff-mode CLI path. Snapshotted and restored around every test (autouse) so
# state never leaks, even when run() sets keys the test never touches itself.
_SETTINGS_KEYS = ["plain_diff.content", "plain_diff.output_path", "plain_diff.json_output_path",
                  "plain_diff.suppress_stdout", "plain_diff.disable_working_tree_enrichment",
                  "config.git_provider", "config.publish_output",
                  "config.propagate_tool_errors", "pr_reviewer.extra_instructions",
                  "local_pair_review.policy_version", "local_pair_review.excluded_paths",
                  "local_pair_review.max_file_bytes", "local_pair_review.cache_enabled",
                  "local_pair_review.cache_max_entries", "config.use_repo_settings_file",
                  "config.reasoning_effort", "config.max_model_tokens"]


@pytest.fixture(autouse=True)
def cfg():
    """Restore all diff-mode settings keys after each test, and expose a setter
    so tests mutate settings through the fixture rather than bare set() calls."""
    s = get_settings()
    saved = {k: s.get(k, None) for k in _SETTINGS_KEYS}

    def _set(key, value):
        s.set(key, value)

    yield _set
    for key, value in saved.items():
        s.set(key, value)


def test_parser_has_diff_flags():
    parser = set_parser()
    args = parser.parse_args([
        "--diff-file", "x.diff", "--output", "out.md",
        "--json-output", "out.json", "review",
    ])
    assert args.diff_file == "x.diff"
    assert args.output == "out.md"
    assert args.json_output == "out.json"
    assert args.command == "review"


def test_parser_stdin_flag():
    parser = set_parser()
    args = parser.parse_args(["--stdin", "review"])
    assert args.stdin is True


def test_missing_diff_file_fails_fast(tmp_path, capsys):
    """A non-existent --diff-file must exit cleanly via parser.error (SystemExit)
    with a clear message, not crash with an uncaught OSError traceback."""
    missing = tmp_path / "does-not-exist.diff"
    with pytest.raises(SystemExit):
        run(inargs=["--diff-file", str(missing), "review"])
    err = capsys.readouterr().err
    assert "Could not read --diff-file" in err


def test_json_output_outside_diff_mode_fails_fast(capsys):
    """Reject --json-output in hosted-provider mode via parser.error instead of
    silently dropping the explicitly requested artifact."""
    with pytest.raises(SystemExit):
        run(inargs=["--pr_url", "https://example/pr/1", "--json-output", "out.json", "review"])
    err = capsys.readouterr().err
    assert "--json-output is only supported in plain-diff mode" in err


_DIFF = (
    "diff --git a/foo.py b/foo.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1,3 +1,3 @@\n"
    " line1\n-line2\n+line2-changed\n line3\n"
)


def test_diff_mode_forces_publish_output(cfg, monkeypatch):
    """Diff mode must force config.publish_output=True so stdout/--output is
    never suppressed by a config/env that disabled publishing."""
    import io

    cfg("config.publish_output", False)
    captured = {}

    class FakeAgent:
        async def handle_request(self, target, request, notify=None):
            captured["publish_output"] = get_settings().config.publish_output
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    monkeypatch.setattr("sys.stdin", io.StringIO(_DIFF))
    run(inargs=["--stdin", "review"])
    assert captured["publish_output"] is True


def test_diff_mode_sets_json_output_path(cfg, monkeypatch, tmp_path):
    import io

    captured = {}

    class FakeAgent:
        async def handle_request(self, target, request, notify=None):
            captured["json_output_path"] = get_settings().plain_diff.json_output_path
            return True

    output = tmp_path / "review.json"
    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    monkeypatch.setattr("sys.stdin", io.StringIO(_DIFF))

    run(inargs=["--stdin", "--json-output", str(output), "review"])

    assert captured["json_output_path"] == str(output)


def test_review_snapshot_cli_emits_snapshot_bound_json(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    result_path = repo / ".git" / "pr-agent" / "snapshot-result.json"
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            assert target == "local_snapshot"
            assert request == ["review"]
            assert get_settings().plain_diff.suppress_stdout is True
            assert get_settings().plain_diff.disable_working_tree_enrichment is True
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({
                    "review": {"key_issues_to_review": []},
                    "usage": {"total_tokens": 12},
                    "metadata": {"review_profile": "full"},
                }),
                encoding="utf-8",
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "changed.py",
        "--no-cache", "--json-output", str(result_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["snapshot_id"].startswith("sha256:")
    assert payload["current_snapshot_id"] == payload["snapshot_id"]
    assert payload["state"] == "no_findings"
    assert payload["usage"] == {"total_tokens": 12}
    assert payload["advisory"] is True
    expected_output = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    assert result_path.read_text(encoding="utf-8") == expected_output
    assert result.state.value == "no_findings"


def test_review_snapshot_loads_repo_policy_before_capture(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess

    repo = tmp_path / "repo-policy"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "secret.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "secret.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    (repo / ".pr_agent.toml").write_text(
        '[local_pair_review]\nexcluded_paths = ["secret.py"]\n', encoding="utf-8"
    )
    monkeypatch.chdir(repo)

    class UnexpectedAgent:
        async def _handle_request(self, target, request, notify=None):
            raise AssertionError("excluded input must not reach the model")

    monkeypatch.setattr("pr_agent.cli.PRAgent", UnexpectedAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "secret.py", "--no-cache",
    ])

    assert result.state.value == "coverage_unavailable"
    assert ("secret.py", "excluded") in {
        (issue.path, issue.reason) for issue in result.coverage_issues
    }
    assert json.loads(capsys.readouterr().out)["state"] == "coverage_unavailable"


def test_review_snapshot_honors_disabled_repo_settings(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-policy-disabled"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "review.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "review.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    (repo / ".pr_agent.toml").write_text(
        '[local_pair_review]\nexcluded_paths = ["review.py"]\n', encoding="utf-8"
    )
    monkeypatch.chdir(repo)
    cfg("config.use_repo_settings_file", False)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "review.py", "--no-cache",
    ])

    assert result.state.value == "no_findings"
    assert json.loads(capsys.readouterr().out)["state"] == "no_findings"


def test_snapshot_configuration_hash_covers_review_settings(cfg):
    from pr_agent.cli import _snapshot_review_configuration_hash

    cfg("config.reasoning_effort", "low")
    first = _snapshot_review_configuration_hash()
    cfg("config.reasoning_effort", "high")

    assert _snapshot_review_configuration_hash() != first
    cfg("config.max_model_tokens", 4096)
    second = _snapshot_review_configuration_hash()
    cfg("config.max_model_tokens", 8192)

    assert _snapshot_review_configuration_hash() != second


def test_review_snapshot_applies_external_policy_before_capture(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess

    repo = tmp_path / "repo-extra-policy"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "shared-secret.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "shared-secret.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    extra = tmp_path / "shared.toml"
    extra.write_text(
        '[local_pair_review]\nexcluded_paths = ["shared-secret.py"]\n', encoding="utf-8"
    )
    monkeypatch.chdir(repo)

    class UnexpectedAgent:
        async def _handle_request(self, target, request, notify=None):
            raise AssertionError("shared exclusions must apply before capture")

    monkeypatch.setattr("pr_agent.cli.PRAgent", UnexpectedAgent)
    result = run(inargs=[
        "--extra_config_url", str(extra), "review-snapshot", "--event", "file-save",
        "--path", "shared-secret.py", "--no-cache",
    ])

    assert result.state.value == "coverage_unavailable"
    assert ("shared-secret.py", "excluded") in {
        (issue.path, issue.reason) for issue in result.coverage_issues
    }
    assert json.loads(capsys.readouterr().out)["state"] == "coverage_unavailable"


def test_review_snapshot_forwards_context_and_publishes_only_fresh_markdown(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-context"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    output = repo / "review.md"
    output.write_text("old output\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            instructions = get_settings().pr_reviewer.extra_instructions
            assert "Check the parser boundary" in instructions
            assert '"name": "lint"' in instructions
            Path(get_settings().plain_diff.output_path).write_text("stale review\n", encoding="utf-8")
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            changed.write_text("value = 3\n", encoding="utf-8")
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "changed.py",
        "--intent", "Check the parser boundary",
        "--deterministic-check", '{"name":"lint","status":"passed"}',
        "--output", str(output), "--no-cache",
    ])

    assert result.state.value == "stale"
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["state"] == "stale"


def test_review_snapshot_bypasses_structured_cache_when_markdown_is_requested(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-cache-output"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    output = repo / "review.md"
    monkeypatch.chdir(repo)
    calls = []

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            calls.append(target)
            Path(get_settings().plain_diff.output_path).write_text("fresh review\n", encoding="utf-8")
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    monkeypatch.setattr("pr_agent.cli.SnapshotCache.read", lambda self, snapshot_id: object())
    result = run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "changed.py",
        "--output", str(output),
    ])

    assert calls == ["local_snapshot"]
    assert result.state.value == "no_findings"
    assert output.read_text(encoding="utf-8") == "fresh review\n"
    assert json.loads(capsys.readouterr().out)["state"] == "no_findings"


def test_review_snapshot_materializes_requested_clean_markdown(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-clean-output"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    output = repo / "review.md"
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "worktree-idle", "--output", str(output), "--no-cache",
    ])

    assert result.state.value == "no_findings"
    assert output.read_text(encoding="utf-8") == "## PR Review\n\nNo findings.\n"
    assert json.loads(capsys.readouterr().out)["state"] == "no_findings"


def test_worktree_snapshot_excludes_requested_output_artifacts(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-output-exclusions"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    output = repo / "review.md"
    result_path = repo / "result.json"
    monkeypatch.chdir(repo)
    calls = []

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            calls.append(target)
            assert "review.md" not in get_settings().plain_diff.content
            assert "result.json" not in get_settings().plain_diff.content
            Path(get_settings().plain_diff.output_path).write_text("fresh review\n", encoding="utf-8")
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    invocation = [
        "review-snapshot", "--event", "worktree-idle", "--output", str(output),
        "--json-output", str(result_path), "--no-cache",
    ]
    first = run(inargs=invocation)
    capsys.readouterr()
    second = run(inargs=invocation)

    assert calls == ["local_snapshot", "local_snapshot"]
    assert first.state.value == second.state.value == "no_findings"
    assert output.read_text(encoding="utf-8") == "fresh review\n"
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == "no_findings"


def test_partial_clean_snapshot_does_not_publish_markdown(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-partial-markdown"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    (repo / "binary.dat").write_bytes(b"\0binary")
    output = repo / "review.md"
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            Path(get_settings().plain_diff.output_path).write_text("clean review\n", encoding="utf-8")
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "worktree-idle", "--output", str(output), "--no-cache",
    ])

    assert result.state.value == "coverage_unavailable"
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["state"] == "coverage_unavailable"
