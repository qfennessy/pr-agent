import pytest

from pr_agent.cli import run, set_parser
from pr_agent.config_loader import get_settings
from pr_agent.tools.local_pair_review import SnapshotCaptureError

# Keys run() mutates on the process-wide settings singleton, directly or via the
# diff-mode CLI path. Snapshotted and restored around every test (autouse) so
# state never leaks, even when run() sets keys the test never touches itself.
_SETTINGS_KEYS = ["plain_diff.content", "plain_diff.output_path", "plain_diff.json_output_path",
                  "plain_diff.suppress_stdout", "plain_diff.disable_working_tree_enrichment",
                  "plain_diff.repo_context_files",
                  "config.git_provider", "config.publish_output",
                  "config.propagate_tool_errors", "pr_reviewer.extra_instructions",
                  "local_pair_review.policy_version", "local_pair_review.excluded_paths",
                  "local_pair_review.max_file_bytes", "local_pair_review.max_snapshot_bytes",
                  "local_pair_review.max_path_discovery_bytes",
                  "local_pair_review.cache_enabled",
                  "local_pair_review.cache_max_entries", "config.use_repo_settings_file",
                  "config.model", "config.reasoning_effort", "config.max_model_tokens", "skills.enabled",
                  "config.repo_context_files", "config.repo_context_max_lines",
                  "skills.paths", "skills.max_skills_tokens"]


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


def test_review_snapshot_rejects_colliding_output_paths(capsys):
    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "worktree-idle",
            "--output", "result.txt", "--json-output", "./result.txt",
        ])

    assert "must reference different paths" in capsys.readouterr().err


def test_review_snapshot_rejects_output_that_aliases_review_input(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-output-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "notes.txt"
    source.write_text("keep this source\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "notes.txt",
            "--output", "notes.txt", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == "keep this source\n"


def test_review_snapshot_rejects_worktree_symlink_into_git_metadata(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-metadata-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    alias = repo / "result.json"
    alias.symlink_to(repo / ".git" / "config")
    original_config = (repo / ".git" / "config").read_text(encoding="utf-8")
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", "result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert (repo / ".git" / "config").read_text(encoding="utf-8") == original_config


def test_review_snapshot_rejects_metadata_symlink_into_worktree(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-reverse-metadata-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    artifact_dir = repo / ".git" / "pr-agent"
    artifact_dir.mkdir()
    (artifact_dir / "result.json").symlink_to(source)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", ".git/pr-agent/result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == "value = 1\n"


def test_review_snapshot_rejects_nonexistent_output_through_repo_symlink(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-parent-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    alias = tmp_path / "repo-link"
    alias.symlink_to(repo, target_is_directory=True)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", str(alias / "new.json"), "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert not (repo / "new.json").exists()


def test_review_snapshot_rejects_nonexistent_output_through_internal_symlink(
    monkeypatch, tmp_path, capsys
):
    import subprocess

    repo = tmp_path / "repo-internal-parent-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    real_parent = repo / "real-output"
    real_parent.mkdir()
    (repo / "output-link").symlink_to(real_parent, target_is_directory=True)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", "output-link/result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert not (real_parent / "result.json").exists()


def test_review_snapshot_rejects_looped_output_symlink(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-output-loop"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    (repo / "changed.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "loop").symlink_to("loop", target_is_directory=True)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", "loop/result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err


def test_review_snapshot_rejects_looped_symlink_before_output_collision_check(
    monkeypatch, tmp_path, capsys
):
    import subprocess

    repo = tmp_path / "repo-output-loop-collision"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    (repo / "loop").symlink_to("loop", target_is_directory=True)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "worktree-idle",
            "--output", "loop/review.md", "--json-output", "result.json",
        ])

    error = capsys.readouterr().err
    assert "aliases an existing repository path" in error
    assert "Traceback" not in error


def test_review_snapshot_rejects_external_symlink_into_git_metadata(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-external-metadata-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    alias = tmp_path / "git-metadata-link"
    alias.symlink_to(repo / ".git", target_is_directory=True)
    original_config = (repo / ".git" / "config").read_text(encoding="utf-8")
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", str(alias / "config"), "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert (repo / ".git" / "config").read_text(encoding="utf-8") == original_config


def test_review_snapshot_rejects_metadata_symlink_to_external_file(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-metadata-external-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    external = tmp_path / "external.txt"
    external.write_text("keep this external file\n", encoding="utf-8")
    artifact_dir = repo / ".git" / "pr-agent"
    artifact_dir.mkdir()
    (artifact_dir / "result.json").symlink_to(external)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", ".git/pr-agent/result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert external.read_text(encoding="utf-8") == "keep this external file\n"


@pytest.mark.parametrize("repository_target", ["changed.py", ".git/config"])
def test_review_snapshot_rejects_external_hard_link_into_repository(
    monkeypatch, tmp_path, capsys, repository_target
):
    import os
    import subprocess

    repo = tmp_path / "repo-hard-link-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    protected = repo / repository_target
    original = protected.read_bytes()
    alias = tmp_path / "hard-linked-output.json"
    os.link(protected, alias)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", str(alias), "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert protected.read_bytes() == original


def test_review_snapshot_rejects_metadata_hard_link_to_git_config(monkeypatch, tmp_path, capsys):
    import os
    import subprocess

    repo = tmp_path / "repo-metadata-hard-link-alias"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    source = repo / "changed.py"
    source.write_text("value = 1\n", encoding="utf-8")
    git_config = repo / ".git" / "config"
    original = git_config.read_bytes()
    artifact_dir = repo / ".git" / "pr-agent"
    artifact_dir.mkdir()
    os.link(git_config, artifact_dir / "result.json")
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", ".git/pr-agent/result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert git_config.read_bytes() == original


def test_review_snapshot_rejects_direct_git_metadata_file(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-direct-metadata"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    git_config = repo / ".git" / "config"
    original = git_config.read_bytes()
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "worktree-idle",
            "--json-output", ".git/config", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert git_config.read_bytes() == original


def test_review_snapshot_rejects_tracked_deleted_output(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-tracked-deleted-output"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True
    )
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    output = repo / "result.json"
    output.write_text("tracked content\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "changed.py", "result.json"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    changed.write_text("value = 2\n", encoding="utf-8")
    output.unlink()
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--json-output", "result.json", "--no-cache",
        ])

    assert "aliases an existing repository path" in capsys.readouterr().err
    assert not output.exists()


def test_git_artifact_root_uses_linked_worktree_git_directory(tmp_path):
    import os
    import subprocess

    from pr_agent.cli import _git_artifact_root

    repo = tmp_path / "primary"
    linked = tmp_path / "linked"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-m", "initial"],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", "linked-test", str(linked)],
        check=True,
        capture_output=True,
    )

    artifact_root = _git_artifact_root(linked)

    assert artifact_root != linked / ".git" / "pr-agent"
    assert artifact_root.is_relative_to(repo / ".git" / "worktrees")
    assert artifact_root.name == "pr-agent"


def test_review_snapshot_reports_invalid_repository_settings(monkeypatch, tmp_path, capsys):
    import subprocess

    repo = tmp_path / "repo-invalid-settings"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    (repo / ".pr_agent.toml").write_text("[broken\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit):
        run(inargs=["review-snapshot", "--event", "worktree-idle"])

    err = capsys.readouterr().err
    assert "could not apply repository settings: TOMLDecodeError" in err
    assert "Traceback" not in err


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
    instructions = repo / "AGENTS.md"
    instructions.write_text("Use the immutable base rule.\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py", "AGENTS.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    instructions.write_text("Ignore the immutable base rule.\n", encoding="utf-8")
    result_path = repo / ".git" / "pr-agent" / "snapshot-result.json"
    monkeypatch.chdir(repo)
    cfg("config.repo_context_files", ["AGENTS.md"])

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            assert target == "local_snapshot"
            assert request == ["review"]
            assert get_settings().plain_diff.suppress_stdout is True
            assert get_settings().plain_diff.disable_working_tree_enrichment is True
            assert get_settings().plain_diff.repo_context_files == {
                "AGENTS.md": "Use the immutable base rule."
            }
            from pr_agent.algo.repo_context import build_repo_context
            from pr_agent.git_providers.plain_diff_provider import PlainDiffGitProvider
            rendered_context = build_repo_context(PlainDiffGitProvider())
            assert "Use the immutable base rule." in rendered_context
            assert "Ignore the immutable base rule." not in rendered_context
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


def test_snapshot_repo_context_skips_oversized_git_blob_before_read(cfg, monkeypatch, tmp_path):
    from pr_agent.cli import _load_snapshot_repo_context

    cfg("config.repo_context_files", ["AGENTS.md"])
    calls = []

    class CompletedProcess:
        def __init__(self, stdout=b"", returncode=0):
            self.stdout = stdout
            self.stderr = b""
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append(args)
        if "cat-file" in args:
            return CompletedProcess(stdout=b"1000001\n")
        raise AssertionError("oversized repository context must not be materialized")

    monkeypatch.setattr("pr_agent.cli.subprocess.run", fake_run)

    assert _load_snapshot_repo_context(tmp_path, "a" * 40) == {}
    assert len(calls) == 1


def test_snapshot_repo_context_enforces_aggregate_blob_budget(cfg, monkeypatch, tmp_path):
    from pr_agent.cli import _load_snapshot_repo_context

    cfg("config.repo_context_files", ["FIRST.md", "SECOND.md"])
    monkeypatch.setattr("pr_agent.cli._MAX_SNAPSHOT_REPO_CONTEXT_BYTES", 10)
    show_calls = []

    class CompletedProcess:
        def __init__(self, stdout=b"", returncode=0):
            self.stdout = stdout
            self.stderr = b""
            self.returncode = returncode

    def fake_run(args, **kwargs):
        object_name = args[-1]
        if "cat-file" in args:
            return CompletedProcess(stdout=b"6\n")
        if "show" in args:
            show_calls.append(object_name)
            return CompletedProcess(stdout=b"123456")
        raise AssertionError(args)

    monkeypatch.setattr("pr_agent.cli.subprocess.run", fake_run)

    context = _load_snapshot_repo_context(tmp_path, "a" * 40)

    assert context == {"FIRST.md": "123456"}
    assert show_calls == [f"{'a' * 40}:FIRST.md"]


def test_review_snapshot_atomically_replaces_json_symlink_swapped_during_review(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-json-output-race"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    protected = repo / ".git" / "config"
    original = protected.read_bytes()
    output = repo / ".git" / "pr-agent" / "result.json"
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.symlink_to(protected)
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "changed.py",
        "--no-cache", "--json-output", str(output),
    ])

    assert protected.read_bytes() == original
    assert not output.is_symlink()
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot_id"] == result.snapshot_id
    capsys.readouterr()


def test_review_snapshot_rejects_output_parent_swapped_during_review(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-json-parent-race"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    output = repo / ".git" / "pr-agent" / "reviews" / "result.json"
    external_parent = tmp_path / "external-reviews"
    external_parent.mkdir()
    protected = external_parent / "result.json"
    protected.write_text("keep external\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            output.parent.rmdir()
            output.parent.symlink_to(external_parent, target_is_directory=True)
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    with pytest.raises(SnapshotCaptureError, match="output parent changed before publication"):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--no-cache", "--json-output", str(output),
        ])

    assert protected.read_text(encoding="utf-8") == "keep external\n"
    capsys.readouterr()


def test_atomic_snapshot_publication_uses_portable_fallback(monkeypatch, tmp_path):
    from pr_agent.cli import _atomic_replace_bytes, _prepare_output_parent

    output = tmp_path / "result.json"
    output.write_text("old", encoding="utf-8")
    identity = _prepare_output_parent(str(output))
    monkeypatch.setattr("pr_agent.cli._supports_descriptor_relative_publication", lambda: False)

    _atomic_replace_bytes(output, b"new", identity)

    assert output.read_bytes() == b"new"
    assert list(tmp_path.glob(".pr-agent-*.tmp")) == []


def test_portable_snapshot_preparation_does_not_delete_existing_markdown(
    monkeypatch, tmp_path
):
    from pr_agent.cli import _prepare_output_parent, _unlink_output

    output = tmp_path / "review.md"
    output.write_text("previous review\n", encoding="utf-8")
    identity = _prepare_output_parent(str(output))
    monkeypatch.setattr("pr_agent.cli._supports_descriptor_relative_publication", lambda: False)

    _unlink_output(output, identity)

    assert output.read_text(encoding="utf-8") == "previous review\n"


def test_repeatable_json_validation_rejects_large_file_without_reading_it(
    monkeypatch, tmp_path
):
    import subprocess
    from pathlib import Path

    from pr_agent.cli import (
        _MAX_REPEATABLE_SNAPSHOT_ARTIFACT_BYTES,
        _is_repeatable_snapshot_artifact,
    )

    repo = tmp_path / "repo-large-artifact"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    output = repo / "result.json"
    with output.open("wb") as handle:
        handle.truncate(_MAX_REPEATABLE_SNAPSHOT_ARTIFACT_BYTES + 1)
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path == output:
            raise AssertionError("oversized artifact must not be materialized")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert _is_repeatable_snapshot_artifact(output, repo, "json") is False


def test_review_snapshot_restores_provider_settings_for_later_hosted_run(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-settings-restore"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    (repo / ".pr_agent.toml").write_text(
        '[config]\nmodel = "repo-only-model"\n', encoding="utf-8"
    )
    monkeypatch.chdir(repo)

    baselines = {
        "config.model": "hosted-model",
        "config.git_provider": "github",
        "config.publish_output": False,
        "config.propagate_tool_errors": False,
        "plain_diff.content": "",
        "plain_diff.output_path": "hosted-output",
        "plain_diff.json_output_path": "hosted-json-output",
        "plain_diff.suppress_stdout": False,
        "plain_diff.disable_working_tree_enrichment": False,
        "pr_reviewer.extra_instructions": "hosted instructions",
    }
    for key, value in baselines.items():
        cfg(key, value)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            assert get_settings().config.model == "repo-only-model"
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}),
                encoding="utf-8",
            )
            return True

        async def handle_request(self, target, request, notify=None):
            assert target == "https://example.test/pr/1"
            for key, value in baselines.items():
                assert get_settings().get(key) == value
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    run(inargs=[
        "review-snapshot", "--event", "file-save", "--path", "changed.py", "--no-cache",
    ])
    capsys.readouterr()
    run(inargs=["--pr_url", "https://example.test/pr/1", "review"])


@pytest.mark.parametrize("excluded_paths_toml", ['["secret.py"]', '"secret.py"'])
def test_review_snapshot_loads_repo_policy_before_capture(
    cfg, monkeypatch, tmp_path, capsys, excluded_paths_toml
):
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
        f"[local_pair_review]\nexcluded_paths = {excluded_paths_toml}\n", encoding="utf-8"
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


def test_snapshot_configuration_hash_covers_resolved_skill_content(cfg, tmp_path):
    from pr_agent.cli import _snapshot_review_configuration_hash

    skill_dir = tmp_path / "review-skill"
    skill_dir.mkdir()
    skill = skill_dir / "SKILL.md"
    skill.write_text(
        "---\nname: local-review\ndescription: Review local changes.\n---\n\nCheck the first rule.\n",
        encoding="utf-8",
    )
    cfg("skills.enabled", True)
    cfg("skills.paths", [str(tmp_path)])
    first = _snapshot_review_configuration_hash()
    skill.write_text(
        "---\nname: local-review\ndescription: Review local changes.\n---\n\nCheck the second rule.\n",
        encoding="utf-8",
    )

    assert _snapshot_review_configuration_hash() != first


def test_pinned_skills_context_is_immutable_and_request_scoped(cfg, tmp_path):
    from pr_agent.algo.skills_loader import get_skills_context, pin_skills_context

    skill_dir = tmp_path / "pinned-review-skill"
    skill_dir.mkdir()
    skill = skill_dir / "SKILL.md"
    skill.write_text(
        "---\nname: pinned-review\ndescription: Review local changes.\n---\n\nFirst rule.\n",
        encoding="utf-8",
    )
    cfg("skills.enabled", True)
    cfg("skills.paths", [str(tmp_path)])
    original = get_skills_context()

    with pin_skills_context(original):
        skill.write_text(
            "---\nname: pinned-review\ndescription: Review local changes.\n---\n\nSecond rule.\n",
            encoding="utf-8",
        )
        assert get_skills_context() == original

    assert get_skills_context() != original


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
    output = tmp_path / "stale-review.md"
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


def test_review_snapshot_returns_stale_when_base_ref_disappears(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-disappearing-base"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "branch", "snapshot-base"], check=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            subprocess.run(
                ["git", "-C", str(repo), "update-ref", "-d", "refs/heads/snapshot-base"],
                check=True,
            )
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "worktree-idle", "--base", "snapshot-base", "--no-cache",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert result.state.value == "stale"
    assert result.current_snapshot_id is None
    assert result.review is None
    assert payload["coverage_issues"] == [
        {"fingerprint": None, "path": None, "reason": "current_snapshot_unavailable"}
    ]


def test_review_snapshot_returns_stale_when_skills_change_during_review(
    cfg, monkeypatch, tmp_path, capsys
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-changing-skills"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    skills = {"content": "before"}
    monkeypatch.setattr("pr_agent.cli.get_skills_context", lambda: skills["content"])

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            skills["content"] = "after"
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    result = run(inargs=[
        "review-snapshot", "--event", "worktree-idle", "--no-cache",
    ])

    assert result.state.value == "stale"
    assert result.review is None
    assert json.loads(capsys.readouterr().out)["state"] == "stale"


@pytest.mark.parametrize("config_source", ["repository", "external"])
@pytest.mark.parametrize("replacement", ["valid", "malformed"])
def test_review_snapshot_returns_stale_when_file_backed_settings_change(
    cfg, monkeypatch, tmp_path, capsys, config_source, replacement
):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / f"repo-changing-{config_source}-config"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    config_path = (
        repo / ".pr_agent.toml"
        if config_source == "repository"
        else tmp_path / "shared-changing.toml"
    )
    config_path.write_text(
        '[pr_reviewer]\nextra_instructions = "before"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            config_path.write_text(
                (
                    '[pr_reviewer]\nextra_instructions = "after"\n'
                    if replacement == "valid"
                    else "[pr_reviewer\n"
                ),
                encoding="utf-8",
            )
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}),
                encoding="utf-8",
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    inargs = ["review-snapshot", "--event", "worktree-idle", "--no-cache"]
    if config_source == "external":
        inargs = ["--extra_config_url", str(config_path), *inargs]
    result = run(inargs=inargs)

    assert result.state.value == "stale"
    assert result.review is None
    assert json.loads(capsys.readouterr().out)["state"] == "stale"


def test_review_snapshot_restores_snapshot_only_instructions(cfg, monkeypatch, tmp_path, capsys):
    import json
    import subprocess
    from pathlib import Path

    repo = tmp_path / "repo-instruction-reset"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Snapshot Test"], check=True)
    changed = repo / "changed.py"
    changed.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "changed.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True)
    changed.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    cfg("pr_reviewer.extra_instructions", "Persistent repository rule")
    observed = []

    class FakeAgent:
        async def _handle_request(self, target, request, notify=None):
            observed.append(str(get_settings().pr_reviewer.extra_instructions))
            Path(get_settings().plain_diff.json_output_path).write_text(
                json.dumps({"review": {"key_issues_to_review": []}}), encoding="utf-8"
            )
            return True

    monkeypatch.setattr("pr_agent.cli.PRAgent", FakeAgent)
    for intent in ("First snapshot", "Second snapshot"):
        run(inargs=[
            "review-snapshot", "--event", "file-save", "--path", "changed.py",
            "--intent", intent, "--no-cache",
        ])
        capsys.readouterr()
        assert get_settings().pr_reviewer.extra_instructions == "Persistent repository rule"

    assert "First snapshot" in observed[0]
    assert "First snapshot" not in observed[1]
    assert "Second snapshot" in observed[1]


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
    assert output.read_text(encoding="utf-8") == (
        "<!-- pr-agent-review-snapshot -->\nfresh review\n"
    )
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
    assert output.read_text(encoding="utf-8") == (
        "<!-- pr-agent-review-snapshot -->\n## PR Review\n\nNo findings.\n"
    )
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
    output = repo / "repeat-review.md"
    result_path = repo / "repeat-result.json"
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
    assert output.read_text(encoding="utf-8") == (
        "<!-- pr-agent-review-snapshot -->\nfresh review\n"
    )
    assert json.loads(result_path.read_text(encoding="utf-8"))["artifact_type"] == (
        "pr-agent-review-snapshot"
    )
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
