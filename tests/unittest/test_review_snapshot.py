import json
import os
import shlex
import stat
import subprocess
from io import BytesIO
from pathlib import Path
from time import monotonic

import pytest

from pr_agent.algo.review_snapshot import CoverageIssue, ReviewEvent, ReviewResultState, ReviewSnapshot
from pr_agent.git_providers.plain_diff_provider import parse_plain_diff
from pr_agent.tools.local_pair_review import (
    LocalPairReview,
    SnapshotCache,
    SnapshotCaptureError,
    build_snapshot_result,
    find_repository_root,
)


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Snapshot Test")
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "rename_me.py").write_text("old = True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_repository_locations_decode_non_utf8_bytes_with_surrogateescape(monkeypatch, tmp_path):
    encoded_root = os.fsencode(str(tmp_path)) + b"/repo-\xff\n"

    class CompletedProcess:
        returncode = 0
        stdout = encoded_root
        stderr = b""

    monkeypatch.setattr("pr_agent.tools.local_pair_review.subprocess.run", lambda *args, **kwargs: CompletedProcess())

    root = find_repository_root(str(tmp_path))

    assert os.fsencode(str(root)).endswith(b"repo-\xff")


def test_snapshot_cache_decodes_non_utf8_git_common_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pr_agent.tools.local_pair_review._run_git",
        lambda *args, **kwargs: b".git-\xff\n",
    )

    cache = SnapshotCache(tmp_path)

    assert os.fsencode(str(cache.cache_dir)).endswith(b".git-\xff/pr-agent/snapshot-cache")


def _snapshot(root: str, *, diff: str = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", policy="v1", config="c1"):
    return ReviewSnapshot(
        event=ReviewEvent.FILE_SAVE,
        repository_root=root,
        base_revision="a" * 40,
        changed_paths=("x",),
        focus_path="x",
        diff=diff,
        policy_version=policy,
        review_configuration_hash=config,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_snapshot_identity_is_stable_and_policy_and_repository_scoped():
    first = _snapshot("/repo/one")
    recreated = ReviewSnapshot(**{
        key: value
        for key, value in first.__dict__.items()
        if key != "snapshot_id"
    })
    assert first.snapshot_id == recreated.snapshot_id
    assert first.snapshot_id != _snapshot("/repo/one", diff=first.diff + "\n").snapshot_id
    assert first.snapshot_id != _snapshot("/repo/one", policy="v2").snapshot_id
    assert first.snapshot_id != _snapshot("/repo/one", config="c2").snapshot_id
    assert first.snapshot_id != _snapshot("/repo/two").snapshot_id
    with_coverage = ReviewSnapshot(**{
        key: value
        for key, value in first.__dict__.items()
        if key not in {"snapshot_id", "coverage_issues"}
    }, coverage_issues=(CoverageIssue(path="x", reason="binary"),))
    assert first.snapshot_id != with_coverage.snapshot_id


def test_worktree_snapshot_captures_modified_untracked_deleted_and_renamed_files(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "new.py").write_text("added = True\n", encoding="utf-8")
    (repo / "deleted.py").write_text("gone = True\n", encoding="utf-8")
    _git(repo, "add", "deleted.py")
    _git(repo, "commit", "-m", "add deleted file")
    (repo / "deleted.py").unlink()
    _git(repo, "mv", "rename_me.py", "renamed.py")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")
    parsed = {item.filename: item for item in parse_plain_diff(snapshot.diff)}

    assert {"tracked.py", "new.py", "deleted.py"}.issubset(set(snapshot.changed_paths))
    assert "renamed.py" not in snapshot.changed_paths
    assert parsed["new.py"].edit_type.name == "ADDED"
    assert parsed["deleted.py"].edit_type.name == "DELETED"
    assert CoverageIssue(path="rename_me.py", reason="metadata_only_diff") in snapshot.coverage_issues
    assert CoverageIssue(path="renamed.py", reason="metadata_only_diff") in snapshot.coverage_issues


def test_worktree_snapshot_round_trips_non_ascii_git_path(tmp_path):
    repo = _repo(tmp_path, "non-ascii-path")
    path = repo / "café.py"
    path.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "café.py")
    _git(repo, "commit", "-m", "add non-ascii path")
    path.write_text("value = 2\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.changed_paths == ("café.py",)
    assert snapshot.coverage_issues == ()
    assert "café.py" in snapshot.diff


def test_worktree_snapshot_decodes_c_quoted_control_path(tmp_path):
    repo = _repo(tmp_path, "control-path")
    filename = "line\nbreak.py"
    path = repo / filename
    path.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", "add control path")
    path.write_text("value = 2\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.changed_paths == (filename,)
    assert snapshot.coverage_issues == ()


def test_mixed_review_reports_metadata_only_path_as_unavailable(tmp_path):
    repo = _repo(tmp_path, "mixed-metadata")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    mode_only = repo / "rename_me.py"
    mode_only.chmod(mode_only.stat().st_mode | stat.S_IXUSR)

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.changed_paths == ("tracked.py",)
    assert CoverageIssue(path="rename_me.py", reason="metadata_only_diff") in snapshot.coverage_issues
    assert "@@" in snapshot.diff


def test_pre_commit_snapshot_uses_only_the_index(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    (repo / "tracked.py").write_text("value = 3\n", encoding="utf-8")
    (repo / "untracked.py").write_text("ignored = True\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="pre-commit")

    assert snapshot.changed_paths == ("tracked.py",)
    assert "+value = 2" in snapshot.diff
    assert "+value = 3" not in snapshot.diff
    assert "untracked.py" not in snapshot.diff


def test_pre_commit_snapshot_supports_an_initial_commit(tmp_path):
    repo = tmp_path / "initial-commit"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Snapshot Test")
    (repo / "first.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "first.py")

    snapshot = LocalPairReview(str(repo)).capture(event="pre-commit")

    assert snapshot.changed_paths == ("first.py",)
    assert "+value = 1" in snapshot.diff
    assert snapshot.base_selector == "HEAD"
    assert snapshot.base_revision


def test_pre_commit_coverage_inspects_the_staged_blob(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "tracked.py"
    path.write_text("x" * 100, encoding="utf-8")
    _git(repo, "add", "tracked.py")
    path.write_text("small\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), max_file_bytes=20).capture(event="pre-commit")

    assert snapshot.diff == ""
    assert CoverageIssue(path="tracked.py", reason="file_too_large") in snapshot.coverage_issues


def test_git_blob_size_is_checked_before_content_is_read(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "bounded-git-blob")
    reviewer = LocalPairReview(str(repo), max_file_bytes=4)
    calls = []

    def fake_run_git(repository_root, *args, **kwargs):
        calls.append(args)
        if args == ("cat-file", "-s", "blob-id"):
            return b"5\n"
        raise AssertionError("oversized Git blob content must not be materialized")

    monkeypatch.setattr("pr_agent.tools.local_pair_review._run_git", fake_run_git)

    assert reviewer._inspect_git_object("100644", "blob-id") == "file_too_large"
    assert calls == [("cat-file", "-s", "blob-id")]


def test_snapshot_diff_disables_textconv_filters(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "no-textconv")
    reviewer = LocalPairReview(str(repo))
    calls = []

    def fake_run_git(repository_root, *args, **kwargs):
        calls.append(args)
        return b""

    monkeypatch.setattr("pr_agent.tools.local_pair_review._run_git", fake_run_git)

    reviewer._capture_diff(ReviewEvent.WORKTREE_IDLE, "HEAD", ["tracked.py"])
    reviewer._capture_untracked_addition("untracked.py")

    diff_calls = [args for args in calls if "diff" in args]
    assert len(diff_calls) == 2
    assert all("--no-textconv" in args for args in diff_calls)


def test_snapshot_diff_disables_forced_git_color(tmp_path):
    repo = _repo(tmp_path, "no-color")
    _git(repo, "config", "color.diff", "always")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "untracked.py").write_text("new = True\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert "\x1b[" not in snapshot.diff
    parsed = parse_plain_diff(snapshot.diff)
    assert {file.filename for file in parsed} == {"tracked.py", "untracked.py"}


def test_snapshot_diff_forces_standard_prefixes(tmp_path):
    repo = _repo(tmp_path, "standard-prefixes")
    _git(repo, "config", "diff.mnemonicPrefix", "true")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "untracked.py").write_text("new = True\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    parsed = parse_plain_diff(snapshot.diff)
    assert {file.filename for file in parsed} == {"tracked.py", "untracked.py"}
    assert "w/tracked.py" not in snapshot.diff
    assert "2/untracked.py" not in snapshot.diff


def test_snapshot_diff_neutralizes_configured_clean_and_process_filters(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "neutral-filters")
    reviewer = LocalPairReview(str(repo))
    calls = []

    def fake_run_git(repository_root, *args, **kwargs):
        calls.append(args)
        if "--get-regexp" in args:
            return b"filter.danger.clean\nfilter.danger.process\nfilter.danger.required\n"
        return b""

    monkeypatch.setattr("pr_agent.tools.local_pair_review._run_git", fake_run_git)

    reviewer._capture_diff(ReviewEvent.WORKTREE_IDLE, "HEAD", ["tracked.py"])

    diff_args = next(args for args in calls if "diff" in args)
    assert "filter.danger.clean=cat" in diff_args
    assert "filter.danger.process=" in diff_args
    assert "filter.danger.required=false" in diff_args


def test_snapshot_capture_does_not_execute_repository_clean_filter(tmp_path):
    repo = _repo(tmp_path, "raw-filter-content")
    tracked = repo / "tracked.py"
    tracked.write_text("raw = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "initial")
    (repo / ".gitattributes").write_text("tracked.py filter=danger\n", encoding="utf-8")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-m", "attributes")
    marker = repo / ".git" / "filter-ran"
    filter_command = f"touch {shlex.quote(str(marker))}; sed s/raw/FILTERED/g"
    _git(repo, "config", "filter.danger.clean", filter_command)
    _git(repo, "config", "filter.danger.process", filter_command)
    _git(repo, "config", "filter.danger.required", "true")
    tracked.write_text("raw = 2\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert not marker.exists()
    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(
        path="tracked.py",
        reason="content_filter_unsupported",
    ) in snapshot.coverage_issues
    assert "FILTERED" not in snapshot.diff


def test_filtered_path_is_reported_when_clean_driver_alone_makes_it_dirty(tmp_path):
    repo = _repo(tmp_path, "filter-only-dirty")
    tracked = repo / "tracked.py"
    tracked.write_text("raw\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("tracked.py filter=changing\n", encoding="utf-8")
    _git(repo, "add", "tracked.py", ".gitattributes")
    _git(repo, "commit", "-m", "filtered fixture")
    _git(repo, "config", "filter.changing.clean", "sed s/raw/OTHER/g")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(
        path="tracked.py",
        reason="content_filter_unsupported",
    ) in snapshot.coverage_issues


def test_pre_commit_uses_index_content_filter_attributes(tmp_path):
    repo = _repo(tmp_path, "index-filter-attributes")
    tracked = repo / "tracked.py"
    tracked.write_text("staged = True\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(
        "tracked.py filter=danger\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tracked.py", ".gitattributes")

    snapshot = LocalPairReview(str(repo)).capture(event="pre-commit")

    assert "tracked.py" not in snapshot.changed_paths
    assert CoverageIssue(
        path="tracked.py",
        reason="content_filter_unsupported",
    ) in snapshot.coverage_issues


def test_pre_commit_ignores_unchanged_filtered_paths(tmp_path):
    repo = _repo(tmp_path, "unchanged-index-filter")
    (repo / ".gitattributes").write_text(
        "large.bin filter=lfs\n",
        encoding="utf-8",
    )
    (repo / "large.bin").write_bytes(b"unchanged binary\n")
    (repo / "changed.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", ".gitattributes", "large.bin", "changed.py")
    _git(repo, "commit", "-m", "filtered fixture")
    (repo / "changed.py").write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "changed.py")

    snapshot = LocalPairReview(str(repo)).capture(event="pre-commit")

    assert snapshot.changed_paths == ("changed.py",)
    assert snapshot.coverage_issues == ()


@pytest.mark.parametrize("event", ["file-save", "worktree-idle", "pre-commit"])
def test_removed_filter_assignment_remains_excluded_from_base(event, tmp_path):
    repo = _repo(tmp_path, f"removed-base-filter-{event}")
    tracked = repo / "tracked.py"
    tracked.write_text("encrypted = True\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(
        "tracked.py filter=crypt\n",
        encoding="utf-8",
    )
    _git(repo, "add", "tracked.py", ".gitattributes")
    _git(repo, "commit", "-m", "filtered fixture")
    tracked.write_text("decrypted = True\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("", encoding="utf-8")
    if event == "pre-commit":
        _git(repo, "add", "tracked.py", ".gitattributes")

    snapshot = LocalPairReview(str(repo)).capture(
        event=event,
        focus_path="tracked.py" if event == "file-save" else None,
    )

    assert "tracked.py" not in snapshot.changed_paths
    assert "decrypted" not in snapshot.diff
    assert CoverageIssue(
        path="tracked.py",
        reason="content_filter_unsupported",
    ) in snapshot.coverage_issues


@pytest.mark.parametrize("event", ["file-save", "worktree-idle"])
def test_untracked_path_uses_index_and_base_filter_attributes(event, tmp_path):
    repo = _repo(tmp_path, f"untracked-filter-attributes-{event}")
    (repo / ".gitattributes").write_text(
        "*.secret filter=crypt\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-m", "filtered fixture")
    (repo / ".gitattributes").write_text("", encoding="utf-8")
    (repo / "new.secret").write_text("plaintext secret\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(
        event=event,
        focus_path="new.secret" if event == "file-save" else None,
    )

    assert "new.secret" not in snapshot.changed_paths
    assert "plaintext secret" not in snapshot.diff
    if event == "file-save":
        assert snapshot.diff == ""
    else:
        assert snapshot.changed_paths == (".gitattributes",)
    assert CoverageIssue(
        path="new.secret",
        reason="content_filter_unsupported",
    ) in snapshot.coverage_issues


@pytest.mark.parametrize("event", ["file-save", "worktree-idle"])
def test_untracked_copy_from_excluded_tracked_source_is_omitted(event, tmp_path):
    repo = _repo(tmp_path, f"untracked-excluded-copy-{event}")
    source = repo / "secrets.txt"
    source.write_text("complete excluded secret\n", encoding="utf-8")
    _git(repo, "add", "secrets.txt")
    _git(repo, "commit", "-m", "add excluded source")
    destination = repo / "public.txt"
    destination.write_bytes(source.read_bytes())

    snapshot = LocalPairReview(
        str(repo), excluded_paths=["secrets.txt"]
    ).capture(
        event=event,
        focus_path="public.txt" if event == "file-save" else None,
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert "complete excluded secret" not in snapshot.diff
    assert CoverageIssue(
        path="secrets.txt", reason="excluded"
    ) in snapshot.coverage_issues
    assert CoverageIssue(
        path="public.txt", reason="rename_group_omitted"
    ) in snapshot.coverage_issues


@pytest.mark.parametrize("event", ["file-save", "worktree-idle"])
def test_edited_untracked_copy_from_excluded_tracked_source_is_omitted(event, tmp_path):
    repo = _repo(tmp_path, f"untracked-edited-excluded-copy-{event}")
    source = repo / "secrets.json"
    source.write_text(
        '{"token":"production-secret-abcdef","mode":"production","region":"east"}\n',
        encoding="utf-8",
    )
    _git(repo, "add", "secrets.json")
    _git(repo, "commit", "-m", "add excluded source")
    destination = repo / "public.json"
    destination.write_text(
        '{"token":"production-secret-abcdef","mode":"production","region":"west"}\n',
        encoding="utf-8",
    )

    snapshot = LocalPairReview(str(repo), excluded_paths=["secrets.json"]).capture(
        event=event,
        focus_path="public.json" if event == "file-save" else None,
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert "production-secret-abcdef" not in snapshot.diff
    assert CoverageIssue(path="secrets.json", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="public.json", reason="rename_group_omitted") in snapshot.coverage_issues


@pytest.mark.parametrize("event", ["file-save", "worktree-idle"])
def test_unrelated_untracked_file_remains_reviewable_with_excluded_tracked_source(event, tmp_path):
    repo = _repo(tmp_path, f"untracked-unrelated-to-excluded-{event}")
    source = repo / "secrets.json"
    source.write_text(
        '{"token":"production-secret-abcdef","mode":"production","region":"east"}\n',
        encoding="utf-8",
    )
    _git(repo, "add", "secrets.json")
    _git(repo, "commit", "-m", "add excluded source")
    destination = repo / "feature.py"
    destination.write_text(
        "def greet(name):\n    return f'hello {name}'\n",
        encoding="utf-8",
    )

    snapshot = LocalPairReview(str(repo), excluded_paths=["secrets.json"]).capture(
        event=event,
        focus_path="feature.py" if event == "file-save" else None,
    )

    assert snapshot.changed_paths == ("feature.py",)
    assert "+def greet(name):" in snapshot.diff
    assert "production-secret-abcdef" not in snapshot.diff
    assert not any(issue.path == "feature.py" for issue in snapshot.coverage_issues)


@pytest.mark.parametrize("event", ["file-save", "worktree-idle"])
def test_untracked_addition_embedding_excluded_source_is_omitted(event, tmp_path):
    repo = _repo(tmp_path, f"untracked-embedded-excluded-copy-{event}")
    source = repo / ".env"
    source.write_text("API_TOKEN=production-secret\n", encoding="utf-8")
    _git(repo, "add", "-f", ".env")
    _git(repo, "commit", "-m", "add excluded source")
    destination = repo / "feature.py"
    destination.write_text(
        "def configure():\n"
        "    settings = '''\n"
        "    API_TOKEN=production-secret\n"
        "    FEATURE_FLAG=enabled\n"
        "    '''\n"
        "    return settings\n",
        encoding="utf-8",
    )

    snapshot = LocalPairReview(str(repo)).capture(
        event=event,
        focus_path="feature.py" if event == "file-save" else None,
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert "production-secret" not in snapshot.diff
    assert CoverageIssue(path=".env", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="feature.py", reason="rename_group_omitted") in snapshot.coverage_issues


def test_deleted_blob_is_validated_before_its_diff_is_captured(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "large.py"
    path.write_text("x" * 100, encoding="utf-8")
    _git(repo, "add", "large.py")
    _git(repo, "commit", "-m", "add large file")
    path.unlink()

    snapshot = LocalPairReview(str(repo), max_file_bytes=20).capture(event="worktree-idle")

    assert "large.py" not in snapshot.changed_paths
    assert CoverageIssue(path="large.py", reason="file_too_large") in snapshot.coverage_issues


def test_modified_file_validates_both_base_and_current_blobs(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "large.py"
    path.write_text("x" * 100, encoding="utf-8")
    _git(repo, "add", "large.py")
    _git(repo, "commit", "-m", "add large base")
    path.write_text("small\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), max_file_bytes=20).capture(event="worktree-idle")

    assert "large.py" not in snapshot.changed_paths
    assert CoverageIssue(path="large.py", reason="file_too_large") in snapshot.coverage_issues


def test_symlink_paths_are_preserved_and_reported_as_unsupported(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "target.py"
    target.write_text("inside = True\n", encoding="utf-8")
    link = repo / "link.py"
    link.symlink_to("target.py")

    snapshot = LocalPairReview(str(repo)).capture(event="file-save", focus_path="link.py")

    assert snapshot.focus_path == "link.py"
    assert snapshot.diff == ""
    assert CoverageIssue(path="link.py", reason="symlink") in snapshot.coverage_issues


def test_symlink_loop_is_reported_as_unsupported_coverage(tmp_path):
    repo = _repo(tmp_path, "symlink-loop")
    link = repo / "loop.py"
    link.symlink_to("loop.py")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.diff == ""
    issue = snapshot.coverage_issues[0]
    assert issue.path == "loop.py"
    assert issue.reason == "symlink"
    assert issue.fingerprint is not None


def test_excluding_either_side_rejects_an_entire_rename(tmp_path):
    repo = _repo(tmp_path)
    source = repo / "ignored.py"
    source.write_text("secret = True\n", encoding="utf-8")
    _git(repo, "add", "ignored.py")
    _git(repo, "commit", "-m", "add ignored source")
    _git(repo, "mv", "ignored.py", "allowed.py")

    snapshot = LocalPairReview(str(repo), excluded_paths=["ignored.py"]).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(path="ignored.py", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="allowed.py", reason="rename_group_omitted") in snapshot.coverage_issues

    (repo / "allowed.py").write_text("secret = False\n", encoding="utf-8")
    current = LocalPairReview(str(repo), excluded_paths=["ignored.py"]).capture(event="worktree-idle")
    assert current.snapshot_id != snapshot.snapshot_id


def test_default_secret_paths_never_enter_snapshot_input(tmp_path):
    repo = _repo(tmp_path, "default-secret-exclusions")
    (repo / ".secrets.toml").write_text(
        '[openai]\nkey = "provider-secret"\n', encoding="utf-8"
    )
    (repo / "credentials.json").write_text('{"token":"secret"}\n', encoding="utf-8")
    (repo / "signing.pem").write_text("PRIVATE KEY\n", encoding="utf-8")
    (repo / "id_ecdsa_sk").write_text("OPENSSH PRIVATE KEY\n", encoding="utf-8")
    (repo / ".git-credentials").write_text(
        "https://user:password@example.test\n", encoding="utf-8"
    )
    (repo / ".aws").mkdir()
    (repo / ".aws" / "credentials").write_text(
        "aws_secret_access_key = must-not-reach-model\n", encoding="utf-8"
    )
    (repo / ".kube").mkdir()
    (repo / ".kube" / "config").write_text(
        "token: must-not-reach-model\n", encoding="utf-8"
    )
    (repo / ".docker").mkdir()
    (repo / ".docker" / "config.json").write_text(
        '{"auths":{"registry.example":{"auth":"must-not-reach-model"}}}\n',
        encoding="utf-8",
    )
    (repo / "visible.py").write_text("value = 1\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), excluded_paths=[]).capture(event="worktree-idle")

    assert snapshot.changed_paths == ("visible.py",)
    assert "secret" not in snapshot.diff
    assert "PRIVATE KEY" not in snapshot.diff
    assert "OPENSSH PRIVATE KEY" not in snapshot.diff
    assert "password" not in snapshot.diff
    assert "aws_secret_access_key" not in snapshot.diff
    assert "token: must-not-reach-model" not in snapshot.diff
    assert '"auth":"must-not-reach-model"' not in snapshot.diff
    assert "provider-secret" not in snapshot.diff
    assert CoverageIssue(path=".secrets.toml", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="credentials.json", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="signing.pem", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="id_ecdsa_sk", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path=".git-credentials", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path=".aws/credentials", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path=".kube/config", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path=".docker/config.json", reason="excluded") in snapshot.coverage_issues


def test_file_save_rejects_a_focused_rename_with_an_excluded_source(tmp_path):
    repo = _repo(tmp_path)
    source = repo / "ignored.py"
    source.write_text("secret = True\n", encoding="utf-8")
    _git(repo, "add", "ignored.py")
    _git(repo, "commit", "-m", "add ignored source")
    _git(repo, "mv", "ignored.py", "allowed.py")

    snapshot = LocalPairReview(str(repo), excluded_paths=["ignored.py"]).capture(
        event="file-save", focus_path="allowed.py"
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(path="ignored.py", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="allowed.py", reason="rename_group_omitted") in snapshot.coverage_issues


def test_file_save_does_not_select_a_copy_by_its_unchanged_source(tmp_path):
    repo = _repo(tmp_path, "file-save-copy-source")
    source = repo / "source.txt"
    source.write_text("shared content\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "add source")
    destination = repo / "other.txt"
    destination.write_bytes(source.read_bytes())
    _git(repo, "add", "other.txt")

    source_snapshot = LocalPairReview(str(repo)).capture(
        event="file-save", focus_path="source.txt"
    )
    destination_snapshot = LocalPairReview(str(repo)).capture(
        event="file-save", focus_path="other.txt"
    )

    assert source_snapshot.changed_paths == ()
    assert source_snapshot.diff == ""
    assert destination_snapshot.changed_paths == ("other.txt",)
    assert "+shared content" in destination_snapshot.diff
    assert destination_snapshot.coverage_issues == ()


def test_file_save_checks_copy_source_filter_from_base(tmp_path):
    repo = _repo(tmp_path, "copy-source-base-filter")
    source = repo / "encrypted.txt"
    source.write_text("decoded secret\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(
        "encrypted.txt filter=crypt\n",
        encoding="utf-8",
    )
    _git(repo, "add", "encrypted.txt", ".gitattributes")
    _git(repo, "commit", "-m", "add filtered source")
    (repo / ".gitattributes").write_text("", encoding="utf-8")
    destination = repo / "public.txt"
    destination.write_bytes(source.read_bytes())
    _git(repo, "add", "public.txt")

    snapshot = LocalPairReview(str(repo)).capture(
        event="file-save", focus_path="public.txt"
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(
        path="encrypted.txt",
        reason="content_filter_unsupported",
    ) in snapshot.coverage_issues
    assert CoverageIssue(
        path="public.txt",
        reason="rename_group_omitted",
    ) in snapshot.coverage_issues


def test_pre_commit_rejects_a_copy_from_an_unchanged_excluded_source(tmp_path):
    repo = _repo(tmp_path)
    source = repo / "secrets.txt"
    source.write_text("production secret\n", encoding="utf-8")
    _git(repo, "add", "secrets.txt")
    _git(repo, "commit", "-m", "add excluded source")
    destination = repo / "public.txt"
    destination.write_bytes(source.read_bytes())
    _git(repo, "add", "public.txt")

    snapshot = LocalPairReview(str(repo), excluded_paths=["secrets.txt"]).capture(
        event="pre-commit"
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(path="secrets.txt", reason="excluded") in snapshot.coverage_issues
    assert CoverageIssue(path="public.txt", reason="rename_group_omitted") in snapshot.coverage_issues


def test_copy_detection_ignores_repository_rename_limit(tmp_path):
    repo = _repo(tmp_path, "unlimited-copy-detection")
    source_paths = []
    destination_paths = []
    for index in range(4):
        source = repo / f"secret_{index}.txt"
        destination = repo / f"public_{index}.txt"
        source.write_text((f"secret-{index}\n" * 100) + "source\n", encoding="utf-8")
        source_paths.append(source)
        destination_paths.append(destination)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add copy sources")
    _git(repo, "config", "diff.renameLimit", "1")
    for source, destination in zip(source_paths, destination_paths, strict=True):
        destination.write_bytes(source.read_bytes().replace(b"source\n", b"copied\n"))
    _git(repo, "add", ".")

    snapshot = LocalPairReview(str(repo), excluded_paths=["secret_*.txt"]).capture(
        event="pre-commit"
    )

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert {
        issue.path for issue in snapshot.coverage_issues
        if issue.reason == "rename_group_omitted"
    } == {path.name for path in destination_paths}


def test_captured_filenames_are_literal_git_pathspecs(tmp_path):
    repo = _repo(tmp_path)
    magic = ":(glob)*.py"
    (repo / magic).write_text("magic = 1\n", encoding="utf-8")
    (repo / "excluded.py").write_text("secret = 1\n", encoding="utf-8")
    (repo / "other.py").write_text("other = 1\n", encoding="utf-8")
    _git(repo, "--literal-pathspecs", "add", "--", magic, "excluded.py", "other.py")
    _git(repo, "commit", "-m", "add pathspec fixture")
    (repo / magic).write_text("magic = 2\n", encoding="utf-8")
    (repo / "excluded.py").write_text("secret = 2\n", encoding="utf-8")
    (repo / "other.py").write_text("other = 2\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), excluded_paths=["excluded.py"]).capture(
        event="file-save", focus_path=magic
    )

    assert snapshot.changed_paths == (magic,)
    assert "excluded.py" not in snapshot.diff
    assert "other.py" not in snapshot.diff


def test_untracked_focus_is_a_literal_git_pathspec(tmp_path):
    repo = _repo(tmp_path, "untracked-literal-focus")
    magic = ":(glob)*.md"
    (repo / magic).write_text("focused\n", encoding="utf-8")
    (repo / "other.md").write_text("unrelated\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(
        event="file-save", focus_path=magic
    )

    assert snapshot.changed_paths == (magic,)
    assert "other.md" not in snapshot.diff


def test_operational_ignores_are_literal_paths(tmp_path):
    repo = _repo(tmp_path)
    (repo / "[ab].md").write_text("artifact\n", encoding="utf-8")
    (repo / "a.md").write_text("review me\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), ignored_paths=["[ab].md"]).capture(event="worktree-idle")

    assert "[ab].md" not in snapshot.changed_paths
    assert "a.md" in snapshot.changed_paths


def test_capture_revalidates_before_accepting_diff_bytes(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "tracked.py"
    path.write_text("small\n", encoding="utf-8")

    class RacingReview(LocalPairReview):
        def _capture_diff(self, event, base_revision, paths, **kwargs):
            path.write_text("x" * 100, encoding="utf-8")
            return super()._capture_diff(
                event,
                base_revision,
                paths,
                **kwargs,
            )

    snapshot = RacingReview(str(repo), max_file_bytes=20).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert CoverageIssue(path="tracked.py", reason="file_too_large") in snapshot.coverage_issues


def test_capture_rejects_new_paths_discovered_after_verified_diff(tmp_path):
    repo = _repo(tmp_path, "late-path-discovery")
    tracked = repo / "tracked.py"
    tracked.write_text("value = 2\n", encoding="utf-8")
    late = repo / "late.py"

    class RacingReview(LocalPairReview):
        captured_once = False

        def _capture_diff(self, event, base_revision, paths, **kwargs):
            result = super()._capture_diff(event, base_revision, paths, **kwargs)
            if not self.captured_once:
                self.captured_once = True
                late.write_text("appeared = True\n", encoding="utf-8")
            return result

    snapshot = RacingReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(reason="content_changed_during_capture") in snapshot.coverage_issues


def test_worktree_idle_includes_index_only_content(tmp_path):
    repo = _repo(tmp_path, "index-only-idle")
    tracked = repo / "tracked.py"
    tracked.write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    tracked.write_text("value = 1\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo)).capture(event="worktree-idle")

    assert snapshot.changed_paths == ("tracked.py",)
    assert "+value = 2" in snapshot.diff
    assert "-value = 2" in snapshot.diff


def test_skipped_content_fingerprint_invalidates_snapshot_identity(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "excluded.txt"
    path.write_text("first\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo), excluded_paths=["excluded.txt"])
    first = reviewer.capture(event="worktree-idle")
    path.write_text("second\n", encoding="utf-8")
    second = reviewer.capture(event="worktree-idle")

    assert first.coverage_issues[0].fingerprint != second.coverage_issues[0].fingerprint
    assert first.snapshot_id != second.snapshot_id


def test_excluded_index_only_content_invalidates_snapshot_identity(tmp_path):
    repo = _repo(tmp_path, "excluded-index-fingerprint")
    path = repo / "excluded.txt"
    path.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "excluded.txt")
    _git(repo, "commit", "-m", "add excluded file")
    reviewer = LocalPairReview(str(repo), excluded_paths=["excluded.txt"])

    path.write_text("first staged value\n", encoding="utf-8")
    _git(repo, "add", "excluded.txt")
    path.write_text("base\n", encoding="utf-8")
    first = reviewer.capture(event="worktree-idle")

    path.write_text("second staged value\n", encoding="utf-8")
    _git(repo, "add", "excluded.txt")
    path.write_text("base\n", encoding="utf-8")
    second = reviewer.capture(event="worktree-idle")

    assert first.diff == second.diff == ""
    assert first.snapshot_id != second.snapshot_id


def test_unfingerprintable_coverage_suppresses_findings(tmp_path):
    repo = _repo(tmp_path, "embedded-repository")
    embedded = repo / "embedded"
    embedded.mkdir()
    _git(embedded, "init")
    (embedded / "nested.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo))
    snapshot = reviewer.capture(event="worktree-idle")
    current = reviewer.recapture(snapshot)

    result = build_snapshot_result(
        snapshot,
        current_snapshot=current,
        structured_review={"review": {"key_issues_to_review": [{
            "relevant_file": "tracked.py",
            "issue_header": "Bug",
            "issue_content": "The changed value is incorrect.",
            "start_line": 1,
            "end_line": 1,
        }]}},
        started_at=monotonic(),
    )

    assert any(
        issue.path == "embedded" and issue.fingerprint is None
        for issue in snapshot.coverage_issues
    )
    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None
    assert CoverageIssue(reason="unfingerprintable_coverage") in result.coverage_issues


def test_modified_submodule_coverage_is_unfingerprintable(tmp_path):
    submodule_source = tmp_path / "submodule-source"
    submodule_source.mkdir()
    _git(submodule_source, "init")
    _git(submodule_source, "config", "user.email", "test@example.com")
    _git(submodule_source, "config", "user.name", "Snapshot Test")
    (submodule_source / "nested.py").write_text("value = 1\n", encoding="utf-8")
    _git(submodule_source, "add", "nested.py")
    _git(submodule_source, "commit", "-m", "initial submodule")

    repo = _repo(tmp_path, "submodule-parent")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_source),
        "vendor",
    )
    _git(repo, "commit", "-am", "add submodule")
    (repo / "vendor" / "nested.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")

    reviewer = LocalPairReview(str(repo))
    snapshot = reviewer.capture(event="worktree-idle")
    current = reviewer.recapture(snapshot)
    submodule_issue = next(
        issue for issue in snapshot.coverage_issues if issue.path == "vendor"
    )
    result = build_snapshot_result(
        snapshot,
        current_snapshot=current,
        structured_review={"review": {"key_issues_to_review": [{
            "relevant_file": "tracked.py",
            "issue_header": "Bug",
            "issue_content": "The changed value is incorrect.",
            "start_line": 1,
            "end_line": 1,
        }]}},
        started_at=monotonic(),
    )

    assert submodule_issue.fingerprint is None
    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None


def test_outside_symlink_target_change_invalidates_snapshot_identity(tmp_path):
    repo = _repo(tmp_path, "outside-symlink")
    link = repo / "outside-link"
    link.symlink_to(tmp_path / "first-target")
    reviewer = LocalPairReview(str(repo))
    first = reviewer.capture(event="worktree-idle")
    link.unlink()
    link.symlink_to(tmp_path / "second-target")
    second = reviewer.capture(event="worktree-idle")

    assert first.coverage_issues[0].reason == "outside_repository_root"
    assert first.coverage_issues[0].fingerprint != second.coverage_issues[0].fingerprint
    assert first.snapshot_id != second.snapshot_id


def test_non_utf8_git_path_has_serializable_snapshot_identity(tmp_path):
    decoded_name = b"changed-\xff.py".decode("utf-8", errors="surrogateescape")
    snapshot = ReviewSnapshot(
        event=ReviewEvent.WORKTREE_IDLE,
        repository_root=str(tmp_path),
        base_revision="a" * 40,
        changed_paths=(decoded_name,),
        diff="",
        policy_version="v1",
        created_at="2026-01-01T00:00:00+00:00",
        coverage_issues=(CoverageIssue(path=decoded_name, reason="binary"),),
    )

    assert snapshot.changed_paths == (decoded_name,)
    assert "\\udcff" in json.dumps(snapshot.to_dict(), ensure_ascii=True)


def test_skipped_mode_change_invalidates_snapshot_identity(tmp_path):
    repo = _repo(tmp_path, "skipped-mode")
    skipped = repo / "skipped.py"
    skipped.write_text("secret = True\n", encoding="utf-8")
    _git(repo, "add", "skipped.py")
    _git(repo, "commit", "-m", "add skipped file")
    skipped.write_text("secret = False\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo), excluded_paths=["skipped.py"])
    first = reviewer.capture(event="worktree-idle")
    skipped.chmod(skipped.stat().st_mode | stat.S_IXUSR)
    second = reviewer.capture(event="worktree-idle")

    assert first.snapshot_id != second.snapshot_id


def test_skipped_file_fingerprint_reads_only_bounded_samples(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "bounded-fingerprint")
    skipped = repo / "skipped.py"
    skipped.write_bytes(b"x" * 100)
    reviewer = LocalPairReview(str(repo), excluded_paths=["skipped.py"], max_file_bytes=10)

    class TrackingStream(BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            chunk = super().read(size)
            self.bytes_read += len(chunk)
            return chunk

    stream = TrackingStream(b"x" * 100)
    monkeypatch.setattr(Path, "open", lambda self, mode="r", **kwargs: stream)

    assert reviewer._path_fingerprint(ReviewEvent.WORKTREE_IDLE, "skipped.py", "HEAD")
    assert stream.bytes_read <= 22


def test_file_save_requires_a_safe_focused_path_and_captures_untracked_addition(tmp_path):
    repo = _repo(tmp_path)
    (repo / "new.py").write_text("added = True\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo))

    snapshot = reviewer.capture(event="file-save", focus_path="new.py")

    assert snapshot.focus_path == "new.py"
    assert snapshot.changed_paths == ("new.py",)
    assert "--- /dev/null" in snapshot.diff
    with pytest.raises(SnapshotCaptureError, match="outside repository root"):
        reviewer.capture(event="file-save", focus_path="../outside.py")
    with pytest.raises(SnapshotCaptureError, match="require --path"):
        reviewer.capture(event="file-save")


def test_file_save_rejects_a_directory_focus_path(tmp_path):
    repo = _repo(tmp_path)
    directory = repo / "saved"
    directory.mkdir()
    (directory / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "saved/tracked.py")
    _git(repo, "commit", "-m", "add saved file")
    (directory / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (directory / "untracked.py").write_text("value = 3\n", encoding="utf-8")

    with pytest.raises(SnapshotCaptureError, match="must identify a file"):
        LocalPairReview(str(repo)).capture(event="file-save", focus_path="saved")


def test_binary_and_excluded_files_are_visible_partial_coverage(tmp_path):
    repo = _repo(tmp_path)
    (repo / "binary.dat").write_bytes(b"\0secret")
    (repo / "ignored.txt").write_text("do not review\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), excluded_paths=["ignored.*"]).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert {(issue.path, issue.reason) for issue in snapshot.coverage_issues} == {
        ("binary.dat", "binary"),
        ("ignored.txt", "excluded"),
    }


def test_negative_file_size_limit_is_clamped_before_reads(tmp_path):
    repo = _repo(tmp_path, "negative-limit")
    (repo / "large.py").write_text("content\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo), max_file_bytes=-2)

    snapshot = reviewer.capture(event="worktree-idle")

    assert reviewer.max_file_bytes == 0
    assert CoverageIssue(path="large.py", reason="file_too_large") in snapshot.coverage_issues


def test_snapshot_diff_budget_reports_all_uncaptured_paths(tmp_path):
    repo = _repo(tmp_path, "snapshot-budget")
    (repo / "first.py").write_text("first = 1\n", encoding="utf-8")
    (repo / "second.py").write_text("second = 2\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo), max_snapshot_bytes=0)

    snapshot = reviewer.capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert {
        issue.path for issue in snapshot.coverage_issues
        if issue.reason == "snapshot_byte_budget"
    } == {"first.py", "second.py"}


def test_snapshot_diff_read_is_bounded_by_remaining_budget(tmp_path):
    repo = _repo(tmp_path, "bounded-snapshot-read")
    path = repo / "tracked.py"
    path.write_text("value = '" + ("x" * 20_000) + "'\n", encoding="utf-8")

    snapshot = LocalPairReview(
        str(repo),
        max_file_bytes=100_000,
        max_snapshot_bytes=128,
    ).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert CoverageIssue(path="tracked.py", reason="snapshot_byte_budget") in snapshot.coverage_issues


def test_tracked_diff_capture_batches_all_selected_paths(monkeypatch, tmp_path):
    repo = _repo(tmp_path, "batched-diff")
    for index in range(3):
        (repo / f"file_{index}.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add batch")
    for index in range(3):
        (repo / f"file_{index}.py").write_text("value = 2\n", encoding="utf-8")

    reviewer = LocalPairReview(str(repo))
    original_capture_diff = reviewer._capture_diff
    captured_path_sets = []

    def capture_diff(event, base_revision, paths, **kwargs):
        captured_path_sets.append(tuple(paths))
        return original_capture_diff(event, base_revision, paths, **kwargs)

    monkeypatch.setattr(reviewer, "_capture_diff", capture_diff)

    snapshot = reviewer.capture(event="worktree-idle")

    assert set(snapshot.changed_paths) == {"file_0.py", "file_1.py", "file_2.py"}
    assert captured_path_sets == [
        ("file_0.py", "file_1.py", "file_2.py"),
        ("file_0.py", "file_1.py", "file_2.py"),
    ]


def test_tracked_diff_capture_bounds_path_arguments(monkeypatch, tmp_path):
    repo = _repo(tmp_path, "bounded-diff-arguments")
    paths = []
    for index in range(80):
        path = f"file_{index:03d}_{'x' * 220}.py"
        (repo / path).write_text("value = 1\n", encoding="utf-8")
        paths.append(path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add many files")
    for path in paths:
        (repo / path).write_text("value = 2\n", encoding="utf-8")

    reviewer = LocalPairReview(str(repo))
    original_capture_diff = reviewer._capture_diff
    captured_path_sets = []

    def capture_diff(event, base_revision, batch, **kwargs):
        captured_path_sets.append(tuple(batch))
        return original_capture_diff(event, base_revision, batch, **kwargs)

    monkeypatch.setattr(reviewer, "_capture_diff", capture_diff)

    snapshot = reviewer.capture(event="worktree-idle")

    assert set(snapshot.changed_paths) == set(paths)
    assert len(captured_path_sets) >= 4
    assert all(
        sum(len(path.encode("utf-8")) + 1 for path in batch) <= 16_384
        for batch in captured_path_sets
    )


def test_changed_path_discovery_is_bounded_and_reports_stage_coverage(tmp_path):
    repo = _repo(tmp_path, "bounded-path-discovery")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "untracked.py").write_text("value = 1\n", encoding="utf-8")

    snapshot = LocalPairReview(
        str(repo),
        max_path_discovery_bytes=1,
    ).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(reason="tracked_path_discovery_budget") in snapshot.coverage_issues
    assert CoverageIssue(reason="untracked_path_discovery_budget") in snapshot.coverage_issues


def test_one_discovery_overflow_discards_the_other_stage_diff(tmp_path):
    repo = _repo(tmp_path, "partial-path-overflow")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "u").write_text("untracked\n", encoding="utf-8")

    snapshot = LocalPairReview(
        str(repo),
        max_path_discovery_bytes=3,
    ).capture(event="worktree-idle")

    assert snapshot.diff == ""
    assert snapshot.changed_paths == ()
    assert CoverageIssue(reason="tracked_path_discovery_budget") in snapshot.coverage_issues
    assert CoverageIssue(reason="untracked_path_discovery_budget") not in snapshot.coverage_issues


def test_superseded_snapshot_is_stale_and_suppresses_review(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "tracked.py"
    path.write_text("value = 2\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo))
    snapshot = reviewer.capture(event="file-save", focus_path="tracked.py")
    path.write_text("value = 3\n", encoding="utf-8")
    current = reviewer.recapture(snapshot)

    result = build_snapshot_result(
        snapshot,
        current_snapshot=current,
        structured_review={"review": {"key_issues_to_review": [{"issue": "old"}]}},
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.STALE
    assert result.review is None
    assert result.current_snapshot_id == current.snapshot_id


def test_recapture_re_resolves_a_moving_base_selector(tmp_path):
    repo = _repo(tmp_path, "moving-head")
    path = repo / "tracked.py"
    path.write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    reviewer = LocalPairReview(str(repo))
    snapshot = reviewer.capture(event="pre-commit", base="HEAD")

    _git(repo, "commit", "-m", "advance head")
    current = reviewer.recapture(snapshot)

    assert snapshot.base_selector == "HEAD"
    assert current.base_selector == "HEAD"
    assert current.base_revision != snapshot.base_revision
    assert current.snapshot_id != snapshot.snapshot_id


def test_recapture_can_recompute_configuration_identity(tmp_path):
    repo = _repo(tmp_path, "moving-configuration")
    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    reviewer = LocalPairReview(str(repo))
    snapshot = reviewer.capture(
        event="worktree-idle",
        review_configuration_hash="sha256:old",
    )

    current = reviewer.recapture(
        snapshot,
        review_configuration_hash_factory=lambda base_revision: "sha256:new",
    )

    assert current.review_configuration_hash == "sha256:new"
    assert current.snapshot_id != snapshot.snapshot_id


def test_result_states_distinguish_findings_clean_and_unavailable():
    snapshot = _snapshot("/repo/one")
    findings = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": [{
            "relevant_file": "x",
            "issue_header": "Bug",
            "issue_content": "The wrong value is returned.",
            "start_line": 1,
            "end_line": 1,
        }]}},
        started_at=monotonic(),
    )
    clean = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    unavailable = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review=None,
        started_at=monotonic(),
        error="AuthenticationError",
    )

    assert findings.state is ReviewResultState.FINDINGS
    assert clean.state is ReviewResultState.NO_FINDINGS
    assert unavailable.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert CoverageIssue(reason="review_failed:AuthenticationError") in unavailable.coverage_issues


def test_partial_coverage_cannot_be_reported_as_clean():
    snapshot = _snapshot("/repo/one")
    snapshot = ReviewSnapshot(**{
        key: value
        for key, value in snapshot.__dict__.items()
        if key not in {"snapshot_id", "coverage_issues"}
    }, coverage_issues=(CoverageIssue(path="large.bin", reason="binary"),))
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None


def test_token_budget_omissions_cannot_be_reported_as_clean():
    snapshot = _snapshot("/repo/one")
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={
            "review": {"key_issues_to_review": []},
            "metadata": {"omitted_files": ["large.py"]},
        },
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None
    assert CoverageIssue(path="large.py", reason="token_budget_omitted") in result.coverage_issues


def test_deleted_files_have_distinct_unsupported_coverage():
    snapshot = _snapshot("/repo/one")
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={
            "review": {"key_issues_to_review": []},
            "metadata": {"deleted_files": ["removed.py"]},
        },
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None
    assert CoverageIssue(path="removed.py", reason="deleted_file_unsupported") in result.coverage_issues
    assert CoverageIssue(path="removed.py", reason="token_budget_omitted") not in result.coverage_issues


@pytest.mark.parametrize(
    ("relevant_file", "start_line", "end_line"),
    [
        ("other.py", 1, 1),
        ("x", 2, 2),
        ("x", 1, 10**30),
    ],
)
def test_findings_must_match_captured_files_and_hunk_lines(
    relevant_file, start_line, end_line
):
    snapshot = _snapshot("/repo/one")

    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": [{
            "relevant_file": relevant_file,
            "issue_header": "Bug",
            "issue_content": "This finding is outside the captured hunk.",
            "start_line": start_line,
            "end_line": end_line,
        }]}},
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None
    assert CoverageIssue(reason="review_failed:InvalidStructuredReview") in result.coverage_issues


def test_finding_range_may_include_adjacent_unchanged_context():
    snapshot = _snapshot("/repo/one")

    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": [{
            "relevant_file": "x",
            "issue_header": "Bug",
            "issue_content": "The changed line breaks the adjacent context.",
            "start_line": 1,
            "end_line": 2,
        }]}},
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.FINDINGS
    assert result.review is not None


@pytest.mark.parametrize(
    "review",
    [
        {},
        {"key_issues_to_review": {}},
        {"key_issues_to_review": "none"},
        {"key_issues_to_review": [None]},
        {"key_issues_to_review": [{}]},
        {"key_issues_to_review": [{
            "relevant_file": "app.py",
            "issue_header": "Bug",
            "issue_content": "Missing line coordinates.",
        }]},
    ],
)
def test_malformed_finding_collection_cannot_be_reported_as_clean(review):
    snapshot = _snapshot("/repo/one")

    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": review},
        started_at=monotonic(),
    )

    assert result.state is ReviewResultState.COVERAGE_UNAVAILABLE
    assert result.review is None
    assert CoverageIssue(reason="review_failed:InvalidStructuredReview") in result.coverage_issues


def test_cache_is_repository_local_and_does_not_store_unavailable_results(tmp_path):
    first_repo = _repo(tmp_path, "one")
    second_repo = _repo(tmp_path, "two")
    first_cache = SnapshotCache(first_repo)
    second_cache = SnapshotCache(second_repo)
    first_snapshot = _snapshot(str(first_repo))
    result = build_snapshot_result(
        first_snapshot,
        current_snapshot=first_snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    first_cache.write(result)

    assert first_cache.read(first_snapshot.snapshot_id).cached is True
    assert second_cache.read(first_snapshot.snapshot_id) is None

    unavailable = build_snapshot_result(
        first_snapshot,
        current_snapshot=first_snapshot,
        structured_review=None,
        started_at=monotonic(),
        error="TimeoutError",
    )
    first_cache.write(unavailable)
    cached = first_cache.read(unavailable.snapshot_id)
    assert cached is None or cached.state is not ReviewResultState.COVERAGE_UNAVAILABLE


def test_cache_eviction_tolerates_entry_removed_before_stat(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "cache-race")
    cache = SnapshotCache(repo, max_entries=1)
    snapshot = _snapshot(str(repo))
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    vanished = cache.cache_dir / "vanished.json"
    vanished.write_text("{}", encoding="utf-8")
    original_stat = Path.stat

    def racing_stat(path, *args, **kwargs):
        if path == vanished:
            vanished.unlink(missing_ok=True)
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)

    cache.write(result)
    assert cache.read(snapshot.snapshot_id) is not None


def test_cache_treats_structurally_invalid_json_as_a_miss(tmp_path):
    repo = _repo(tmp_path, "invalid-cache")
    cache = SnapshotCache(repo)
    snapshot = _snapshot(str(repo))
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    invalid_payloads = [
        {"snapshot_id": snapshot.snapshot_id},
        {"snapshot_id": snapshot.snapshot_id, "state": "unknown"},
        {
            "snapshot_id": snapshot.snapshot_id,
            "state": "no_findings",
            "latency_seconds": "not-a-number",
        },
        {
            "snapshot_id": snapshot.snapshot_id,
            "state": "no_findings",
            "coverage_issues": [{"path": "missing-reason.py"}],
        },
    ]

    for payload in invalid_payloads:
        cache._path(snapshot.snapshot_id).write_text(json.dumps(payload), encoding="utf-8")
        assert cache.read(snapshot.snapshot_id) is None


def test_cache_treats_semantically_inconsistent_results_as_misses(tmp_path):
    repo = _repo(tmp_path, "inconsistent-cache")
    cache = SnapshotCache(repo)
    snapshot = _snapshot(str(repo))
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    valid = result.to_dict()
    invalid_payloads = [
        {**valid, "coverage_issues": [{"reason": "unreadable", "path": "lost.py"}]},
        {**valid, "current_snapshot_id": "sha256:" + "0" * 64},
        {**valid, "state": "findings"},
        {
            **valid,
            "review": {"key_issues_to_review": [{"issue": "contradicts state"}]},
        },
        {**valid, "state": "stale"},
        {**valid, "advisory": False},
        {**valid, "review": {}},
        {**valid, "review": {"key_issues_to_review": "none"}},
        {
            **valid,
            "state": "findings",
            "review": {"key_issues_to_review": [{
                "relevant_file": "other.py",
                "issue_header": "Bug",
                "issue_content": "This file was not captured.",
                "start_line": 1,
                "end_line": 1,
            }]},
        },
    ]
    cache.cache_dir.mkdir(parents=True, exist_ok=True)

    for payload in invalid_payloads:
        cache._path(snapshot.snapshot_id).write_text(json.dumps(payload), encoding="utf-8")
        assert cache.read(snapshot.snapshot_id, snapshot=snapshot) is None


def test_cache_write_failure_does_not_abort_completed_review(tmp_path):
    repo = _repo(tmp_path, "cache-write-failure")
    cache = SnapshotCache(repo)
    snapshot = _snapshot(str(repo))
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    cache.cache_dir.parent.write_text("not a directory", encoding="utf-8")

    cache.write(result)

    assert result.state is ReviewResultState.NO_FINDINGS


@pytest.mark.parametrize("symlink_component", ["pr-agent", "snapshot-cache"])
def test_cache_refuses_symlinked_directories(tmp_path, symlink_component):
    repo = _repo(tmp_path, f"symlinked-cache-{symlink_component}")
    cache = SnapshotCache(repo)
    snapshot = _snapshot(str(repo))
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    external = tmp_path / f"external-{symlink_component}"
    external.mkdir()
    if symlink_component == "pr-agent":
        cache.cache_dir.parent.symlink_to(external, target_is_directory=True)
    else:
        cache.cache_dir.parent.mkdir()
        cache.cache_dir.symlink_to(external, target_is_directory=True)

    cache.write(result)

    assert cache.read(snapshot.snapshot_id) is None
    assert list(external.iterdir()) == []


def test_cache_write_stays_bound_when_validated_path_is_swapped(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "cache-path-swap")
    cache = SnapshotCache(repo)
    snapshot = _snapshot(str(repo))
    result = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": []}},
        started_at=monotonic(),
    )
    cache.cache_dir.mkdir(parents=True)
    original_open_cache_dir = cache._open_cache_dir
    original_cache_parent = cache.cache_dir.parent.with_name("pr-agent-original")
    external = tmp_path / "external-cache-swap"
    external.mkdir()

    def open_then_swap(*, create):
        cache_fd = original_open_cache_dir(create=create)
        cache.cache_dir.parent.rename(original_cache_parent)
        cache.cache_dir.parent.symlink_to(external, target_is_directory=True)
        return cache_fd

    monkeypatch.setattr(cache, "_open_cache_dir", open_then_swap)

    cache.write(result)

    assert list(external.iterdir()) == []
    assert (original_cache_parent / "snapshot-cache" / cache._path(snapshot.snapshot_id).name).is_file()
