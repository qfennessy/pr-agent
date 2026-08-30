import subprocess
from pathlib import Path
from time import monotonic

import pytest

from pr_agent.algo.review_snapshot import (CoverageIssue, ReviewEvent,
                                           ReviewResultState, ReviewSnapshot)
from pr_agent.git_providers.plain_diff_provider import parse_plain_diff
from pr_agent.tools.local_pair_review import (LocalPairReview, SnapshotCache,
                                              SnapshotCaptureError,
                                              build_snapshot_result)


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


def _snapshot(root: str, *, diff: str = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", policy="v1"):
    return ReviewSnapshot(
        event=ReviewEvent.FILE_SAVE,
        repository_root=root,
        base_revision="a" * 40,
        changed_paths=("x",),
        focus_path="x",
        diff=diff,
        policy_version=policy,
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
    assert first.snapshot_id != _snapshot("/repo/two").snapshot_id


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

    assert {"tracked.py", "new.py", "deleted.py", "renamed.py"}.issubset(set(snapshot.changed_paths))
    assert parsed["new.py"].edit_type.name == "ADDED"
    assert parsed["deleted.py"].edit_type.name == "DELETED"
    assert parsed["renamed.py"].edit_type.name == "RENAMED"
    assert parsed["renamed.py"].old_filename == "rename_me.py"


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


def test_pre_commit_coverage_inspects_the_staged_blob(tmp_path):
    repo = _repo(tmp_path)
    path = repo / "tracked.py"
    path.write_text("x" * 100, encoding="utf-8")
    _git(repo, "add", "tracked.py")
    path.write_text("small\n", encoding="utf-8")

    snapshot = LocalPairReview(str(repo), max_file_bytes=20).capture(event="pre-commit")

    assert snapshot.diff == ""
    assert CoverageIssue(path="tracked.py", reason="file_too_large") in snapshot.coverage_issues


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


def test_result_states_distinguish_findings_clean_and_unavailable():
    snapshot = _snapshot("/repo/one")
    findings = build_snapshot_result(
        snapshot,
        current_snapshot=snapshot,
        structured_review={"review": {"key_issues_to_review": [{"issue": "bug"}]}},
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
