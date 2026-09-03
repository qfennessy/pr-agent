import hashlib
import json

import pytest

from pr_agent.algo.checkpoint_evaluation import CheckpointCase, EvaluationCohort, EvaluationValidationError
from pr_agent.algo.checkpoint_evaluation_snapshot import load_review_snapshot_artifact
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewEvent, ReviewSnapshot


def _snapshot(*, deterministic_results=()) -> ReviewSnapshot:
    return ReviewSnapshot(
        event=ReviewEvent.FILE_SAVE,
        repository_root="/private/cocos-story",
        base_revision="a" * 40,
        base_selector="HEAD",
        changed_paths=("src/example.py",),
        diff="--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new\n",
        policy_version="snapshot-policy-v1",
        created_at="2026-08-30T12:00:00+00:00",
        focus_path="src/example.py",
        task_intent="fix the example",
        deterministic_results=deterministic_results,
        review_configuration_hash="configuration-v1",
        coverage_issues=(CoverageIssue(reason="binary", path="asset.bin"),),
    )


def _serialize(snapshot: ReviewSnapshot) -> bytes:
    return (
        json.dumps(snapshot.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _artifact_hash(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _case(
    snapshot: ReviewSnapshot,
    payload: bytes,
    *,
    event: ReviewEvent | None = None,
) -> CheckpointCase:
    return CheckpointCase(
        case_id="checkpoint-1",
        snapshot_id=snapshot.snapshot_id,
        snapshot_artifact_hash=_artifact_hash(payload),
        event=event or snapshot.event,
        cohort=EvaluationCohort.HOLDOUT,
    )


def test_review_snapshot_loader_binds_exact_bytes_content_and_checkpoint(tmp_path):
    snapshot = _snapshot(deterministic_results=({"check": "unit-tests", "passed": True},))
    payload = _serialize(snapshot)
    path = tmp_path / "snapshot.json"
    path.write_bytes(payload)

    loaded = load_review_snapshot_artifact(path, _case(snapshot, payload))

    assert loaded.to_dict() == snapshot.to_dict()
    assert loaded.snapshot_id == snapshot.snapshot_id


def test_review_snapshot_loader_rejects_changed_bytes_before_parsing(tmp_path):
    snapshot = _snapshot()
    payload = _serialize(snapshot)
    path = tmp_path / "snapshot.json"
    path.write_bytes(payload + b" ")

    with pytest.raises(EvaluationValidationError, match="artifact bytes"):
        load_review_snapshot_artifact(path, _case(snapshot, payload))


def test_review_snapshot_loader_rejects_content_id_and_event_mismatches(tmp_path):
    snapshot = _snapshot()
    value = snapshot.to_dict()
    value["snapshot_id"] = "sha256:" + "0" * 64
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / "snapshot.json"
    path.write_bytes(payload)

    with pytest.raises(EvaluationValidationError, match="reconstructed content"):
        load_review_snapshot_artifact(path, _case(snapshot, payload))

    valid_payload = _serialize(snapshot)
    path.write_bytes(valid_payload)
    with pytest.raises(EvaluationValidationError, match="event does not match"):
        load_review_snapshot_artifact(
            path,
            _case(snapshot, valid_payload, event=ReviewEvent.PRE_COMMIT),
        )


def test_review_snapshot_loader_rejects_answer_leakage_and_unknown_fields(tmp_path):
    leaking = _snapshot(deterministic_results=({
        "results": [{
            "nested": {"expected_withdrawn_fingerprints": ["known-bug"]},
        }],
    },))
    payload = _serialize(leaking)
    path = tmp_path / "snapshot.json"
    path.write_bytes(payload)
    with pytest.raises(EvaluationValidationError, match="answer-only fields"):
        load_review_snapshot_artifact(path, _case(leaking, payload))

    clean = _snapshot()
    value = clean.to_dict()
    value["truth"] = {"is_clean": False}
    unknown_payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(unknown_payload)
    with pytest.raises(EvaluationValidationError, match="unknown fields"):
        load_review_snapshot_artifact(path, _case(clean, unknown_payload))


def test_review_snapshot_loader_deep_freezes_deterministic_results(tmp_path):
    evidence = {"check": {"items": [{"status": "passed"}]}}
    snapshot = _snapshot(deterministic_results=(evidence,))
    evidence["check"]["items"][0]["status"] = "changed-after-capture"
    payload = _serialize(snapshot)
    path = tmp_path / "snapshot.json"
    path.write_bytes(payload)

    loaded = load_review_snapshot_artifact(path, _case(snapshot, payload))
    nested = loaded.deterministic_results[0]["check"]["items"][0]

    assert nested["status"] == "passed"
    with pytest.raises(TypeError):
        nested["status"] = "mutated"
    with pytest.raises(AttributeError):
        loaded.deterministic_results[0]["check"]["items"].append({"status": "mutated"})
    assert json.loads(json.dumps(loaded.to_dict())) == json.loads(json.dumps(snapshot.to_dict()))


def test_review_snapshot_loader_rejects_symlinks_duplicate_keys_and_oversized_files(tmp_path):
    snapshot = _snapshot()
    payload = _serialize(snapshot)
    target = tmp_path / "snapshot.json"
    target.write_bytes(payload)
    symlink = tmp_path / "snapshot-link.json"
    symlink.symlink_to(target)
    case = _case(snapshot, payload)

    with pytest.raises(EvaluationValidationError, match="regular file, not a symlink"):
        load_review_snapshot_artifact(symlink, case)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested_target = real_parent / "snapshot.json"
    nested_target.write_bytes(payload)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(EvaluationValidationError, match="parent components"):
        load_review_snapshot_artifact(linked_parent / "snapshot.json", case)

    duplicate_payload = b'{"schema_version":"review-snapshot-v1","schema_version":"review-snapshot-v1"}'
    target.write_bytes(duplicate_payload)
    with pytest.raises(EvaluationValidationError, match="duplicate JSON key"):
        load_review_snapshot_artifact(target, _case(snapshot, duplicate_payload))

    target.write_bytes(payload)
    with pytest.raises(EvaluationValidationError, match="byte validation limit"):
        load_review_snapshot_artifact(target, case, max_bytes=len(payload) - 1)
