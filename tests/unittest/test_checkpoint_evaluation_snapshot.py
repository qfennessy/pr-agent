import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from functools import lru_cache

import pytest

from pr_agent.algo.checkpoint_evaluation import CheckpointCase, EvaluationCohort, EvaluationValidationError
from pr_agent.algo.checkpoint_evaluation_snapshot import (
    load_review_snapshot_and_configuration_artifacts,
    load_review_snapshot_artifact,
)
from pr_agent.algo.review_configuration import (
    ReviewConfigurationBundle,
    materialize_review_configuration,
    review_configuration_artifact_name,
    review_configuration_canonical_bytes,
)
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewEvent, ReviewSnapshot


def _snapshot(*, deterministic_results=(), review_configuration_hash="configuration-v1") -> ReviewSnapshot:
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
        review_configuration_hash=review_configuration_hash,
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


@lru_cache(maxsize=1)
def _captured_configuration_bundle() -> ReviewConfigurationBundle:
    return materialize_review_configuration(
        skills_context="frozen skill instructions",
        repo_context_files={"AGENTS.md": "frozen repository instructions"},
        repo_context_max_lines=100,
        prompt_date="2026-09-03",
    )


def _configuration_bundle(**changes) -> ReviewConfigurationBundle:
    captured = _captured_configuration_bundle()
    if not changes:
        return captured
    values = {
        "runtime_version": captured.runtime_version,
        "runtime_artifact_hash": captured.runtime_artifact_hash,
        "settings": captured.settings,
        "missing_settings": captured.missing_settings,
        "skills_context": captured.skills_context,
        "repo_context_files": captured.repo_context_files,
        "repo_context_max_lines": captured.repo_context_max_lines,
        "prompt_date": captured.prompt_date,
        "schema_version": captured.schema_version,
    }
    values.update(changes)
    return ReviewConfigurationBundle(**values)


def _write_private_pair(tmp_path, bundle, *, configuration_payload=None, snapshot=None):
    tmp_path.chmod(0o700)
    snapshot = snapshot or _snapshot(review_configuration_hash=bundle.configuration_hash)
    snapshot_payload = _serialize(snapshot)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(snapshot_payload)
    snapshot_path.chmod(0o600)
    configuration_payload = configuration_payload or review_configuration_canonical_bytes(bundle)
    configuration_path = tmp_path / review_configuration_artifact_name(snapshot.review_configuration_hash)
    configuration_path.write_bytes(configuration_payload)
    configuration_path.chmod(0o600)
    return snapshot, snapshot_payload, snapshot_path, configuration_payload, configuration_path


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


def test_paired_loader_returns_exact_frozen_snapshot_and_configuration(tmp_path):
    bundle = _configuration_bundle()
    snapshot, snapshot_payload, snapshot_path, configuration_payload, configuration_path = _write_private_pair(
        tmp_path,
        bundle,
    )

    loaded = load_review_snapshot_and_configuration_artifacts(
        snapshot_path,
        _case(snapshot, snapshot_payload),
        review_configuration_artifact_hash=_artifact_hash(configuration_payload),
    )

    assert loaded.snapshot.to_dict() == snapshot.to_dict()
    assert loaded.review_configuration.to_dict() == bundle.to_dict()
    assert configuration_path.name == review_configuration_artifact_name(bundle.configuration_hash)
    with pytest.raises(FrozenInstanceError):
        loaded.snapshot = snapshot


@pytest.mark.parametrize("configuration_hash", [None, "configuration-v1"])
def test_paired_loader_rejects_missing_or_legacy_configuration_hash(tmp_path, configuration_hash):
    bundle = _configuration_bundle()
    snapshot = _snapshot(review_configuration_hash=configuration_hash)
    snapshot_payload = _serialize(snapshot)
    tmp_path.chmod(0o700)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(snapshot_payload)
    snapshot_path.chmod(0o600)

    with pytest.raises(EvaluationValidationError, match="must identify an immutable configuration"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(review_configuration_canonical_bytes(bundle)),
        )


def test_paired_loader_requires_expected_configuration_filename_and_byte_hash(tmp_path):
    bundle = _configuration_bundle()
    snapshot = _snapshot(review_configuration_hash=bundle.configuration_hash)
    snapshot_payload = _serialize(snapshot)
    tmp_path.chmod(0o700)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_bytes(snapshot_payload)
    snapshot_path.chmod(0o600)
    wrong_path = tmp_path / "review-configuration.json"
    configuration_payload = review_configuration_canonical_bytes(bundle)
    wrong_path.write_bytes(configuration_payload)
    wrong_path.chmod(0o600)

    with pytest.raises(EvaluationValidationError, match="cannot inspect ReviewConfiguration"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )

    configuration_path = tmp_path / review_configuration_artifact_name(bundle.configuration_hash)
    wrong_path.rename(configuration_path)
    with pytest.raises(EvaluationValidationError, match="expected artifact hash"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash="sha256:" + "0" * 64,
        )


def test_paired_loader_rejects_wrong_bundle_even_when_its_artifact_hash_matches(tmp_path):
    expected_bundle = _configuration_bundle()
    wrong_bundle = _configuration_bundle(prompt_date="2026-09-04")
    snapshot = _snapshot(review_configuration_hash=expected_bundle.configuration_hash)
    wrong_payload = review_configuration_canonical_bytes(wrong_bundle)
    snapshot, snapshot_payload, snapshot_path, _, _ = _write_private_pair(
        tmp_path,
        expected_bundle,
        configuration_payload=wrong_payload,
        snapshot=snapshot,
    )

    with pytest.raises(EvaluationValidationError, match="does not match ReviewSnapshot"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(wrong_payload),
        )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda value: value + b"\n", "not canonical"),
        (
            lambda value: b'{"configuration_hash":"sha256:' + b"0" * 64 + b'",' + value[1:],
            "duplicate JSON key",
        ),
        (
            lambda value: value.replace(b'"repo_context_max_lines":100', b'"repo_context_max_lines":NaN'),
            "non-finite JSON number",
        ),
        (lambda value: value[:-1] + b"\xff", "strict UTF-8 JSON"),
    ],
)
def test_paired_loader_rejects_noncanonical_duplicate_nonfinite_and_invalid_utf8(
    tmp_path,
    mutate,
    error,
):
    bundle = _configuration_bundle()
    configuration_payload = mutate(review_configuration_canonical_bytes(bundle))
    snapshot, snapshot_payload, snapshot_path, _, _ = _write_private_pair(
        tmp_path,
        bundle,
        configuration_payload=configuration_payload,
    )

    with pytest.raises(EvaluationValidationError, match=error):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )


def test_paired_loader_rejects_tampered_and_oversized_configuration_before_parsing(tmp_path):
    bundle = _configuration_bundle()
    snapshot, snapshot_payload, snapshot_path, configuration_payload, configuration_path = _write_private_pair(
        tmp_path,
        bundle,
    )
    configuration_path.write_bytes(configuration_payload + b" ")
    configuration_path.chmod(0o600)

    with pytest.raises(EvaluationValidationError, match="expected artifact hash"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )
    with pytest.raises(EvaluationValidationError, match="byte validation limit"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload + b" "),
            max_configuration_bytes=len(configuration_payload),
        )


def test_paired_loader_rejects_incompatible_runtime(tmp_path):
    bundle = _configuration_bundle(runtime_version="incompatible-runtime")
    snapshot, snapshot_payload, snapshot_path, configuration_payload, _ = _write_private_pair(tmp_path, bundle)

    with pytest.raises(EvaluationValidationError, match="runtime mismatch"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )


@pytest.mark.parametrize(("target", "mode"), [("snapshot", 0o644), ("configuration", 0o640)])
def test_paired_loader_requires_owner_only_files(tmp_path, target, mode):
    bundle = _configuration_bundle()
    snapshot, snapshot_payload, snapshot_path, configuration_payload, configuration_path = _write_private_pair(
        tmp_path,
        bundle,
    )
    (snapshot_path if target == "snapshot" else configuration_path).chmod(mode)

    with pytest.raises(EvaluationValidationError, match="owner-only single-link"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )


def test_paired_loader_requires_owner_only_directory(tmp_path):
    bundle = _configuration_bundle()
    snapshot, snapshot_payload, snapshot_path, configuration_payload, _ = _write_private_pair(tmp_path, bundle)
    tmp_path.chmod(0o755)

    with pytest.raises(EvaluationValidationError, match="parent must be an owner-only directory"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_paired_loader_rejects_linked_configuration_artifacts(tmp_path, link_kind):
    bundle = _configuration_bundle()
    snapshot, snapshot_payload, snapshot_path, configuration_payload, configuration_path = _write_private_pair(
        tmp_path,
        bundle,
    )
    configuration_path.unlink()
    link_target = tmp_path / "configuration-target.json"
    link_target.write_bytes(configuration_payload)
    link_target.chmod(0o600)
    if link_kind == "symlink":
        configuration_path.symlink_to(link_target)
        error = "regular file, not a symlink"
    else:
        os.link(link_target, configuration_path)
        error = "owner-only single-link"

    with pytest.raises(EvaluationValidationError, match=error):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )


def test_paired_loader_rejects_snapshot_path_replacement_during_read(tmp_path, monkeypatch):
    bundle = _configuration_bundle()
    snapshot, snapshot_payload, snapshot_path, configuration_payload, _ = _write_private_pair(tmp_path, bundle)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(snapshot_payload)
    replacement.chmod(0o600)
    real_read = os.read
    replaced = False

    def replacing_read(descriptor, amount):
        nonlocal replaced
        value = real_read(descriptor, amount)
        if value and not replaced:
            replaced = True
            os.replace(replacement, snapshot_path)
        return value

    monkeypatch.setattr(os, "read", replacing_read)

    with pytest.raises(EvaluationValidationError, match="changed while it was read"):
        load_review_snapshot_and_configuration_artifacts(
            snapshot_path,
            _case(snapshot, snapshot_payload),
            review_configuration_artifact_hash=_artifact_hash(configuration_payload),
        )
