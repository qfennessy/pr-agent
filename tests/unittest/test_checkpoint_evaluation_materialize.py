import hashlib
import json
import os
import stat
from dataclasses import replace
from functools import lru_cache

import pytest

import pr_agent.algo.checkpoint_evaluation_materialize as materialize_module
from pr_agent.algo.checkpoint_evaluation import EvaluationArm, EvaluationArmKind, EvaluationValidationError
from pr_agent.algo.checkpoint_evaluation_snapshot import (
    load_review_snapshot_and_configuration_artifacts,
    load_review_snapshot_artifact,
)
from pr_agent.algo.review_configuration import ReviewConfigurationBundle, materialize_review_configuration
from pr_agent.algo.review_snapshot import CoverageIssue, ReviewEvent, ReviewSnapshot

CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION = (
    materialize_module.CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION
)
materialize_checkpoint_controls = materialize_module.materialize_checkpoint_controls
write_review_snapshot_artifact = materialize_module.write_review_snapshot_artifact


def _hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _arm() -> EvaluationArm:
    return EvaluationArm(
        arm_id="deterministic",
        kind=EvaluationArmKind.DETERMINISTIC,
        configuration_hash=_hash("arm-configuration"),
        prompt_hash=_hash("no-prompt"),
    )


@lru_cache(maxsize=1)
def _review_configuration() -> ReviewConfigurationBundle:
    return materialize_review_configuration("pinned skills", {"AGENTS.md": "pinned repository rules"})


def _fixture() -> tuple[dict, dict[str, ReviewSnapshot]]:
    review_configuration = _review_configuration()
    scenarios = (
        "partial_correct_save",
        "coherent_clean_worktree",
        "temporary_mistake_corrected",
        "staged_checkpoint",
        "stale_candidate_withdrawn",
    )
    events = (
        ReviewEvent.FILE_SAVE,
        ReviewEvent.WORKTREE_IDLE,
        ReviewEvent.FILE_SAVE,
        ReviewEvent.PRE_COMMIT,
        ReviewEvent.WORKTREE_IDLE,
    )
    entries = []
    snapshots: dict[str, ReviewSnapshot] = {}
    for index in range(15):
        case_id = f"checkpoint-{index}"
        scenario = scenarios[index % len(scenarios)]
        parent_id = f"checkpoint-{index - 1}" if index % len(scenarios) in {2, 4} else None
        parent_snapshot_id = snapshots[parent_id].snapshot_id if parent_id else None
        path = f"src/control_{index}.py"
        event = events[index % len(events)]
        snapshots[case_id] = ReviewSnapshot(
            event=event,
            repository_root="/private/cocos-story",
            base_revision=f"{index + 1:040x}",
            base_selector="HEAD",
            changed_paths=(path,),
            diff=(
                f"diff --git a/{path} b/{path}\n"
                f"--- a/{path}\n"
                f"+++ b/{path}\n"
                "@@ -1 +1 @@\n"
                f"-value = {index}\n"
                f"+value = {index + 1}\n"
            ),
            policy_version="snapshot-policy-v1",
            created_at=f"2026-08-30T12:{index:02d}:00+00:00",
            review_configuration_hash=review_configuration.configuration_hash,
            parent_snapshot_id=parent_snapshot_id,
        )
        entries.append({
            "id": case_id,
            "parent_id": parent_id,
            "stage": event.value,
            "scenario": scenario,
            "is_final_pr_head": False,
            "independently_adjudicated": True,
            "adjudication_hash": _hash(f"adjudication-{index}"),
            "is_clean": True,
            "expected_withdrawn_fingerprints": (
                [f"finding-{index}"] if scenario == "stale_candidate_withdrawn" else []
            ),
        })
    return {
        "schema_version": CHECKPOINT_CONTROL_MATERIALIZATION_SCHEMA_VERSION,
        "answer_only": True,
        "entries": entries,
    }, snapshots


def _materialize(tmp_path, spec=None, snapshots=None, review_configurations=None):
    default_spec, default_snapshots = _fixture()
    selected_snapshots = snapshots or default_snapshots
    selected_configurations = (
        {case_id: _review_configuration() for case_id in selected_snapshots}
        if review_configurations is None
        else review_configurations
    )
    return materialize_checkpoint_controls(
        spec or default_spec,
        selected_snapshots,
        tmp_path / "checkpoint-bundle",
        review_configurations=selected_configurations,
        name="cocos-clean-checkpoints",
        policy_hash=_hash("policy"),
        configuration_hash=_hash("configuration"),
        arms=(_arm(),),
    )


def _assert_no_bundle_or_staging(parent) -> None:
    assert not (parent / "checkpoint-bundle").exists()
    assert not [path for path in parent.iterdir() if path.name.startswith(".checkpoint-bundle.")]


def test_materializes_exact_private_snapshots_and_source_free_artifacts(tmp_path, monkeypatch):
    spec, snapshots = _fixture()

    def network_forbidden(*args, **kwargs):
        raise AssertionError("checkpoint materialization must not perform network or model requests")

    monkeypatch.setattr("socket.create_connection", network_forbidden)
    materialized = _materialize(tmp_path, spec, snapshots)

    assert len(materialized.manifest.cases) == 15
    assert len(materialized.truth.truths) == 15
    assert {entry["scenario"] for entry in materialized.checkpoint_control_ledger["entries"]} == {
        "partial_correct_save",
        "coherent_clean_worktree",
        "temporary_mistake_corrected",
        "staged_checkpoint",
        "stale_candidate_withdrawn",
    }
    serialized_public_contracts = json.dumps({
        "manifest": materialized.manifest.to_dict(),
        "truth": materialized.truth.to_dict(),
        "index": materialized.artifact_index_dict(),
        "ledger": materialized.checkpoint_control_ledger_dict(),
    })
    assert "/private/cocos-story" not in serialized_public_contracts
    assert "diff --git" not in serialized_public_contracts
    assert "value =" not in serialized_public_contracts
    assert "pinned skills" not in serialized_public_contracts
    assert "pinned repository rules" not in serialized_public_contracts

    cases_by_id = {case.case_id: case for case in materialized.manifest.cases}
    for artifact in materialized.artifact_index["artifacts"]:
        path = tmp_path / "checkpoint-bundle" / artifact["relative_path"]
        metadata = path.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
        assert _hash_bytes(path.read_bytes()) == artifact["snapshot_artifact_hash"]
        loaded = load_review_snapshot_artifact(path, cases_by_id[artifact["case_id"]])
        assert loaded.to_dict() == snapshots[artifact["case_id"]].to_dict()
        configuration_path = (
            tmp_path / "checkpoint-bundle" / artifact["review_configuration_relative_path"]
        )
        assert _hash_bytes(configuration_path.read_bytes()) == artifact[
            "review_configuration_artifact_hash"
        ]
        assert artifact["review_configuration_hash"] == snapshots[
            artifact["case_id"]
        ].review_configuration_hash
        loaded_pair = load_review_snapshot_and_configuration_artifacts(
            path,
            cases_by_id[artifact["case_id"]],
            review_configuration_artifact_hash=artifact[
                "review_configuration_artifact_hash"
            ],
        )
        assert loaded_pair.snapshot.to_dict() == snapshots[artifact["case_id"]].to_dict()
        assert loaded_pair.review_configuration == _review_configuration()
    for filename in (
        "checkpoint-controls.json",
        "evaluation-manifest.json",
        "evaluation-truth.json",
        "snapshot-index.json",
    ):
        metadata = (tmp_path / "checkpoint-bundle" / filename).stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
    configuration_artifacts = list((tmp_path / "checkpoint-bundle").glob("review-configuration-*.json"))
    assert len(configuration_artifacts) == 1
    assert b"pinned skills" in configuration_artifacts[0].read_bytes()
    assert stat.S_IMODE(configuration_artifacts[0].stat().st_mode) == 0o600
    assert configuration_artifacts[0].stat().st_nlink == 1

    repeated = _materialize(tmp_path, spec, snapshots)
    assert repeated.manifest.manifest_id == materialized.manifest.manifest_id
    assert repeated.artifact_hashes == materialized.artifact_hashes


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def test_writes_one_supplied_snapshot_with_exact_canonical_hash(tmp_path):
    _, snapshots = _fixture()
    path = tmp_path / "snapshot.json"

    artifact_hash = write_review_snapshot_artifact(snapshots["checkpoint-0"], path)

    assert artifact_hash == _hash_bytes(path.read_bytes())
    assert path.read_bytes().endswith(b"\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_nlink == 1
    assert write_review_snapshot_artifact(snapshots["checkpoint-0"], path) == artifact_hash


def test_materialization_requires_one_exact_configuration_per_snapshot(tmp_path):
    spec, snapshots = _fixture()
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    configurations = {case_id: _review_configuration() for case_id in snapshots}
    configurations.pop("checkpoint-14")

    with pytest.raises(EvaluationValidationError, match="review configurations do not match"):
        _materialize(output, spec, snapshots, configurations)
    _assert_no_bundle_or_staging(output)

    configurations["checkpoint-14"] = materialize_review_configuration(
        "different pinned skills",
        {"AGENTS.md": "pinned repository rules"},
    )
    with pytest.raises(EvaluationValidationError, match="not bound to its review configuration"):
        _materialize(output, spec, snapshots, configurations)
    _assert_no_bundle_or_staging(output)


def test_accepts_the_generated_answer_only_ledger_as_a_replay_lock(tmp_path):
    spec, snapshots = _fixture()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir(mode=0o700)
    second_root.mkdir(mode=0o700)
    first = _materialize(first_root, spec, snapshots)

    replay = _materialize(second_root, first.checkpoint_control_ledger, snapshots)

    assert replay.manifest.corpus_hash == first.manifest.corpus_hash
    assert replay.checkpoint_control_ledger == first.checkpoint_control_ledger


def test_returned_contracts_are_deep_frozen_but_export_as_json_native_values(tmp_path):
    materialized = _materialize(tmp_path)

    with pytest.raises(TypeError):
        materialized.checkpoint_control_ledger["entries"][0]["is_clean"] = False
    with pytest.raises(AttributeError):
        materialized.artifact_index["artifacts"].append({"source": "leak"})

    ledger = materialized.checkpoint_control_ledger_dict()
    index = materialized.artifact_index_dict()
    assert json.loads(json.dumps(ledger)) == ledger
    assert json.loads(json.dumps(index)) == index
    ledger["entries"][0]["is_clean"] = False
    assert materialized.checkpoint_control_ledger["entries"][0]["is_clean"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda spec: spec.update({"source": "secret"}), "unknown fields"),
        (lambda spec: spec["entries"][0].update({"diff": "secret"}), "unknown fields"),
        (lambda spec: spec["entries"].pop(), "15 to 20"),
        (lambda spec: spec["entries"][0].update({"is_final_pr_head": True}), "final clean PR heads"),
        (lambda spec: spec["entries"][2].update({"parent_id": "missing"}), "unknown parent_id"),
        (lambda spec: spec["entries"][1].update({"id": "checkpoint-0"}), "ids must be unique"),
        (
            lambda spec: spec["entries"][1].update({
                "adjudication_hash": spec["entries"][0]["adjudication_hash"]
            }),
            "adjudication hashes must be unique",
        ),
        (
            lambda spec: spec["entries"][0].update({"independently_adjudicated": False}),
            "independent, affirmative clean adjudication",
        ),
        (lambda spec: spec["entries"][0].update({"adjudication_hash": "weak"}), "sha256 identity"),
        (
            lambda spec: [entry.update({"scenario": "partial_correct_save"}) for entry in spec["entries"]],
            "do not cover required intermediate scenarios",
        ),
        (
            lambda spec: spec["entries"][4].update({"expected_withdrawn_fingerprints": []}),
            "require a withdrawn finding fingerprint",
        ),
    ],
)
def test_materialization_spec_fails_closed(tmp_path, mutate, message):
    spec, snapshots = _fixture()
    mutate(spec)
    tmp_path.chmod(0o700)

    with pytest.raises(EvaluationValidationError, match=message):
        _materialize(tmp_path, spec, snapshots)


def test_snapshot_validation_rejects_coverage_mismatch_duplicates_and_bad_lineage(tmp_path):
    spec, snapshots = _fixture()
    with_coverage = snapshots["checkpoint-0"]
    snapshots["checkpoint-0"] = ReviewSnapshot(
        **{
            key: value
            for key, value in with_coverage.to_dict().items()
            if key not in {"snapshot_id", "event", "coverage_issues"}
        },
        event=with_coverage.event,
        coverage_issues=(CoverageIssue(reason="capture_budget"),),
    )
    tmp_path.chmod(0o700)
    with pytest.raises(EvaluationValidationError, match="cannot have unavailable snapshot coverage"):
        _materialize(tmp_path, spec, snapshots)

    spec, snapshots = _fixture()
    snapshots["checkpoint-1"] = snapshots["checkpoint-0"]
    with pytest.raises(EvaluationValidationError, match="stage does not match"):
        _materialize(tmp_path, spec, snapshots)

    spec, snapshots = _fixture()
    child = snapshots["checkpoint-2"]
    snapshots["checkpoint-2"] = ReviewSnapshot(
        **{
            key: value
            for key, value in child.to_dict().items()
            if key not in {"snapshot_id", "event", "parent_snapshot_id"}
        },
        event=child.event,
        parent_snapshot_id="sha256:" + "0" * 64,
    )
    with pytest.raises(EvaluationValidationError, match="not bound to its declared parent"):
        _materialize(tmp_path, spec, snapshots)

    spec, snapshots = _fixture()
    snapshots["checkpoint-0"] = replace(
        snapshots["checkpoint-0"],
        parent_snapshot_id=_hash("undeclared-parent"),
    )
    with pytest.raises(EvaluationValidationError, match="root cannot name a parent"):
        _materialize(tmp_path, spec, snapshots)


def test_validates_every_snapshot_before_persisting_source_bearing_files(tmp_path):
    spec, snapshots = _fixture()
    invalid = snapshots["checkpoint-9"]
    snapshots["checkpoint-9"] = ReviewSnapshot(
        **{
            key: value
            for key, value in invalid.to_dict().items()
            if key not in {"snapshot_id", "event", "coverage_issues"}
        },
        event=invalid.event,
        coverage_issues=(CoverageIssue(reason="capture_budget"),),
    )
    output = tmp_path / "output"
    output.mkdir(mode=0o700)

    with pytest.raises(EvaluationValidationError, match="cannot have unavailable snapshot coverage"):
        _materialize(output, spec, snapshots)

    assert list(output.iterdir()) == []


def test_runtime_incompatible_configuration_is_rejected_before_persisting(tmp_path):
    spec, snapshots = _fixture()
    incompatible_configuration = replace(
        _review_configuration(),
        runtime_version="incompatible-runtime",
    )
    incompatible_snapshots = {
        case_id: replace(
            snapshot,
            review_configuration_hash=incompatible_configuration.configuration_hash,
        )
        for case_id, snapshot in snapshots.items()
    }
    configurations = {
        case_id: incompatible_configuration
        for case_id in incompatible_snapshots
    }
    output = tmp_path / "runtime-mismatch"
    output.mkdir(mode=0o700)

    with pytest.raises(EvaluationValidationError, match="review configuration runtime mismatch"):
        _materialize(
            output,
            spec,
            incompatible_snapshots,
            configurations,
        )

    _assert_no_bundle_or_staging(output)


def test_shared_configuration_runtime_is_validated_once(tmp_path, monkeypatch):
    calls = 0
    real_require_compatible_runtime = ReviewConfigurationBundle.require_compatible_runtime

    def count_runtime_validation(review_configuration):
        nonlocal calls
        calls += 1
        real_require_compatible_runtime(review_configuration)

    monkeypatch.setattr(
        ReviewConfigurationBundle,
        "require_compatible_runtime",
        count_runtime_validation,
    )

    _materialize(tmp_path)

    assert calls == 1


def test_private_write_rejects_symlink_hard_link_and_permissive_output_root(tmp_path):
    spec, snapshots = _fixture()
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    (output / "checkpoint-bundle").symlink_to(external, target_is_directory=True)
    with pytest.raises(EvaluationValidationError, match="real directory, not a symlink"):
        _materialize(output, spec, snapshots)
    assert list(external.iterdir()) == []

    (output / "checkpoint-bundle").unlink()
    _materialize(output, spec, snapshots)
    artifact = output / "checkpoint-bundle" / "checkpoint-controls.json"
    alias = tmp_path / "hard-link-alias.json"
    os.link(artifact, alias)
    with pytest.raises(EvaluationValidationError, match="private and single-link"):
        _materialize(output, spec, snapshots)
    assert alias.read_bytes() == artifact.read_bytes()

    permissive = tmp_path / "permissive"
    permissive.mkdir(mode=0o755)
    with pytest.raises(EvaluationValidationError, match="parent must be owner-only"):
        _materialize(permissive, spec, snapshots)


def test_file_spec_rejects_duplicate_json_keys_and_symlink_parent(tmp_path):
    tmp_path.chmod(0o700)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    _, snapshots = _fixture()
    with pytest.raises(EvaluationValidationError, match="duplicate JSON key"):
        _materialize(tmp_path, duplicate, snapshots)

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    spec, _ = _fixture()
    (real / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvaluationValidationError, match="parent components"):
        _materialize(tmp_path, alias / "spec.json", snapshots)


def test_partial_single_file_write_never_exposes_final_bytes_and_retry_succeeds(tmp_path, monkeypatch):
    _, snapshots = _fixture()
    output = tmp_path / "snapshot.json"
    real_write = materialize_module.os.write
    call_count = 0

    def partial_then_fail(descriptor, payload):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_write(descriptor, payload[:3])
        raise OSError("injected partial-write failure")

    monkeypatch.setattr(materialize_module.os, "write", partial_then_fail)
    with pytest.raises(OSError, match="partial-write"):
        write_review_snapshot_artifact(snapshots["checkpoint-0"], output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(materialize_module.os, "write", real_write)
    artifact_hash = write_review_snapshot_artifact(snapshots["checkpoint-0"], output)
    assert artifact_hash == _hash_bytes(output.read_bytes())


@pytest.mark.parametrize("boundary", range(1, 21))
def test_bundle_failure_at_each_completed_file_boundary_is_invisible_and_retryable(
    tmp_path,
    monkeypatch,
    boundary,
):
    output = tmp_path / f"boundary-{boundary}"
    output.mkdir(mode=0o700)
    real_write_staged_file = materialize_module._write_staged_file
    completed = 0

    def fail_after_boundary(directory_fd, filename, payload):
        nonlocal completed
        identity = real_write_staged_file(directory_fd, filename, payload)
        completed += 1
        if completed == boundary:
            raise RuntimeError(f"injected file-boundary failure {boundary}")
        return identity

    monkeypatch.setattr(materialize_module, "_write_staged_file", fail_after_boundary)
    with pytest.raises(RuntimeError, match="file-boundary"):
        _materialize(output)

    assert completed == boundary
    _assert_no_bundle_or_staging(output)

    monkeypatch.setattr(materialize_module, "_write_staged_file", real_write_staged_file)
    _materialize(output)
    assert len(list((output / "checkpoint-bundle").iterdir())) == 20


@pytest.mark.parametrize(
    ("fsync_call", "fail_after_fsync"),
    [
        (1, False),
        (1, True),
        (21, False),
        (21, True),
        (22, False),
        (22, True),
        (23, False),
        (23, True),
    ],
)
def test_bundle_fsync_faults_before_and_after_flush_leave_no_visible_bundle(
    tmp_path,
    monkeypatch,
    fsync_call,
    fail_after_fsync,
):
    output = tmp_path / f"fsync-{fsync_call}-{fail_after_fsync}"
    output.mkdir(mode=0o700)
    real_fsync = materialize_module.os.fsync
    calls = 0

    def fail_at_selected_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == fsync_call:
            if fail_after_fsync:
                real_fsync(descriptor)
            raise OSError(f"injected fsync failure {fsync_call}")
        return real_fsync(descriptor)

    monkeypatch.setattr(materialize_module.os, "fsync", fail_at_selected_fsync)
    with pytest.raises(OSError, match="injected fsync"):
        _materialize(output)

    _assert_no_bundle_or_staging(output)


def test_pre_rename_failure_cleans_staging_and_retry_succeeds(tmp_path, monkeypatch):
    output = tmp_path / "pre-rename"
    output.mkdir(mode=0o700)
    real_rename = materialize_module._rename_no_replace

    def fail_before_rename(*args, **kwargs):
        raise OSError("injected pre-rename failure")

    monkeypatch.setattr(materialize_module, "_rename_no_replace", fail_before_rename)
    with pytest.raises(OSError, match="pre-rename"):
        _materialize(output)
    _assert_no_bundle_or_staging(output)

    monkeypatch.setattr(materialize_module, "_rename_no_replace", real_rename)
    _materialize(output)
    assert (output / "checkpoint-bundle" / "snapshot-index.json").is_file()


def test_atomic_rename_collision_is_rejected_without_accepting_partial_bundle(tmp_path, monkeypatch):
    output = tmp_path / "collision"
    output.mkdir(mode=0o700)
    real_rename = materialize_module._rename_no_replace

    def collide_then_rename(directory_fd, source, destination):
        os.mkdir(destination, mode=0o700, dir_fd=directory_fd)
        real_rename(directory_fd, source, destination)

    monkeypatch.setattr(materialize_module, "_rename_no_replace", collide_then_rename)
    with pytest.raises(EvaluationValidationError, match="appeared during atomic publication"):
        _materialize(output)

    collision = output / "checkpoint-bundle"
    assert collision.is_dir()
    assert list(collision.iterdir()) == []
    assert not [path for path in output.iterdir() if path.name.startswith(".checkpoint-bundle.")]
    with pytest.raises(EvaluationValidationError, match="incomplete"):
        _materialize(output)

    collision.rmdir()
    monkeypatch.setattr(materialize_module, "_rename_no_replace", real_rename)
    _materialize(output)
    assert (collision / "evaluation-manifest.json").is_file()


def test_hard_link_race_after_staging_is_rejected_and_final_bundle_is_removed(tmp_path, monkeypatch):
    output = tmp_path / "hard-link-race"
    output.mkdir(mode=0o700)
    alias = tmp_path / "escaped-snapshot.json"
    real_rename = materialize_module._rename_no_replace

    def link_then_rename(directory_fd, source, destination):
        staging_fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            snapshot_name = next(
                name for name in os.listdir(staging_fd) if name.startswith("review-snapshot-")
            )
            os.link(snapshot_name, alias, src_dir_fd=staging_fd)
        finally:
            os.close(staging_fd)
        real_rename(directory_fd, source, destination)

    monkeypatch.setattr(materialize_module, "_rename_no_replace", link_then_rename)
    with pytest.raises(EvaluationValidationError, match="private and single-link"):
        _materialize(output)

    _assert_no_bundle_or_staging(output)
    assert alias.is_file()


def test_same_name_replacement_during_verification_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "replacement-race"
    output.mkdir(mode=0o700)
    real_read = materialize_module._read_private_file_at
    replaced = False

    def replace_after_first_read(directory_fd, filename, payload, **kwargs):
        nonlocal replaced
        identity = real_read(directory_fd, filename, payload, **kwargs)
        if not replaced and filename.startswith("review-snapshot-"):
            replacement_name = f"replacement-{filename}"
            descriptor = os.open(
                replacement_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(descriptor, b"corrupted")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(replacement_name, filename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            replaced = True
        return identity

    monkeypatch.setattr(materialize_module, "_read_private_file_at", replace_after_first_read)

    with pytest.raises(EvaluationValidationError, match="identity changed while it was verified"):
        _materialize(output)

    _assert_no_bundle_or_staging(output)
