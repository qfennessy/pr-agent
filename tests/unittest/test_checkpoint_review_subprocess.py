import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pr_agent.algo import checkpoint_review_subprocess as review_subprocess
from pr_agent.algo.review_execution_context import review_execution_is_isolated
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewSnapshot


def _snapshot() -> ReviewSnapshot:
    return ReviewSnapshot(
        event=ReviewEvent.PRE_COMMIT,
        repository_root="/private/checkpoint/repository",
        base_revision="a" * 40,
        base_selector="main",
        changed_paths=("example.py",),
        diff=(
            "diff --git a/example.py b/example.py\n"
            "--- a/example.py\n"
            "+++ b/example.py\n"
            "@@ -1 +1 @@\n"
            "-old = True\n"
            "+new = True\n"
        ),
        policy_version="policy-v1",
        created_at="2026-09-03T12:00:00Z",
    )


def _request_bytes(snapshot: ReviewSnapshot, *, allow_model_execution: bool = True) -> bytes:
    return json.dumps({
        "schema_version": review_subprocess.CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION,
        "allow_model_execution": allow_model_execution,
        "snapshot": snapshot.to_dict(),
    }).encode("utf-8")


def _completed_outcome(snapshot: ReviewSnapshot) -> review_subprocess.CheckpointReviewSubprocessOutcome:
    return review_subprocess.CheckpointReviewSubprocessOutcome(
        state=review_subprocess.CheckpointReviewSubprocessState.COMPLETED,
        snapshot_id=snapshot.snapshot_id,
        review={"review": {"key_issues_to_review": []}},
        latency_seconds=0.25,
    )


class _FakeStdin:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, value):
        self.written += value

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _FakeStdout:
    def __init__(self, value: bytes | list[bytes]):
        self.values = list(value) if isinstance(value, list) else [value]

    async def read(self, limit: int):
        if not self.values:
            return b""
        value = self.values.pop(0)
        if len(value) > limit:
            self.values.insert(0, value[limit:])
        return value[:limit]


class _FakeProcess:
    def __init__(self, output: bytes):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(output)
        self.returncode = None
        self.terminated = False
        self.killed = False

    async def wait(self):
        self.returncode = 0 if not self.killed else -9
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class _TimeoutProcess(_FakeProcess):
    async def wait(self):
        if self.returncode is None:
            await asyncio.Event().wait()
        return self.returncode


class _AbnormalExitProcess(_FakeProcess):
    async def wait(self):
        self.returncode = 3
        return self.returncode


@pytest.mark.asyncio
async def test_parent_refuses_without_spawning(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    outcome = await review_subprocess.run_checkpoint_review_subprocess(_snapshot())

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.REFUSED
    assert outcome.failure_reason_code == "model_execution_not_authorized"
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_rejects_non_canonical_snapshot_without_spawning(monkeypatch):
    snapshot = replace(_snapshot(), changed_paths=("../example.py",))
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        snapshot,
        allow_model_execution=True,
    )

    assert outcome.failure_reason_code == "invalid_snapshot"
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_uses_current_interpreter_without_a_shell(monkeypatch):
    snapshot = _snapshot()
    encoded_outcome = review_subprocess._encode_worker_outcome(_completed_outcome(snapshot))
    process = _FakeProcess(encoded_outcome)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        snapshot,
        allow_model_execution=True,
    )

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED
    spawn.assert_awaited_once_with(
        sys.executable,
        "-m",
        "pr_agent.algo.checkpoint_review_subprocess",
        "--worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin.closed is True
    assert json.loads(process.stdin.written)["allow_model_execution"] is True


@pytest.mark.asyncio
async def test_parent_drains_chunked_worker_output_before_waiting(monkeypatch):
    snapshot = _snapshot()
    encoded_outcome = review_subprocess._encode_worker_outcome(_completed_outcome(snapshot))
    split_at = len(encoded_outcome) // 2
    process = _FakeProcess([encoded_outcome[:split_at], encoded_outcome[split_at:]])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        snapshot,
        allow_model_execution=True,
    )

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED
    assert outcome.snapshot_id == snapshot.snapshot_id


@pytest.mark.asyncio
async def test_parent_rejects_oversized_worker_output(monkeypatch):
    process = _FakeProcess(b"x" * (review_subprocess.MAX_REVIEW_SUBPROCESS_OUTPUT_BYTES + 1))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        _snapshot(),
        allow_model_execution=True,
    )

    assert outcome.failure_reason_code == "worker_output_too_large"
    assert process.terminated is True


@pytest.mark.asyncio
async def test_parent_maps_malformed_worker_output_to_bounded_failure(monkeypatch):
    process = _FakeProcess(b'{"secret_source":"contents"}')
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        _snapshot(),
        allow_model_execution=True,
    )

    assert outcome.to_dict()["failure_reason_code"] == "worker_protocol_failed"
    assert "secret_source" not in json.dumps(outcome.to_dict())


@pytest.mark.asyncio
async def test_parent_terminates_worker_on_timeout(monkeypatch):
    process = _TimeoutProcess(b"")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        _snapshot(),
        allow_model_execution=True,
        timeout_seconds=0.001,
    )

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.TIMEOUT
    assert outcome.failure_reason_code == "worker_timeout"
    assert process.terminated is True


@pytest.mark.asyncio
async def test_parent_maps_abnormal_worker_exit_to_bounded_failure(monkeypatch):
    process = _AbnormalExitProcess(b'provider exception included source text')
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        _snapshot(),
        allow_model_execution=True,
    )

    assert outcome.failure_reason_code == "worker_process_failed"
    assert "source text" not in json.dumps(outcome.to_dict())


@pytest.mark.asyncio
async def test_worker_refuses_permission_without_calling_executor():
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        _request_bytes(_snapshot(), allow_model_execution=False),
        executor=executor,
    )

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.REFUSED
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_rejects_unknown_request_fields_before_execution():
    payload = json.loads(_request_bytes(_snapshot()))
    payload["unexpected"] = "source text"
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        json.dumps(payload).encode("utf-8"),
        executor=executor,
    )

    assert outcome.failure_reason_code == "invalid_request"
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_validates_snapshot_hash_before_calling_executor():
    snapshot = _snapshot()
    payload = json.loads(_request_bytes(snapshot))
    payload["snapshot"]["diff"] += "+private source\n"
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        json.dumps(payload).encode("utf-8"),
        executor=executor,
    )

    assert outcome.failure_reason_code == "invalid_request"
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_rejects_non_canonical_snapshot_before_execution():
    snapshot = replace(_snapshot(), changed_paths=("../example.py",))
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        _request_bytes(snapshot),
        executor=executor,
    )

    assert outcome.failure_reason_code == "invalid_request"
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_rejects_answer_only_snapshot_data_before_execution():
    snapshot = _snapshot()
    payload = json.loads(_request_bytes(snapshot))
    payload["snapshot"]["deterministic_results"] = [{"ground_truth": "private"}]
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        json.dumps(payload).encode("utf-8"),
        executor=executor,
    )

    assert outcome.failure_reason_code == "invalid_request"
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_does_not_leak_execution_exception_text():
    async def fail(_snapshot):
        raise RuntimeError("source line and credential")

    outcome = await review_subprocess._handle_worker_request(
        _request_bytes(_snapshot()),
        executor=fail,
    )

    encoded = review_subprocess._encode_worker_outcome(outcome)
    assert outcome.failure_reason_code == "review_execution_failed"
    assert b"source line" not in encoded
    assert b"credential" not in encoded


@pytest.mark.asyncio
async def test_worker_executes_valid_request_and_binds_snapshot():
    snapshot = _snapshot()
    executor = AsyncMock(return_value=_completed_outcome(snapshot))

    outcome = await review_subprocess._handle_worker_request(
        _request_bytes(snapshot),
        executor=executor,
    )

    assert outcome.snapshot_id == snapshot.snapshot_id
    executor.assert_awaited_once()
    assert executor.await_args.args[0].snapshot_id == snapshot.snapshot_id


@pytest.mark.asyncio
async def test_execution_constructs_fresh_reviewer_inside_isolation_and_closes_sinks(monkeypatch):
    snapshot = replace(_snapshot(), review_configuration_hash="sha256:" + "b" * 64)
    configured = {}
    drain = AsyncMock()
    monkeypatch.setattr(
        review_subprocess,
        "_current_review_configuration_hash",
        lambda: snapshot.review_configuration_hash,
    )

    class FakeSettings:
        def set(self, key, value):
            configured[key] = value

        def get(self, key, default=None):
            return configured.get(key, default)

    class FakeReviewer:
        def __init__(self, pr_url):
            assert pr_url == "checkpoint-review-subprocess"
            assert review_execution_is_isolated() is True
            assert configured["plain_diff.disable_working_tree_enrichment"] is True
            assert configured["plain_diff.output_path"] is None
            assert configured["plain_diff.json_output_path"] is None
            assert configured["config.propagate_tool_errors"] is True

        async def _run_structured_no_publish_once(self):
            assert review_execution_is_isolated() is True
            return SimpleNamespace(
                review={"review": {"key_issues_to_review": []}},
                run_details=SimpleNamespace(
                    model_used="model",
                    review_profile="bugs_only",
                    fallback_used=False,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    num_ai_calls=1,
                    total_cost_usd=Decimal("0.01"),
                    known_cost_call_count=1,
                    model_costs_usd={"model": Decimal("0.01")},
                    duration_seconds=0.2,
                ),
            )

    monkeypatch.setattr("pr_agent.config_loader.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.PRReviewer", FakeReviewer)
    monkeypatch.setattr("pr_agent.algo.ai_handlers.litellm_helpers.drain_litellm_callbacks", drain)

    outcome = await review_subprocess._execute_review(snapshot)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED
    assert outcome.run_details["total_tokens"] == 15
    assert configured["config.publish_output"] is False
    assert configured["plain_diff.suppress_stdout"] is True
    assert configured["plain_diff.repo_context_files"] == {}
    assert "caller-supplied context" in configured["pr_reviewer.extra_instructions"]
    drain.assert_awaited_once_with(timeout=review_subprocess._CALLBACK_DRAIN_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_execution_passes_snapshot_intent_and_deterministic_evidence_to_reviewer(monkeypatch):
    snapshot = replace(
        _snapshot(),
        task_intent="focus on concurrency",
        deterministic_results=({"check": "lint", "status": "failed"},),
        review_configuration_hash="sha256:" + "b" * 64,
    )
    configured = {"pr_reviewer.extra_instructions": "persistent rule"}
    monkeypatch.setattr(
        review_subprocess,
        "_current_review_configuration_hash",
        lambda: snapshot.review_configuration_hash,
    )

    class FakeSettings:
        def set(self, key, value):
            configured[key] = value

        def get(self, key, default=None):
            return configured.get(key, default)

    class FakeReviewer:
        def __init__(self, _pr_url):
            instructions = configured["pr_reviewer.extra_instructions"]
            assert instructions.startswith("persistent rule\n\n")
            assert "focus on concurrency" in instructions
            assert '"check": "lint"' in instructions
            assert '"status": "failed"' in instructions

        async def _run_structured_no_publish_once(self):
            return SimpleNamespace(review=None, run_details=None)

    monkeypatch.setattr("pr_agent.config_loader.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.PRReviewer", FakeReviewer)
    monkeypatch.setattr(
        "pr_agent.algo.ai_handlers.litellm_helpers.drain_litellm_callbacks",
        AsyncMock(),
    )

    outcome = await review_subprocess._execute_review(snapshot)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED


@pytest.mark.asyncio
async def test_execution_refuses_unverified_review_configuration():
    outcome = await review_subprocess._execute_review(_snapshot())

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.FAILED
    assert outcome.failure_reason_code == "review_configuration_unverified"


@pytest.mark.asyncio
async def test_execution_refuses_mismatched_review_configuration(monkeypatch):
    snapshot = replace(_snapshot(), review_configuration_hash="sha256:" + "a" * 64)
    monkeypatch.setattr(
        review_subprocess,
        "_current_review_configuration_hash",
        lambda: "sha256:" + "b" * 64,
    )

    outcome = await review_subprocess._execute_review(snapshot)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.FAILED
    assert outcome.failure_reason_code == "review_configuration_mismatch"
