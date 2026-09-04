import asyncio
import builtins
import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pr_agent.algo import checkpoint_review_subprocess as review_subprocess
from pr_agent.algo.review_configuration import (
    materialize_review_configuration,
    snapshot_review_configuration_hash,
)
from pr_agent.algo.review_execution_context import get_review_prompt_date, review_execution_is_isolated
from pr_agent.algo.review_snapshot import ReviewEvent, ReviewSnapshot
from pr_agent.algo.review_specialists import get_specialist_snapshot_context
from pr_agent.algo.run_details import RunDetails
from pr_agent.algo.skills_loader import get_skills_context


def _configuration(*, skills_context=None, repo_context_files=None, repo_context_max_lines=None):
    return materialize_review_configuration(
        skills_context,
        repo_context_files or {},
        repo_context_max_lines=repo_context_max_lines,
    )


def _snapshot(*, review_configuration=None) -> ReviewSnapshot:
    configuration = review_configuration or _configuration()
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
        review_configuration_hash=configuration.configuration_hash,
    )


def _request_bytes(
    snapshot: ReviewSnapshot,
    *,
    review_configuration=None,
    allow_model_execution: bool = True,
) -> bytes:
    configuration = review_configuration or _configuration()
    return json.dumps({
        "schema_version": review_subprocess.CHECKPOINT_REVIEW_SUBPROCESS_SCHEMA_VERSION,
        "allow_model_execution": allow_model_execution,
        "snapshot": snapshot.to_dict(),
        "review_configuration": configuration.to_dict(),
        "evaluation_stage_plan": [],
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


class _ExitBeforeTerminateProcess(_FakeProcess):
    def __init__(self):
        super().__init__(b"")
        self.waited = False

    def terminate(self):
        raise ProcessLookupError

    async def wait(self):
        self.waited = True
        self.returncode = 0
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
    monkeypatch.setenv("PYTHONPATH", "/attacker-controlled-checkout")
    monkeypatch.setenv("PYTHONHOME", "/attacker-controlled-runtime")
    monkeypatch.setenv("PR_AGENT_CONFIG__MODEL", "attacker-model")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=secret")
    monkeypatch.setenv("OPENAI_API_BASE", "https://attacker.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-credential")
    monkeypatch.setenv("HTTPS_PROXY", "https://attacker.invalid")
    monkeypatch.setenv("AWS_USE_IMDS", "true")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        snapshot,
        allow_model_execution=True,
    )

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED
    spawn.assert_awaited_once()
    args = spawn.await_args.args
    kwargs = spawn.await_args.kwargs
    assert args == (
        sys.executable,
        "-I",
        "-c",
        review_subprocess._WORKER_BOOTSTRAP,
        review_subprocess._TRUSTED_PACKAGE_ROOT,
        "--worker",
    )
    assert kwargs["stdin"] is asyncio.subprocess.PIPE
    assert kwargs["stdout"] is asyncio.subprocess.PIPE
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["cwd"] == review_subprocess._TRUSTED_PACKAGE_ROOT
    assert "PYTHONPATH" not in kwargs["env"]
    assert "PYTHONHOME" not in kwargs["env"]
    assert "PR_AGENT_CONFIG__MODEL" not in kwargs["env"]
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in kwargs["env"]
    assert "OPENAI_API_BASE" not in kwargs["env"]
    assert "HTTPS_PROXY" not in kwargs["env"]
    assert "AWS_USE_IMDS" not in kwargs["env"]
    assert kwargs["env"]["OPENAI_API_KEY"] == "provider-credential"
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert process.stdin.closed is True
    request = json.loads(process.stdin.written)
    assert request["allow_model_execution"] is True
    assert request["review_configuration"]["configuration_hash"] == snapshot.review_configuration_hash


@pytest.mark.asyncio
async def test_parent_rejects_legacy_cli_snapshot_without_an_immutable_bundle(monkeypatch):
    snapshot = replace(
        _snapshot(),
        review_configuration_hash=snapshot_review_configuration_hash(get_skills_context(), {}),
    )
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    outcome = await review_subprocess.run_checkpoint_review_subprocess(
        snapshot,
        allow_model_execution=True,
    )

    assert outcome.failure_reason_code == "review_configuration_mismatch"
    spawn.assert_not_awaited()


def test_worker_environment_prefers_effective_settings_credential(monkeypatch):
    class FakeSettings:
        @staticmethod
        def get(key, default=None):
            return "settings-credential" if key == "openai.key" else default

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-credential")
    monkeypatch.setattr("pr_agent.config_loader.get_settings", lambda: FakeSettings())

    assert review_subprocess._worker_environment()["OPENAI_API_KEY"] == "settings-credential"


def test_isolated_worker_bootstrap_ignores_untrusted_import_paths(tmp_path):
    attacker_package = tmp_path / "pr_agent"
    attacker_package.mkdir()
    (attacker_package / "__init__.py").write_text(
        'raise RuntimeError("loaded attacker-controlled package")\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            review_subprocess._WORKER_BOOTSTRAP,
            review_subprocess._TRUSTED_PACKAGE_ROOT,
            "--worker",
        ],
        input=_request_bytes(_snapshot(), allow_model_execution=False),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
    )

    outcome = json.loads(completed.stdout)
    assert outcome["state"] == review_subprocess.CheckpointReviewSubprocessState.REFUSED.value
    assert outcome["failure_reason_code"] == "model_execution_not_authorized"


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
async def test_stop_process_reaps_worker_that_exits_before_terminate():
    process = _ExitBeforeTerminateProcess()

    await review_subprocess._stop_process(process)

    assert process.waited is True
    assert process.returncode == 0


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
async def test_worker_rejects_oversized_request_before_decoding():
    executor = AsyncMock()

    outcome = await review_subprocess._handle_worker_request(
        b"x" * (review_subprocess.MAX_REVIEW_SUBPROCESS_REQUEST_BYTES + 1),
        executor=executor,
    )

    assert outcome.failure_reason_code == "request_too_large"
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
    async def fail(_snapshot, _review_configuration):
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
    configuration = _configuration()
    snapshot = _snapshot(review_configuration=configuration)
    executor = AsyncMock(return_value=_completed_outcome(snapshot))

    outcome = await review_subprocess._handle_worker_request(
        _request_bytes(snapshot, review_configuration=configuration),
        executor=executor,
    )

    assert outcome.snapshot_id == snapshot.snapshot_id
    executor.assert_awaited_once()
    assert executor.await_args.args[0].snapshot_id == snapshot.snapshot_id
    assert executor.await_args.args[1].configuration_hash == configuration.configuration_hash


@pytest.mark.asyncio
async def test_execution_constructs_fresh_reviewer_inside_isolation_and_closes_sinks(monkeypatch):
    configuration = _configuration(
        skills_context="pinned skill content",
        repo_context_files={"CLAUDE.md": "pinned repository context"},
        repo_context_max_lines=7,
    )
    snapshot = _snapshot(review_configuration=configuration)
    configured = {"anthropic.key": "unrelated-secret", "openrouter.key": "unrelated-secret"}
    drain = AsyncMock()

    class FakeSettings:
        def unset(self, key, **_kwargs):
            prefix = f"{key}."
            for configured_key in tuple(configured):
                if configured_key.casefold().startswith(prefix.casefold()):
                    configured.pop(configured_key)

        def set(self, key, value, **_kwargs):
            configured[key] = value

        def get(self, key, default=None):
            return configured.get(key, default)

    class FakeReviewer:
        def __init__(self, pr_url):
            assert pr_url == "checkpoint-review-subprocess"
            assert review_execution_is_isolated() is True
            assert get_review_prompt_date() == ""
            assert configured["plain_diff.disable_working_tree_enrichment"] is True
            assert configured["plain_diff.output_path"] is None
            assert configured["plain_diff.json_output_path"] is None
            assert configured["plain_diff.repo_context_files"] == {
                "CLAUDE.md": "pinned repository context"
            }
            assert configured["config.repo_context_files"] == ["CLAUDE.md"]
            assert configured["config.repo_context_max_lines"] == 7
            assert configured["config.propagate_tool_errors"] is True
            assert configured["config.use_repo_settings_file"] is False
            assert configured["config.use_global_settings_file"] is False
            assert configured["config.add_user_to_requests"] is False
            assert configured["config.output_run_cost"] is True
            assert configured["config.output_run_details"] is False
            assert configured["litellm.enable_callbacks"] is False
            assert configured["litellm.extra_headers"] == {}
            assert configured["otel.is_enabled"] is False
            assert "anthropic.key" not in configured
            assert "openrouter.key" not in configured
            assert get_skills_context() == "pinned skill content"
            specialist_context = get_specialist_snapshot_context()
            assert specialist_context is not None
            assert specialist_context.snapshot is snapshot
            assert specialist_context.current_snapshot_id() == snapshot.snapshot_id

        async def _run_structured_no_publish_once(self):
            assert review_execution_is_isolated() is True
            return SimpleNamespace(
                review={"review": {"key_issues_to_review": []}},
                run_details=RunDetails(
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
                    start_time=0.0,
                    finish_time=0.2,
                ),
            )

    monkeypatch.setattr("pr_agent.config_loader.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("pr_agent.tools.pr_reviewer.PRReviewer", FakeReviewer)
    monkeypatch.setattr("pr_agent.algo.ai_handlers.litellm_helpers.drain_litellm_callbacks", drain)

    outcome = await review_subprocess._execute_review(snapshot, configuration)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED
    assert outcome.run_details["total_tokens"] == 15
    assert configured["config.publish_output"] is False
    assert configured["plain_diff.suppress_stdout"] is True
    assert configured["plain_diff.repo_context_files"] == {
        "CLAUDE.md": "pinned repository context"
    }
    assert "caller-supplied context" in configured["pr_reviewer.extra_instructions"]
    assert get_review_prompt_date() != ""
    drain.assert_awaited_once_with(timeout=review_subprocess._CALLBACK_DRAIN_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_execution_passes_snapshot_intent_and_deterministic_evidence_to_reviewer(monkeypatch):
    configuration = _configuration()
    snapshot = replace(
        _snapshot(review_configuration=configuration),
        task_intent="focus on concurrency",
        deterministic_results=({"check": "lint", "status": "failed"},),
    )
    configured = {"pr_reviewer.extra_instructions": "persistent rule"}

    class FakeSettings:
        def unset(self, key, **_kwargs):
            prefix = f"{key}."
            for configured_key in tuple(configured):
                if configured_key.casefold().startswith(prefix.casefold()):
                    configured.pop(configured_key)

        def set(self, key, value, **_kwargs):
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

    outcome = await review_subprocess._execute_review(snapshot, configuration)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED


@pytest.mark.asyncio
async def test_execution_refuses_unverified_review_configuration():
    configuration = _configuration()
    snapshot = replace(_snapshot(review_configuration=configuration), review_configuration_hash=None)
    outcome = await review_subprocess._execute_review(snapshot, configuration)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.FAILED
    assert outcome.failure_reason_code == "review_configuration_unverified"


@pytest.mark.asyncio
async def test_execution_refuses_mismatched_review_configuration():
    configuration = _configuration()
    snapshot = replace(_snapshot(review_configuration=configuration), review_configuration_hash="sha256:" + "a" * 64)

    outcome = await review_subprocess._execute_review(snapshot, configuration)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.FAILED
    assert outcome.failure_reason_code == "review_configuration_mismatch"


def test_configuration_hash_does_not_import_cli_logging(monkeypatch):
    real_import = builtins.__import__

    def reject_cli_import(name, *args, **kwargs):
        if name == "pr_agent.cli":
            raise AssertionError("worker configuration hashing must not import the CLI")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_cli_import)

    configuration_hash = review_subprocess._current_review_configuration().configuration_hash

    assert review_subprocess._SNAPSHOT_ID_PATTERN.fullmatch(configuration_hash)


def test_configuration_hash_ignores_foreign_cwd_version(monkeypatch, tmp_path):
    expected = snapshot_review_configuration_hash("", repo_context_files={})
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "attacker-project"\nversion = "999.0.0"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert snapshot_review_configuration_hash("", repo_context_files={}) == expected


@pytest.mark.asyncio
async def test_empty_snapshot_completes_without_constructing_reviewer(monkeypatch):
    configuration = _configuration()
    snapshot = replace(
        _snapshot(review_configuration=configuration),
        changed_paths=(),
        diff="",
    )

    class UnexpectedReviewer:
        def __init__(self, _pr_url):
            raise AssertionError("empty snapshots must not construct a reviewer")

    monkeypatch.setattr("pr_agent.tools.pr_reviewer.PRReviewer", UnexpectedReviewer)

    outcome = await review_subprocess._execute_review(snapshot, configuration)

    assert outcome.state is review_subprocess.CheckpointReviewSubprocessState.COMPLETED
    assert outcome.review == {"review": {"key_issues_to_review": []}}
    assert outcome.run_details is None
    assert outcome.latency_seconds == 0.0
