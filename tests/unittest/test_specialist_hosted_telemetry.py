import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pr_agent.algo.review_specialists import SpecialistBatchResult
from pr_agent.tools.pr_reviewer import PRReviewer


class _ReadOnlyProvider:
    """GitHub/GitLab-like input surface with no publication or comment hooks."""

    def __init__(self, head_sha):
        self.head_sha = head_sha

    def get_pr_head_sha(self, *, refresh):
        return self.head_sha

    def get_diff_files(self):
        return []


def _reviewer(provider):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.vars = {"title": "SECRET TITLE"}
    reviewer.pr_description = "SECRET DESCRIPTION"
    reviewer.ai_handler = MagicMock()
    return reviewer


def _batch(*, state="success", failure_reason=None):
    return SpecialistBatchResult(
        snapshot_id="snapshot-1",
        head_sha="head-1",
        input_hash="sha256:input",
        configuration_hash="sha256:configuration",
        records=(),
        role_records={
            "risk_recommendation": {
                "role": "risk_recommendation",
                "model": "low-cost-model",
                "deployment": None,
                "fallback_used": False,
                "prompt_version": "risk-recommendation-prompt-v2",
                "input_schema_version": "risk-recommendation-input-v2",
                "schema_version": "risk-recommendation-output-v2",
                "state": state,
                "latency_seconds": 0.25,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                    "ai_calls": 1,
                },
                "cost": {
                    "status": "complete",
                    "total_usd": "0.00012",
                    "by_model_usd": {"low-cost-model": "0.00012"},
                },
                "confidence": 0.91,
                "failure_reason": failure_reason,
                "cached": False,
                "reservation": {"input_tokens": 64, "output_tokens": 32},
                "output": {
                    "schema_version": "risk-recommendation-output-v2",
                    "confidence": 0.91,
                    "recommendation": "escalate",
                    "reasons": [
                        {
                            "reason": "Authorization checks were deleted",
                            "evidence": [
                                {
                                    "source": "diff_hunk",
                                    "path": "auth.py",
                                    "hunk_id": "sha256:hunk",
                                    "side": "old",
                                    "line": 10,
                                }
                            ],
                        }
                    ],
                },
            }
        },
        changed_path_count=2,
        hunk_count=3,
    )


def _assert_source_free_artifact(logger, batch):
    logger.info.assert_called_once_with(
        "Specialist shadow telemetry",
        artifact=batch.to_dict(),
    )
    artifact = logger.info.call_args.kwargs["artifact"]
    assert artifact["roles"]["risk_recommendation"]["output"]["recommendation"] == "escalate"
    assert artifact["roles"]["risk_recommendation"]["output"]["reasons"][0][
        "evidence"
    ][0] == {
        "source": "diff_hunk",
        "path": "auth.py",
        "hunk_id": "sha256:hunk",
        "side": "old",
        "line": 10,
    }
    assert artifact["roles"]["risk_recommendation"]["confidence"] == 0.91
    assert artifact["roles"]["risk_recommendation"]["usage"]["total_tokens"] == 28
    assert artifact["roles"]["risk_recommendation"]["cost"]["total_usd"] == "0.00012"
    serialized = json.dumps(artifact)
    assert "SECRET TITLE" not in serialized
    assert "SECRET DESCRIPTION" not in serialized
    assert "diff" not in artifact


@pytest.mark.asyncio
async def test_completed_specialist_batch_is_logged_without_provider_publication_api():
    provider = _ReadOnlyProvider("head-1")
    reviewer = _reviewer(provider)
    batch = _batch()
    pipeline = SimpleNamespace(allowed_change_labels=("feature",))
    specialist_input = SimpleNamespace()
    logger = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.load_specialist_pipeline_config", return_value=pipeline),
        patch("pr_agent.tools.pr_reviewer.get_specialist_snapshot_context", return_value=None),
        patch("pr_agent.tools.pr_reviewer.build_specialist_input", return_value=specialist_input),
        patch(
            "pr_agent.tools.pr_reviewer.run_shadow_specialists",
            new_callable=AsyncMock,
            return_value=batch,
        ),
        patch("pr_agent.tools.pr_reviewer.get_logger", return_value=logger),
    ):
        await reviewer._run_shadow_specialists_once()

    assert reviewer.specialist_shadow_result is batch
    _assert_source_free_artifact(logger, batch)
    assert not hasattr(provider, "publish_structured_review")
    assert not hasattr(provider, "publish_comment")


@pytest.mark.asyncio
async def test_unavailable_specialist_batch_is_logged_without_provider_publication_api():
    provider = _ReadOnlyProvider(None)
    reviewer = _reviewer(provider)
    batch = _batch(state="unavailable", failure_reason="stable_head_identity_unavailable")
    pipeline = SimpleNamespace()
    logger = MagicMock()

    with (
        patch("pr_agent.tools.pr_reviewer.load_specialist_pipeline_config", return_value=pipeline),
        patch("pr_agent.tools.pr_reviewer.get_specialist_snapshot_context", return_value=None),
        patch("pr_agent.tools.pr_reviewer.unavailable_specialist_batch", return_value=batch),
        patch("pr_agent.tools.pr_reviewer.get_logger", return_value=logger),
    ):
        await reviewer._run_shadow_specialists_once()

    assert reviewer.specialist_shadow_result is batch
    _assert_source_free_artifact(logger, batch)
    assert logger.warning.call_count == 1
    assert not hasattr(provider, "publish_structured_review")
    assert not hasattr(provider, "publish_comment")


@pytest.mark.asyncio
async def test_specialist_telemetry_export_failure_remains_non_blocking():
    provider = _ReadOnlyProvider("head-1")
    reviewer = _reviewer(provider)
    batch = _batch()
    pipeline = SimpleNamespace(allowed_change_labels=("feature",))
    logger = MagicMock()
    logger.info.side_effect = RuntimeError("telemetry backend unavailable")

    with (
        patch("pr_agent.tools.pr_reviewer.load_specialist_pipeline_config", return_value=pipeline),
        patch("pr_agent.tools.pr_reviewer.get_specialist_snapshot_context", return_value=None),
        patch("pr_agent.tools.pr_reviewer.build_specialist_input", return_value=SimpleNamespace()),
        patch(
            "pr_agent.tools.pr_reviewer.run_shadow_specialists",
            new_callable=AsyncMock,
            return_value=batch,
        ),
        patch("pr_agent.tools.pr_reviewer.get_logger", return_value=logger),
    ):
        await reviewer._run_shadow_specialists_once()

    assert reviewer.specialist_shadow_result is batch
    logger.warning.assert_called_once_with(
        "Specialist shadow batch failed; continuing the ordinary review",
        artifact={"error_class": "RuntimeError"},
    )
