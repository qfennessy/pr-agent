from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
from pr_agent.tools.pr_reviewer import PRReviewer


class _SettingsDict(dict):
    __getattr__ = dict.__getitem__


def _candidate(**overrides):
    candidate = {
        "relevant_file": "src/service.py",
        "issue_header": "Null error",
        "issue_content": "The new call dereferences a missing result.",
        "start_line": 12,
        "end_line": 12,
        "trigger": "The lookup returns None.",
        "impact": "The request crashes.",
        "root_cause": "unchecked lookup result",
        "context_files": ["src/caller.py"],
        "context_symbols": ["call_service"],
    }
    candidate.update(overrides)
    return candidate


def _diff_file():
    return FilePatchInfo(
        "\n".join("old" for _ in range(30)),
        "\n".join("new" for _ in range(30)),
        "@@ -10,1 +12,4 @@\n+one\n+two\n+three\n+four",
        "src/service.py",
        edit_type=EDIT_TYPE.UNKNOWN,
    )


def _settings():
    settings = SimpleNamespace(
        pr_reviewer=_SettingsDict(
            candidate_verification_max_candidates=3,
            candidate_verification_max_sensitive_candidates=6,
            candidate_verification_deployment="",
            candidate_verification_fallback_models=[],
            candidate_verification_fallback_deployments=[],
            candidate_verification_max_output_tokens=0,
            candidate_verification_max_files=3,
            candidate_verification_max_lines_per_file=30,
            candidate_verification_max_total_lines=60,
            candidate_verification_max_context_tokens=300,
            candidate_verification_timeout_seconds=1,
            candidate_verification_sensitive_path_globs=[],
            candidate_verification_max_model_calls=1,
            candidate_verification_model="",
            num_max_findings=3,
        ),
        config=_SettingsDict(model="model", temperature=0),
        pr_review_verification_prompt=SimpleNamespace(
            system="verify",
            user="{{ verification_payload }}",
        ),
    )
    settings.get = lambda key, default=None: default
    return settings


def _reviewer(second_candidate):
    provider = MagicMock()
    provider.supports_repo_file_fetching.return_value = True
    provider.get_diff_files.return_value = [_diff_file()]
    provider.get_repo_file_content.return_value = "def call_service(): return service().value"

    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.prediction = "review: {}"
    reviewer.patches_diff = "+changed"
    reviewer.remaining_files_list = []
    reviewer.deleted_files_list = []
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.candidate_verification_artifact = None
    reviewer.verified_review_data = None
    reviewer._parse_review_prediction = MagicMock(
        return_value={
            "review": {
                "key_issues_to_review": [_candidate(), second_candidate],
            }
        }
    )
    reviewer.ai_handler = SimpleNamespace(
        chat_completion=AsyncMock(
            return_value=(
                "verification:\n"
                "  decisions:\n"
                "    - candidate_id: candidate-1\n"
                "      verdict: rejected\n"
                "      reason: disproved by repository evidence",
                None,
            )
        )
    )
    return reviewer


@pytest.mark.asyncio
async def test_malformed_mixed_candidate_blocks_a_false_clean_review():
    reviewer = _reviewer(
        _candidate(
            start_line=99,
            end_line=99,
            root_cause="different off-diff proposal",
        )
    )

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_settings()),
        patch("pr_agent.tools.pr_reviewer.get_max_tokens", return_value=20_000),
    ):
        await reviewer._run_candidate_verification()

    artifact = reviewer.candidate_verification_artifact
    assert artifact["candidate_rejections"] == [
        {
            "candidate_id": "candidate-2",
            "reason": "invalid_candidate",
        }
    ]
    assert artifact["model_candidate_coverage"]["status"] == "incomplete"
    assert artifact["status"] == "candidate_validation_incomplete"
    assert artifact["publication_safe"] is False
    assert artifact["verified_count"] == 0
    assert reviewer.verified_review_data["review"]["key_issues_to_review"] == []
    assert reviewer._candidate_verification_blocks_publication() is True


@pytest.mark.asyncio
async def test_deliberate_duplicate_collapse_can_still_complete_cleanly():
    reviewer = _reviewer(_candidate())

    with (
        patch("pr_agent.tools.pr_reviewer.get_settings", return_value=_settings()),
        patch("pr_agent.tools.pr_reviewer.get_max_tokens", return_value=20_000),
    ):
        await reviewer._run_candidate_verification()

    artifact = reviewer.candidate_verification_artifact
    assert artifact["candidate_rejections"] == [
        {
            "candidate_id": "candidate-2",
            "reason": "duplicate_candidate",
        }
    ]
    assert artifact["model_candidate_coverage"]["status"] == "partial"
    assert artifact["status"] == "complete"
    assert artifact["publication_safe"] is True
    assert artifact["verified_count"] == 0
    assert reviewer.verified_review_data["review"]["key_issues_to_review"] == []
    assert reviewer._candidate_verification_blocks_publication() is False
