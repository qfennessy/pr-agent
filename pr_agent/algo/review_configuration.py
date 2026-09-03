"""Stable identity helpers for effective review configuration."""

import hashlib
import json

from pr_agent.algo.skills_loader import get_skills_context
from pr_agent.algo.utils import get_version
from pr_agent.config_loader import get_settings


def snapshot_review_configuration_hash(
    skills_context: str | None = None,
    repo_context_files: dict[str, str] | None = None,
) -> str:
    """Hash source-free review inputs without importing CLI side effects."""

    settings = get_settings()

    credential_names = {"key", "token", "secret", "password", "credential", "credentials", "private"}
    credential_suffixes = (
        "_api_key", "_token", "_access_token", "_private_token", "_client_secret",
        "_webhook_secret", "_password", "_private_key", "_secret_access_key",
        "_auth_header", "_authorization", "_credential", "_credentials",
    )
    transient_config_keys = {"cli_mode", "git_provider", "publish_output", "propagate_tool_errors"}

    def sanitized(value, *, section: str = ""):
        if isinstance(value, dict):
            cleaned = {}
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in credential_names or normalized.endswith(credential_suffixes):
                    continue
                if section == "config" and normalized in transient_config_keys:
                    continue
                cleaned[str(key)] = sanitized(child, section=section)
            return cleaned
        if isinstance(value, (list, tuple)):
            return [sanitized(item, section=section) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    all_settings = settings.as_dict()
    all_settings.pop("PLAIN_DIFF", None)
    effective = {
        "runtime_version": get_version(),
        "skills_context_sha256": hashlib.sha256(
            (get_skills_context() if skills_context is None else skills_context).encode("utf-8")
        ).hexdigest(),
        "repo_context_files": repo_context_files or {},
        "settings": {
            str(section): sanitized(contents, section=str(section).lower())
            for section, contents in all_settings.items()
        },
    }
    payload = json.dumps(effective, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
