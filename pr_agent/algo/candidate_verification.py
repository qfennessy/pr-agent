"""Bounded repository-evidence retrieval and result validation for review candidates."""

import asyncio
import fnmatch
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterable, Mapping, Optional

from pr_agent.algo.git_patch_processing import (
    RE_HUNK_HEADER,
    iter_git_patch_lines,
    split_git_file_lines,
    strip_git_line_ending,
)
from pr_agent.algo.review_specialists import (
    SpecialistBatchResult,
    SpecialistInput,
    SpecialistRole,
    SpecialistState,
)
from pr_agent.algo.types import EDIT_TYPE
from pr_agent.log import get_logger

_TELEMETRY_SAFE_DECISION_REASONS = frozenset({
    "duplicate_decision",
    "duplicate_verified_finding",
    "location_not_in_changed_lines",
    "missing_decision",
    "required_context_unavailable",
    "changed_code_evidence_unavailable",
    "trusted_identity_collision",
    "trusted_identity_unavailable",
    "unknown_candidate",
    "unverified_or_incomplete_evidence",
})
_REPO_FETCH_MAX_WORKERS = 4
_REPO_FETCH_SLOTS = threading.BoundedSemaphore(_REPO_FETCH_MAX_WORKERS)
_ATOMIC_CHANGED_EVIDENCE_SOURCES = frozenset({
    "changed_head",
    "changed_patch",
    "changed_context_patch",
})
_PATCH_CHANGED_EVIDENCE_SOURCES = frozenset({"changed_patch", "changed_context_patch"})
_CHANGED_EVIDENCE_SOURCES = _ATOMIC_CHANGED_EVIDENCE_SOURCES | {"changed_context_head"}


def _is_atomic_prompt_evidence(item: Mapping) -> bool:
    return bool(
        item.get("source") in _ATOMIC_CHANGED_EVIDENCE_SOURCES
        or item.get("required_evidence")
    )


class _RepositoryFetchCapacityExhausted(RuntimeError):
    """Raised when timed-out provider work already occupies every bounded fetch slot."""


class _ChangedContextCollectionStopped(RuntimeError):
    """Raised when a retrieval budget stops lazy changed-context collection."""


async def _bounded_repo_file_fetch(git_provider, path: str, timeout_seconds: float,
                                   from_pr_head: bool = False):
    """Run a synchronous provider fetch without allowing abandoned work to grow unboundedly."""
    if timeout_seconds <= 0:
        raise asyncio.TimeoutError
    if not _REPO_FETCH_SLOTS.acquire(blocking=False):
        raise _RepositoryFetchCapacityExhausted

    loop = asyncio.get_running_loop()
    result = loop.create_future()

    def deliver(value=None, error=None):
        if result.done():
            return
        if error is not None:
            result.set_exception(error)
        else:
            result.set_result(value)

    def fetch():
        try:
            if from_pr_head:
                value = git_provider.get_pr_head_file_content(path)
            else:
                value = git_provider.get_repo_file_content(path, False)
            outcome = (value, None)
        except Exception as exc:
            outcome = (None, exc)
        finally:
            _REPO_FETCH_SLOTS.release()
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:
            # The timed-out caller may have closed its event loop. The bounded
            # slot is already released and no result remains to deliver.
            pass

    try:
        threading.Thread(
            target=fetch,
            name="pr-agent-repo-context-fetch",
            daemon=True,
        ).start()
    except Exception:
        _REPO_FETCH_SLOTS.release()
        raise
    return await asyncio.wait_for(result, timeout=timeout_seconds)


@dataclass(frozen=True)
class VerificationBudgets:
    max_candidates: int = 3
    max_files: int = 6
    max_lines_per_file: int = 160
    max_total_lines: int = 600
    max_context_tokens: int = 6000
    timeout_seconds: float = 10.0


def telemetry_safe_artifact(artifact: dict) -> dict:
    """Return provider-neutral telemetry without repository or model-generated text."""
    safe = {}
    scalar_keys = {
        "enabled", "status", "model_calls", "verifier_attempts", "candidate_count",
        "verified_count", "model", "proposal_source", "proposal_shape",
        "proposed_candidate_count", "accepted_model_candidate_count",
        "sensitive_candidate_count", "candidate_rejection_count",
        "verifier_verified_count", "finding_limit_dropped", "rejected_count", "failure",
        "verifier_latency_seconds", "publication_safe",
    }
    mapping_keys = {
        "specialist_prioritization", "sensitive_audit_coverage", "prompt_budget",
        "verifier_usage", "verifier_cost", "model_candidate_coverage",
        "prompt_evidence_coverage",
    }
    for key in scalar_keys:
        if key in artifact and isinstance(artifact[key], (str, int, float, bool, type(None))):
            safe[key] = artifact[key]
    for key in mapping_keys:
        value = artifact.get(key)
        if isinstance(value, dict):
            safe[key] = {
                item_key: item_value for item_key, item_value in value.items()
                if isinstance(item_key, str) and isinstance(item_value, (str, int, float, bool, type(None)))
            }

    rejections = artifact.get("candidate_rejections")
    if isinstance(rejections, list):
        safe["candidate_rejections"] = [
            {
                key: item[key]
                for key in (
                    "candidate_id", "reason", "sensitive_path", "total_count",
                    "selected_count", "omitted_count",
                )
                if key in item
            }
            for item in rejections if isinstance(item, dict)
        ]

    decisions = artifact.get("decisions")
    if isinstance(decisions, list):
        safe["decisions"] = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            decision = {
                key: item[key] for key in ("candidate_id", "verdict", "evidence_paths") if key in item
            }
            if "reason" in item:
                reason = str(item.get("reason") or "")
                decision["reason"] = (
                    reason if reason in _TELEMETRY_SAFE_DECISION_REASONS else "rejected_by_verifier"
                )
            safe["decisions"].append(decision)

    retrieval = artifact.get("retrieval")
    if isinstance(retrieval, dict):
        safe_retrieval = {
            key: retrieval[key]
            for key in (
                "budget_exhausted", "files_read", "changed_evidence_count", "lines_retrieved",
                "context_tokens", "duration_seconds",
            )
            if key in retrieval and isinstance(retrieval[key], (str, int, float, bool, type(None)))
        }
        requests = retrieval.get("requests")
        if isinstance(requests, list):
            safe_retrieval["requests"] = [
                {
                    key: item[key]
                    for key in (
                        "candidate_id", "path", "status", "error", "required", "source",
                        "start_line", "end_line",
                    )
                    if key in item and isinstance(item[key], (str, int, float, bool, type(None)))
                }
                for item in requests if isinstance(item, dict)
            ]
        retrieved = retrieval.get("retrieved_evidence")
        if isinstance(retrieved, list):
            safe_retrieval["retrieved_evidence"] = []
            for item in retrieved:
                if not isinstance(item, dict):
                    continue
                summary = {
                    key: item[key]
                    for key in (
                        "candidate_id", "candidate_ids", "source", "path", "start_line", "end_line",
                    )
                    if key in item and isinstance(item[key], (str, int, float, bool, type(None)))
                }
                candidate_ids = item.get("candidate_ids")
                if isinstance(candidate_ids, list):
                    summary["candidate_ids"] = [
                        candidate_id for candidate_id in candidate_ids
                        if isinstance(candidate_id, str)
                    ]
                summary["content_characters"] = len(str(item.get("content") or ""))
                safe_retrieval["retrieved_evidence"].append(summary)
        safe["retrieval"] = safe_retrieval
    return safe


def safe_repo_path(value) -> Optional[str]:
    """Return a normalized relative repository path, or None for unsafe input."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 512 or "\x00" in value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    normalized = str(path)
    return normalized if normalized == value else None


def _first_changed_line(patch: str) -> int:
    new_line = None
    for record in iter_git_patch_lines(patch or ""):
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            new_line = int(header.group(3))
            continue
        if new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            return new_line
        if not line.startswith("-"):
            new_line += 1
    return 1


def _consecutive_line_ranges(lines) -> list[tuple[int, int]]:
    ranges = []
    for line in lines:
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], line)
        else:
            ranges.append((line, line))
    return ranges


def _added_line_ranges(patch: str) -> list[tuple[int, int]]:
    added_lines = []
    new_line = None
    for record in iter_git_patch_lines(patch or ""):
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            new_line = int(header.group(3))
            continue
        if new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            added_lines.append(new_line)
        if not line.startswith("-"):
            new_line += 1
    return _consecutive_line_ranges(added_lines)


def _deleted_line_ranges(patch: str) -> list[tuple[int, int]]:
    deleted_lines = []
    old_line = None
    for record in iter_git_patch_lines(patch or ""):
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            old_line = int(header.group(1))
            continue
        if old_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("-"):
            deleted_lines.append(old_line)
        if not line.startswith("+"):
            old_line += 1
    return _consecutive_line_ranges(deleted_lines)


def _first_changed_anchor(patch: str) -> tuple[str, int, list[tuple[int, int]]]:
    added = _added_line_ranges(patch)
    if added:
        return "new", added[0][0], added
    deleted = _deleted_line_ranges(patch)
    if deleted:
        return "old", deleted[0][0], deleted
    return "new", _first_changed_line(patch), []


def _sensitive_change_anchors(
    patch: str,
) -> Iterable[tuple[str, int, int, list[tuple[int, int]]]]:
    """Return one audit for every visible contiguous changed range on both sides."""
    for side, ranges in (("new", _added_line_ranges(patch)), ("old", _deleted_line_ranges(patch))):
        for start, end in ranges:
            yield side, start, end, [(start, end)]


def _line_is_changed(line: int, ranges: list) -> bool:
    return any(start <= line <= end for start, end in ranges)


def _candidate_key(candidate: dict) -> tuple:
    root_cause = " ".join(str(candidate.get("root_cause") or "").casefold().split())
    if root_cause:
        return ("root_cause", root_cause)
    return (
        candidate.get("relevant_file"),
        candidate.get("start_line"),
        " ".join(str(candidate.get("issue_content") or "").casefold().split()),
    )


_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER_TOKEN = re.compile(r"\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)\b")
_STRUCTURAL_KEYWORDS = frozenset({
    "and", "async", "await", "break", "case", "catch", "class", "continue", "def", "do",
    "else", "except", "false", "finally", "for", "foreach", "function", "if", "in", "is",
    "lambda", "match", "new", "none", "not", "null", "or", "raise", "return", "switch",
    "throw", "true", "try", "while", "with", "yield",
})


def _replace_string_literals(value: str) -> str:
    """Replace quoted content in linear time without parsing repository code as a language."""
    shaped = []
    quote = None
    escaped = False
    for character in str(value or ""):
        if quote is None:
            if character in {"'", '"'}:
                shaped.append("<literal>")
                quote = character
            else:
                shaped.append(character)
            continue
        if character in {"\n", "\r"}:
            quote = None
            escaped = False
            shaped.append(character)
        elif escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            quote = None
    return "".join(shaped)


def _normalized_evidence_shape(content: str) -> str:
    """Normalize names, literals, paths, whitespace, and line numbers while preserving control flow."""
    shaped = _replace_string_literals(content)
    shaped = _NUMBER_TOKEN.sub("<number>", shaped)
    shaped = _IDENTIFIER_TOKEN.sub(
        lambda match: match.group(0).casefold()
        if match.group(0).casefold() in _STRUCTURAL_KEYWORDS else "<identifier>",
        shaped,
    )
    return "".join(shaped.split())


def _changed_anchor_identity(
    patch: str,
    start_line: int,
    end_line: int,
    side: str = "new",
) -> tuple[str, int]:
    """Derive the range shape and equal-shape start ordinal in one patch traversal."""
    old_line = None
    new_line = None
    segment = 0
    last_changed_line = None
    changed_records = []
    for record in iter_git_patch_lines(patch or ""):
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            old_line = int(header.group(1))
            new_line = int(header.group(3))
            segment += 1
            last_changed_line = None
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        is_changed = (
            side == "new" and line.startswith("+")
        ) or (
            side == "old" and line.startswith("-")
        )
        changed_line = new_line if side == "new" else old_line
        if is_changed:
            content = line[1:]
            line_shape = _normalized_evidence_shape(content)
            if last_changed_line is not None and changed_line != last_changed_line + 1:
                segment += 1
            changed_records.append((changed_line, segment, line_shape))
            last_changed_line = changed_line
        if not line.startswith("+"):
            old_line += 1
        if not line.startswith("-"):
            new_line += 1

    target_record_indexes = [
        index
        for index, (line_number, _, _) in enumerate(changed_records)
        if start_line <= line_number <= end_line
    ]
    if not target_record_indexes:
        return "", 0

    def tokenized(records: list[tuple[int, int, str]]) -> list[Optional[str]]:
        tokens = []
        previous_segment = None
        for _, record_segment, line_shape in records:
            if previous_segment is not None and record_segment != previous_segment:
                tokens.append(None)
            tokens.append(line_shape)
            previous_segment = record_segment
        return tokens

    changed_tokens = tokenized(changed_records)
    target_records = [changed_records[index] for index in target_record_indexes]
    target_shapes = tokenized(target_records)
    if not any(isinstance(token, str) and token for token in target_shapes):
        return "", 0
    anchor_shape = json.dumps(target_shapes, separators=(",", ":"))
    if target_records[0][0] != start_line:
        return anchor_shape, 0

    target_record_index = target_record_indexes[0]
    target_start_index = target_record_index
    for index in range(1, target_record_index + 1):
        if changed_records[index][1] != changed_records[index - 1][1]:
            target_start_index += 1

    # Count exact normalized range-shape occurrences using KMP so a multi-line
    # candidate is not renumbered by an earlier range that only shares its first
    # line or crosses a hunk/context boundary. This stays linear even for
    # generated patches with repeated anchors.
    prefix = [0] * len(target_shapes)
    matched = 0
    for index in range(1, len(target_shapes)):
        while matched and target_shapes[index] != target_shapes[matched]:
            matched = prefix[matched - 1]
        if target_shapes[index] == target_shapes[matched]:
            matched += 1
            prefix[index] = matched

    anchor_ordinal = 0
    matched = 0
    scan_limit = target_start_index + len(target_shapes)
    for index, line_shape in enumerate(changed_tokens[:scan_limit]):
        while matched and line_shape != target_shapes[matched]:
            matched = prefix[matched - 1]
        if line_shape == target_shapes[matched]:
            matched += 1
        if matched == len(target_shapes):
            occurrence_start = index - len(target_shapes) + 1
            if occurrence_start <= target_start_index:
                anchor_ordinal += 1
            matched = prefix[matched - 1]
    return anchor_shape, anchor_ordinal


def _changed_anchor_shape(patch: str, start_line: int, end_line: int, side: str = "new") -> str:
    return _changed_anchor_identity(patch, start_line, end_line, side)[0]


def _changed_anchor_ordinal(patch: str, start_line: int, side: str = "new") -> int:
    """Return the ordinal among equal-shaped anchors within one file lineage."""
    return _changed_anchor_identity(patch, start_line, start_line, side)[1]


def _trusted_file_lineage(diff_file) -> Optional[str]:
    """Use the base-side path as stable rename lineage when the provider supplies it."""
    old_path = safe_repo_path(getattr(diff_file, "old_filename", None))
    current_path = safe_repo_path(getattr(diff_file, "filename", None))
    path = old_path or current_path
    return f"file:{path}" if path else None


def _base_fetch_path(path: str, diff_file) -> str:
    """Return the base-side path while keeping evidence keyed to the current path."""
    if getattr(diff_file, "edit_type", None) == EDIT_TYPE.RENAMED:
        return safe_repo_path(getattr(diff_file, "old_filename", None)) or path
    return path


def _trusted_side_line_count(diff_file, side: str) -> Optional[int]:
    """Return a complete provider-supplied side's line count, if available."""
    if diff_file is None:
        return None
    content_attribute = "head_file" if side == "new" else "base_file"
    complete_attribute = f"{content_attribute}_is_complete"
    if not hasattr(diff_file, complete_attribute) or not getattr(diff_file, complete_attribute):
        return None
    content = getattr(diff_file, content_attribute, "") or ""
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return len(split_git_file_lines(str(content)))


def _bounded_sensitive_audit_specs(
    diff_files: list,
    normalized_globs: list[str],
    max_sensitive_candidates: Optional[int],
) -> tuple[list[tuple], int]:
    """Select sensitive ranges deterministically without constructing an unbounded payload."""
    limit = None if max_sensitive_candidates is None else max(0, int(max_sensitive_candidates))
    selected = []
    total_count = 0
    for diff_file in diff_files or []:
        relevant_file = safe_repo_path(getattr(diff_file, "filename", ""))
        old_file = safe_repo_path(getattr(diff_file, "old_filename", ""))
        sensitive_paths = {path for path in (relevant_file, old_file) if path}
        matched_glob_indexes = [
            index
            for index, pattern in enumerate(normalized_globs)
            if any(fnmatch.fnmatch(path, pattern) for path in sensitive_paths)
        ]
        if not relevant_file or not matched_glob_indexes:
            continue
        patch = getattr(diff_file, "patch", "")
        patch_digest = hashlib.sha256(str(patch or "").encode("utf-8")).hexdigest()
        lineage_path = old_file or relevant_file
        for anchor_index, (side, start_line, end_line, changed_line_ranges) in enumerate(
            _sensitive_change_anchors(patch), start=1
        ):
            total_count += 1
            # Earlier configured globs are the explicit risk order. A removed
            # guard is prioritized before added code within the same risk tier.
            # All remaining tie-breakers come from repository evidence, not
            # provider iteration order.
            priority = (
                min(matched_glob_indexes),
                0 if side == "old" else 1,
                lineage_path.casefold(),
                lineage_path,
                start_line,
                end_line,
                relevant_file.casefold(),
                relevant_file,
                patch_digest,
            )
            if limit == 0:
                continue
            selected.append((
                priority,
                diff_file,
                relevant_file,
                patch,
                anchor_index,
                side,
                start_line,
                end_line,
                changed_line_ranges,
            ))
            selected.sort(key=lambda item: item[0])
            if limit is not None and len(selected) > limit:
                selected.pop()
    return selected, total_count


def prepare_candidates(review_data: dict, diff_files: list, sensitive_globs: list,
                       max_candidates: int,
                       max_sensitive_candidates: Optional[int] = None) -> tuple[list[dict], list[dict]]:
    """Normalize model candidates, deduplicate them, and add deterministic sensitive-file audits."""
    raw_candidates = (review_data.get("review") or {}).get("key_issues_to_review") or []
    candidates = []
    rejected = []
    seen = set()
    diff_by_file = {
        path: diff_file
        for diff_file in (diff_files or [])
        if (path := safe_repo_path(getattr(diff_file, "filename", "")))
    }
    normalized_globs = [pattern.strip() for pattern in sensitive_globs if isinstance(pattern, str) and pattern.strip()]
    sensitive_specs, sensitive_total_count = _bounded_sensitive_audit_specs(
        diff_files,
        normalized_globs,
        max_sensitive_candidates,
    )
    for sensitive_count, spec in enumerate(sensitive_specs, start=1):
        (
            _, diff_file, relevant_file, patch, anchor_index, side,
            start_line, end_line, changed_line_ranges,
        ) = spec
        trusted_side_line_count = _trusted_side_line_count(diff_file, side)
        if trusted_side_line_count is not None and not (
            1 <= start_line <= end_line <= trusted_side_line_count
        ):
            rejected.append({
                "candidate_id": f"sensitive-{sensitive_count}",
                "reason": "invalid_candidate",
                "sensitive_path": True,
            })
            continue
        anchor_shape, anchor_ordinal = _changed_anchor_identity(
            patch, start_line, end_line, side
        )
        candidates.append({
            "candidate_id": f"sensitive-{sensitive_count}",
            "candidate_type": "sensitive_path_audit",
            "relevant_file": relevant_file,
            "issue_header": "Sensitive path audit",
            "issue_content": "Independently inspect this configured sensitive path change for introduced defects.",
            "trigger": "A configured sensitive path range changed.",
            "impact": "A high-risk regression could otherwise be suppressed before verification.",
            "root_cause": f"sensitive audit for {relevant_file} change {anchor_index}",
            "start_line": start_line,
            "end_line": end_line,
            "side": side,
            "context_files": [relevant_file],
            "context_symbols": [],
            "sensitive_path": True,
            "_changed_line_ranges": changed_line_ranges,
            "_changed_anchor_shape": anchor_shape,
            "_changed_anchor_ordinal": anchor_ordinal,
            "_trusted_lineage_key": _trusted_file_lineage(diff_file),
            "_trusted_side_line_count": trusted_side_line_count,
            "_display_file": (
                safe_repo_path(getattr(diff_file, "old_filename", None))
                if side == "old" else relevant_file
            ) or relevant_file,
        })
    sensitive_omitted_count = sensitive_total_count - len(sensitive_specs)
    if sensitive_omitted_count:
        rejected.append({
            "candidate_id": "sensitive-overflow",
            "reason": "sensitive_audit_budget_exhausted",
            "sensitive_path": True,
            "total_count": sensitive_total_count,
            "selected_count": len(sensitive_specs),
            "omitted_count": sensitive_omitted_count,
        })

    model_candidate_count = 0
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            rejected.append({"candidate_id": f"candidate-{index + 1}", "reason": "invalid_candidate"})
            continue
        candidate = dict(raw_candidate)
        for key in list(candidate):
            if str(key).startswith("_"):
                candidate.pop(key, None)
        candidate["candidate_id"] = f"candidate-{index + 1}"
        candidate["candidate_type"] = "model_finding"
        candidate["sensitive_path"] = False
        proposed_side_value = candidate.get("side", "new")
        proposed_side = (
            proposed_side_value.strip().lower()
            if isinstance(proposed_side_value, str) else ""
        )
        start_line = candidate.get("start_line")
        end_line = candidate.get("end_line")
        required_text = ("issue_header", "issue_content", "trigger", "impact", "root_cause")
        required_text_valid = all(
            isinstance(candidate.get(key), str) and candidate[key].strip()
            for key in required_text
        )
        context_files_present = "context_files" in candidate
        context_symbols_present = "context_symbols" in candidate
        context_files = candidate.get("context_files")
        context_symbols = candidate.get("context_symbols")
        normalized_context_files = (
            [safe_repo_path(value) for value in context_files]
            if isinstance(context_files, list) else []
        )
        context_files_valid = (
            context_files_present
            and isinstance(context_files, list)
            and all(normalized_context_files)
        )
        context_symbols_valid = (
            context_symbols_present
            and isinstance(context_symbols, list)
            and all(isinstance(value, str) and value.strip() for value in context_symbols)
        )
        candidate["side"] = "new"
        candidate["relevant_file"] = safe_repo_path(candidate.get("relevant_file"))
        candidate["start_line"] = start_line
        candidate["end_line"] = end_line
        candidate["context_files"] = normalized_context_files
        candidate["context_symbols"] = [
            value.strip() for value in context_symbols
        ] if context_symbols_valid else []
        diff_file = diff_by_file.get(candidate["relevant_file"])
        changed_line_ranges = _added_line_ranges(getattr(diff_file, "patch", "")) if diff_file else []
        trusted_side_line_count = _trusted_side_line_count(diff_file, "new")
        if (proposed_side != "new" or not isinstance(start_line, int) or
                isinstance(start_line, bool) or not isinstance(end_line, int) or
                isinstance(end_line, bool) or not required_text_valid or
                not context_files_valid or not context_symbols_valid or
                not candidate["relevant_file"] or not diff_file or
                not _line_is_changed(candidate["start_line"], changed_line_ranges) or
                candidate["end_line"] < candidate["start_line"] or
                (trusted_side_line_count is not None
                 and candidate["end_line"] > trusted_side_line_count)):
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": "invalid_candidate"})
            continue
        key = _candidate_key(candidate)
        if key in seen:
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": "duplicate_candidate"})
            continue
        if model_candidate_count >= max_candidates:
            rejected.append({
                "candidate_id": candidate["candidate_id"],
                "reason": "candidate_budget_exhausted",
            })
            continue
        seen.add(key)
        candidate["_changed_line_ranges"] = changed_line_ranges
        candidate["_changed_anchor_shape"], candidate["_changed_anchor_ordinal"] = (
            _changed_anchor_identity(
                getattr(diff_file, "patch", ""),
                candidate["start_line"],
                candidate["end_line"],
            )
        )
        candidate["_trusted_lineage_key"] = _trusted_file_lineage(diff_file)
        candidate["_trusted_side_line_count"] = trusted_side_line_count
        candidates.append(candidate)
        model_candidate_count += 1
    return candidates, rejected


def validated_specialist_prioritization(
    batch_result: Optional[SpecialistBatchResult],
    specialist_input: Optional[SpecialistInput],
) -> Optional[Mapping]:
    """Return only the validated diff-prioritization output for the exact shared input.

    Successful and cached role records have already crossed the specialist output
    validator. Other states can contain rejected or incomplete output and must not
    influence candidate verification.
    """
    if (
        batch_result is None
        or specialist_input is None
        or batch_result.stale
        or batch_result.input_hash != specialist_input.input_hash
        or batch_result.snapshot_id != specialist_input.snapshot_id
        or batch_result.head_sha != specialist_input.head_sha
    ):
        return None
    for record in batch_result.records:
        if (
            record.role is SpecialistRole.DIFF_PRIORITIZATION
            and record.state in (SpecialistState.SUCCESS, SpecialistState.CACHED)
            and isinstance(record.output, Mapping)
            and isinstance(record.output.get("ranked_hunks"), (list, tuple))
            and isinstance(record.output.get("context_requests"), (list, tuple))
        ):
            return record.output
    return None


def _candidate_hunk_id(candidate: dict, specialist_input: SpecialistInput) -> Optional[str]:
    try:
        line = int(candidate.get("start_line"))
    except (TypeError, ValueError):
        return None
    path = candidate.get("relevant_file")
    for hunk in specialist_input.hunks:
        if hunk.path == path and hunk.start_line <= line <= hunk.end_line:
            return hunk.hunk_id
    return None


def _append_context_hint(candidate: dict, kind: str, target: str) -> bool:
    target = str(target or "").strip()
    if not target or len(target) > 256 or "\x00" in target or "\n" in target or "\r" in target:
        return False
    path = safe_repo_path(target)
    path_name = PurePosixPath(path).name if path else ""
    if kind != "symbol" and path and not any(character.isspace() for character in path) and (
        "/" in path or "." in path_name
    ):
        key = "context_files"
        value = path
    else:
        key = "context_symbols"
        value = target
    current = candidate.get(key) or []
    if isinstance(current, str):
        current = [current]
    elif not isinstance(current, list):
        current = []
    if value in current:
        return False
    candidate[key] = [*current, value]
    if key == "context_files":
        optional = candidate.get("_specialist_optional_context_files") or []
        candidate["_specialist_optional_context_files"] = [*optional, value]
    return True


def apply_specialist_prioritization(
    candidates: list[dict],
    prioritization: Mapping,
    specialist_input: SpecialistInput,
) -> tuple[list[dict], dict]:
    """Apply validated hunk ranks and anchored context requests without dropping candidates."""
    prepared = [dict(candidate) for candidate in candidates]
    candidate_hunks = {
        candidate["candidate_id"]: _candidate_hunk_id(candidate, specialist_input)
        for candidate in prepared
    }
    ranks = {
        str(item.get("hunk_id")): int(item["rank"])
        for item in prioritization.get("ranked_hunks", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("rank"), int)
        and not isinstance(item.get("rank"), bool)
    }
    original_order = {candidate["candidate_id"]: index for index, candidate in enumerate(prepared)}
    prepared.sort(key=lambda candidate: (
        0 if candidate.get("sensitive_path") else 1,
        ranks.get(candidate_hunks.get(candidate["candidate_id"]), float("inf")),
        original_order[candidate["candidate_id"]],
    ))

    context_hints_added = 0
    matched_requests = 0
    for request in prioritization.get("context_requests", ()):
        if not isinstance(request, Mapping):
            continue
        anchor_hunk_id = str(request.get("anchor_hunk_id") or "")
        anchor_path = request.get("anchor_path")
        matched = False
        for candidate in prepared:
            if (
                candidate.get("relevant_file") != anchor_path
                or candidate_hunks.get(candidate["candidate_id"]) != anchor_hunk_id
            ):
                continue
            matched = True
            if _append_context_hint(candidate, str(request.get("kind") or ""), request.get("target")):
                context_hints_added += 1
        if matched:
            matched_requests += 1

    return prepared, {
        "status": "applied",
        "ranked_candidate_count": sum(
            1 for hunk_id in candidate_hunks.values() if hunk_id in ranks
        ),
        "context_request_count": len(prioritization.get("context_requests", ())),
        "matched_context_request_count": matched_requests,
        "context_hints_added": context_hints_added,
    }


def _requested_path_specs(candidate: dict) -> list[dict]:
    relevant_file = candidate.get("relevant_file")
    paths = [(relevant_file, False)]
    optional_context = {
        path for value in (candidate.get("_specialist_optional_context_files") or [])
        if (path := safe_repo_path(value))
    }
    context_files = candidate.get("context_files") or []
    if isinstance(context_files, str):
        context_files = [context_files]
    if isinstance(context_files, list):
        paths.extend(
            (value, value != relevant_file and safe_repo_path(value) not in optional_context)
            for value in context_files
        )
    normalized = []
    seen = set()
    for value, required in paths:
        path = safe_repo_path(value)
        if path:
            if path not in seen:
                normalized.append({"path": path, "required": required})
                seen.add(path)
        elif required:
            normalized.append({"path": None, "required": True, "status": "unsafe_path"})
    return normalized


def _excerpt_anchor_line(content: str, candidate: dict, path: str) -> int:
    lines = split_git_file_lines(content)
    if not lines:
        return 0
    center = None
    if path == candidate.get("relevant_file"):
        center = candidate.get("start_line")
    if center is None:
        symbols = candidate.get("context_symbols") or []
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols if isinstance(symbols, list) else []:
            symbol = str(symbol).strip()
            if not symbol:
                continue
            for index, line in enumerate(lines, start=1):
                if symbol in line:
                    center = index
                    break
            if center is not None:
                break
    return max(1, min(len(lines), int(center or 1)))


def _select_excerpt(content: str, candidate: dict, path: str, max_lines: int) -> tuple[str, int, int]:
    lines = split_git_file_lines(content)
    if not lines or max_lines < 1:
        return "", 0, 0
    center = _excerpt_anchor_line(content, candidate, path)
    before = max_lines // 2
    start = max(1, center - before)
    end = min(len(lines), start + max_lines - 1)
    start = max(1, end - max_lines + 1)
    return "\n".join(lines[start - 1:end]), start, end


def _matching_static_evidence(candidate: dict, static_evidence: list) -> list[dict]:
    matches = []
    for item in static_evidence if isinstance(static_evidence, list) else []:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        path = safe_repo_path(item.get("path"))
        content = str(item.get("content") or "").strip()
        if not content or (candidate_id and candidate_id != candidate["candidate_id"]):
            continue
        if not candidate_id and path != candidate.get("relevant_file"):
            continue
        match = dict(item)
        match.update({"path": path or "", "content": content})
        match.setdefault("source", "static_analyzer")
        matches.append(match)
    return matches


def _changed_patch_range_evidence(
    diff_file,
    side: str,
    start_line: int,
    end_line: int,
    source: str,
    *,
    max_lines: Optional[int] = None,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_budget_exhausted: Optional[Callable[[], None]] = None,
) -> Optional[dict]:
    """Return a complete side-specific patch range with at least one changed line."""
    if diff_file is None:
        return None
    range_line_count = end_line - start_line + 1
    if start_line < 1 or range_line_count < 1:
        return None
    if max_lines is not None and range_line_count > max(0, int(max_lines)):
        # This exact range cannot fit the prompt-visible evidence budget.  Fail
        # before walking the patch or materializing any range-sized lists.
        if on_budget_exhausted is not None:
            on_budget_exhausted()
        return None
    if max_tokens is not None and int(max_tokens) < 1:
        if on_budget_exhausted is not None:
            on_budget_exhausted()
        return None
    count_tokens = token_counter or (lambda value: len(str(value).encode("utf-8")))
    newline_tokens = count_tokens("\n") if max_tokens is not None else 0
    selected_tokens = 0
    old_line = None
    new_line = None
    selected = []
    selected_lines = []
    start_line_is_changed = False
    for record in iter_git_patch_lines(getattr(diff_file, "patch", "") or ""):
        if stop_requested is not None and stop_requested():
            return None
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            old_line = int(header.group(1))
            new_line = int(header.group(3))
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if side == "new" and not line.startswith("-") and start_line <= new_line <= end_line:
            content_line = line[1:] if line.startswith(("+", " ")) else line
            if max_tokens is not None:
                next_tokens = count_tokens(content_line) + (newline_tokens if selected else 0)
                if selected_tokens + next_tokens > int(max_tokens):
                    if on_budget_exhausted is not None:
                        on_budget_exhausted()
                    return None
                selected_tokens += next_tokens
            selected.append(content_line)
            selected_lines.append(new_line)
            if line.startswith("+") and new_line == start_line:
                start_line_is_changed = True
        elif side == "old" and not line.startswith("+") and start_line <= old_line <= end_line:
            content_line = line[1:] if line.startswith(("-", " ")) else line
            if max_tokens is not None:
                next_tokens = count_tokens(content_line) + (newline_tokens if selected else 0)
                if selected_tokens + next_tokens > int(max_tokens):
                    if on_budget_exhausted is not None:
                        on_budget_exhausted()
                    return None
                selected_tokens += next_tokens
            selected.append(content_line)
            selected_lines.append(old_line)
            if line.startswith("-") and old_line == start_line:
                start_line_is_changed = True
        if not line.startswith("+"):
            old_line += 1
        if not line.startswith("-"):
            new_line += 1
    has_complete_range = (
        len(selected_lines) == range_line_count
        and all(line_number == start_line + offset
                for offset, line_number in enumerate(selected_lines))
    )
    if not has_complete_range or not start_line_is_changed or not selected:
        return None
    return {
        "source": source,
        "side": side,
        "content": "\n".join(selected),
        "start_line": min(selected_lines),
        "end_line": max(selected_lines),
        "anchor_start_line": start_line,
        "anchor_end_line": end_line,
    }


def _candidate_changed_patch_evidence(
    diff_file,
    candidate: dict,
    *,
    max_lines: Optional[int] = None,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_budget_exhausted: Optional[Callable[[], None]] = None,
) -> Optional[dict]:
    """Return exact candidate-scoped changed lines from one trusted provider patch."""
    try:
        start_line = int(candidate.get("start_line"))
        end_line = int(candidate.get("end_line"))
    except (TypeError, ValueError):
        return None
    return _changed_patch_range_evidence(
        diff_file,
        candidate.get("side", "new"),
        start_line,
        end_line,
        "changed_patch",
        max_lines=max_lines,
        max_tokens=max_tokens,
        token_counter=token_counter,
        stop_requested=stop_requested,
        on_budget_exhausted=on_budget_exhausted,
    )


def _changed_context_patch_evidence(
    diff_file,
    stop_requested: Optional[Callable[[], bool]] = None,
    remaining_line_budget: Optional[Callable[[], int]] = None,
    remaining_token_budget: Optional[Callable[[], int]] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> Iterable[dict]:
    """Yield changed ranges from one patch traversal so retrieval budgets can stop work early."""
    if diff_file is None:
        return

    old_line = None
    new_line = None
    pending = {"new": None, "old": None}

    def completed(side: str) -> Optional[dict]:
        current = pending[side]
        if current is None:
            return None
        pending[side] = None
        return {
            "source": "changed_context_patch",
            "side": side,
            "content": "\n".join(current["content"]),
            "start_line": current["start_line"],
            "end_line": current["end_line"],
            "anchor_start_line": current["start_line"],
            "anchor_end_line": current["end_line"],
        }

    def append_changed(side: str, line_number: int, content: str) -> Optional[dict]:
        current = pending[side]
        finished = None
        if current is not None and line_number != current["end_line"] + 1:
            finished = completed(side)
            current = None
        line_budget = remaining_line_budget() if remaining_line_budget is not None else None
        token_budget = remaining_token_budget() if remaining_token_budget is not None else None
        added_tokens = 0
        if token_counter is not None:
            added_tokens = max(1, int(token_counter(content)))
            if current is not None:
                added_tokens += max(1, int(token_counter("\n")))
        if line_budget is not None and (
            line_budget < 1
            or (current is not None and len(current["content"]) >= line_budget)
        ):
            raise _ChangedContextCollectionStopped
        if token_budget is not None and (
            token_budget < 1
            or (current["token_count"] if current is not None else 0) + added_tokens > token_budget
        ):
            raise _ChangedContextCollectionStopped
        if current is None:
            pending[side] = {
                "start_line": line_number,
                "end_line": line_number,
                "content": [content],
                "token_count": added_tokens,
            }
        else:
            current["end_line"] = line_number
            current["content"].append(content)
            current["token_count"] += added_tokens
        return finished

    for record in iter_git_patch_lines(getattr(diff_file, "patch", "") or ""):
        if stop_requested is not None and stop_requested():
            raise _ChangedContextCollectionStopped
        line = strip_git_line_ending(record)
        header = RE_HUNK_HEADER.match(line)
        if header:
            next_old_line = int(header.group(1))
            next_new_line = int(header.group(3))
            for side, next_line in (("new", next_new_line), ("old", next_old_line)):
                current = pending[side]
                if current is not None and next_line != current["end_line"] + 1:
                    item = completed(side)
                    if item is not None:
                        yield item
            old_line = next_old_line
            new_line = next_new_line
            continue
        if old_line is None or new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            item = append_changed("new", new_line, line[1:])
            if item is not None:
                yield item
        elif line.startswith("-"):
            item = append_changed("old", old_line, line[1:])
            if item is not None:
                yield item
        else:
            for side in ("new", "old"):
                item = completed(side)
                if item is not None:
                    yield item
        if not line.startswith("+"):
            old_line += 1
        if not line.startswith("-"):
            new_line += 1

    for side in ("new", "old"):
        item = completed(side)
        if item is not None:
            yield item


def _retrieval_evidence_id(candidate_id: str, path: str, source: str) -> str:
    payload = json.dumps(
        {"candidate_id": candidate_id, "path": path, "source": source},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


async def retrieve_evidence(git_provider, candidates: list[dict], budgets: VerificationBudgets,
                            static_evidence: list, diff_files: Optional[list] = None,
                            token_counter: Optional[Callable[[str], int]] = None,
                            prefer_pr_head: bool = False) -> tuple[list[dict], dict]:
    """Fetch bounded repository excerpts and return them with an auditable retrieval record."""
    started = time.monotonic()
    deadline = started + max(0.0, budgets.timeout_seconds)
    evidence = []
    requests = []
    cache = {}
    unique_files = set()
    lines_by_path = {}
    total_lines = 0
    total_tokens = 0
    count_tokens = token_counter or (lambda value: len(str(value).encode("utf-8")))
    budget_exhausted = False
    time_budget_exhausted = False
    changed_evidence_count = 0
    changed_context_head_available = {}
    incomplete_changed_context = set()
    shared_repo_evidence = {}
    diff_by_file = {
        path: diff_file
        for diff_file in (diff_files or [])
        if (path := safe_repo_path(getattr(diff_file, "filename", "")))
    }

    def remaining_lines(path: str) -> int:
        return max(0, min(
            budgets.max_lines_per_file - lines_by_path.get(path, 0),
            budgets.max_total_lines - total_lines,
        ))

    def append_evidence(item: dict, candidate: dict, path: str) -> bool:
        nonlocal budget_exhausted, changed_evidence_count, time_budget_exhausted, total_lines, total_tokens
        if time.monotonic() >= deadline:
            budget_exhausted = True
            time_budget_exhausted = True
            return False
        content = str(item.get("content") or "")
        line_budget = remaining_lines(path)
        token_budget = budgets.max_context_tokens - total_tokens
        if line_budget < 1 or token_budget < 1:
            budget_exhausted = True
            return False
        if item.get("source") in _PATCH_CHANGED_EVIDENCE_SOURCES:
            if content.count("\n") + 1 > line_budget:
                budget_exhausted = True
                return False
            excerpt = content
            start_line = item.get("start_line")
            end_line = item.get("end_line")
        else:
            excerpt, start_line, end_line = _select_excerpt(content, candidate, path, line_budget)
        if not excerpt:
            budget_exhausted = True
            return False
        prepared = dict(item)
        prepared.update({"path": path, "content": excerpt})
        prepared.setdefault("start_line", start_line)
        prepared.setdefault("end_line", end_line)
        if _is_atomic_prompt_evidence(prepared):
            prepared.setdefault("anchor_start_line", candidate.get("start_line"))
            prepared.setdefault("anchor_end_line", candidate.get("end_line"))
        evidence_tokens = count_tokens(excerpt)
        if evidence_tokens > token_budget:
            budget_exhausted = True
            if _is_atomic_prompt_evidence(prepared):
                best = None
                lower = 0.0
                upper = 1.0
                for _ in range(24):
                    midpoint = (lower + upper) / 2
                    bounded = bounded_verification_evidence([prepared], midpoint)
                    if bounded and count_tokens(bounded[0]["content"]) <= token_budget:
                        best = bounded[0]
                        lower = midpoint
                    else:
                        upper = midpoint
                if best is None:
                    return False
                prepared = best
            else:
                lower = 0
                upper = len(excerpt)
                while lower < upper:
                    midpoint = (lower + upper + 1) // 2
                    if count_tokens(excerpt[:midpoint]) <= token_budget:
                        lower = midpoint
                    else:
                        upper = midpoint - 1
                if lower < 1:
                    return False
                prepared["content"] = excerpt[:lower]
                prepared["content_truncated"] = True
                prepared["end_line"] = prepared["start_line"] + prepared["content"].count("\n")
            evidence_tokens = count_tokens(prepared["content"])
        if time.monotonic() > deadline:
            budget_exhausted = True
            time_budget_exhausted = True
            return False
        line_count = prepared["content"].count("\n") + 1
        evidence.append(prepared)
        lines_by_path[path] = lines_by_path.get(path, 0) + line_count
        total_lines += line_count
        total_tokens += evidence_tokens
        if prepared.get("source") in _CHANGED_EVIDENCE_SOURCES:
            changed_evidence_count += 1
        return True

    def append_complete_context_head(candidate: dict, candidate_id: str, context_path: str,
                                     request_spec: dict, context_diff) -> bool:
        head_file = getattr(context_diff, "head_file", "")
        if not head_file or not getattr(context_diff, "head_file_is_complete", True):
            return False
        source = "changed_context_head"
        evidence_id = _retrieval_evidence_id(candidate_id, context_path, source)
        anchor_line = _excerpt_anchor_line(str(head_file), candidate, context_path)
        if not append_evidence({
            "candidate_id": candidate_id,
            "source": source,
            "content": str(head_file),
            "evidence_id": evidence_id,
            "required_evidence": bool(request_spec.get("required")),
            "anchor_start_line": anchor_line,
            "anchor_end_line": anchor_line,
        }, candidate, context_path):
            return False
        changed_context_head_available[(candidate_id, context_path)] = evidence_id
        return True

    # Reserve each exact candidate anchor before any larger same-file or
    # repository excerpt can consume its path budget. This keeps one candidate
    # from starving another candidate in the same file of changed-code proof.
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relevant_file = candidate.get("relevant_file")
        candidate_line_budget = remaining_lines(relevant_file)
        try:
            candidate_range_lines = (
                int(candidate.get("end_line")) - int(candidate.get("start_line")) + 1
            )
        except (TypeError, ValueError):
            candidate_range_lines = 0
        changed_patch = None
        if candidate_range_lines < 1 or candidate_range_lines > candidate_line_budget:
            budget_exhausted = True
        elif time.monotonic() >= deadline or total_tokens >= budgets.max_context_tokens:
            budget_exhausted = True
            time_budget_exhausted = time.monotonic() >= deadline
        else:
            candidate_budget_stop = []
            changed_patch = _candidate_changed_patch_evidence(
                diff_by_file.get(relevant_file),
                candidate,
                max_lines=candidate_line_budget,
                max_tokens=budgets.max_context_tokens - total_tokens,
                token_counter=count_tokens,
                stop_requested=lambda: time.monotonic() >= deadline,
                on_budget_exhausted=(
                    lambda stops=candidate_budget_stop: stops.append(True)
                ),
            )
            if candidate_budget_stop:
                budget_exhausted = True
            if changed_patch is None and time.monotonic() >= deadline:
                budget_exhausted = True
                time_budget_exhausted = True
        if changed_patch is not None:
            append_evidence(
                {"candidate_id": candidate_id, **changed_patch},
                candidate,
                relevant_file,
            )

    # Supplemental patches for requested changed files come only after every
    # candidate has had its mandatory own-change anchor reserved.
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relevant_file = candidate.get("relevant_file")
        for request_spec in _requested_path_specs(candidate):
            context_path = request_spec.get("path")
            if not context_path or context_path == relevant_file:
                continue
            context_diff = diff_by_file.get(context_path)
            context_key = (candidate_id, context_path)
            if context_diff is None:
                continue
            if (
                getattr(context_diff, "edit_type", None) == EDIT_TYPE.ADDED
                and getattr(context_diff, "head_file", "")
                and getattr(context_diff, "head_file_is_complete", True)
            ):
                if not append_complete_context_head(
                    candidate, candidate_id, context_path, request_spec, context_diff
                ):
                    incomplete_changed_context.add(context_key)
                continue
            context_item_count = 0
            try:
                context_items = _changed_context_patch_evidence(
                    context_diff,
                    stop_requested=lambda path=context_path: (
                        time.monotonic() >= deadline
                        or remaining_lines(path) < 1
                        or total_tokens >= budgets.max_context_tokens
                    ),
                    remaining_line_budget=lambda path=context_path: remaining_lines(path),
                    remaining_token_budget=lambda: budgets.max_context_tokens - total_tokens,
                    token_counter=count_tokens,
                )
                appended_all = True
                for context_item in context_items:
                    context_item_count += 1
                    if not append_evidence(
                        {"candidate_id": candidate_id, **context_item},
                        candidate,
                        context_path,
                    ):
                        appended_all = False
                        break
            except _ChangedContextCollectionStopped:
                budget_exhausted = True
                if time.monotonic() >= deadline:
                    time_budget_exhausted = True
                appended_all = False
            if context_item_count == 0 and appended_all:
                appended_all = append_complete_context_head(
                    candidate, candidate_id, context_path, request_spec, context_diff
                )
            if not appended_all:
                incomplete_changed_context.add(context_key)

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        relevant_file = candidate.get("relevant_file")
        diff_file = diff_by_file.get(candidate.get("relevant_file"))
        head_file = getattr(diff_file, "head_file", "") if diff_file is not None else ""
        changed_head_available = False
        changed_head_evidence_id = _retrieval_evidence_id(
            candidate_id, relevant_file, "changed_head"
        )
        for static_index, static_item in enumerate(_matching_static_evidence(candidate, static_evidence)):
            evidence_item = {"candidate_id": candidate_id, **static_item}
            static_path = static_item.get("path") or f"@static/{candidate_id}/{static_index + 1}"
            append_evidence(evidence_item, candidate, static_path)
        if (candidate.get("side", "new") == "new" and head_file
                and getattr(diff_file, "head_file_is_complete", True)):
            changed_head_available = append_evidence({
                "candidate_id": candidate_id,
                "source": "changed_head",
                "content": str(head_file),
                "evidence_id": changed_head_evidence_id,
            }, candidate, relevant_file)
        for request_spec in _requested_path_specs(candidate):
            request = {"candidate_id": candidate_id, **request_spec}
            if request.get("status") == "unsafe_path":
                requests.append(request)
                continue
            path = request["path"]
            request["status"] = "pending"
            requests.append(request)
            if path == relevant_file and changed_head_available:
                request["status"] = "satisfied_by_changed_head"
                request["source"] = "changed_head"
                request["evidence_id"] = changed_head_evidence_id
                continue
            context_key = (candidate_id, path)
            if context_key in changed_context_head_available:
                request["status"] = "satisfied_by_changed_head"
                request["source"] = "changed_context_head"
                request["evidence_id"] = changed_context_head_available[context_key]
                continue
            if context_key in incomplete_changed_context:
                request["status"] = "context_budget_exhausted"
                budget_exhausted = True
                continue
            source = "pr_head_file" if prefer_pr_head else "repository_file"
            fetch_path = path if prefer_pr_head else _base_fetch_path(path, diff_by_file.get(path))
            cache_key = ("pr_head" if prefer_pr_head else "base", fetch_path)
            shared_item = None
            cached_content = cache.get(cache_key)
            if isinstance(cached_content, bytes):
                cached_content = cached_content.decode("utf-8", errors="replace")
            resolved_anchor = _excerpt_anchor_line(
                str(cached_content or ""), candidate, path
            )
            for existing in shared_repo_evidence.get((source, path), ()):
                try:
                    visible_start = int(existing.get("start_line"))
                    visible_end = int(existing.get("end_line"))
                except (TypeError, ValueError):
                    continue
                visible_lines = split_git_file_lines(str(existing.get("content") or ""))
                source_lines = split_git_file_lines(str(cached_content or ""))
                visible_anchor_index = resolved_anchor - visible_start
                if (
                    resolved_anchor > 0
                    and resolved_anchor == existing.get("anchor_start_line")
                    and resolved_anchor == existing.get("anchor_end_line")
                    and visible_start <= resolved_anchor <= visible_end
                    and 0 <= visible_anchor_index < len(visible_lines)
                    and resolved_anchor <= len(source_lines)
                    and visible_lines[visible_anchor_index] == source_lines[resolved_anchor - 1]
                ):
                    shared_item = existing
                    break
            if shared_item is not None:
                candidate_ids = shared_item.get("candidate_ids")
                if not isinstance(candidate_ids, list):
                    candidate_ids = [shared_item.pop("candidate_id")]
                    shared_item["candidate_ids"] = candidate_ids
                if candidate_id not in candidate_ids:
                    candidate_ids.append(candidate_id)
                shared_item["required_evidence"] = bool(
                    shared_item.get("required_evidence") or request.get("required")
                )
                request.update({
                    "status": "retrieved",
                    "source": source,
                    "start_line": shared_item["start_line"],
                    "end_line": shared_item["end_line"],
                    "evidence_id": shared_item["evidence_id"],
                })
                continue
            if path not in unique_files and len(unique_files) >= budgets.max_files:
                request["status"] = "file_budget_exhausted"
                budget_exhausted = True
                continue
            if remaining_lines(path) < 1 or total_tokens >= budgets.max_context_tokens:
                request["status"] = "context_budget_exhausted"
                budget_exhausted = True
                continue
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                request["status"] = "time_budget_exhausted"
                budget_exhausted = True
                continue
            unique_files.add(path)
            if cache_key not in cache:
                try:
                    cache[cache_key] = await _bounded_repo_file_fetch(
                        git_provider, fetch_path, remaining_time, from_pr_head=prefer_pr_head
                    )
                except asyncio.TimeoutError:
                    request["status"] = "time_budget_exhausted"
                    budget_exhausted = True
                    continue
                except _RepositoryFetchCapacityExhausted:
                    request["status"] = "fetch_capacity_exhausted"
                    budget_exhausted = True
                    continue
                except Exception as exc:
                    cache[cache_key] = None
                    request["status"] = "fetch_failed"
                    request["error"] = type(exc).__name__
                    continue
            content = cache[cache_key]
            if not content:
                request["status"] = "missing"
                continue
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            evidence_id = _retrieval_evidence_id(candidate_id, path, source)
            anchor_line = _excerpt_anchor_line(str(content), candidate, path)
            evidence_item = {
                "candidate_id": candidate_id,
                "source": source,
                "path": path,
                "content": str(content),
                "evidence_id": evidence_id,
                "required_evidence": bool(request.get("required")),
                "anchor_start_line": anchor_line,
                "anchor_end_line": anchor_line,
            }
            before = len(evidence)
            if not append_evidence(evidence_item, candidate, path):
                request["status"] = (
                    "time_budget_exhausted" if time_budget_exhausted else "context_budget_exhausted"
                )
                continue
            appended = evidence[before]
            shared_repo_evidence.setdefault((source, path), []).append(appended)
            request.update({
                "status": "retrieved",
                "source": source,
                "start_line": appended["start_line"],
                "end_line": appended["end_line"],
                "evidence_id": evidence_id,
            })

    artifact = {
        "requests": requests,
        "retrieved_evidence": evidence,
        "budget_exhausted": budget_exhausted,
        "files_read": len(unique_files),
        "changed_evidence_count": changed_evidence_count,
        "lines_retrieved": total_lines,
        "context_tokens": total_tokens,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    return evidence, artifact


def render_verification_payload(candidates: list[dict], changed_diff: str, evidence: list[dict],
                                content_fraction: float = 1.0,
                                changed_diff_fraction: Optional[float] = None) -> str:
    """Serialize all model-controlled material as JSON data for the verifier prompt."""
    content_fraction = min(1.0, max(0.0, float(content_fraction)))
    if changed_diff_fraction is None:
        changed_diff_fraction = content_fraction
    changed_diff_fraction = min(1.0, max(0.0, float(changed_diff_fraction)))

    bounded_evidence = bounded_verification_evidence(evidence, content_fraction)
    return json.dumps({"candidates": candidates,
                       "changed_diff": _bounded_content(changed_diff, changed_diff_fraction),
                       "evidence": bounded_evidence},
                      ensure_ascii=False, indent=2)


def _bounded_content(value, fraction: float) -> str:
    value = str(value or "")
    if fraction >= 1.0 or not value:
        return value
    kept_characters = int(len(value) * fraction)
    if kept_characters < 1:
        return ""
    if kept_characters >= len(value):
        return value
    return f"{value[:kept_characters]}\n...(truncated)"


def bounded_verification_evidence(evidence: list[dict], content_fraction: float) -> list[dict]:
    """Return the exact evidence material made visible in a bounded verifier prompt."""
    fraction = min(1.0, max(0.0, float(content_fraction)))
    bounded = []
    for item in evidence:
        bounded_item = dict(item)
        if "content" in bounded_item:
            content = str(bounded_item["content"] or "")
            allowed_characters = int(len(content) * fraction)
            anchor_start = bounded_item.get("anchor_start_line")
            anchor_end = bounded_item.get("anchor_end_line")
            excerpt_start = bounded_item.get("start_line")
            if (fraction < 1.0 and _is_atomic_prompt_evidence(bounded_item)
                    and all(isinstance(value, int) for value in (anchor_start, anchor_end, excerpt_start))):
                lines = split_git_file_lines(content)
                relative_start = anchor_start - excerpt_start
                relative_end = anchor_end - excerpt_start
                if (relative_start < 0 or relative_end >= len(lines) or relative_end < relative_start):
                    bounded_item["content"] = ""
                else:
                    selected_start = relative_start
                    selected_end = relative_end
                    selected = "\n".join(lines[selected_start:selected_end + 1])
                    if not selected or len(selected) > allowed_characters:
                        bounded_item["content"] = ""
                    else:
                        while True:
                            before = selected_start - 1
                            after = selected_end + 1
                            choices = []
                            if before >= 0:
                                choices.append((before, selected_end))
                            if after < len(lines):
                                choices.append((selected_start, after))
                            expanded = None
                            for candidate_start, candidate_end in choices:
                                candidate_content = "\n".join(lines[candidate_start:candidate_end + 1])
                                if len(candidate_content) <= allowed_characters:
                                    expanded = candidate_start, candidate_end, candidate_content
                                    break
                            if expanded is None:
                                break
                            selected_start, selected_end, selected = expanded
                        bounded_item["content"] = selected
                        bounded_item["start_line"] = excerpt_start + selected_start
                        bounded_item["end_line"] = excerpt_start + selected_end
                        bounded_item["content_truncated"] = True
            else:
                bounded_item["content"] = _bounded_content(content, fraction)
        if str(bounded_item.get("content") or "").strip():
            bounded.append(bounded_item)
    return bounded


def _candidate_prompt_evidence_gaps(
    candidate: dict,
    available_evidence: list[dict],
    retrieval_requests: list[dict],
) -> tuple[bool, int]:
    """Report changed-anchor and required-context gaps in prompt-visible evidence."""
    visible_evidence_ids = {
        item.get("evidence_id") for item in available_evidence
        if str(item.get("content") or "").strip() and item.get("evidence_id")
    }
    missing_required_request_count = sum(
        1 for request in retrieval_requests
        if request.get("required") and (
            request.get("status") not in {"retrieved", "satisfied_by_changed_head"}
            or not request.get("evidence_id")
            or request.get("evidence_id") not in visible_evidence_ids
        )
    )
    candidate_side = candidate.get("side", "new")
    candidate_start = candidate.get("start_line")
    candidate_end = candidate.get("end_line")
    missing_changed_anchor = not any(
        item.get("source") in {"changed_head", "changed_patch"}
        and str(item.get("content") or "").strip()
        and item.get("path") == candidate.get("relevant_file")
        and item.get("side", "new") == candidate_side
        and isinstance(candidate_start, int)
        and isinstance(candidate_end, int)
        and isinstance(item.get("start_line"), int)
        and isinstance(item.get("end_line"), int)
        and item["start_line"] <= candidate_start
        and candidate_end <= item["end_line"]
        for item in available_evidence
    )
    return missing_changed_anchor, missing_required_request_count


def prompt_evidence_coverage(
    candidates: list[dict],
    evidence: list[dict],
    retrieval_requests: Optional[list[dict]] = None,
) -> dict:
    """Summarize whether every candidate retained its required prompt evidence."""
    evidence_by_candidate = {}
    for item in evidence:
        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = [item.get("candidate_id")]
        for candidate_id in candidate_ids:
            evidence_by_candidate.setdefault(candidate_id, []).append(item)
    requests_by_candidate = {}
    for request in retrieval_requests if isinstance(retrieval_requests, list) else []:
        if isinstance(request, dict):
            requests_by_candidate.setdefault(request.get("candidate_id"), []).append(request)

    missing_changed_candidate_count = 0
    missing_required_request_count = 0
    complete_candidate_count = 0
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        missing_changed, missing_required = _candidate_prompt_evidence_gaps(
            candidate,
            evidence_by_candidate.get(candidate_id, []),
            requests_by_candidate.get(candidate_id, []),
        )
        missing_changed_candidate_count += int(missing_changed)
        missing_required_request_count += missing_required
        complete_candidate_count += int(not missing_changed and not missing_required)

    status = (
        "complete"
        if complete_candidate_count == len(candidates)
        else "incomplete"
    )
    return {
        "status": status,
        "candidate_count": len(candidates),
        "complete_candidate_count": complete_candidate_count,
        "missing_changed_candidate_count": missing_changed_candidate_count,
        "missing_required_request_count": missing_required_request_count,
    }


def _verified_finding_identity(candidate: dict) -> Optional[tuple[str, str]]:
    """Derive identities internally from a verified assertion and its cited proof.

    The verifier cannot supply either identifier. The stable key ignores paths,
    line numbers, and identifier spelling so a file or symbol rename preserves it,
    while code shape keeps distinct verified defects from collapsing together.
    """
    anchor_shape = str(candidate.get("_changed_anchor_shape") or "")
    anchor_ordinal = candidate.get("_changed_anchor_ordinal")
    lineage_key = str(candidate.get("_trusted_lineage_key") or "")
    if (not anchor_shape or not lineage_key
            or not isinstance(anchor_ordinal, int) or anchor_ordinal < 1):
        return None
    root_digest = hashlib.sha256(
        json.dumps({
            "schema": "verified-root-cause-v1",
            "changed_anchor_shape": anchor_shape,
            "changed_anchor_ordinal": anchor_ordinal,
        },
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    root_cause_id = f"sha256:{root_digest}"
    stable_digest = hashlib.sha256(
        json.dumps({
            "schema": "verified-finding-stable-key-v1",
            "root_cause_id": root_cause_id,
            "trusted_lineage_key": lineage_key,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return root_cause_id, f"sha256:{stable_digest}"


def apply_verification_decisions(
    candidates: list[dict],
    evidence: list[dict],
    verification_data: dict,
    retrieval_requests: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Accept only complete verified decisions backed by retrieved evidence."""
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    evidence_by_candidate = {}
    for item in evidence:
        candidate_ids = item.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = [item.get("candidate_id")]
        for candidate_id in candidate_ids:
            evidence_by_candidate.setdefault(candidate_id, []).append(item)
    requests_by_candidate = {}
    for request in retrieval_requests if isinstance(retrieval_requests, list) else []:
        if isinstance(request, dict):
            requests_by_candidate.setdefault(request.get("candidate_id"), []).append(request)
    decisions = (verification_data.get("verification") or {}).get("decisions") or []
    verified_findings = []
    result_records = []
    seen_findings = set()
    seen_identities = set()
    decision_ids = [
        str(decision.get("candidate_id") or "").strip()
        for decision in decisions if isinstance(decision, dict)
    ] if isinstance(decisions, list) else []
    duplicate_decision_ids = {
        candidate_id for candidate_id in decision_ids if candidate_id and decision_ids.count(candidate_id) > 1
    }
    recorded_duplicate_ids = set()
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict):
            continue
        candidate_id = str(decision.get("candidate_id") or "").strip()
        if candidate_id in duplicate_decision_ids:
            if candidate_id not in recorded_duplicate_ids:
                result_records.append({
                    "candidate_id": candidate_id,
                    "verdict": "rejected",
                    "reason": "duplicate_decision",
                })
                recorded_duplicate_ids.add(candidate_id)
            continue
        candidate = candidates_by_id.get(candidate_id)
        verdict = str(decision.get("verdict") or "").strip().lower()
        record = {"candidate_id": candidate_id, "verdict": verdict or "invalid"}
        if candidate is None:
            record["reason"] = "unknown_candidate"
            result_records.append(record)
            continue
        if verdict != "verified":
            record["reason"] = str(decision.get("reason") or "rejected_by_verifier").strip()
            result_records.append(record)
            continue
        cited_paths = decision.get("evidence_paths") or []
        if isinstance(cited_paths, str):
            cited_paths = [cited_paths]
        available_evidence = evidence_by_candidate.get(candidate_id, [])
        _, missing_required_request_count = _candidate_prompt_evidence_gaps(
            candidate,
            available_evidence,
            requests_by_candidate.get(candidate_id, []),
        )
        if missing_required_request_count:
            record["verdict"] = "rejected"
            record["reason"] = "required_context_unavailable"
            result_records.append(record)
            continue
        candidate_side = candidate.get("side", "new")
        available_paths = {item.get("path") for item in available_evidence}
        normalized_citations = [safe_repo_path(path) for path in cited_paths]
        normalized_citations = [path for path in normalized_citations if path in available_paths]
        trigger = str(decision.get("trigger") or candidate.get("trigger") or "").strip()
        impact = str(decision.get("impact") or candidate.get("impact") or "").strip()
        explanation = str(decision.get("issue_content") or candidate.get("issue_content") or "").strip()
        relevant_file = safe_repo_path(decision.get("relevant_file") or candidate.get("relevant_file"))
        try:
            start_line = int(decision.get("start_line") or candidate.get("start_line"))
            end_line = int(decision.get("end_line") or candidate.get("end_line"))
        except (TypeError, ValueError):
            start_line = 0
            end_line = 0
        if not any(
            item.get("source") in {"changed_head", "changed_patch"}
            and str(item.get("content") or "").strip()
            and item.get("path") == candidate.get("relevant_file")
            and item.get("side", "new") == candidate_side
            and isinstance(item.get("start_line"), int)
            and isinstance(item.get("end_line"), int)
            and item["start_line"] <= start_line
            and end_line <= item["end_line"]
            for item in available_evidence
        ):
            record["verdict"] = "rejected"
            record["reason"] = "changed_code_evidence_unavailable"
            result_records.append(record)
            continue
        changed_line_ranges = candidate.get("_changed_line_ranges") or []
        trusted_side_line_count = candidate.get("_trusted_side_line_count")
        if (not normalized_citations or not trigger or not impact or not explanation or
                relevant_file != candidate.get("relevant_file") or start_line < 1 or end_line < start_line or
                (isinstance(trusted_side_line_count, int)
                 and end_line > trusted_side_line_count)):
            record["verdict"] = "rejected"
            record["reason"] = "unverified_or_incomplete_evidence"
            result_records.append(record)
            continue
        if not _line_is_changed(start_line, changed_line_ranges):
            record["verdict"] = "rejected"
            record["reason"] = "location_not_in_changed_lines"
            result_records.append(record)
            continue
        display_file = safe_repo_path(candidate.get("_display_file")) or relevant_file
        finding = {
            "_candidate_id": candidate_id,
            "relevant_file": display_file,
            "issue_header": str(decision.get("issue_header") or candidate.get("issue_header") or "Issue").strip(),
            "issue_content": (
                f"{explanation}\n\n**Trigger:** {trigger}\n\n**Impact:** {impact}\n\n"
                f"**Verified with:** {', '.join(f'`{path}`' for path in normalized_citations)}"
            ),
            "start_line": start_line,
            "end_line": end_line,
            "side": candidate_side,
            "trigger": trigger,
            "impact": impact,
            "verification_evidence": normalized_citations,
        }
        identity = _verified_finding_identity(candidate)
        if identity is None:
            record["verdict"] = "rejected"
            record["reason"] = "trusted_identity_unavailable"
            result_records.append(record)
            continue
        finding["root_cause_id"], finding["trusted_stable_key"] = identity
        if identity in seen_identities:
            record["verdict"] = "rejected"
            record["reason"] = "trusted_identity_collision"
            result_records.append(record)
            continue
        finding_key = _candidate_key(finding)
        if finding_key in seen_findings:
            record["verdict"] = "rejected"
            record["reason"] = "duplicate_verified_finding"
            result_records.append(record)
            continue
        seen_findings.add(finding_key)
        seen_identities.add(identity)
        verified_findings.append(finding)
        record["evidence_paths"] = normalized_citations
        result_records.append(record)

    decided_ids = {record["candidate_id"] for record in result_records}
    for candidate_id in candidates_by_id.keys() - decided_ids:
        result_records.append({"candidate_id": candidate_id, "verdict": "rejected", "reason": "missing_decision"})
    candidate_order = {candidate["candidate_id"]: index for index, candidate in enumerate(candidates)}
    verified_findings.sort(
        key=lambda finding: candidate_order.get(finding["_candidate_id"], len(candidate_order))
    )
    for finding in verified_findings:
        finding.pop("_candidate_id", None)
    get_logger().debug(
        "Candidate verification decisions validated",
        artifact=telemetry_safe_artifact({"decisions": result_records}),
    )
    return verified_findings, result_records
