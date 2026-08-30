"""Bounded repository-evidence retrieval and result validation for review candidates."""

import asyncio
import fnmatch
import json
import re
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

from pr_agent.log import get_logger

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class VerificationBudgets:
    max_candidates: int = 3
    max_files: int = 6
    max_lines_per_file: int = 160
    max_total_lines: int = 600
    max_context_tokens: int = 6000
    timeout_seconds: float = 10.0


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
    for line in (patch or "").splitlines():
        header = _HUNK_HEADER_RE.match(line)
        if header:
            new_line = int(header.group(1))
            continue
        if new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            return new_line
        if not line.startswith("-"):
            new_line += 1
    return 1


def _added_line_ranges(patch: str) -> list[tuple[int, int]]:
    added_lines = []
    new_line = None
    for line in (patch or "").splitlines():
        header = _HUNK_HEADER_RE.match(line)
        if header:
            new_line = int(header.group(1))
            continue
        if new_line is None or line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(new_line)
        if not line.startswith("-"):
            new_line += 1
    ranges = []
    for line in added_lines:
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], line)
        else:
            ranges.append((line, line))
    return ranges


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


def prepare_candidates(review_data: dict, diff_files: list, sensitive_globs: list,
                       max_candidates: int, max_sensitive_files: int) -> tuple[list[dict], list[dict]]:
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
    sensitive_count = 0
    for diff_file in diff_files or []:
        relevant_file = safe_repo_path(getattr(diff_file, "filename", ""))
        if not relevant_file or not any(fnmatch.fnmatch(relevant_file, pattern) for pattern in normalized_globs):
            continue
        if sensitive_count >= max_sensitive_files:
            rejected.append({"candidate_id": f"sensitive-{relevant_file}", "reason": "sensitive_file_budget_exhausted",
                             "sensitive_path": True})
            continue
        sensitive_count += 1
        line = _first_changed_line(getattr(diff_file, "patch", ""))
        candidates.append({
            "candidate_id": f"sensitive-{sensitive_count}",
            "candidate_type": "sensitive_path_audit",
            "relevant_file": relevant_file,
            "issue_header": "Sensitive path audit",
            "issue_content": "Independently inspect this configured sensitive path for introduced defects.",
            "trigger": "A configured sensitive path changed.",
            "impact": "A high-risk regression could otherwise be suppressed before verification.",
            "root_cause": f"sensitive audit for {relevant_file}",
            "start_line": line,
            "end_line": line,
            "context_files": [relevant_file],
            "context_symbols": [],
            "sensitive_path": True,
            "_changed_line_ranges": _added_line_ranges(getattr(diff_file, "patch", "")),
        })

    model_candidate_count = 0
    for index, raw_candidate in enumerate(raw_candidates):
        if model_candidate_count >= max_candidates:
            rejected.append({"candidate_id": f"candidate-{index + 1}", "reason": "candidate_budget_exhausted"})
            continue
        if not isinstance(raw_candidate, dict):
            rejected.append({"candidate_id": f"candidate-{index + 1}", "reason": "invalid_candidate"})
            continue
        candidate = dict(raw_candidate)
        candidate["candidate_id"] = f"candidate-{index + 1}"
        candidate["candidate_type"] = "model_finding"
        candidate["relevant_file"] = safe_repo_path(candidate.get("relevant_file"))
        try:
            candidate["start_line"] = int(candidate.get("start_line"))
            candidate["end_line"] = int(candidate.get("end_line"))
        except (TypeError, ValueError):
            candidate["start_line"] = 0
            candidate["end_line"] = 0
        required_text = ("issue_header", "issue_content", "trigger", "impact")
        diff_file = diff_by_file.get(candidate["relevant_file"])
        changed_line_ranges = _added_line_ranges(getattr(diff_file, "patch", "")) if diff_file else []
        if (not candidate["relevant_file"] or not diff_file or
                not _line_is_changed(candidate["start_line"], changed_line_ranges) or
                candidate["end_line"] < candidate["start_line"] or
                any(not str(candidate.get(key) or "").strip() for key in required_text)):
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": "invalid_candidate"})
            continue
        key = _candidate_key(candidate)
        if key in seen:
            rejected.append({"candidate_id": candidate["candidate_id"], "reason": "duplicate_candidate"})
            continue
        seen.add(key)
        candidate["_changed_line_ranges"] = changed_line_ranges
        candidates.append(candidate)
        model_candidate_count += 1
    return candidates, rejected


def _requested_paths(candidate: dict) -> list[str]:
    paths = [candidate.get("relevant_file")]
    context_files = candidate.get("context_files") or []
    if isinstance(context_files, str):
        context_files = [context_files]
    if isinstance(context_files, list):
        paths.extend(context_files)
    normalized = []
    for value in paths:
        path = safe_repo_path(value)
        if path and path not in normalized:
            normalized.append(path)
    return normalized


def _select_excerpt(content: str, candidate: dict, path: str, max_lines: int) -> tuple[str, int, int]:
    lines = content.splitlines()
    if not lines or max_lines < 1:
        return "", 0, 0
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
    center = max(1, min(len(lines), int(center or 1)))
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


async def retrieve_evidence(git_provider, candidates: list[dict], budgets: VerificationBudgets,
                            static_evidence: list, diff_files: Optional[list] = None) -> tuple[list[dict], dict]:
    """Fetch bounded base-branch excerpts and return them with an auditable retrieval record."""
    started = time.monotonic()
    deadline = started + max(0.0, budgets.timeout_seconds)
    evidence = []
    requests = []
    cache = {}
    unique_files = set()
    total_lines = 0
    total_chars = 0
    max_chars = max(0, budgets.max_context_tokens) * 4
    budget_exhausted = False
    changed_evidence_count = 0
    diff_by_file = {
        path: diff_file
        for diff_file in (diff_files or [])
        if (path := safe_repo_path(getattr(diff_file, "filename", "")))
    }

    for candidate in candidates:
        diff_file = diff_by_file.get(candidate.get("relevant_file"))
        head_file = getattr(diff_file, "head_file", "") if diff_file is not None else ""
        if head_file and getattr(diff_file, "head_file_is_complete", True):
            available_lines = min(budgets.max_lines_per_file, budgets.max_total_lines - total_lines)
            excerpt, start_line, end_line = _select_excerpt(
                str(head_file), candidate, candidate.get("relevant_file"), available_lines
            )
            available_chars = max_chars - total_chars
            if len(excerpt) > available_chars:
                excerpt = excerpt[:available_chars]
                budget_exhausted = True
            if excerpt:
                evidence.append({
                    "candidate_id": candidate["candidate_id"],
                    "source": "changed_head",
                    "path": candidate["relevant_file"],
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": excerpt,
                })
                line_count = excerpt.count("\n") + 1
                total_lines += line_count
                total_chars += len(excerpt)
                changed_evidence_count += 1
        for static_item in _matching_static_evidence(candidate, static_evidence):
            if total_chars + len(static_item["content"]) > max_chars:
                budget_exhausted = True
                break
            evidence_item = {"candidate_id": candidate["candidate_id"], **static_item}
            evidence.append(evidence_item)
            total_chars += len(static_item["content"])
        for path in _requested_paths(candidate):
            request = {"candidate_id": candidate["candidate_id"], "path": path, "status": "pending"}
            requests.append(request)
            if path not in unique_files and len(unique_files) >= budgets.max_files:
                request["status"] = "file_budget_exhausted"
                budget_exhausted = True
                continue
            if total_lines >= budgets.max_total_lines or total_chars >= max_chars:
                request["status"] = "context_budget_exhausted"
                budget_exhausted = True
                continue
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                request["status"] = "time_budget_exhausted"
                budget_exhausted = True
                continue
            unique_files.add(path)
            if path not in cache:
                try:
                    cache[path] = await asyncio.wait_for(
                        asyncio.to_thread(git_provider.get_repo_file_content, path, False),
                        timeout=remaining_time,
                    )
                except asyncio.TimeoutError:
                    request["status"] = "time_budget_exhausted"
                    budget_exhausted = True
                    continue
                except Exception as exc:
                    cache[path] = None
                    request["status"] = "fetch_failed"
                    request["error"] = type(exc).__name__
                    continue
            content = cache[path]
            if not content:
                request["status"] = "missing"
                continue
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            available_lines = min(budgets.max_lines_per_file, budgets.max_total_lines - total_lines)
            excerpt, start_line, end_line = _select_excerpt(str(content), candidate, path, available_lines)
            available_chars = max_chars - total_chars
            if len(excerpt) > available_chars:
                excerpt = excerpt[:available_chars]
                budget_exhausted = True
            if not excerpt:
                request["status"] = "context_budget_exhausted"
                budget_exhausted = True
                continue
            evidence.append({
                "candidate_id": candidate["candidate_id"],
                "source": "repository_file",
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "content": excerpt,
            })
            request.update({"status": "retrieved", "start_line": start_line, "end_line": end_line})
            line_count = excerpt.count("\n") + 1
            total_lines += line_count
            total_chars += len(excerpt)

    artifact = {
        "requests": requests,
        "retrieved_evidence": evidence,
        "budget_exhausted": budget_exhausted,
        "files_read": len(unique_files),
        "changed_evidence_count": changed_evidence_count,
        "lines_retrieved": total_lines,
        "estimated_context_tokens": (total_chars + 3) // 4,
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

    def bounded_content(value, fraction: float) -> str:
        value = str(value or "")
        if fraction >= 1.0 or not value:
            return value
        kept_characters = int(len(value) * fraction)
        if kept_characters < 1:
            return ""
        if kept_characters >= len(value):
            return value
        return f"{value[:kept_characters]}\n...(truncated)"

    bounded_evidence = []
    for item in evidence:
        bounded_item = dict(item)
        if "content" in bounded_item:
            bounded_item["content"] = bounded_content(bounded_item["content"], content_fraction)
        bounded_evidence.append(bounded_item)
    return json.dumps({"candidates": candidates,
                       "changed_diff": bounded_content(changed_diff, changed_diff_fraction),
                       "evidence": bounded_evidence},
                      ensure_ascii=False, indent=2)


def apply_verification_decisions(candidates: list[dict], evidence: list[dict], verification_data: dict
                                 ) -> tuple[list[dict], list[dict]]:
    """Accept only complete verified decisions backed by retrieved evidence."""
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    evidence_by_candidate = {}
    for item in evidence:
        evidence_by_candidate.setdefault(item.get("candidate_id"), []).append(item)
    decisions = (verification_data.get("verification") or {}).get("decisions") or []
    verified_findings = []
    result_records = []
    seen_findings = set()
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
        changed_line_ranges = candidate.get("_changed_line_ranges") or []
        if (not normalized_citations or not trigger or not impact or not explanation or
                relevant_file != candidate.get("relevant_file") or start_line < 1 or end_line < start_line):
            record["verdict"] = "rejected"
            record["reason"] = "unverified_or_incomplete_evidence"
            result_records.append(record)
            continue
        if not _line_is_changed(start_line, changed_line_ranges):
            record["verdict"] = "rejected"
            record["reason"] = "location_not_in_changed_lines"
            result_records.append(record)
            continue
        finding = {
            "relevant_file": relevant_file,
            "issue_header": str(decision.get("issue_header") or candidate.get("issue_header") or "Issue").strip(),
            "issue_content": (
                f"{explanation}\n\n**Trigger:** {trigger}\n\n**Impact:** {impact}\n\n"
                f"**Verified with:** {', '.join(f'`{path}`' for path in normalized_citations)}"
            ),
            "start_line": start_line,
            "end_line": end_line,
            "trigger": trigger,
            "impact": impact,
            "verification_evidence": normalized_citations,
        }
        finding_key = _candidate_key(finding)
        if finding_key in seen_findings:
            record["verdict"] = "rejected"
            record["reason"] = "duplicate_verified_finding"
            result_records.append(record)
            continue
        seen_findings.add(finding_key)
        verified_findings.append(finding)
        record["evidence_paths"] = normalized_citations
        result_records.append(record)

    decided_ids = {record["candidate_id"] for record in result_records}
    for candidate_id in candidates_by_id.keys() - decided_ids:
        result_records.append({"candidate_id": candidate_id, "verdict": "rejected", "reason": "missing_decision"})
    get_logger().debug("Candidate verification decisions validated", artifact={"decisions": result_records})
    return verified_findings, result_records
