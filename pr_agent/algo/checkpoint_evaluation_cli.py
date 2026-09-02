"""Credential-free CLI support for checkpoint evaluation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pr_agent.algo.checkpoint_evaluation import (EvaluationManifest,
                                                 EvaluationValidationError,
                                                 TruthArtifact,
                                                 build_evaluation_plan)
from pr_agent.algo.checkpoint_evaluation_cocos import (
    CocosCorpusLock, validate_cocos_story_corpus)

_MAX_EVALUATION_ARTIFACT_BYTES = 10_000_000


def _load_json_object(path: str, label: str) -> Mapping[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise EvaluationValidationError(f"{label} must name an existing local file")
    try:
        size = candidate.stat().st_size
    except OSError as exc:
        raise EvaluationValidationError(f"could not inspect {label}: {exc}") from exc
    if size > _MAX_EVALUATION_ARTIFACT_BYTES:
        raise EvaluationValidationError(
            f"{label} exceeds the {_MAX_EVALUATION_ARTIFACT_BYTES}-byte validation limit"
        )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(f"could not load {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{label} must contain one JSON object")
    return value


def load_evaluation_manifest(path: str) -> EvaluationManifest:
    try:
        return EvaluationManifest.from_dict(_load_json_object(path, "--manifest"))
    except EvaluationValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationValidationError(f"--manifest does not match the evaluation schema: {exc}") from exc


def load_truth_artifact(path: str, manifest: EvaluationManifest) -> TruthArtifact:
    try:
        artifact = TruthArtifact.from_dict(_load_json_object(path, "--truth"))
    except EvaluationValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationValidationError(f"--truth does not match the truth schema: {exc}") from exc
    artifact.validate_for_manifest(manifest)
    return artifact


def evaluation_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pr-agent evaluation-plan")
    parser.add_argument("--manifest", required=True, help="Local immutable evaluation manifest JSON")
    parser.add_argument(
        "--truth",
        default=None,
        help="Optional local answer-only truth JSON to validate separately from the plan",
    )
    parser.add_argument(
        "--cocos-corpus-root",
        default=None,
        help="Optional path to the external Cocos Story corpus; requires --cocos-lock",
    )
    parser.add_argument(
        "--cocos-lock",
        default=None,
        help="Approved immutable hash lock for the external Cocos Story corpus",
    )
    parser.add_argument(
        "--checkpoint-controls",
        default=None,
        help="Optional separate 15-20 case intermediate-checkpoint control ledger",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="List validated cases and enabled arms")
    mode.add_argument("--dry-run", action="store_true", help="Print the deterministic paired execution plan")
    return parser


def run_evaluation_plan(inargs: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Validate and list a local plan without importing or invoking model clients."""
    parser = evaluation_plan_parser()
    args = parser.parse_args(inargs)
    try:
        manifest = load_evaluation_manifest(args.manifest)
        truth = load_truth_artifact(args.truth, manifest) if args.truth else None
        if bool(args.cocos_corpus_root) != bool(args.cocos_lock):
            raise EvaluationValidationError("--cocos-corpus-root and --cocos-lock must be supplied together")
        if args.checkpoint_controls and not args.cocos_corpus_root:
            raise EvaluationValidationError("--checkpoint-controls requires the Cocos external corpus adapter")
        external_corpus = None
        if args.cocos_corpus_root:
            lock = CocosCorpusLock.from_dict(_load_json_object(args.cocos_lock, "--cocos-lock"))
            external_corpus = validate_cocos_story_corpus(
                args.cocos_corpus_root,
                lock,
                checkpoint_controls_path=args.checkpoint_controls,
            )
            if manifest.corpus_hash != external_corpus.corpus_hash:
                raise EvaluationValidationError(
                    "evaluation manifest corpus_hash does not match the validated Cocos corpus"
                )
    except EvaluationValidationError as exc:
        parser.error(str(exc))

    if args.list:
        payload = {
            "schema_version": manifest.schema_version,
            "schema_hash": manifest.schema_hash,
            "manifest_id": manifest.manifest_id,
            "network_calls": 0,
            "model_calls": 0,
            "cases": [
                {
                    "case_id": case.case_id,
                    "snapshot_id": case.snapshot_id,
                    "event": case.event.value,
                    "cohort": case.cohort.value,
                }
                for case in sorted(manifest.cases, key=lambda item: item.case_id)
            ],
            "arms": [
                arm.to_dict()
                for arm in sorted((item for item in manifest.arms if item.enabled), key=lambda item: item.arm_id)
            ],
        }
    else:
        payload = build_evaluation_plan(manifest).to_dict()
    if truth is not None:
        payload["truth_validation"] = {
            "status": "valid",
            "truth_artifact_id": truth.truth_artifact_id,
        }
    if external_corpus is not None:
        payload["external_corpus"] = external_corpus.to_dict()
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2))
    return payload
