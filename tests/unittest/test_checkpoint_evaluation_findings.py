import hashlib
import json

import pytest

from pr_agent.algo.checkpoint_evaluation import (
    EvaluationValidationError,
    FindingLifecycleState,
    FindingSeverity,
    ObservedFinding,
)
from pr_agent.algo.checkpoint_evaluation_findings import (
    CHECKPOINT_FINDING_NORMALIZATION_SCHEMA_VERSION,
    FRONTIER_FINDING_STAGE,
    GENERAL_REVIEW_FINDING_STAGE,
    VERIFIED_FINDING_STAGE,
    carry_forward_active_findings,
    derive_finding_lifecycle,
    normalize_frontier_findings,
    normalize_general_review_findings,
    normalize_verified_findings,
)


def _general(**overrides):
    finding = {
        "relevant_file": "src/service.py",
        "root_cause": "Cache key omits tenant identifier",
        "start_line": 12,
        "end_line": 14,
        "side": "new",
        "symbol": "load_record",
        "normalized_severity": "high",
        "issue_header": "Authorization bug",
        "issue_content": "Mutable explanation.",
        "trigger": "Mutable trigger.",
        "impact": "Mutable impact.",
    }
    finding.update(overrides)
    return finding


def _verified(**overrides):
    finding = {
        "root_cause_id": f"sha256:{'a' * 64}",
        "trusted_stable_key": f"sha256:{'b' * 64}",
        "relevant_file": "src/service.py",
        "start_line": 12,
        "end_line": 14,
        "side": "new",
        "normalized_severity": "high",
        "issue_content": "Mutable verified explanation.",
    }
    finding.update(overrides)
    return finding


def _observation(
    fingerprint: str,
    *,
    severity: FindingSeverity = FindingSeverity.HIGH,
    lifecycle_state: FindingLifecycleState = FindingLifecycleState.ACTIVE,
    stage: str = GENERAL_REVIEW_FINDING_STAGE,
    deterministic_overlap=None,
):
    return ObservedFinding(
        fingerprint=fingerprint,
        severity=severity,
        lifecycle_state=lifecycle_state,
        stage=stage,
        deterministic_overlap=deterministic_overlap,
    )


def test_general_review_fingerprint_uses_versioned_canonical_identity():
    observation = normalize_general_review_findings([_general()])[0]
    identity = {
        "schema_version": CHECKPOINT_FINDING_NORMALIZATION_SCHEMA_VERSION,
        "source": "general_review",
        "root_cause": "cache key omits tenant identifier",
    }
    expected = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert observation == ObservedFinding(
        fingerprint=f"sha256:{expected}",
        severity=FindingSeverity.HIGH,
        lifecycle_state=FindingLifecycleState.ACTIVE,
        stage=GENERAL_REVIEW_FINDING_STAGE,
    )


def test_general_review_fingerprint_ignores_mutable_prose_and_response_order():
    first = _general()
    second = _general(
        relevant_file="src/other.py",
        root_cause="A different defect",
        start_line=30,
        end_line=30,
        symbol=None,
        normalized_severity="medium",
    )
    original = normalize_general_review_findings([first, second])
    rewritten = normalize_general_review_findings([
        {**second, "issue_header": "Reworded", "issue_content": "Completely different prose."},
        {
            **first,
            "issue_header": "Different title",
            "issue_content": "Different explanation",
            "trigger": "Different trigger",
            "impact": "Different impact",
            "confidence": 0.1,
        },
    ])

    assert original == rewritten
    assert tuple(item.fingerprint for item in original) == tuple(sorted(item.fingerprint for item in original))


def test_general_review_normalizes_cosmetic_identity_differences():
    original = normalize_general_review_findings([_general()])[0]
    cosmetic = normalize_general_review_findings([_general(
        relevant_file=" src/service.py ",
        root_cause="  CACHE key\n omits tenant identifier  ",
        side=" NEW ",
        symbol="  load_record  ",
    )])[0]

    assert cosmetic.fingerprint == original.fingerprint


def test_general_review_fingerprint_changes_with_stable_root_cause():
    original = normalize_general_review_findings([_general()])[0]

    changed = normalize_general_review_findings([_general(root_cause="Cache key includes the wrong tenant")])[0]

    assert changed.fingerprint != original.fingerprint


@pytest.mark.parametrize(
    "field,value",
    [
        ("relevant_file", "src/other.py"),
        ("side", "old"),
        ("start_line", 13),
        ("end_line", 15),
        ("symbol", "store_record"),
    ],
)
def test_general_review_stable_root_cause_survives_location_movement(field, value):
    original = normalize_general_review_findings([_general()])[0]

    moved = normalize_general_review_findings([_general(**{field: value})])[0]

    assert moved.fingerprint == original.fingerprint


@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
def test_general_review_accepts_the_exact_severity_vocabulary(severity):
    observation = normalize_general_review_findings([_general(normalized_severity=severity.upper())])[0]

    assert observation.severity is FindingSeverity(severity)


def test_general_review_accepts_a_trusted_severity_enum():
    finding = _general()
    finding.pop("normalized_severity")
    finding["severity"] = FindingSeverity.CRITICAL

    assert normalize_general_review_findings([finding])[0].severity is FindingSeverity.CRITICAL


def test_general_review_rejects_conflicting_severity_fields():
    with pytest.raises(EvaluationValidationError, match="cannot conflict"):
        normalize_general_review_findings([_general(severity="medium")])


@pytest.mark.parametrize("severity", [None, "", "urgent", 2, True])
def test_general_review_rejects_missing_or_unknown_explicit_severity(severity):
    with pytest.raises(EvaluationValidationError, match="severity"):
        normalize_general_review_findings([_general(normalized_severity=severity)])


@pytest.mark.parametrize("field", ["fingerprint", "lifecycle_state", "stage", "deterministic_overlap"])
def test_model_output_cannot_control_derived_observation_fields(field):
    with pytest.raises(EvaluationValidationError, match="cannot control derived fields"):
        normalize_general_review_findings([_general(**{field: "model-value"})])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"root_cause": ""}, "root_cause"),
        ({"root_cause": None}, "root_cause"),
        ({"relevant_file": "../secret"}, "relevant_file"),
        ({"relevant_file": "/src/service.py"}, "relevant_file"),
        ({"relevant_file": "src//service.py"}, "relevant_file"),
        ({"relevant_file": "src\\service.py"}, "relevant_file"),
        ({"relevant_file": ""}, "relevant_file"),
        ({"side": "RIGHT"}, "side"),
        ({"start_line": True}, "start_line"),
        ({"start_line": 0}, "start_line"),
        ({"end_line": 11}, "cannot precede"),
        ({"symbol": ""}, "symbol"),
    ],
)
def test_general_review_rejects_incomplete_or_unsafe_identity(overrides, error):
    with pytest.raises(EvaluationValidationError, match=error):
        normalize_general_review_findings([_general(**overrides)])


def test_general_review_defaults_a_missing_side_to_new():
    finding = _general()
    finding.pop("side")

    assert normalize_general_review_findings([finding]) == normalize_general_review_findings([_general(side="new")])


def test_actual_legacy_general_review_shape_fails_closed_without_durable_metadata():
    finding = {
        "relevant_file": "src/service.py",
        "issue_header": "Possible Bug",
        "issue_content": "The new call dereferences a missing result.",
        "start_line": 12,
        "end_line": 12,
    }

    with pytest.raises(EvaluationValidationError, match="root_cause"):
        normalize_general_review_findings([finding])


def test_actual_general_review_shape_accepts_caller_owned_root_cause_and_severity():
    finding = {
        "relevant_file": "src/service.py",
        "issue_header": "Possible Bug",
        "issue_content": "The new call dereferences a missing result.",
        "start_line": 12,
        "end_line": 12,
    }
    moved = {**finding, "relevant_file": "src/moved.py", "start_line": 80, "end_line": 80}

    first = normalize_general_review_findings(
        [finding],
        severity_by_index={0: FindingSeverity.HIGH},
        stable_root_cause_by_index={0: "Unchecked optional result"},
    )[0]
    second = normalize_general_review_findings(
        [moved], severity_by_index={0: "high"}, stable_root_cause_by_index={0: "unchecked optional result"}
    )[0]

    assert first.fingerprint == second.fingerprint
    assert first.severity is FindingSeverity.HIGH


def test_actual_bugs_only_shape_requires_caller_owned_severity():
    finding = {
        "relevant_file": "src/cache.py",
        "issue_header": "Security",
        "issue_content": "The new cache key can reuse another tenant's data.",
        "start_line": 12,
        "end_line": 14,
        "finding_type": "security",
        "duplicates_ci_failure": False,
        "matching_ci_failure": "",
        "trigger": "Two tenants request one record identifier.",
        "impact": "The second tenant receives the first tenant's data.",
        "root_cause": "Cache key omits tenant identifier.",
    }

    with pytest.raises(EvaluationValidationError, match="severity"):
        normalize_general_review_findings([finding])

    observation = normalize_general_review_findings([finding], severity_by_index={0: "medium"})[0]

    assert observation.fingerprint.startswith("sha256:")
    assert observation.severity is FindingSeverity.MEDIUM


@pytest.mark.parametrize("header", ["[P0] Critical bypass", "P1 regression", "Possible bug", "[P3] Cleanup"])
def test_general_review_does_not_infer_severity_from_model_header(header):
    finding = _general(issue_header=header)
    finding.pop("normalized_severity")

    with pytest.raises(EvaluationValidationError, match="severity"):
        normalize_general_review_findings([finding])


@pytest.mark.parametrize(
    "mapping,error",
    [
        ({1: "high"}, "valid finding indexes"),
        ({True: "high"}, "valid finding indexes"),
        ({0: "urgent"}, "severity"),
    ],
)
def test_general_review_rejects_invalid_caller_owned_severity_mapping(mapping, error):
    with pytest.raises(EvaluationValidationError, match=error):
        normalize_general_review_findings([_general()], severity_by_index=mapping)


def test_general_review_preserves_duplicate_fingerprints_for_duplicate_scoring():
    observations = normalize_general_review_findings([_general(), _general(issue_content="Different prose")])

    assert len(observations) == 2
    assert observations[0].fingerprint == observations[1].fingerprint


def test_general_review_preserves_duplicate_fingerprints_with_different_metadata():
    observations = normalize_general_review_findings([
        _general(normalized_severity="high"),
        _general(normalized_severity="low"),
    ])

    assert len(observations) == 2
    assert observations[0].fingerprint == observations[1].fingerprint
    assert {item.severity for item in observations} == {FindingSeverity.HIGH, FindingSeverity.LOW}


def test_verified_finding_reuses_exact_trusted_stable_key_across_location_and_wording_changes():
    stable_key = f"sha256:{'b' * 64}"
    before = normalize_verified_findings([_verified()])[0]
    after = normalize_verified_findings([_verified(
        relevant_file="src/renamed.py",
        start_line=90,
        end_line=91,
        side="old",
        issue_content="Reworded after verification.",
    )])[0]

    assert before.fingerprint == after.fingerprint == stable_key
    assert before.stage == after.stage == VERIFIED_FINDING_STAGE


def test_verified_finding_without_stable_key_uses_location_independent_root_cause_identity():
    finding = _verified(trusted_stable_key=None)
    first = normalize_verified_findings([finding])[0]
    second = normalize_verified_findings([{**finding, "issue_content": "Reworded"}])[0]
    moved = normalize_verified_findings([{**finding, "relevant_file": "src/moved.py"}])[0]

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.startswith("sha256:")
    assert moved.fingerprint == first.fingerprint


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"root_cause_id": "model-id"}, "root_cause_id"),
        ({"root_cause_id": None}, "root_cause_id"),
        ({"trusted_stable_key": "model-key"}, "trusted_stable_key"),
        ({"trusted_stable_key": f"sha256:{'B' * 64}"}, "trusted_stable_key"),
    ],
)
def test_verified_finding_rejects_untrusted_identity_values(overrides, error):
    with pytest.raises(EvaluationValidationError, match=error):
        normalize_verified_findings([_verified(**overrides)])


def test_verified_finding_preserves_duplicate_trusted_keys():
    observations = normalize_verified_findings([_verified(), _verified(relevant_file="src/other.py")])

    assert len(observations) == 2
    assert observations[0].fingerprint == observations[1].fingerprint


def test_actual_issue_9_verified_shape_accepts_caller_owned_decision_severity():
    stable_key = f"sha256:{'b' * 64}"
    finding = {
        "root_cause_id": f"sha256:{'a' * 64}",
        "trusted_stable_key": stable_key,
        "relevant_file": "src/auth.py",
        "issue_header": "Authorization bypass",
        "issue_content": "The authorization guard can be skipped.",
        "trigger": "Pass an unowned object id.",
        "impact": "Another tenant's object is returned.",
        "verification_evidence": ["src/auth.py"],
        "start_line": 10,
        "end_line": 10,
        "side": "new",
        "_trusted_anchor_shape_id": f"sha256:{'c' * 64}",
        "_trusted_anchor_shape_occurrence_count": 1,
        "_trusted_same_anchor_candidate_count": 1,
        "_trusted_patch_is_complete": True,
    }

    observation = normalize_verified_findings(
        [finding], severity_by_fingerprint={stable_key: "critical"}
    )[0]

    assert observation == ObservedFinding(
        fingerprint=stable_key,
        severity=FindingSeverity.CRITICAL,
        lifecycle_state=FindingLifecycleState.ACTIVE,
        stage=VERIFIED_FINDING_STAGE,
    )


def test_actual_issue_9_verified_shape_fails_closed_without_decision_severity_map():
    finding = _verified()
    finding.pop("normalized_severity")

    with pytest.raises(EvaluationValidationError, match="severity"):
        normalize_verified_findings([finding])


def test_frontier_telemetry_shape_keeps_only_confirmed_findings():
    confirmed_key = f"sha256:{'a' * 64}"
    results = [{
        "schema_version": "frontier-adjudication-output-v1",
        "decision": "confirm",
        "state": "confirmed",
        "confidence": 0.9,
        "evidence_citation_count": 1,
        "stable_finding_id": confirmed_key,
        "normalized_severity": "high",
        "failure_reason": None,
        "telemetry": {},
        "publication_safe": False,
    }, {
        "schema_version": "frontier-adjudication-output-v1",
        "decision": "reject",
        "state": "rejected",
        "confidence": 0.9,
        "evidence_citation_count": 1,
        "stable_finding_id": f"sha256:{'b' * 64}",
        "normalized_severity": None,
        "failure_reason": None,
        "telemetry": {},
        "publication_safe": False,
    }, {
        "stable_finding_id": None,
        "state": "unavailable",
        "failure_reason": "trusted_candidate_identity_unavailable",
        "publication_safe": False,
    }]

    assert normalize_frontier_findings(results) == (ObservedFinding(
        fingerprint=confirmed_key,
        severity=FindingSeverity.HIGH,
        lifecycle_state=FindingLifecycleState.ACTIVE,
        stage=FRONTIER_FINDING_STAGE,
    ),)


@pytest.mark.parametrize(
    "state,decision,failure_reason",
    [
        ("rejected", "reject", None),
        ("unavailable", "unavailable", "model_unavailable"),
        ("stale", "unavailable", "snapshot_changed"),
        ("malformed_output", "unavailable", "malformed_output"),
        ("timeout", "unavailable", "stage_timeout"),
        ("provider_failure", "unavailable", "provider_failure"),
        ("not_required", "unavailable", "escalation_not_required"),
    ],
)
def test_frontier_accepts_every_production_non_confirmed_state_pair(state, decision, failure_reason):
    result = {
        "schema_version": "frontier-adjudication-output-v1",
        "decision": decision,
        "state": state,
        "stable_finding_id": f"sha256:{'a' * 64}",
        "normalized_severity": None,
        "failure_reason": failure_reason,
        "publication_safe": False,
    }

    assert normalize_frontier_findings([result]) == ()


def test_normalized_frontier_finding_shape_is_accepted_without_location_in_identity():
    stable_key = f"sha256:{'a' * 64}"
    finding = {
        "schema_version": "normalized-review-finding-v1",
        "stable_finding_id": stable_key,
        "root_cause_id": f"sha256:{'b' * 64}",
        "severity": "critical",
        "location": {"path": "src/auth.py", "side": "new", "start_line": 10, "end_line": 10},
        "evidence_citations": ["evidence-1"],
    }

    assert normalize_frontier_findings([finding])[0].fingerprint == stable_key


def test_frontier_preserves_duplicate_confirmations_for_duplicate_scoring():
    result = {
        "schema_version": "frontier-adjudication-output-v1",
        "decision": "confirm",
        "state": "confirmed",
        "stable_finding_id": f"sha256:{'a' * 64}",
        "normalized_severity": "high",
    }

    observations = normalize_frontier_findings([result, result])

    assert len(observations) == 2
    assert observations[0] == observations[1]


def test_frontier_preserves_duplicate_confirmations_with_different_severity():
    stable_key = f"sha256:{'a' * 64}"
    high = {
        "decision": "confirm",
        "state": "confirmed",
        "schema_version": "frontier-adjudication-output-v1",
        "stable_finding_id": stable_key,
        "normalized_severity": "high",
    }
    low = {**high, "normalized_severity": "low"}

    observations = normalize_frontier_findings([high, low])

    assert len(observations) == 2
    assert {item.severity for item in observations} == {FindingSeverity.HIGH, FindingSeverity.LOW}


@pytest.mark.parametrize(
    "result,error",
    [
        ({"state": "invented"}, "synthetic"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "confirmed",
            "decision": "reject",
        }, "inconsistent"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "confirmed",
            "decision": "confirm",
            "stable_finding_id": "model-id",
        }, "sha256"),
        ({
            "state": "confirmed",
            "decision": "confirm",
            "schema_version": "frontier-adjudication-output-v1",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": "urgent",
        }, "severity"),
        ({"state": "confirmed", "schema_version": "unknown"}, "schema"),
        ({"schema_version": "unknown"}, "schema"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "rejected",
            "decision": "confirm",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": None,
        }, "inconsistent"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "rejected",
            "decision": "reject",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": "low",
        }, "cannot assign severity"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "timeout",
            "decision": "reject",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": None,
        }, "inconsistent"),
        ({
            "state": "timeout",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "failure_reason": "stage_timeout_exhausted",
            "publication_safe": False,
            "decision": "unavailable",
        }, "synthetic"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "unavailable",
            "decision": "unavailable",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": None,
            "failure_reason": None,
        }, "requires a failure reason"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "rejected",
            "decision": "reject",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": None,
            "failure_reason": "rejected",
        }, "cannot assign a failure reason"),
        ({
            "schema_version": "frontier-adjudication-output-v1",
            "state": "confirmed",
            "decision": "confirm",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "normalized_severity": "high",
            "failure_reason": "unexpected",
        }, "cannot assign a failure reason"),
        ({
            "schema_version": "normalized-review-finding-v1",
            "stable_finding_id": f"sha256:{'a' * 64}",
            "root_cause_id": f"sha256:{'b' * 64}",
            "severity": "high",
            "location": {"path": "/src/auth.py", "side": "new", "start_line": 10, "end_line": 10},
        }, "repository-relative"),
    ],
)
def test_frontier_rejects_malformed_production_shapes(result, error):
    with pytest.raises(EvaluationValidationError, match=error):
        normalize_frontier_findings([result])


@pytest.mark.parametrize("stage", ["", "General Review", "1general", "general-review", "a" * 65])
def test_normalizers_reject_invalid_caller_owned_stage(stage):
    with pytest.raises(EvaluationValidationError, match="stage"):
        normalize_general_review_findings([_general()], stage=stage)


def test_normalizers_reject_non_mapping_findings():
    with pytest.raises(EvaluationValidationError, match="must be a mapping"):
        normalize_general_review_findings(["not-a-mapping"])


@pytest.mark.parametrize("findings", [None, "model-output", {"finding": "mapping"}])
def test_normalizers_reject_non_sequence_batches(findings):
    with pytest.raises(EvaluationValidationError, match="must be a sequence"):
        normalize_general_review_findings(findings)


def test_lifecycle_marks_only_missing_parent_active_findings_withdrawn():
    continuing = _observation(f"sha256:{'a' * 64}")
    missing = _observation(f"sha256:{'b' * 64}", severity=FindingSeverity.MEDIUM)
    historical = _observation(
        f"sha256:{'c' * 64}",
        lifecycle_state=FindingLifecycleState.WITHDRAWN,
    )
    new = _observation(f"sha256:{'d' * 64}", severity=FindingSeverity.CRITICAL)

    result = derive_finding_lifecycle(
        [new, continuing],
        [historical, missing, continuing],
        arm_id="arm-general",
        parent_arm_id="arm-general",
    )

    assert result == (
        continuing,
        _observation(
            f"sha256:{'b' * 64}",
            severity=FindingSeverity.MEDIUM,
            lifecycle_state=FindingLifecycleState.WITHDRAWN,
        ),
        new,
    )


def test_partial_lifecycle_carries_unresolved_findings_until_complete_checkpoint():
    continuing = _observation(f"sha256:{'a' * 64}")
    unresolved = _observation(f"sha256:{'b' * 64}", severity=FindingSeverity.MEDIUM)

    partial_middle = carry_forward_active_findings(
        [continuing],
        [continuing, unresolved],
        arm_id="arm-general",
        parent_arm_id="arm-general",
    )
    complete_child = derive_finding_lifecycle(
        [continuing],
        partial_middle,
        arm_id="arm-general",
        parent_arm_id="arm-general",
    )

    carried = _observation(
        unresolved.fingerprint,
        severity=FindingSeverity.MEDIUM,
        lifecycle_state=FindingLifecycleState.CARRIED_FORWARD,
    )
    assert partial_middle == (continuing, carried)
    assert complete_child == (
        continuing,
        _observation(
            unresolved.fingerprint,
            severity=FindingSeverity.MEDIUM,
            lifecycle_state=FindingLifecycleState.WITHDRAWN,
        ),
    )


def test_partial_lifecycle_keeps_carried_findings_distinct_from_current_observations():
    observed = _observation(f"sha256:{'a' * 64}")
    inherited = _observation(f"sha256:{'b' * 64}", severity=FindingSeverity.MEDIUM)

    partial = carry_forward_active_findings(
        [observed],
        [inherited],
        arm_id="arm-general",
        parent_arm_id="arm-general",
    )

    assert [finding for finding in partial if finding.lifecycle_state is FindingLifecycleState.ACTIVE] == [observed]
    assert partial[1].lifecycle_state is FindingLifecycleState.CARRIED_FORWARD


def test_lifecycle_does_not_repeat_historical_withdrawals():
    historical = _observation(
        f"sha256:{'a' * 64}",
        lifecycle_state=FindingLifecycleState.WITHDRAWN,
    )

    assert derive_finding_lifecycle(
        [], [historical], arm_id="arm-general", parent_arm_id="arm-general"
    ) == ()


def test_lifecycle_allows_a_previously_withdrawn_finding_to_reappear_active():
    active = _observation(f"sha256:{'a' * 64}")
    withdrawn = _observation(
        active.fingerprint,
        lifecycle_state=FindingLifecycleState.WITHDRAWN,
    )

    assert derive_finding_lifecycle(
        [active], [withdrawn], arm_id="arm-general", parent_arm_id="arm-general"
    ) == (active,)


@pytest.mark.parametrize(
    "changed",
    [
        _observation(f"sha256:{'a' * 64}", severity=FindingSeverity.LOW),
        _observation(f"sha256:{'a' * 64}", stage="candidate_verification"),
        _observation(f"sha256:{'a' * 64}", deterministic_overlap=True),
    ],
)
def test_lifecycle_uses_current_metadata_for_a_continuing_fingerprint(changed):
    original = _observation(f"sha256:{'a' * 64}")

    assert derive_finding_lifecycle(
        [changed], [original], arm_id="arm-general", parent_arm_id="arm-general"
    ) == (changed,)


@pytest.mark.parametrize("source", ["current", "parent"])
def test_lifecycle_preserves_duplicate_fingerprints(source):
    finding = _observation(f"sha256:{'a' * 64}")
    current = [finding, finding] if source == "current" else []
    parent = [finding, finding] if source == "parent" else []

    result = derive_finding_lifecycle(
        current, parent, arm_id="arm-general", parent_arm_id="arm-general"
    )

    assert len(result) == 2
    assert [finding.lifecycle_state for finding in result] == [
        FindingLifecycleState.ACTIVE if source == "current" else FindingLifecycleState.WITHDRAWN,
    ] * 2


def test_lifecycle_preserves_current_duplicates_with_different_metadata():
    fingerprint = f"sha256:{'a' * 64}"
    current = [
        _observation(fingerprint, severity=FindingSeverity.HIGH),
        _observation(fingerprint, severity=FindingSeverity.LOW, stage="candidate_verification"),
        _observation(fingerprint, deterministic_overlap=True),
    ]

    assert derive_finding_lifecycle(
        current,
        [_observation(fingerprint)],
        arm_id="arm-general",
        parent_arm_id="arm-general",
    ) == tuple(current)


def test_lifecycle_rejects_cross_arm_parentage():
    with pytest.raises(EvaluationValidationError, match="same evaluation arm"):
        derive_finding_lifecycle([], [], arm_id="arm-general", parent_arm_id="arm-cascade")


def test_lifecycle_rejects_model_supplied_current_withdrawal():
    withdrawn = _observation(
        f"sha256:{'a' * 64}",
        lifecycle_state=FindingLifecycleState.WITHDRAWN,
    )

    with pytest.raises(EvaluationValidationError, match="current normalized findings must be active"):
        derive_finding_lifecycle(
            [withdrawn], [], arm_id="arm-general", parent_arm_id="arm-general"
        )


def test_lifecycle_rejects_non_observation_input():
    with pytest.raises(EvaluationValidationError, match="must use ObservedFinding"):
        derive_finding_lifecycle(
            ["model-finding"], [], arm_id="arm-general", parent_arm_id="arm-general"
        )


def test_lifecycle_rejects_non_sequence_input():
    with pytest.raises(EvaluationValidationError, match="current findings must be a sequence"):
        derive_finding_lifecycle(
            None, [], arm_id="arm-general", parent_arm_id="arm-general"
        )
