"""Canonical, credential-free inventory of checkpoint production bindings.

The evaluation runner accepts injected adapters so its safety contracts can be
tested independently.  This module is the production boundary: it names the
remaining capability gaps and returns fail-closed bindings until each gap is
implemented by shipped review orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from pr_agent.algo.checkpoint_evaluation import (
    EvaluationArm,
    EvaluationArmKind,
    EvaluationManifest,
    content_hash,
)
from pr_agent.algo.checkpoint_evaluation_runner import ModelTelemetryShape, ProductionArmBinding

PRODUCTION_BINDING_INVENTORY_SCHEMA_VERSION = "checkpoint-production-binding-inventory-v1"
_PRODUCTION_BINDING_INVENTORY_SCHEMA = {
    "schema_version": PRODUCTION_BINDING_INVENTORY_SCHEMA_VERSION,
    "fields": (
        "schema_version",
        "schema_hash",
        "manifest_id",
        "bindings",
        "inventory_id",
    ),
    "binding_fields": ("arm_id", "kind", "available", "blocker_codes"),
}


@dataclass(frozen=True)
class ProductionBindingReadiness:
    """Source-free readiness for one manifest arm."""

    arm_id: str
    kind: EvaluationArmKind
    available: bool
    blocker_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "kind": self.kind.value,
            "available": self.available,
            "blocker_codes": list(self.blocker_codes),
        }


@dataclass(frozen=True)
class ProductionBindingInventory:
    """Versioned readiness envelope tied to one immutable evaluation manifest."""

    manifest_id: str
    bindings: tuple[ProductionBindingReadiness, ...]
    schema_version: str = PRODUCTION_BINDING_INVENTORY_SCHEMA_VERSION

    @property
    def schema_hash(self) -> str:
        return content_hash(_PRODUCTION_BINDING_INVENTORY_SCHEMA)

    @property
    def inventory_id(self) -> str:
        return content_hash({
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "manifest_id": self.manifest_id,
            "bindings": [binding.to_dict() for binding in self.bindings],
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "manifest_id": self.manifest_id,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "inventory_id": self.inventory_id,
        }


_BLOCKERS: Mapping[EvaluationArmKind, tuple[str, ...]] = MappingProxyType({
    EvaluationArmKind.DETERMINISTIC: (
        "deterministic_finding_contract_unavailable",
    ),
    EvaluationArmKind.GENERAL_REVIEW: (
        "finding_fingerprint_contract_unavailable",
        "hard_cost_cap_enforcement_unavailable",
    ),
    EvaluationArmKind.SPECIALISTS: (
        "specialist_finding_contract_unavailable",
        "hard_cost_cap_enforcement_unavailable",
    ),
    EvaluationArmKind.VERIFIED_SPECIALISTS: (
        "verified_candidate_source_contract_unavailable",
        "hard_cost_cap_enforcement_unavailable",
    ),
    EvaluationArmKind.FULL_CASCADE: (
        "finding_fingerprint_contract_unavailable",
        "verified_candidate_source_contract_unavailable",
        "frontier_stage_telemetry_contract_unavailable",
        "frontier_decision_semantics_unavailable",
        "hard_cost_cap_enforcement_unavailable",
    ),
})


def production_binding_inventory(manifest: EvaluationManifest) -> ProductionBindingInventory:
    """Return exact disabled production readiness without importing model clients."""

    return ProductionBindingInventory(
        manifest_id=manifest.manifest_id,
        bindings=tuple(
            ProductionBindingReadiness(
                arm_id=arm.arm_id,
                kind=arm.kind,
                available=False,
                blocker_codes=_BLOCKERS[arm.kind],
            )
            for arm in sorted((item for item in manifest.arms if item.enabled), key=lambda item: item.arm_id)
        )
    )


def build_production_arm_bindings(manifest: EvaluationManifest) -> tuple[ProductionArmBinding, ...]:
    """Bind every enabled manifest arm to its current fail-closed production state."""

    readiness_by_arm = {
        item.arm_id: item for item in production_binding_inventory(manifest).bindings
    }
    return tuple(
        _unavailable_binding(arm, readiness_by_arm[arm.arm_id])
        for arm in manifest.arms
        if arm.enabled
    )


def _unavailable_binding(
    arm: EvaluationArm,
    readiness: ProductionBindingReadiness,
) -> ProductionArmBinding:
    telemetry_shape = (
        ModelTelemetryShape.NONE
        if arm.kind is EvaluationArmKind.DETERMINISTIC
        else (
            ModelTelemetryShape.SINGLE_SELECTED
            if arm.kind is EvaluationArmKind.GENERAL_REVIEW
            else ModelTelemetryShape.PER_STAGE
        )
    )
    return ProductionArmBinding(
        kind=arm.kind,
        configuration_hash=arm.configuration_hash,
        prompt_hash=arm.prompt_hash,
        model_identities=arm.model_identities(),
        stage_plan=arm.stage_plan,
        telemetry_shape=telemetry_shape,
        adapter=None,
        available=False,
        enforces_hard_cost_cap=False,
        unavailable_reason=readiness.blocker_codes[0],
        publish_output=False,
    )
