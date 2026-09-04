"""Frozen pre-call spending authority for checkpoint evaluation workers.

The local ledger does not estimate prices. It consumes a maximum-charge guarantee
issued by the provider or an enforcing gateway before every underlying provider
request. A missing guarantee, an unbounded output, or SDK-internal retrying denies
the request before the model client is called.
"""

from __future__ import annotations

import hmac
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence

from pr_agent.algo.checkpoint_evaluation import (
    CheckpointCase,
    EvaluationArm,
    EvaluationManifest,
    EvaluationValidationError,
    content_hash,
    deployment_identity_hash,
)
from pr_agent.algo.checkpoint_evaluation_execution import PaidExecutionRequest

CHECKPOINT_COST_AUTHORITY_SCHEMA_VERSION = "checkpoint-cost-authority-v1"
CHECKPOINT_COST_ENFORCEMENT_KIND = "provider_gateway_maximum_charge"
GENERAL_REVIEW_COST_STAGE = "general_review"
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_STAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MUTABLE_REVISIONS = frozenset({"default", "latest", "main", "stable"})


class CheckpointCostAuthorityError(EvaluationValidationError):
    """Raised before a provider request when its hard spending proof is absent."""


def _require_hash(name: str, value: object) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise CheckpointCostAuthorityError(f"{name} must be a sha256 identity")
    return value


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise CheckpointCostAuthorityError(f"{name} must be a bounded non-empty string")
    return value


def _require_stage(value: object) -> str:
    if not isinstance(value, str) or not _STAGE_PATTERN.fullmatch(value):
        raise CheckpointCostAuthorityError("cost quote stage must be a bounded identifier")
    return value


def _positive_decimal(name: str, value: object) -> Decimal:
    if not isinstance(value, (str, Decimal)) or isinstance(value, bool):
        raise CheckpointCostAuthorityError(f"{name} must be an exact decimal string")
    try:
        decimal_value = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise CheckpointCostAuthorityError(f"{name} must be an exact decimal string") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise CheckpointCostAuthorityError(f"{name} must be finite and positive")
    return decimal_value


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _parse_expiry(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CheckpointCostAuthorityError("cost authority expiry must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise CheckpointCostAuthorityError("cost authority expiry must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise CheckpointCostAuthorityError("cost authority expiry must use UTC")
    return parsed


@dataclass(frozen=True)
class ProviderMaximumCharge:
    """One externally guaranteed maximum charge for an exact provider request."""

    stage: str
    model_id: str
    provider_id: str
    model_revision: str
    deployment_id_hash: Optional[str]
    max_output_tokens: int
    maximum_charge_usd: Decimal
    quote_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _require_stage(self.stage))
        for name in ("model_id", "provider_id", "model_revision"):
            object.__setattr__(self, name, _require_identifier(f"cost quote {name}", getattr(self, name)))
        if self.model_revision.strip().casefold() in _MUTABLE_REVISIONS:
            raise CheckpointCostAuthorityError("cost quote model revision must be immutable")
        if self.deployment_id_hash is not None:
            _require_hash("cost quote deployment identity", self.deployment_id_hash)
        if (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens < 1
        ):
            raise CheckpointCostAuthorityError("cost quote max_output_tokens must be a positive integer")
        object.__setattr__(
            self,
            "maximum_charge_usd",
            _positive_decimal("cost quote maximum_charge_usd", self.maximum_charge_usd),
        )
        object.__setattr__(self, "quote_id", content_hash(self._identity_payload()))

    def route_key(self) -> tuple[str, str, str, str, Optional[str]]:
        return (
            self.stage,
            self.model_id,
            self.provider_id,
            self.model_revision,
            self.deployment_id_hash,
        )

    def runtime_key(self) -> tuple[str, str, Optional[str]]:
        return self.stage, self.model_id, self.deployment_id_hash

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "model_revision": self.model_revision,
            "deployment_id_hash": self.deployment_id_hash,
            "max_output_tokens": self.max_output_tokens,
            "maximum_charge_usd": _decimal_text(self.maximum_charge_usd),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "quote_id": self.quote_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderMaximumCharge":
        expected = {
            "stage",
            "model_id",
            "provider_id",
            "model_revision",
            "deployment_id_hash",
            "max_output_tokens",
            "maximum_charge_usd",
            "quote_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CheckpointCostAuthorityError("cost quote fields do not match its schema")
        quote = cls(
            stage=value["stage"],
            model_id=value["model_id"],
            provider_id=value["provider_id"],
            model_revision=value["model_revision"],
            deployment_id_hash=value["deployment_id_hash"],
            max_output_tokens=value["max_output_tokens"],
            maximum_charge_usd=value["maximum_charge_usd"],
        )
        if value["quote_id"] != quote.quote_id:
            raise CheckpointCostAuthorityError("cost quote identity does not match its content")
        return quote


@dataclass(frozen=True)
class FrozenCostAuthority:
    """Source-free provider/gateway authority bound to one paid case/arm attempt."""

    manifest_id: str
    paid_request_id: str
    case_id: str
    arm_id: str
    snapshot_id: str
    arm_configuration_hash: str
    review_configuration_hash: str
    hard_cost_cap_usd: Decimal
    authority_name: str
    authority_revision: str
    authority_reference_hash: str
    expires_at: str
    quotes: tuple[ProviderMaximumCharge, ...]
    enforcement_kind: str = CHECKPOINT_COST_ENFORCEMENT_KIND
    schema_version: str = CHECKPOINT_COST_AUTHORITY_SCHEMA_VERSION
    authority_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_COST_AUTHORITY_SCHEMA_VERSION:
            raise CheckpointCostAuthorityError("unsupported checkpoint cost authority schema")
        if self.enforcement_kind != CHECKPOINT_COST_ENFORCEMENT_KIND:
            raise CheckpointCostAuthorityError("unsupported checkpoint cost enforcement kind")
        for name in (
            "manifest_id",
            "paid_request_id",
            "snapshot_id",
            "arm_configuration_hash",
            "review_configuration_hash",
        ):
            _require_hash(f"cost authority {name}", getattr(self, name))
        for name in ("case_id", "arm_id", "authority_name", "authority_revision"):
            object.__setattr__(self, name, _require_identifier(f"cost authority {name}", getattr(self, name)))
        if self.authority_revision.strip().casefold() in _MUTABLE_REVISIONS:
            raise CheckpointCostAuthorityError("cost authority revision must be immutable")
        _require_hash("cost authority reference", self.authority_reference_hash)
        _parse_expiry(self.expires_at)
        object.__setattr__(
            self,
            "hard_cost_cap_usd",
            _positive_decimal("cost authority hard_cost_cap_usd", self.hard_cost_cap_usd),
        )
        object.__setattr__(self, "quotes", tuple(self.quotes))
        if not self.quotes or any(not isinstance(quote, ProviderMaximumCharge) for quote in self.quotes):
            raise CheckpointCostAuthorityError("cost authority requires provider maximum-charge quotes")
        route_keys = [quote.route_key() for quote in self.quotes]
        runtime_keys = [quote.runtime_key() for quote in self.quotes]
        if len(route_keys) != len(set(route_keys)) or len(runtime_keys) != len(set(runtime_keys)):
            raise CheckpointCostAuthorityError("cost authority contains ambiguous provider routes")
        if any(quote.maximum_charge_usd > self.hard_cost_cap_usd for quote in self.quotes):
            raise CheckpointCostAuthorityError("one provider maximum charge exceeds the attempt cap")
        object.__setattr__(self, "authority_id", content_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enforcement_kind": self.enforcement_kind,
            "manifest_id": self.manifest_id,
            "paid_request_id": self.paid_request_id,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "snapshot_id": self.snapshot_id,
            "arm_configuration_hash": self.arm_configuration_hash,
            "review_configuration_hash": self.review_configuration_hash,
            "hard_cost_cap_usd": _decimal_text(self.hard_cost_cap_usd),
            "authority_name": self.authority_name,
            "authority_revision": self.authority_revision,
            "authority_reference_hash": self.authority_reference_hash,
            "expires_at": self.expires_at,
            "quotes": [quote.to_dict() for quote in self.quotes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "authority_id": self.authority_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenCostAuthority":
        expected = {
            "schema_version",
            "enforcement_kind",
            "manifest_id",
            "paid_request_id",
            "case_id",
            "arm_id",
            "snapshot_id",
            "arm_configuration_hash",
            "review_configuration_hash",
            "hard_cost_cap_usd",
            "authority_name",
            "authority_revision",
            "authority_reference_hash",
            "expires_at",
            "quotes",
            "authority_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value.get("quotes"), list):
            raise CheckpointCostAuthorityError("cost authority fields do not match its schema")
        authority = cls(
            schema_version=value["schema_version"],
            enforcement_kind=value["enforcement_kind"],
            manifest_id=value["manifest_id"],
            paid_request_id=value["paid_request_id"],
            case_id=value["case_id"],
            arm_id=value["arm_id"],
            snapshot_id=value["snapshot_id"],
            arm_configuration_hash=value["arm_configuration_hash"],
            review_configuration_hash=value["review_configuration_hash"],
            hard_cost_cap_usd=value["hard_cost_cap_usd"],
            authority_name=value["authority_name"],
            authority_revision=value["authority_revision"],
            authority_reference_hash=value["authority_reference_hash"],
            expires_at=value["expires_at"],
            quotes=tuple(ProviderMaximumCharge.from_dict(quote) for quote in value["quotes"]),
        )
        if value["authority_id"] != authority.authority_id:
            raise CheckpointCostAuthorityError("cost authority identity does not match its content")
        return authority

    def require_active(self, *, now: Optional[datetime] = None) -> None:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise CheckpointCostAuthorityError("cost authority clock must be timezone-aware")
        if current.astimezone(timezone.utc) >= _parse_expiry(self.expires_at):
            raise CheckpointCostAuthorityError("cost authority has expired")

    def require_context(
        self,
        *,
        manifest_id: str,
        case_id: str,
        arm_id: str,
        snapshot_id: str,
        arm_configuration_hash: str,
        review_configuration_hash: str,
    ) -> None:
        self.require_pair_context(
            manifest_id=manifest_id,
            case_id=case_id,
            arm_id=arm_id,
            snapshot_id=snapshot_id,
            arm_configuration_hash=arm_configuration_hash,
        )
        if not hmac.compare_digest(review_configuration_hash, self.review_configuration_hash):
            raise CheckpointCostAuthorityError("cost authority belongs to a different review configuration")

    def require_pair_context(
        self,
        *,
        manifest_id: str,
        case_id: str,
        arm_id: str,
        snapshot_id: str,
        arm_configuration_hash: str,
    ) -> None:
        expected = (
            manifest_id,
            case_id,
            arm_id,
            snapshot_id,
            arm_configuration_hash,
        )
        actual = (
            self.manifest_id,
            self.case_id,
            self.arm_id,
            self.snapshot_id,
            self.arm_configuration_hash,
        )
        if any(not hmac.compare_digest(left, right) for left, right in zip(expected, actual, strict=True)):
            raise CheckpointCostAuthorityError("cost authority belongs to a different evaluation context")
        self.require_active()


def _expected_routes(arm: EvaluationArm) -> set[tuple[str, str, str, str, Optional[str]]]:
    routes = {
        (
            GENERAL_REVIEW_COST_STAGE,
            model_id,
            provider_id,
            model_revision,
            None,
        )
        for model_id, provider_id, model_revision in arm.aggregate_model_identities()
        if model_id is not None and provider_id is not None and model_revision is not None
    }
    routes.update(
        (
            stage.stage,
            identity.model_id,
            identity.provider_id,
            identity.model_revision,
            identity.deployment_id_hash,
        )
        for stage in arm.stage_plan
        for identity in stage.model_route
    )
    return routes


def validate_cost_authority_for_pair(
    manifest: EvaluationManifest,
    request: PaidExecutionRequest,
    case: CheckpointCase,
    arm: EvaluationArm,
    authority: FrozenCostAuthority,
) -> None:
    """Validate exact route coverage without consulting a provider or source artifact."""

    if not all(isinstance(value, expected) for value, expected in (
        (manifest, EvaluationManifest),
        (request, PaidExecutionRequest),
        (case, CheckpointCase),
        (arm, EvaluationArm),
        (authority, FrozenCostAuthority),
    )):
        raise TypeError("cost authority validation requires evaluation contract objects")
    budget = next(
        (
            item
            for item in request.plan_item_budgets
            if item.case_id == case.case_id and item.arm_id == arm.arm_id
        ),
        None,
    )
    if budget is None:
        raise CheckpointCostAuthorityError("cost authority pair has no immutable paid budget")
    authority.require_pair_context(
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        arm_id=arm.arm_id,
        snapshot_id=case.snapshot_id,
        arm_configuration_hash=arm.configuration_hash,
    )
    if not hmac.compare_digest(authority.paid_request_id, request.request_id):
        raise CheckpointCostAuthorityError("cost authority belongs to a different paid request")
    if authority.hard_cost_cap_usd != Decimal(str(budget.hard_cost_cap_per_attempt_usd)):
        raise CheckpointCostAuthorityError("cost authority cap does not match the immutable paid budget")
    if {quote.route_key() for quote in authority.quotes} != _expected_routes(arm):
        raise CheckpointCostAuthorityError("cost authority does not quote every exact arm route")


@dataclass(frozen=True)
class ProviderAttemptReservation:
    authority_id: str
    sequence: int
    quote_id: str
    stage: str
    model_id: str
    maximum_charge_usd: Decimal
    cumulative_reserved_usd: Decimal
    reservation_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reservation_id", content_hash({
            "authority_id": self.authority_id,
            "sequence": self.sequence,
            "quote_id": self.quote_id,
            "stage": self.stage,
            "model_id": self.model_id,
            "maximum_charge_usd": _decimal_text(self.maximum_charge_usd),
            "cumulative_reserved_usd": _decimal_text(self.cumulative_reserved_usd),
        }))


class CostAuthorityLedger:
    """Atomic, consume-only reservations shared by concurrent evaluation stages."""

    def __init__(self, authority: FrozenCostAuthority):
        if not isinstance(authority, FrozenCostAuthority):
            raise TypeError("cost ledger requires a FrozenCostAuthority")
        authority.require_active()
        self.authority = authority
        self._quotes = MappingProxyType({quote.runtime_key(): quote for quote in authority.quotes})
        self._lock = threading.Lock()
        self._reservations: list[ProviderAttemptReservation] = []
        self._reserved_usd = Decimal("0")

    @property
    def reserved_usd(self) -> Decimal:
        with self._lock:
            return self._reserved_usd

    @property
    def reservations(self) -> tuple[ProviderAttemptReservation, ...]:
        with self._lock:
            return tuple(self._reservations)

    def reserve(
        self,
        *,
        stage: str,
        model_id: str,
        deployment_id: Optional[str],
        max_output_tokens: object,
        provider_max_retries: object,
    ) -> ProviderAttemptReservation:
        """Consume the guaranteed maximum immediately before one provider call."""

        self.authority.require_active()
        if provider_max_retries != 0 or isinstance(provider_max_retries, bool):
            raise CheckpointCostAuthorityError(
                "checkpoint cost authority requires provider SDK retries to be exactly zero"
            )
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens < 1
        ):
            raise CheckpointCostAuthorityError("checkpoint provider request requires a bounded output cap")
        key = (_require_stage(stage), _require_identifier("provider request model_id", model_id),
               deployment_identity_hash(deployment_id))
        quote = self._quotes.get(key)
        if quote is None:
            raise CheckpointCostAuthorityError("checkpoint provider request has no authoritative maximum-charge quote")
        if max_output_tokens > quote.max_output_tokens:
            raise CheckpointCostAuthorityError("checkpoint provider request exceeds its quoted output cap")
        with self._lock:
            cumulative = self._reserved_usd + quote.maximum_charge_usd
            if cumulative > self.authority.hard_cost_cap_usd:
                raise CheckpointCostAuthorityError("checkpoint provider request would exceed the hard cost cap")
            reservation = ProviderAttemptReservation(
                authority_id=self.authority.authority_id,
                sequence=len(self._reservations) + 1,
                quote_id=quote.quote_id,
                stage=stage,
                model_id=model_id,
                maximum_charge_usd=quote.maximum_charge_usd,
                cumulative_reserved_usd=cumulative,
            )
            self._reservations.append(reservation)
            self._reserved_usd = cumulative
            return reservation


_cost_authority_ledger: ContextVar[Optional[CostAuthorityLedger]] = ContextVar(
    "pr_agent_checkpoint_cost_authority_ledger",
    default=None,
)
_cost_authority_scope_lock = threading.Lock()
_active_cost_authority_ledger: Optional[CostAuthorityLedger] = None


def get_checkpoint_cost_authority_ledger() -> Optional[CostAuthorityLedger]:
    contextual = _cost_authority_ledger.get()
    if contextual is not None:
        return contextual
    with _cost_authority_scope_lock:
        return _active_cost_authority_ledger


@contextmanager
def use_checkpoint_cost_authority(authority: FrozenCostAuthority) -> Iterator[CostAuthorityLedger]:
    """Install one shared ledger process-wide in the single-review worker."""

    global _active_cost_authority_ledger
    ledger = CostAuthorityLedger(authority)
    with _cost_authority_scope_lock:
        if _active_cost_authority_ledger is not None:
            raise CheckpointCostAuthorityError("checkpoint worker already has an active cost authority")
        _active_cost_authority_ledger = ledger
    token = _cost_authority_ledger.set(ledger)
    try:
        yield ledger
    finally:
        _cost_authority_ledger.reset(token)
        with _cost_authority_scope_lock:
            if _active_cost_authority_ledger is not ledger:
                raise CheckpointCostAuthorityError("checkpoint cost authority scope was replaced")
            _active_cost_authority_ledger = None


def reserve_checkpoint_provider_attempt(
    *,
    model_id: str,
    deployment_id: Optional[str],
    max_output_tokens: object,
    provider_max_retries: object,
    attribution: Optional[str],
) -> Optional[ProviderAttemptReservation]:
    """Reserve one provider call when checkpoint enforcement is active."""

    ledger = get_checkpoint_cost_authority_ledger()
    if ledger is None:
        return None
    return ledger.reserve(
        stage=attribution or GENERAL_REVIEW_COST_STAGE,
        model_id=model_id,
        deployment_id=deployment_id,
        max_output_tokens=max_output_tokens,
        provider_max_retries=provider_max_retries,
    )


def validate_cost_authorities(
    manifest: EvaluationManifest,
    request: PaidExecutionRequest,
    authorities: Sequence[FrozenCostAuthority],
) -> Mapping[tuple[str, str], FrozenCostAuthority]:
    """Return an exact pair map only when every model-backed pair is enforceable."""

    cases = {case.case_id: case for case in manifest.cases}
    arms = {arm.arm_id: arm for arm in manifest.arms if arm.enabled}
    expected_pairs = {
        (case.case_id, arm.arm_id)
        for case in manifest.cases
        for arm in arms.values()
        if arm.model_id is not None
    }
    by_pair: dict[tuple[str, str], FrozenCostAuthority] = {}
    for authority in authorities:
        if not isinstance(authority, FrozenCostAuthority):
            raise TypeError("cost authorities must use FrozenCostAuthority")
        pair = authority.case_id, authority.arm_id
        if pair in by_pair:
            raise CheckpointCostAuthorityError("multiple cost authorities cover one paid pair")
        if authority.case_id not in cases or authority.arm_id not in arms:
            raise CheckpointCostAuthorityError("cost authority names an unknown paid pair")
        validate_cost_authority_for_pair(
            manifest,
            request,
            cases[authority.case_id],
            arms[authority.arm_id],
            authority,
        )
        by_pair[pair] = authority
    if set(by_pair) != expected_pairs:
        raise CheckpointCostAuthorityError("cost authorities must cover every model-backed pair exactly")
    return MappingProxyType(by_pair)
