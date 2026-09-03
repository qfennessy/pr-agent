"""Details of a single PR-Agent run, collected while the command executes.

The data is held in a ``ContextVar`` so that the AI handler can record token
usage without changing ``chat_completion``'s return signature. Context vars are
copied into ``asyncio`` child tasks while still referencing the same mutable
``RunDetails`` object, so concurrent AI calls accumulate into one instance and
stay isolated between concurrent requests.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator, Mapping, Optional

from pr_agent.algo.ai_request_context import get_ai_request_options

_run_details: ContextVar[Optional["RunDetails"]] = ContextVar(
    "pr_agent_run_details", default=None
)


@dataclass
class SpecialistRunDetails:
    """Telemetry for one shadow specialist, separate from the main review footer."""

    role: str
    model_used: Optional[str] = None
    deployment_id: Optional[str] = None
    fallback_used: bool = False
    prompt_version: Optional[str] = None
    input_schema_version: Optional[str] = None
    schema_version: Optional[str] = None
    state: Optional[str] = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_ai_calls: int = 0
    total_cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    known_cost_call_count: int = 0
    model_costs_usd: dict[str, Decimal] = field(default_factory=dict)
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None
    cached: bool = False
    input_token_reservation: int = 0
    output_token_reservation: int = 0
    output: Optional[Mapping[str, Any]] = None

    @property
    def cost_status(self) -> str:
        if self.known_cost_call_count == 0:
            return "unavailable"
        if self.known_cost_call_count == self.num_ai_calls:
            return "complete"
        return "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model_used,
            "deployment": self.deployment_id,
            "fallback_used": self.fallback_used,
            "prompt_version": self.prompt_version,
            "input_schema_version": self.input_schema_version,
            "schema_version": self.schema_version,
            "state": self.state,
            "latency_seconds": self.latency_seconds,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "ai_calls": self.num_ai_calls,
            },
            "cost": {
                "status": self.cost_status,
                "total_usd": str(self.total_cost_usd) if self.known_cost_call_count else None,
                "by_model_usd": {model: str(cost) for model, cost in self.model_costs_usd.items()},
            },
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
            "cached": self.cached,
            "reservation": {
                "input_tokens": self.input_token_reservation,
                "output_tokens": self.output_token_reservation,
            },
            "output": deepcopy(self.output),
        }


@dataclass
class AdjudicationRunDetails:
    """Telemetry for one frontier adjudication, isolated from specialists and publication."""

    finding_id: str
    model_used: Optional[str] = None
    provider: Optional[str] = None
    model_revision: Optional[str] = None
    deployment_id: Optional[str] = None
    fallback_used: bool = False
    route_attempts: int = 0
    model_attempts: Optional[int] = None
    model_attempts_configured: Optional[int] = None
    provider_attempts: Optional[int] = None
    provider_retries_configured: Optional[int] = None
    prompt_version: Optional[str] = None
    input_schema_version: Optional[str] = None
    schema_version: Optional[str] = None
    state: Optional[str] = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_ai_calls: int = 0
    known_usage_call_count: int = 0
    total_cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    known_cost_call_count: int = 0
    model_costs_usd: dict[str, Decimal] = field(default_factory=dict)
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None
    cache_state: str = "not_requested"

    @property
    def cost_status(self) -> str:
        if self.known_cost_call_count == 0:
            return "unavailable"
        if self.known_cost_call_count == self.num_ai_calls:
            return "complete"
        return "partial"

    @property
    def usage_status(self) -> str:
        if self.num_ai_calls == 0:
            return "unavailable"
        if self.known_usage_call_count == self.num_ai_calls:
            return "complete"
        return "partial"

    @property
    def model_retry_attempts(self) -> Optional[int]:
        """Observed handler retries beyond the first attempt for each route entry."""

        if self.model_attempts is None or self.model_attempts < self.route_attempts:
            return None
        return self.model_attempts - self.route_attempts

    @property
    def provider_retry_attempts(self) -> Optional[int]:
        """Observed provider retries when SDK-internal retrying is disabled."""

        if (
            self.provider_retries_configured != 0
            or self.provider_attempts is None
            or self.model_attempts is None
            or self.provider_attempts < self.model_attempts
        ):
            return None
        return self.provider_attempts - self.model_attempts

    @property
    def provider_attempts_unavailable_reason(self) -> Optional[str]:
        if self.provider_retry_attempts is not None:
            return None
        if self.provider_retries_configured == 0:
            return "provider_attempts_not_observed"
        return "provider_internal_attempts_not_exposed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "model": self.model_used,
            "provider": self.provider,
            "model_revision": self.model_revision,
            "deployment": self.deployment_id,
            "fallback_used": self.fallback_used,
            "route_attempts": self.route_attempts,
            "retries": {
                "model": {
                    "status": "complete" if self.model_retry_attempts is not None else "unavailable",
                    "configured_attempts_per_model": self.model_attempts_configured,
                    "attempts": self.model_attempts,
                    "retry_attempts": self.model_retry_attempts,
                },
                "provider": {
                    "status": "complete" if self.provider_retry_attempts is not None else "unavailable",
                    "configured_retries_per_model_attempt": self.provider_retries_configured,
                    "attempts": (
                        self.provider_attempts if self.provider_retry_attempts is not None else None
                    ),
                    "retry_attempts": self.provider_retry_attempts,
                    "unavailable_reason": self.provider_attempts_unavailable_reason,
                },
            },
            "prompt_version": self.prompt_version,
            "input_schema_version": self.input_schema_version,
            "schema_version": self.schema_version,
            "state": self.state,
            "latency_seconds": self.latency_seconds,
            "usage": {
                "status": self.usage_status,
                "prompt_tokens": self.prompt_tokens or None,
                "completion_tokens": self.completion_tokens or None,
                "total_tokens": self.total_tokens or None,
                "ai_calls": self.num_ai_calls or None,
            },
            "cost": {
                "status": self.cost_status,
                "total_usd": str(self.total_cost_usd) if self.known_cost_call_count else None,
                "by_model_usd": {model: str(cost) for model, cost in self.model_costs_usd.items()},
            },
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
            "cache_state": self.cache_state,
        }


@dataclass
class RunDetails:
    """Counters and identifiers accumulated over a single command run.

    Every field is filled opportunistically: whatever the provider does not report
    stays at its default, and the renderer omits the corresponding line rather than
    displaying a zero.
    """

    # Model that produced the answer, which differs from `config.model` when a fallback
    # took over. Stays None when no prediction succeeded, which the renderer reads as
    # "nothing worth showing".
    model_used: Optional[str] = None
    # Review mode selected for this run. Other tools leave it unset.
    review_profile: Optional[str] = None
    # Provider-neutral deterministic review-depth decision. Kept separate from
    # specialist shadow records so observational model telemetry remains unchanged.
    review_route: Optional[Mapping[str, Any]] = None
    # Sticky: once a fallback has won, a later success on the primary model must not
    # clear this, or the comment would hide that a fallback ran at all.
    fallback_used: bool = False
    # Input/output tokens summed over every AI call of the run. Named after litellm's
    # normalized usage object, which is what the collector reads. Both stay 0 when no usage
    # reaches the collector, e.g. streaming responses or the langchain handler.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Provider-reported total when available, otherwise derived from prompt + completion.
    # Counts failed fallback attempts as well, so it reflects what the run really cost,
    # while `model_used` names only the model behind the final answer.
    total_tokens: int = 0
    # Successful LLM invocations, counted even when their token usage is unavailable.
    num_ai_calls: int = 0
    # Accumulate costs only when cost output is enabled and LiteLLM can synchronously
    # price a successful response with a positive amount. Use the known-call count to
    # distinguish priced calls from missing pricing data. Retain per-model totals to
    # keep fallback and multi-call runs auditable.
    total_cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    known_cost_call_count: int = 0
    model_costs_usd: dict[str, Decimal] = field(default_factory=dict)
    # Shadow specialists are intentionally excluded from the aggregate fields above.
    # Otherwise enabling shadow mode would change the published main-review footer.
    specialist_runs: dict[str, SpecialistRunDetails] = field(default_factory=dict)
    # Frontier adjudication is a distinct production stage. Keeping a separate
    # collection prevents availability fallbacks from masquerading as specialist
    # decisions and gives every stable finding its own accounting record.
    adjudication_runs: dict[str, AdjudicationRunDetails] = field(default_factory=dict)
    # Monotonic reference taken when the collector is installed, i.e. at the top of the
    # tool's run(). Monotonic so that wall-clock adjustments cannot yield a negative duration.
    start_time: float = field(default_factory=time.monotonic)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.start_time)

    @property
    def has_token_usage(self) -> bool:
        return (
            self.total_tokens > 0
            or self.prompt_tokens > 0
            or self.completion_tokens > 0
        )

    @property
    def cost_status(self) -> str:
        """Return whether every, some, or none of the successful calls were priced."""
        if self.known_cost_call_count == 0:
            return "unavailable"
        if self.known_cost_call_count == self.num_ai_calls:
            return "complete"
        return "partial"


def init_run_details() -> RunDetails:
    """Install a fresh collector for the current run and return it."""
    details = RunDetails()
    _run_details.set(details)
    return details


def get_run_details() -> Optional[RunDetails]:
    """Return the collector for the current run, or None if not initialized."""
    return _run_details.get()


@contextmanager
def isolate_run_details() -> Iterator[None]:
    """Install an empty request-local telemetry slot and restore the caller's slot."""

    token = _run_details.set(None)
    try:
        yield
    finally:
        _run_details.reset(token)


def _get_specialist_details(attribution: str) -> Optional[SpecialistRunDetails]:
    details = get_run_details()
    if details is None:
        return None
    return details.specialist_runs.setdefault(attribution, SpecialistRunDetails(role=attribution))


_FRONTIER_ATTRIBUTION_PREFIX = "frontier_adjudication:"


def _frontier_finding_id(attribution: Optional[str]) -> Optional[str]:
    if not attribution or not attribution.startswith(_FRONTIER_ATTRIBUTION_PREFIX):
        return None
    finding_id = attribution.removeprefix(_FRONTIER_ATTRIBUTION_PREFIX).strip()
    return finding_id or None


def _get_adjudication_details(attribution: Optional[str]) -> Optional[AdjudicationRunDetails]:
    finding_id = _frontier_finding_id(attribution)
    details = get_run_details()
    if finding_id is None or details is None:
        return None
    return details.adjudication_runs.setdefault(
        finding_id, AdjudicationRunDetails(finding_id=finding_id)
    )


def record_model_used(
    model: str,
    is_fallback: bool,
    attribution: Optional[str] = None,
    deployment_id: Optional[str] = None,
) -> None:
    """Record the model that produced a successful completion."""
    details = get_run_details()
    if details is None:
        return
    if attribution:
        adjudication = _get_adjudication_details(attribution)
        if adjudication is not None:
            adjudication.model_used = model
            adjudication.deployment_id = deployment_id
            if is_fallback:
                adjudication.fallback_used = True
            return
        specialist = _get_specialist_details(attribution)
        if specialist is not None:
            specialist.model_used = model
            specialist.deployment_id = deployment_id
            if is_fallback:
                specialist.fallback_used = True
        return
    details.model_used = model
    if is_fallback:
        # sticky: later primary success must not hide that a fallback ran
        details.fallback_used = True


def record_specialist_model_attempt(
    model: str,
    *,
    attribution: Optional[str],
    deployment_id: Optional[str],
    is_fallback: bool,
    model_attempts_configured: Optional[int] = None,
    provider_retries_configured: Optional[int] = None,
) -> None:
    """Preserve the last attempted specialist route even when its output is rejected."""

    if not attribution:
        return
    adjudication = _get_adjudication_details(attribution)
    if adjudication is not None:
        adjudication.model_used = model
        adjudication.deployment_id = deployment_id
        adjudication.route_attempts += 1
        if adjudication.model_attempts is None:
            adjudication.model_attempts = 0
        adjudication.model_attempts += 1
        adjudication.model_attempts_configured = model_attempts_configured
        adjudication.provider_retries_configured = provider_retries_configured
        if is_fallback:
            adjudication.fallback_used = True
        return
    specialist = _get_specialist_details(attribution)
    if specialist is None:
        return
    specialist.model_used = model
    specialist.deployment_id = deployment_id
    if is_fallback:
        specialist.fallback_used = True


def record_model_request_attempt(attribution: Optional[str] = None) -> None:
    """Count one observable retry beyond a route entry's first model attempt.

    ``retry_with_fallback_models`` records the first attempt for every selected
    model/deployment route. Handler retry hooks call this function only before
    subsequent invocations, so ``model_attempts - route_attempts`` is the exact
    retry-attempt count.
    """

    request_options = get_ai_request_options()
    attribution = attribution or (
        request_options.attribution if request_options is not None else None
    )
    adjudication = _get_adjudication_details(attribution)
    if adjudication is None:
        return
    if adjudication.model_attempts is None:
        adjudication.model_attempts = 0
    adjudication.model_attempts += 1


def record_provider_request_attempt(attribution: Optional[str] = None) -> None:
    """Count one exact provider request when SDK-internal retries are disabled.

    A positive or unknown provider retry budget means the SDK may issue hidden
    requests, so those counts remain unavailable rather than being understated.
    """

    request_options = get_ai_request_options()
    attribution = attribution or (
        request_options.attribution if request_options is not None else None
    )
    if request_options is None or request_options.provider_retries != 0:
        return
    adjudication = _get_adjudication_details(attribution)
    if adjudication is None:
        return
    if adjudication.provider_attempts is None:
        adjudication.provider_attempts = 0
    adjudication.provider_attempts += 1


def record_review_profile(profile: str) -> None:
    """Record the selected reviewer profile in the current run metadata."""
    details = get_run_details()
    if details is not None:
        details.review_profile = profile


def record_review_route(route: Mapping[str, Any]) -> None:
    """Record an isolated structured snapshot of the applied review route."""

    details = get_run_details()
    if details is not None:
        details.review_route = deepcopy(dict(route))


def _read_token_field(usage, name: str) -> int:
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    return value if isinstance(value, int) else 0


def _add_token_usage(details, usage) -> None:
    prompt_tokens = _read_token_field(usage, "prompt_tokens")
    completion_tokens = _read_token_field(usage, "completion_tokens")
    total_tokens = _read_token_field(usage, "total_tokens") or (
        prompt_tokens + completion_tokens
    )
    details.prompt_tokens += prompt_tokens
    details.completion_tokens += completion_tokens
    details.total_tokens += total_tokens


def _has_complete_token_usage(usage) -> bool:
    prompt_tokens = _read_token_field(usage, "prompt_tokens")
    completion_tokens = _read_token_field(usage, "completion_tokens")
    total_tokens = _read_token_field(usage, "total_tokens") or (
        prompt_tokens + completion_tokens
    )
    return prompt_tokens > 0 and completion_tokens > 0 and total_tokens > 0


def add_token_usage(usage) -> None:
    """Accumulate token counts from a litellm usage object or dict."""
    details = get_run_details()
    if details is None or usage is None:
        return
    _add_token_usage(details, usage)


def _as_decimal_cost(cost_usd) -> Optional[Decimal]:
    """Normalize a positive finite USD value without introducing float math.

    Zero is rejected on purpose: litellm.completion_cost returns 0.0 both for
    zero-priced model entries (e.g. local/ollama models) and for usage without
    billable tokens, so a zero here means "could not be priced", not "free" —
    recording it would render a false "$0.00" with cost status complete.
    """
    if cost_usd is None or isinstance(cost_usd, bool):
        return None
    try:
        cost = cost_usd if isinstance(cost_usd, Decimal) else Decimal(str(cost_usd))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not cost.is_finite() or cost <= 0:
        return None
    return cost


def record_ai_call(
    usage=None,
    model: Optional[str] = None,
    cost_usd=None,
    attribution: Optional[str] = None,
    provider: Optional[str] = None,
    model_revision: Optional[str] = None,
) -> None:
    """Count one successful AI call and accumulate usage and known cost."""
    details = get_run_details()
    if details is None:
        return
    request_options = get_ai_request_options()
    attribution = attribution or (request_options.attribution if request_options is not None else None)
    if attribution:
        target = _get_adjudication_details(attribution) or _get_specialist_details(attribution)
    else:
        target = details
    if target is None:
        return
    if isinstance(target, AdjudicationRunDetails):
        target.provider = provider
        target.model_revision = model_revision
    target.num_ai_calls += 1
    if usage is not None:
        _add_token_usage(target, usage)
        if isinstance(target, AdjudicationRunDetails) and _has_complete_token_usage(usage):
            target.known_usage_call_count += 1
    cost = _as_decimal_cost(cost_usd)
    if cost is not None:
        target.total_cost_usd += cost
        target.known_cost_call_count += 1
        model_name = model or "unknown"
        target.model_costs_usd[model_name] = target.model_costs_usd.get(model_name, Decimal("0")) + cost


def record_specialist_result(
    role: str,
    *,
    prompt_version: str,
    input_schema_version: str,
    schema_version: str,
    state: str,
    latency_seconds: float,
    confidence: Optional[float] = None,
    failure_reason: Optional[str] = None,
    cached: bool = False,
    input_token_reservation: int = 0,
    output_token_reservation: int = 0,
    output: Optional[Mapping[str, Any]] = None,
    model: Optional[str] = None,
    deployment_id: Optional[str] = None,
    fallback_used: Optional[bool] = None,
) -> None:
    """Finish one role record without changing primary-review telemetry."""

    specialist = _get_specialist_details(role)
    if specialist is None:
        return
    specialist.prompt_version = prompt_version
    specialist.input_schema_version = input_schema_version
    specialist.schema_version = schema_version
    specialist.state = state
    specialist.latency_seconds = max(0.0, float(latency_seconds))
    specialist.confidence = confidence
    specialist.failure_reason = failure_reason
    specialist.cached = cached
    specialist.input_token_reservation = max(0, int(input_token_reservation))
    specialist.output_token_reservation = max(0, int(output_token_reservation))
    specialist.output = deepcopy(output) if output is not None else None
    if model is not None:
        specialist.model_used = model
    if deployment_id is not None:
        specialist.deployment_id = deployment_id
    if fallback_used is not None:
        specialist.fallback_used = fallback_used


def specialist_runs_to_dict(details: Optional[RunDetails] = None) -> dict[str, dict[str, Any]]:
    """Serialize the versioned shadow records for structured telemetry."""

    details = details or get_run_details()
    if details is None:
        return {}
    return {role: run.to_dict() for role, run in sorted(details.specialist_runs.items())}


def record_adjudication_result(
    finding_id: str,
    *,
    provider: Optional[str],
    model_revision: Optional[str],
    model_attempts_configured: Optional[int],
    provider_retries_configured: Optional[int],
    prompt_version: str,
    input_schema_version: str,
    schema_version: str,
    state: str,
    latency_seconds: float,
    confidence: Optional[float] = None,
    failure_reason: Optional[str] = None,
    cache_state: str = "not_requested",
) -> None:
    """Finish one frontier record without changing main-review or specialist totals."""

    adjudication = _get_adjudication_details(f"{_FRONTIER_ATTRIBUTION_PREFIX}{finding_id}")
    if adjudication is None:
        return
    if provider is not None:
        adjudication.provider = provider
    if model_revision is not None:
        adjudication.model_revision = model_revision
    adjudication.model_attempts_configured = model_attempts_configured
    adjudication.provider_retries_configured = provider_retries_configured
    adjudication.prompt_version = prompt_version
    adjudication.input_schema_version = input_schema_version
    adjudication.schema_version = schema_version
    adjudication.state = state
    adjudication.latency_seconds = max(0.0, float(latency_seconds))
    adjudication.confidence = confidence
    adjudication.failure_reason = failure_reason
    adjudication.cache_state = cache_state


def adjudication_runs_to_dict(details: Optional[RunDetails] = None) -> dict[str, dict[str, Any]]:
    """Serialize frontier records without model-generated repository text."""

    details = details or get_run_details()
    if details is None:
        return {}
    return {
        finding_id: run.to_dict()
        for finding_id, run in sorted(details.adjudication_runs.items())
    }
