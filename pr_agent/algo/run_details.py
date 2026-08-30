"""Details of a single PR-Agent run, collected while the command executes.

The data is held in a ``ContextVar`` so that the AI handler can record token
usage without changing ``chat_completion``'s return signature. Context vars are
copied into ``asyncio`` child tasks while still referencing the same mutable
``RunDetails`` object, so concurrent AI calls accumulate into one instance and
stay isolated between concurrent requests.
"""

import time
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional

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


def _get_specialist_details(attribution: str) -> Optional[SpecialistRunDetails]:
    details = get_run_details()
    if details is None:
        return None
    return details.specialist_runs.setdefault(attribution, SpecialistRunDetails(role=attribution))


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
) -> None:
    """Preserve the last attempted specialist route even when its output is rejected."""

    if not attribution:
        return
    specialist = _get_specialist_details(attribution)
    if specialist is None:
        return
    specialist.model_used = model
    specialist.deployment_id = deployment_id
    if is_fallback:
        specialist.fallback_used = True


def record_review_profile(profile: str) -> None:
    """Record the selected reviewer profile in the current run metadata."""
    details = get_run_details()
    if details is not None:
        details.review_profile = profile


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
) -> None:
    """Count one successful AI call and accumulate usage and known cost."""
    details = get_run_details()
    if details is None:
        return
    request_options = get_ai_request_options()
    attribution = attribution or (request_options.attribution if request_options is not None else None)
    target = _get_specialist_details(attribution) if attribution else details
    if target is None:
        return
    target.num_ai_calls += 1
    if usage is not None:
        _add_token_usage(target, usage)
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
