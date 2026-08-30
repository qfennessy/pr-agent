"""Immutable, coroutine-local controls for one AI model route."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class AIRequestOptions:
    """Request controls that must not be written into shared Dynaconf state."""

    deployment_id: Optional[str] = None
    timeout_seconds: Optional[float] = None
    model_retries: Optional[int] = None
    provider_retries: Optional[int] = None
    max_output_tokens: Optional[int] = None
    attribution: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.model_retries is not None and self.model_retries < 1:
            raise ValueError("model_retries must be at least 1")
        if self.provider_retries is not None and self.provider_retries < 0:
            raise ValueError("provider_retries cannot be negative")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if self.attribution is not None and not self.attribution.strip():
            raise ValueError("attribution cannot be blank")


@dataclass(frozen=True)
class AIModelRoute:
    """An ordered primary/fallback route plus immutable request controls."""

    models: tuple[str, ...]
    deployments: tuple[Optional[str], ...]
    timeout_seconds: Optional[float] = None
    model_retries: Optional[int] = None
    provider_retries: Optional[int] = None
    max_output_tokens: Optional[int] = None
    attribution: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.models or any(not isinstance(model, str) or not model.strip() for model in self.models):
            raise ValueError("models must contain at least one non-blank model")
        if len(self.deployments) != len(self.models):
            raise ValueError("deployments must contain one entry per model")
        # Reuse the per-attempt validator for the route-wide controls.
        AIRequestOptions(
            timeout_seconds=self.timeout_seconds,
            model_retries=self.model_retries,
            provider_retries=self.provider_retries,
            max_output_tokens=self.max_output_tokens,
            attribution=self.attribution,
        )

    def options_for_attempt(self, attempt: int) -> AIRequestOptions:
        return AIRequestOptions(
            deployment_id=self.deployments[attempt],
            timeout_seconds=self.timeout_seconds,
            model_retries=self.model_retries,
            provider_retries=self.provider_retries,
            max_output_tokens=self.max_output_tokens,
            attribution=self.attribution,
        )


_ai_request_options: ContextVar[Optional[AIRequestOptions]] = ContextVar(
    "pr_agent_ai_request_options", default=None
)


def get_ai_request_options() -> Optional[AIRequestOptions]:
    """Return the controls for the current coroutine, when one is active."""

    return _ai_request_options.get()


@contextmanager
def use_ai_request_options(options: AIRequestOptions) -> Iterator[None]:
    """Install request controls for one retry attempt and restore the prior context."""

    token = _ai_request_options.set(options)
    try:
        yield
    finally:
        _ai_request_options.reset(token)
