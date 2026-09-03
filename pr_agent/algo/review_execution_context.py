"""Coroutine-local controls for isolated review execution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_isolated_review_execution: ContextVar[bool] = ContextVar(
    "pr_agent_isolated_review_execution",
    default=False,
)


def review_execution_is_isolated() -> bool:
    """Return whether the current review must avoid output and shared request state."""

    return _isolated_review_execution.get()


@contextmanager
def isolate_review_execution() -> Iterator[None]:
    """Suppress review output and shared request-state writes for one coroutine."""

    token = _isolated_review_execution.set(True)
    try:
        yield
    finally:
        _isolated_review_execution.reset(token)
