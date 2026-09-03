"""Coroutine-local controls for isolated review execution."""

from __future__ import annotations

import datetime
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_isolated_review_execution: ContextVar[bool] = ContextVar(
    "pr_agent_isolated_review_execution",
    default=False,
)
_review_prompt_date: ContextVar[Optional[str]] = ContextVar(
    "pr_agent_review_prompt_date",
    default=None,
)


def review_execution_is_isolated() -> bool:
    """Return whether the current review must avoid output and shared request state."""

    return _isolated_review_execution.get()


def get_review_prompt_date() -> str:
    """Return the request-pinned prompt date, or today's date for normal reviews."""

    pinned_date = _review_prompt_date.get()
    if pinned_date is not None:
        return pinned_date
    return datetime.datetime.now().strftime("%Y-%m-%d")


@contextmanager
def pin_review_prompt_date(prompt_date: str) -> Iterator[None]:
    """Pin the prompt date for one review request, including an intentionally blank date."""

    token = _review_prompt_date.set(prompt_date)
    try:
        yield
    finally:
        _review_prompt_date.reset(token)


@contextmanager
def isolate_review_execution() -> Iterator[None]:
    """Suppress review output and shared request-state writes for one coroutine."""

    token = _isolated_review_execution.set(True)
    try:
        yield
    finally:
        _isolated_review_execution.reset(token)
