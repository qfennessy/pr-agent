import os

import pytest

# LiteLLM downloads its live model registry from GitHub at import time unless this is
# set, so upstream edits to that file change test results without any code change
# (e.g. cohere/command-r-plus disappearing from it after 2026-09-14). Use the registry
# bundled with the pinned litellm version instead. Set before any test imports litellm;
# an explicit environment value still wins.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def isolate_run_details():
    """Start each test from a clean run-details ContextVar and restore it.

    The collector lives in a module-level ContextVar, so a test that leaves
    details behind would otherwise be visible to whichever test runs next.
    """
    from pr_agent.algo import run_details

    token = run_details._run_details.set(None)
    yield
    run_details._run_details.reset(token)
