import time

from pr_agent.log import get_logger


class _ProviderRegistry:
    """The providers pr-agent created — the only ones its flush/shutdown may touch."""

    def __init__(self):
        self._providers = []

    def __len__(self):
        return len(self._providers)

    def register(self, provider):
        self._providers.append(provider)

    def flush_all(self, timeout_millis=3000):
        timeout_millis = max(0, int(timeout_millis))
        deadline_ns = time.monotonic_ns() + timeout_millis * 1_000_000

        for index, provider in enumerate(list(self._providers)):
            remaining_millis = timeout_millis
            if index:
                remaining_ns = max(0, deadline_ns - time.monotonic_ns())
                remaining_millis = remaining_ns // 1_000_000
            try:
                provider.force_flush(remaining_millis)
            except Exception as e:
                get_logger().warning(f"Error flushing telemetry: {e}")

    def shutdown_all(self):
        for provider in list(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                get_logger().warning(f"Error shutting down telemetry: {e}")
        self._providers.clear()

    def reset(self):
        """Forget registrations without shutting anything down (test seam)."""
        self._providers.clear()


provider_registry = _ProviderRegistry()
