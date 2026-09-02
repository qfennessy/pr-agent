from typing import Any


def parse_env_bool(value: Any) -> bool | None:
    """Parse boolean values commonly supplied through environment-backed settings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None
