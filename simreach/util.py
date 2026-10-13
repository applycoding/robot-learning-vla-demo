"""Small shared helpers."""

from __future__ import annotations

from typing import Never


def assert_never(value: Never) -> Never:
    """Fail if a match over an enum or union grows a new variant."""
    raise AssertionError(f"Unhandled variant: {value!r}")
