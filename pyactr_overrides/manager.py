"""Activation manager for reversible experimental pyactr overrides."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from typing import Any

from pyactr_overrides import (
    chunks_override,
    declarative_override,
    productions_override,
    simulation_override,
)

_LOCK = RLock()
_ACTIVE = False
_MISSING = object()
_ORIGINALS: dict[str, tuple[Any, str, Any]] = {}
_SUPPORTED_PYACTR = "0.3.2"


def remember(key: str, owner: Any, attribute: str) -> None:
    """Remember an original attribute exactly once."""
    if key not in _ORIGINALS:
        _ORIGINALS[key] = (
            owner,
            attribute,
            getattr(owner, attribute, _MISSING),
        )


def _restore_all() -> None:
    for owner, attribute, original in reversed(list(_ORIGINALS.values())):
        if original is _MISSING:
            try:
                delattr(owner, attribute)
            except AttributeError:
                pass
        else:
            setattr(owner, attribute, original)
    chunks_override.clear_caches()
    declarative_override.clear_caches()
    productions_override.clear_caches()
    simulation_override.clear_caches()


def _validate_version() -> None:
    try:
        installed = version("pyactr")
    except PackageNotFoundError as exc:
        raise RuntimeError("pyactr is not installed.") from exc
    if installed != _SUPPORTED_PYACTR:
        raise RuntimeError(
            "Experimental pyactr performance boost supports pyactr "
            f"{_SUPPORTED_PYACTR}; installed version is {installed}."
        )


def configure(enabled: bool) -> bool:
    """Apply or restore all overrides and return the resulting active state."""
    global _ACTIVE
    requested = bool(enabled)
    with _LOCK:
        if requested == _ACTIVE:
            return _ACTIVE
        if not requested:
            _restore_all()
            _ACTIVE = False
            return False

        _validate_version()
        try:
            chunks_override.apply(remember)
            declarative_override.apply(remember)
            productions_override.apply(remember)
            simulation_override.apply(remember)
        except Exception:
            _restore_all()
            _ACTIVE = False
            raise
        _ACTIVE = True
        return True


def is_active() -> bool:
    return _ACTIVE


def status() -> dict[str, Any]:
    return {
        "active": _ACTIVE,
        "supported_pyactr": _SUPPORTED_PYACTR,
        "overrides": (
            "cached chunk parser",
            "indexed declarative retrieval",
            "goal-state production prefilter",
            "lower-priority procedural scheduling barrier",
            "pyactr event peek/step helpers",
        )
        if _ACTIVE
        else (),
    }
