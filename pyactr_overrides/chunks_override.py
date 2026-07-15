"""Cache pyactr's pyparsing grammar and parsed chunk templates."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from threading import RLock
from typing import Any, Callable

import pyactr
from pyactr import chunks, utilities

_LOCK = RLock()
_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = utilities.getchunk()
    return _PARSER


@lru_cache(maxsize=8192)
def _parsed_template(string: str):
    with _LOCK:
        parsed = _parser().parse_string(string, parse_all=True)
        try:
            typename, values = chunks.createchunkdict(parsed)
        except utilities.ACTRError as exc:
            raise utilities.ACTRError(
                f"The chunk string {string} is not defined correctly; {exc}"
            ) from exc
        return typename, values


def cached_chunkstring(name: str = "", string: str = ""):
    typename, values = _parsed_template(str(string))
    # Chunk slot wrappers carry mutable binding state.  Copy the compact parsed
    # template before constructing a new runtime chunk.
    return chunks.makechunk(name, typename, **deepcopy(values))


def apply(remember: Callable[[str, Any, str], None]) -> None:
    remember("chunks.chunkstring", chunks, "chunkstring")
    remember("pyactr.chunkstring", pyactr, "chunkstring")
    chunks.chunkstring = cached_chunkstring
    pyactr.chunkstring = cached_chunkstring


def clear_caches() -> None:
    global _PARSER
    _parsed_template.cache_clear()
    _PARSER = None
