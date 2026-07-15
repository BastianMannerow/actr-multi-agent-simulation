"""Indexed candidate selection for pyactr declarative retrieval."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable

from pyactr import chunks, declarative, utilities

_ORIGINAL_SETITEM = None
_ORIGINAL_DELITEM = None


def _touch(memory: Any) -> None:
    memory._experimental_perf_revision = int(
        getattr(memory, "_experimental_perf_revision", 0)
    ) + 1


def indexed_setitem(self, key, time):
    if _ORIGINAL_SETITEM is None:
        raise RuntimeError("Declarative performance override was not initialized.")
    _ORIGINAL_SETITEM(self, key, time)
    _touch(self)


def indexed_delitem(self, key):
    if _ORIGINAL_DELITEM is None:
        raise RuntimeError("Declarative performance override was not initialized.")
    _ORIGINAL_DELITEM(self, key)
    _touch(self)


def _slot_value(value: Any) -> Any | None:
    try:
        split = utilities.splitting(value)
    except Exception:
        return None
    if split.variables or split.negvariables or split.negvalues:
        return None
    return split.values


def _ensure_index(
    memory: Any,
) -> tuple[dict[str, tuple[Any, ...]], dict[tuple[str, Any], frozenset[Any]]]:
    revision = int(getattr(memory, "_experimental_perf_revision", 0))
    cached_revision = getattr(memory, "_experimental_perf_index_revision", None)
    if cached_revision == revision:
        return memory._experimental_perf_type_index, memory._experimental_perf_slot_index

    by_type: dict[str, list[Any]] = defaultdict(list)
    by_slot: dict[tuple[str, Any], set[Any]] = defaultdict(set)
    for chunk in memory:
        by_type[str(getattr(chunk, "typename", ""))].append(chunk)
        try:
            for slot_name, raw_value in chunk:
                value = _slot_value(raw_value)
                if value is not None:
                    by_slot[(str(slot_name), value)].add(chunk)
        except Exception:
            continue
    type_index = {name: tuple(values) for name, values in by_type.items()}
    slot_index = {key: frozenset(values) for key, values in by_slot.items()}
    memory._experimental_perf_type_index = type_index
    memory._experimental_perf_slot_index = slot_index
    memory._experimental_perf_index_revision = revision
    return type_index, slot_index


def _candidates(memory: Any, query: Any, *, partial_matching: bool) -> Iterable[Any]:
    type_index, slot_index = _ensure_index(memory)
    typename = str(getattr(query, "typename", ""))
    candidates: set[Any] = set(type_index.get(typename, ()))
    if not candidates:
        # Conservative fallback protects custom models that intentionally rely
        # on pyactr's permissive cross-type matching behavior.
        candidates = set(memory)
    if partial_matching:
        return tuple(candidates)

    try:
        for slot_name, raw_value in query.removeunused():
            value = _slot_value(raw_value)
            if value is None:
                continue
            exact = slot_index.get((str(slot_name), value))
            if exact is None:
                return ()
            candidates.intersection_update(exact)
            if not candidates:
                return ()
    except Exception:
        return tuple(candidates)
    return tuple(candidates)


def indexed_retrieve(
    self,
    time,
    otherchunk,
    actrvariables,
    buffers,
    extra_tests,
    model_parameters,
):
    """Equivalent to pyactr 0.3.2 retrieval with an indexed candidate set."""
    model_parameters = model_parameters.copy()
    model_parameters.update(self.model_parameters)

    if actrvariables is None:
        actrvariables = {}
    try:
        mod_attr_val = {
            item[0]: utilities.check_bound_vars(
                actrvariables, item[1], negative_impossible=False
            )
            for item in otherchunk.removeunused()
        }
    except utilities.ACTRError as exc:
        raise utilities.ACTRError(
            f"Retrieving the chunk '{otherchunk}' is impossible; {exc}"
        ) from exc
    chunk_tobe_matched = chunks.Chunk(otherchunk.typename, **mod_attr_val)

    max_activation = float("-inf")
    retrieved = None
    partial = bool(model_parameters["partial_matching"])
    for chunk in _candidates(
        self.dm, chunk_tobe_matched, partial_matching=partial
    ):
        try:
            recently = extra_tests["recently_retrieved"]
            if recently is False or recently == "False":
                if self._DecMemBuffer__finst and chunk in self.recent:
                    continue
            elif self._DecMemBuffer__finst and chunk not in self.recent:
                continue
        except KeyError:
            pass

        if model_parameters["subsymbolic"]:
            activation_partial = 0
            if partial:
                activation_partial = chunk_tobe_matched.match(
                    chunk,
                    partialmatching=True,
                    mismatch_penalty=model_parameters["mismatch_penalty"],
                )
            elif not chunk_tobe_matched <= chunk:
                continue
            try:
                activation_base = utilities.baselevel_learning(
                    time,
                    self.dm[chunk],
                    model_parameters["baselevel_learning"],
                    model_parameters["decay"],
                    self.dm.activations.get(chunk),
                    optimized_learning=model_parameters["optimized_learning"],
                )
            except UnboundLocalError:
                continue
            if math.isnan(activation_base):
                raise utilities.ACTRError(
                    "The following chunk cannot receive base activation: "
                    f"{chunk}. One of its traces did not appear in the past."
                )
            activation_spreading = utilities.spreading_activation(
                chunk,
                buffers,
                self.dm,
                model_parameters["buffer_spreading_activation"],
                model_parameters["strength_of_association"],
                model_parameters["spreading_activation_restricted"],
                model_parameters["association_only_from_chunks"],
            )
            noise = utilities.calculate_instantaneous_noise(
                model_parameters["instantaneous_noise"]
            )
            activation = (
                activation_base
                + activation_spreading
                + activation_partial
                + noise
            )
            if utilities.retrieval_success(
                activation, model_parameters["retrieval_threshold"]
            ) and max_activation < activation:
                max_activation = activation
                self.activation = max_activation
                retrieved = chunk
                extra_time = utilities.retrieval_latency(
                    activation,
                    model_parameters["latency_factor"],
                    model_parameters["latency_exponent"],
                )
                if model_parameters["activation_trace"]:
                    print("(Partially) matching chunk:", chunk)
                    print("Base level learning:", activation_base)
                    print("Spreading activation", activation_spreading)
                    print("Partial matching", activation_partial)
                    print("Noise:", noise)
                    print("Total activation", activation)
                    print("Time to retrieve", extra_time)
        elif chunk_tobe_matched <= chunk and self.dm[chunk][0] != time:
            retrieved = chunk
            extra_time = model_parameters["rule_firing"]

    if not retrieved:
        if model_parameters["subsymbolic"]:
            extra_time = utilities.retrieval_latency(
                model_parameters["retrieval_threshold"],
                model_parameters["latency_factor"],
                model_parameters["latency_exponent"],
            )
        else:
            extra_time = model_parameters["rule_firing"]
    if self._DecMemBuffer__finst:
        self.recent.append(retrieved)
        if self._DecMemBuffer__finst < len(self.recent):
            self.recent.popleft()
    return retrieved, extra_time


def apply(remember: Callable[[str, Any, str], None]) -> None:
    global _ORIGINAL_SETITEM, _ORIGINAL_DELITEM
    remember("DecMem.__setitem__", declarative.DecMem, "__setitem__")
    remember("DecMem.__delitem__", declarative.DecMem, "__delitem__")
    remember("DecMemBuffer.retrieve", declarative.DecMemBuffer, "retrieve")
    if _ORIGINAL_SETITEM is None:
        _ORIGINAL_SETITEM = declarative.DecMem.__setitem__
    if _ORIGINAL_DELITEM is None:
        _ORIGINAL_DELITEM = declarative.DecMem.__delitem__
    declarative.DecMem.__setitem__ = indexed_setitem
    declarative.DecMem.__delitem__ = indexed_delitem
    declarative.DecMemBuffer.retrieve = indexed_retrieve


def clear_caches() -> None:
    return None
