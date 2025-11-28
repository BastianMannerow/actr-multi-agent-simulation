"""
Enhanced utility layer for pyACT-R integration.

Purpose
-------
- Extend pyACT-R with additional helper functions and ergonomic accessors.
- Patch known issues in the visual search subsystem (``VisualLocation.find``).
- Simplify interaction with ACT-R goal, imaginal, production utilities
  and declarative memory.

Scope
-----
- This module is purely an extension; it does not modify ACT-R theory.
- Designed for applied simulations, debugging, and GUI synchronization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyactr
import pyactr.vision as vision
from pyactr import chunks, utilities
from pyactr.utilities import ACTRError


def fix_pyactr() -> None:
    """
    Monkey-patch pyACT-R's :class:`VisualLocation` search routine.

    The original implementation of :meth:`VisualLocation.find` can
    occasionally mismatch attended objects when
    ``automatic_visual_search=True``. This helper replaces the method
    at runtime with a variant that:

    * resolves attributes from the production RHS more robustly,
    * enforces consistency of the ``attended`` flag and FINST history,
    * treats screen coordinates as absolute constraints when provided, and
    * synthesizes well-formed ``_visuallocation`` chunks for matches.

    Notes
    -----
    This function mutates the global pyACT-R class definition
    (:class:`pyactr.vision.VisualLocation`). Call it once during
    application start-up, before running simulations.
    """
    _original_find = vision.VisualLocation.find  # kept for potential restoration

    def patched_find(self, otherchunk, actrvariables=None, extra_tests=None):
        """
        Search for a visual stimulus that matches the request chunk.

        Parameters
        ----------
        otherchunk :
            The request pattern created from the production RHS.
        actrvariables : dict, optional
            Mapping from variable names to bound values; used to resolve
            variables in ``otherchunk``.
        extra_tests : dict, optional
            Extra constraints used by pyACT-R (for example
            ``{\"attended\": True}``).

        Returns
        -------
        tuple
            ``(visuallocation_chunk, stimulus_dict)``, where the first
            element is a ``_visuallocation`` chunk or ``None`` if no
            match is found, and the second element is the raw stimulus
            dictionary from the environment or ``None``.
        """
        if extra_tests is None:
            extra_tests = {}
        if actrvariables is None:
            actrvariables = {}

        # Resolve all attributes from the production RHS pattern
        try:
            mod_attr_val = {
                x[0]: utilities.check_bound_vars(actrvariables, x[1], negative_impossible=False)
                for x in otherchunk.removeunused()
            }
        except ACTRError as e:
            raise ACTRError(f"The chunk '{otherchunk}' is not defined correctly; {e}")
        chunk_used_for_search = chunks.Chunk(utilities.VISUALLOCATION, **mod_attr_val)

        found, found_stim = None, None

        # Iterate over all stimuli present in the environment
        for each in self.environment.stimulus:
            stim_attrs = self.environment.stimulus[each]

            # Enforce attended flag and FINST history
            attended_flag = extra_tests.get("attended", None)
            if attended_flag in (False, "False"):
                # Request: unattended item
                if self.finst and stim_attrs in self.recent:
                    continue
            elif attended_flag not in (False, "False") and attended_flag is not None:
                # Request: attended item
                if self.finst and stim_attrs not in self.recent:
                    continue

            # Optional text-value filter
            if (
                chunk_used_for_search.value != chunk_used_for_search.EmptyValue()
                and chunk_used_for_search.value.values != stim_attrs.get("text")
            ):
                continue

            # Extract pixel coordinates
            position = (int(stim_attrs["position"][0]), int(stim_attrs["position"][1]))

            # Screen coordinate constraints (absolute equality)
            try:
                if (
                    chunk_used_for_search.screen_x.values
                    and int(chunk_used_for_search.screen_x.values) != position[0]
                ):
                    continue
            except (TypeError, ValueError, AttributeError):
                pass
            try:
                if (
                    chunk_used_for_search.screen_y.values
                    and int(chunk_used_for_search.screen_y.values) != position[1]
                ):
                    continue
            except (TypeError, ValueError, AttributeError):
                pass

            # Build a visible-location chunk from the stimulus attributes
            found_stim = stim_attrs
            filtered = {
                k: stim_attrs[k]
                for k in stim_attrs
                if k not in ("position", "text", "vis_delay")
            }
            visible_chunk = chunks.makechunk(
                nameofchunk="vis1",
                typename="_visuallocation",
                **filtered,
            )

            # Check for structural compatibility with the query chunk
            if visible_chunk <= chunk_used_for_search:
                temp_dict = visible_chunk._asdict()
                temp_dict.update({"screen_x": position[0], "screen_y": position[1]})
                found = chunks.Chunk(utilities.VISUALLOCATION, **temp_dict)
                break  # return first compatible match

        return found, found_stim

    vision.VisualLocation.find = patched_find


# ---------------------------------------------------------------------------
# ACT-R event utilities
# ---------------------------------------------------------------------------


def request_if_production_fired(agent_construct: Any) -> bool:
    """
    Determine whether the current event corresponds to a fired production.

    Parameters
    ----------
    agent_construct :
        Object exposing ``simulation.current_event`` (for example an
        AgentConstruct wrapper).

    Returns
    -------
    bool
        ``True`` if the current event encodes a fired production,
        otherwise ``False``.
    """
    return get_production_fired(agent_construct) is not None


def get_production_fired(agent: Any) -> Optional[str]:
    """
    Return the name of the currently fired production, if any.

    Parameters
    ----------
    agent :
        Object exposing ``simulation.current_event``.

    Returns
    -------
    str or None
        The production name if the current event contains a
        ``"RULE FIRED: <name>"`` marker, otherwise ``None``.
    """
    try:
        event = agent.simulation.current_event
    except AttributeError:
        return None

    # pyactr events typically expose the action both as attribute
    # and as tuple index.
    action = getattr(event, "action", None)
    if action is None:
        try:
            action = event[2]
        except Exception:
            return None

    if isinstance(action, str) and "RULE FIRED: " in action:
        return action.replace("RULE FIRED: ", "")
    return None


def request_if_key_pressed(agent_construct: Any) -> bool:
    """
    Determine whether the current event represents a manual key press.

    Parameters
    ----------
    agent_construct :
        Object exposing ``simulation.current_event``.

    Returns
    -------
    bool
        ``True`` if the current event is a KEY PRESSED event from the
        manual module, otherwise ``False``.
    """
    return key_pressed(agent_construct) is not None


def key_pressed(agent_construct: Any) -> Optional[str]:
    """
    Extract a manual key press from the current event, if present.

    Parameters
    ----------
    agent_construct :
        Object exposing ``simulation.current_event``.

    Returns
    -------
    str or None
        The last pressed character (single-character string) or
        ``None`` if the current event is not a key press.
    """
    try:
        event = agent_construct.simulation.current_event
    except AttributeError:
        return None

    module = getattr(event, "module", None)
    if module is None:
        try:
            module = event[1]
        except Exception:
            module = None

    action = getattr(event, "action", None)
    if action is None:
        try:
            action = event[2]
        except Exception:
            return None

    if module == "manual" and isinstance(action, str) and "KEY PRESSED:" in action:
        # For simple alphanumeric keys the last character is the key;
        # for multi-character labels (for example "SPACE") this preserves
        # the previous behavior by returning the last character.
        return action[-1]
    return None


# ---------------------------------------------------------------------------
# Goal and imaginal utilities
# ---------------------------------------------------------------------------


def get_goal(agent_construct: Any):
    """
    Return the agent's primary goal buffer (key ``"g"``).

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.

    Returns
    -------
    Buffer or None
        The goal buffer if present, otherwise ``None``.
    """
    key = "g"
    return agent_construct.actr_agent.goals.get(key, None)


def set_goal(agent_construct: Any, chunk: chunks.Chunk) -> None:
    """
    Insert a chunk into the primary goal buffer.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    chunk : Chunk
        :class:`pyactr.chunks.Chunk` instance to be added to the goal buffer.
    """
    first_goal = next(iter(agent_construct.actr_agent.goals.values()))
    first_goal.add(chunk)


def get_imaginal(agent_construct: Any, key: str):
    """
    Retrieve a buffer from ``actr_agent.goals`` by name.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    key : str
        Buffer name, for example ``"imaginal"``.

    Returns
    -------
    Buffer or None
        The buffer object if it exists; otherwise ``None`` and a
        short diagnostic message is printed.
    """
    goals = agent_construct.actr_agent.goals
    if key not in goals:
        print(f"'{key}' not found. Available buffers: {list(goals.keys())}")
        return None
    return goals[key]


def set_imaginal(agent_construct: Any, new_chunk: chunks.Chunk, key: str) -> None:
    """
    Write a chunk into a named buffer (for example the imaginal buffer).

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    new_chunk : Chunk
        :class:`pyactr.chunks.Chunk` to be inserted into the buffer.
    key : str
        Buffer name.

    Raises
    ------
    TypeError
        If the target buffer does not implement an ``add`` method.
    """
    goals = agent_construct.actr_agent.goals
    if key not in goals:
        print(f"Buffer '{key}' not found. Available keys: {list(goals.keys())}")
        return

    target = goals[key]
    try:
        target.add(new_chunk)
    except AttributeError as exc:
        raise TypeError(f"Goal object for '{key}' does not support '.add()'.") from exc


# ---------------------------------------------------------------------------
# Production rule utilities
# ---------------------------------------------------------------------------


def update_utility(agent_construct: Any, production_name: str, utility: float) -> None:
    """
    Set the utility value of an existing production.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    production_name : str
        Name of the production to update.
    utility : float
        New utility value.
    """
    agent_construct.actr_agent.productions[production_name]["utility"] = utility


def get_production_utility(
    agent_construct: Any,
    production_name: str,
) -> Optional[float]:
    """
    Return the utility of a production, if available.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    production_name : str
        Name of the production.

    Returns
    -------
    float or None
        Utility value, or ``None`` if the production or its utility
        entry does not exist.
    """
    try:
        return agent_construct.actr_agent.productions[production_name]["utility"]
    except KeyError:
        return None


def add_production(
    agent_construct: Any,
    name: str,
    string: str,
    utility: Optional[float] = None,
) -> None:
    """
    Add a new production to the model.

    This is a convenience wrapper around
    :meth:`ACTRModel.productionstring` that optionally sets an initial
    utility.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    name : str
        Symbolic name of the production.
    string : str
        Production specification in pyACT-R's string format.
    utility : float, optional
        Initial utility value. If ``None``, the pyACT-R default is used.
    """
    model = agent_construct.actr_agent
    model.productionstring(name=name, string=string)
    if utility is not None:
        update_utility(agent_construct, name, utility)


def get_all_productions(agent_construct: Any) -> Dict[str, Dict[str, Any]]:
    """
    Return a shallow copy of the internal production structure.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.

    Returns
    -------
    dict
        Mapping from production names to their metadata dictionaries.
        Mutating this copy does not affect the underlying model.
    """
    return dict(agent_construct.actr_agent.productions)


# ---------------------------------------------------------------------------
# Declarative memory utilities
# ---------------------------------------------------------------------------


def get_declarative_memory(agent_construct: Any):
    """
    Return the agent's declarative memory (ACTRDM instance).

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.

    Returns
    -------
    DeclarativeMemory
        The declarative memory object (typically an ``ACTRDM`` instance).
    """
    return agent_construct.actr_agent.decmem


def add_to_declarative_memory(agent_construct: Any, chunk: chunks.Chunk) -> None:
    """
    Insert a chunk into declarative memory.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    chunk : Chunk
        :class:`pyactr.chunks.Chunk` to be stored.
    """
    agent_construct.actr_agent.decmem.add(chunk)


def get_declarative_chunk_type(agent_construct: Any, typename: str):
    """
    Collect all chunks of a given type from declarative memory.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    typename : str
        Chunk type (``chunk.typename``) to filter on.

    Returns
    -------
    list of Chunk
        All chunks in declarative memory whose ``typename`` matches
        ``typename``.
    """
    dm = agent_construct.actr_agent.decmem
    # In pyactr, the declarative memory is dict-like with chunks as keys.
    return [chunk for chunk in dm.keys() if getattr(chunk, "typename", None) == typename]


def delete_declarative_chunk_type(agent_construct: Any, typename: str) -> int:
    """
    Remove all chunks of a given type from declarative memory.

    Parameters
    ----------
    agent_construct :
        Wrapper exposing ``agent_construct.actr_agent``.
    typename : str
        Chunk type (``chunk.typename``) to remove.

    Returns
    -------
    int
        Number of deleted chunks.
    """
    dm = agent_construct.actr_agent.decmem
    to_delete = [chunk for chunk in dm.keys() if getattr(chunk, "typename", None) == typename]
    for chunk in to_delete:
        del dm[chunk]
    return len(to_delete)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


def build_chunkstring_by_tuples(pairs: Sequence[Tuple[str, Any]]):
    """
    Build a chunk from a sequence of ``(slot, value)`` pairs.

    The first tuple is expected to be ``("isa", <chunk_type>)``; all
    subsequent pairs are interpreted as regular slot-value assignments.

    Parameters
    ----------
    pairs : sequence of (str, Any)
        Slot/value pairs in the desired order of appearance in the
        chunk specification.

    Returns
    -------
    Chunk
        The resulting :class:`pyactr.chunks.Chunk` instance.

    Raises
    ------
    ValueError
        If ``pairs`` is empty.
    """
    if not pairs:
        raise ValueError("At least one (slot, value) tuple is required to build a chunk.")

    lines: List[str] = []
    for slot, value in pairs:
        # pyactr.chunkstring expects a simple "slot value" syntax per line.
        # Values are converted to strings; quoting for multi-word values
        # must be handled by the caller if required.
        val_str = "None" if value is None else str(value)
        lines.append(f"{slot} {val_str}")

    chunk_spec = "\n".join(lines)
    return pyactr.chunkstring(string=chunk_spec)
