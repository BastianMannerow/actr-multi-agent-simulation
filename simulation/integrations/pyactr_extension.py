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
            # Application metadata is deliberately kept outside the pyactr
            # stimulus dictionary. Standard _visuallocation chunks only need
            # screen coordinates; arbitrary keys would become unsupported
            # chunk slots and can crash chunk construction.
            filtered = {}
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


def sanitize_visual_frame(frame: Any) -> dict[str, dict[str, Any]]:
    """Return a pyactr-safe visual frame.

    pyactr converts every non-reserved stimulus key into a chunk slot. Only
    ``text``, ``position`` and optional ``vis_delay`` are therefore accepted
    here. This protects both visual-location and visual-buffer construction.
    """
    if not isinstance(frame, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_stimulus in frame.items():
        if not isinstance(raw_stimulus, dict):
            continue
        text = raw_stimulus.get("text")
        position = raw_stimulus.get("position")
        try:
            if text is None or position is None or len(position) != 2:
                continue
            x, y = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            continue
        stimulus: dict[str, Any] = {
            "text": str(text),
            "position": (x, y),
        }
        if raw_stimulus.get("vis_delay") is not None:
            try:
                stimulus["vis_delay"] = float(raw_stimulus["vis_delay"])
            except (TypeError, ValueError):
                pass
        result[str(raw_key)] = stimulus
    return result


def publish_visual_stimulus(agent_construct: Any) -> dict[str, dict[str, Any]]:
    """Publish the latest frame and refresh automatic pyactr visual buffers.

    The environment is shared by the model's visual modules. The currently
    scheduled agent publishes its own frame immediately before a cognitive
    step. Explicit visual requests then read the updated environment, while
    automatic visual-location/visual buffers are refreshed using pyactr's own
    chunk constructors and buffer APIs.
    """
    stimuli = getattr(agent_construct, "stimuli", None)
    raw_frame = stimuli[0] if isinstance(stimuli, list) and stimuli else stimuli
    frame = sanitize_visual_frame(raw_frame)
    environment = getattr(agent_construct, "actr_environment", None)
    if environment is None:
        return frame
    # ``Environment.output`` prints every update when pyactr GUI mode is off.
    # Direct assignment is equivalent for perception and keeps the application
    # console clean; scheduled pyactr environment events still use output().
    environment.stimulus = frame

    simulation = getattr(agent_construct, "simulation", None)
    buffers = getattr(simulation, "_Simulation__buffers", None)
    if not isinstance(buffers, dict):
        return frame

    model = getattr(agent_construct, "actr_agent", None)
    parameters = dict(getattr(model, "model_parameters", {}) or {})
    values = list(frame.values())
    current_time = float(getattr(agent_construct, "actr_time", 0.0))
    buffers_changed = False

    for buffer in buffers.values():
        try:
            if getattr(buffer, "state", None) == getattr(buffer, "_BUSY", object()):
                continue

            if isinstance(buffer, vision.VisualLocation):
                if not bool(parameters.get("automatic_visual_search", True)):
                    continue
                new_chunk, found_stimulus = buffer.automatic_search(values)
                current_chunk = next(iter(buffer), None) if buffer else None
                if new_chunk is None:
                    if buffer:
                        buffer.clear(current_time)
                        buffers_changed = True
                elif current_chunk is None:
                    buffer.add(new_chunk, found_stimulus, current_time)
                    buffers_changed = True
                elif str(current_chunk) != str(new_chunk):
                    buffer.modify(new_chunk, found_stimulus)
                    buffers_changed = True
                buffer.state = buffer._FREE
                continue

            if isinstance(buffer, vision.Visual) and bool(
                getattr(buffer, "attend_automatic", False)
            ):
                selected = _foveal_stimulus(buffer, values)
                if selected is None:
                    if buffer:
                        buffer.clear(current_time)
                        buffers_changed = True
                    buffer.state = buffer._FREE
                    buffer.autoattending = buffer._FREE
                    continue
                new_chunk, _encoding = buffer.automatic_buffering(
                    selected,
                    parameters,
                )
                current_chunk = next(iter(buffer), None) if buffer else None
                if current_chunk is None:
                    buffer.add(new_chunk, current_time)
                    buffers_changed = True
                elif str(current_chunk) != str(new_chunk):
                    buffer.modify(new_chunk)
                    buffers_changed = True
                buffer.state = buffer._FREE
                buffer.autoattending = buffer._FREE
        except (ACTRError, AttributeError, KeyError, TypeError, ValueError):
            # A malformed individual object must not invalidate the complete
            # frame. Explicit production requests can still use the published
            # safe environment stimulus.
            continue

    if buffers_changed:
        activation = getattr(simulation, "_Simulation__proc_activate", None)
        try:
            if activation is not None and not activation.triggered:
                activation.succeed()
        except (AttributeError, RuntimeError):
            pass
    return frame


def _foveal_stimulus(
    visual_buffer: Any,
    stimuli: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not stimuli:
        return None
    environment = visual_buffer.environment
    try:
        foveal_distance = utilities.calculate_distance(
            1,
            environment.size,
            environment.simulated_screen_size,
            environment.viewing_distance,
        )
        focus = tuple(visual_buffer.current_focus)
        candidates = [
            stimulus
            for stimulus in stimuli
            if abs(float(stimulus["position"][0]) - float(focus[0])) < foveal_distance
            and abs(float(stimulus["position"][1]) - float(focus[1])) < foveal_distance
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        candidates = list(stimuli)
        focus = (0.0, 0.0)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda stimulus: (
            float(stimulus["position"][0]) - float(focus[0])
        ) ** 2
        + (
            float(stimulus["position"][1]) - float(focus[1])
        ) ** 2,
    )


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
    try:
        first_goal.clear()
    except AttributeError:
        pass
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


def get_buffer(agent_construct: Any, name: str):
    """Return any named ACT-R buffer if it exists on the model or running simulation."""
    simulation = getattr(agent_construct, "simulation", None)
    simulation_buffers = getattr(simulation, "_Simulation__buffers", None)
    if isinstance(simulation_buffers, dict) and name in simulation_buffers:
        return simulation_buffers[name]

    model = getattr(agent_construct, "actr_agent", None)
    for attribute in ("_ACTRModel__buffers", "goals", "retrievals", "visbuffers"):
        mapping = getattr(model, attribute, None)
        if isinstance(mapping, dict) and name in mapping:
            return mapping[name]
    return None


def replace_buffer(agent_construct: Any, name: str, chunk: chunks.Chunk) -> None:
    """Replace the content of a named buffer with exactly one chunk."""
    buffer = get_buffer(agent_construct, name)
    if buffer is None:
        raise KeyError(f"Buffer '{name}' does not exist.")
    try:
        buffer.clear()
    except AttributeError:
        pass
    if chunk is not None:
        buffer.add(chunk)


def set_buffer(agent_construct: Any, name: str, chunk: chunks.Chunk) -> None:
    """Compatibility alias for :func:`replace_buffer`."""
    replace_buffer(agent_construct, name, chunk)


def chunk_from_string(string: str) -> chunks.Chunk:
    """Parse a textual ACT-R chunk definition into a chunk object."""
    return pyactr.chunkstring(string=string)


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
        target.clear()
    except AttributeError:
        pass
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
    model_production = agent_construct.actr_agent.productions[production_name]
    model_production["utility"] = utility
    try:
        model_production.utility = utility
    except AttributeError:
        pass

    simulation = getattr(agent_construct, "simulation", None)
    production_rules = getattr(simulation, "_Simulation__pr", None)
    runtime_rules = getattr(production_rules, "rules", None)
    if runtime_rules is not None and production_name in runtime_rules:
        runtime_production = runtime_rules[production_name]
        runtime_production["utility"] = utility
        try:
            runtime_production.utility = utility
        except AttributeError:
            pass


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
        simulation = getattr(agent_construct, "simulation", None)
        production_rules = getattr(simulation, "_Simulation__pr", None)
        runtime_rules = getattr(production_rules, "rules", None)
        if runtime_rules is not None and production_name in runtime_rules:
            return float(runtime_rules[production_name]["utility"])
        return float(agent_construct.actr_agent.productions[production_name]["utility"])
    except (KeyError, TypeError, ValueError):
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
    model = agent_construct.actr_agent
    model.decmem.add(chunk)
    registry = getattr(model, "_explicit_declarative_chunks", None)
    if registry is None:
        registry = set()
        model._explicit_declarative_chunks = registry
    registry.add(chunk)


def unregister_explicit_declarative_chunk(
    agent_construct: Any, chunk: chunks.Chunk
) -> None:
    """Remove a replaced chunk from the explicit-memory provenance registry."""
    model = getattr(agent_construct, "actr_agent", None)
    registry = getattr(model, "_explicit_declarative_chunks", None)
    if registry is not None:
        registry.discard(chunk)


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
