"""
Enhanced utility layer for pyACT-R integration.

Purpose
-------
- Extend pyACT-R with additional helper functions and ergonomic accessors.
- Patch known issues in the visual search subsystem (`VisualLocation.find`).
- Simplify interaction with ACT-R goal, imaginal, and production utilities.

Scope
-----
- This module is purely an extension; it does not modify ACT-R theory.
- Designed for applied simulations, debugging, and GUI synchronization.
"""

import pyactr.vision as vision
from pyactr import chunks, utilities
from pyactr.utilities import ACTRError


def fix_pyactr():
    """
    Monkey-patch pyACT-R's `VisualLocation.find` to fix incorrect attribute
    resolution during automatic visual search.

    Context
    -------
    - The original implementation occasionally mismatches attended objects
      when `automatic_visual_search=True`.
    - This patch corrects how chunk attributes are extracted and compared.

    Behavior
    --------
    - Retains full compatibility with original API.
    - Correctly filters stimuli by `attended` and `position`.
    - Generates consistent `_visuallocation` chunks for matched stimuli.

    Warning
    -------
    This patch mutates the pyACT-R global class definition at runtime.
    Apply it once during initialization.
    """

    _original_find = vision.VisualLocation.find  # backup

    def patched_find(self, otherchunk, actrvariables=None, extra_tests=None):
        """Corrected search routine for the visual buffer."""
        if extra_tests is None:
            extra_tests = {}
        if actrvariables is None:
            actrvariables = {}

        # Build chunk from production RHS
        try:
            mod_attr_val = {
                x[0]: utilities.check_bound_vars(actrvariables, x[1], negative_impossible=False)
                for x in otherchunk.removeunused()
            }
        except ACTRError as e:
            raise ACTRError(f"The chunk '{otherchunk}' is not defined correctly; {e}")
        chunk_used_for_search = chunks.Chunk(utilities.VISUALLOCATION, **mod_attr_val)

        found, found_stim = None, None
        closest = x_closest = y_closest = float("inf")

        # Iterate over all stimuli in the environment
        for each in self.environment.stimulus:
            stim_attrs = self.environment.stimulus[each]

            # Attended flag consistency check
            try:
                attended_flag = extra_tests.get("attended")
                if attended_flag in (False, 'False') and self.finst and stim_attrs in self.recent:
                    continue
                if attended_flag not in (False, 'False') and self.finst and stim_attrs not in self.recent:
                    continue
            except KeyError:
                pass

            # Text value filter
            if (
                chunk_used_for_search.value != chunk_used_for_search.EmptyValue()
                and chunk_used_for_search.value.values != stim_attrs.get("text")
            ):
                continue

            # Extract pixel coordinates
            position = (int(stim_attrs['position'][0]), int(stim_attrs['position'][1]))

            # Screen coordinate constraints (absolute matching)
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

            # If stimulus passes all filters, synthesize a visible chunk
            found_stim = stim_attrs
            filtered = {
                k: stim_attrs[k]
                for k in stim_attrs
                if k not in ('position', 'text', 'vis_delay')
            }
            visible_chunk = chunks.makechunk(
                nameofchunk="vis1",
                typename="_visuallocation",
                **filtered
            )

            # Check for structural compatibility
            if visible_chunk <= chunk_used_for_search:
                temp_dict = visible_chunk._asdict()
                temp_dict.update({"screen_x": position[0], "screen_y": position[1]})
                found = chunks.Chunk(utilities.VISUALLOCATION, **temp_dict)

                # Update spatial distance metrics
                closest = utilities.calculate_pythagorean_distance(self.environment.current_focus, position)
                x_closest = utilities.calculate_onedimensional_distance(
                    self.environment.current_focus, position, horizontal=True
                )
                y_closest = utilities.calculate_onedimensional_distance(
                    self.environment.current_focus, position, horizontal=False
                )

        return found, found_stim

    vision.VisualLocation.find = patched_find


# ---------------------------
# ACT-R event utilities
# ---------------------------

def production_fired(agent):
    """
    Extract the name of the currently fired production rule.

    Returns
    -------
    str | None
        Production name if the event corresponds to a "RULE FIRED" entry.
    """
    event = agent.simulation.current_event
    if "RULE FIRED: " in event[2]:
        return event[2].replace("RULE FIRED: ", "")
    return None


def key_pressed(agent_construct):
    """
    Check if a manual keypress event occurred in the ACT-R event queue.

    Returns
    -------
    str | None
        The last key pressed (single character) or None if no event is detected.
    """
    event = agent_construct.simulation.current_event
    if event[1] == "manual" and "KEY PRESSED:" in event[2]:
        return event[2][-1]
    return None


# ---------------------------
# Goal and Imaginal utilities
# ---------------------------

def set_goal(agent, chunk):
    """Add a new chunk to the agent’s goal buffer."""
    first_goal = next(iter(agent.goals.values()))
    first_goal.add(chunk)


def get_goal(agent):
    """
    Retrieve the agent’s primary goal chunk under key `'g'`.

    Returns
    -------
    Chunk | None
        The goal chunk if available, otherwise None.
    """
    key = "g"
    return agent.goals.get(key, None)


def get_imaginal(agent, key):
    """
    Retrieve the imaginal chunk associated with a given buffer key.

    Logs available buffers if the key is invalid.
    """
    goals = agent.goals
    if key not in goals:
        print(f"'{key}' not found. Available buffers: {list(goals.keys())}")
        return None
    return goals[key]


def set_imaginal(agent, new_chunk, key):
    """
    Assign a new chunk to the imaginal buffer with the given key.

    Raises
    ------
    TypeError
        If the target buffer does not support `.add()`.
    """
    goals = agent.goals
    if key not in goals:
        print(f"Buffer '{key}' not found. Available keys: {list(goals.keys())}")
        return

    target = goals[key]
    try:
        target.add(new_chunk)
    except AttributeError:
        raise TypeError(f"Goal object for '{key}' does not support `.add()`.")


# ---------------------------
# Production rule utility
# ---------------------------

def update_utility(actr_agent, production_name, utility):
    """
    Update the numeric utility value of a given production.

    Parameters
    ----------
    actr_agent : ACT-R agent
        The agent whose production set is being modified.
    production_name : str
        Name of the production rule to update.
    utility : float
        New utility value.
    """
    actr_agent.productions[production_name]["utility"] = utility
