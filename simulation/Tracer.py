import datetime
from typing import Any, Dict, List, Set


class Tracer:
    """
    Lightweight in-memory event tracer for ACT-R style simulations.

    Design goals
    ------------
    - **Schema clarity:** Each log record is a flat dict with four keys:
      `timestamp`, `type`, `event`, `agent_name`.
    - **Agent discovery:** On first encounter of an agent name, an
      `agent_added` record is emitted before the actual event.
    - **Zero dependencies:** Uses only built-in types for speed and portability.
    - **Order-preserving:** Records are appended in call order; no sorting.
    - **Non-blocking:** No I/O or locks; suitable for single-threaded sims.

    Notes
    -----
    - `timestamp` is expected to be the *simulation time* (e.g., `agent.actr_time`),
      not wall-clock time. The `datetime` import is unused and kept only as a
      placeholder if you later extend this to real-time stamps.
    - `type` and `event` are derived from the ACT-R event tuple. This class
      assumes the tuple format is `(ignored, event_type, event_description, ...)`.
    - This tracer does not deduplicate events, enforce schemas, or manage memory.
      If your simulation is long-running, periodically drain or persist `records`.
    """

    def __init__(self) -> None:
        # Append-only event store. Each entry is a dict with the canonical keys:
        # "timestamp": simulation time at which the event occurred
        # "type":      event category (string label, e.g., "production_fired")
        # "event":     human-readable description or payload
        # "agent_name":identifier of the agent that produced the event
        self.records: List[Dict[str, Any]] = []

        # Agent registry to emit exactly one 'agent_added' per agent name.
        # Using names avoids coupling to agent object identity.
        self.known_agents: Set[str] = set()

    def trace(self, agent: Any, event: Any) -> None:
        """
        Record a simulation event and, if needed, the agent's first-seen marker.

        Parameters
        ----------
        agent : Any
            Object with at least:
              - `actr_time`: numeric or comparable simulation timestamp
              - `actr_agent_type_name`: string type label (not stored here, but kept for future use)
              - `name`: stable string identifier for the agent instance
        event : Any
            ACT-R event tuple or similar indexable structure where:
              - `event[1]` is the event type (str)
              - `event[2]` is the event description/payload (str or Any)

        Behavior
        --------
        - Emits an 'agent_added' record the first time an agent name is observed.
        - Emits the actual event record using the agent's simulation timestamp.
        - Does not mutate `event` or `agent`; reads only.

        Raises
        ------
        IndexError
            If `event` does not contain at least three items.
        AttributeError
            If `agent` lacks required attributes (`actr_time`, `name`).
        """
        # Read simulation timestamp and agent identity early to surface errors quickly.
        ts = agent.actr_time
        name = agent.name

        # First encounter of this agent name → log a synthetic discovery event.
        if name not in self.known_agents:
            self.known_agents.add(name)
            self.records.append({
                "timestamp": ts,        # simulation time at first sighting
                "type": "agent_added",  # sentinel to aid downstream consumers
                "event": None,          # no payload; presence of this record is the signal
                "agent_name": name
            })

        # Log the user-supplied ACT-R event. Assumes tuple-like indexing:
        # event[1] = type label, event[2] = human-readable description/payload.
        self.records.append({
            "timestamp": ts,
            "type": event[1],
            "event": event[2],
            "agent_name": name
        })

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Return a snapshot of all collected records.

        Returns
        -------
        List[Dict[str, Any]]
            Ordered list of event dicts. Mutating the returned list does not
            affect the internal store, but mutating dict items will. Copy if needed.
        """
        return self.records
