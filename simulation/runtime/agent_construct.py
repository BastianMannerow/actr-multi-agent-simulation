from simulation.world.entities import SpatialAgent
from simulation.integrations import pyactr_extension

class AgentConstruct(SpatialAgent):
    """
    Container class connecting an ACT-R agent, its environment bindings,
    adapters, and runtime metadata.

    Purpose
    -------
    - Encapsulate one cognitive agent’s state within the simulation.
    - Manage ACT-R environment coupling, adapter extension, and visual input updates.
    - Serve as the data backbone for GUI logging and Middleman communication.

    Design principles
    -----------------
    - Keeps references to ACT-R core objects but avoids circular initialization.
    - Designed for flexibility: can run in headless (no GUI) or interactive setups.
    - Provides lightweight methods to rebuild, reset, or extend the agent simulation.
    """

    def __init__(self, actr_agent_type_name, actr_environment, simulation, middleman, name, name_number, los):
        """
        Initialize the agent construct.

        Parameters
        ----------
        actr_agent_type_name : str
            Class name of the agent model in the `/agents` directory.
        actr_environment : pyactr.Environment
            ACT-R visual environment handle used to locate visual stimuli.
        simulation : Any
            Simulation context. May be reassigned after full initialization.
        middleman : Middleman
            Communication interface between agent and environment.
        name : str
            Human-readable name for logs and GUI.
        name_number : str
            Display identifier, typically the full name used in GUI rendering.
        los : int
            Line-of-sight distance for perceptual range.
        """
        super().__init__(name)

        # --- ACT-R binding and synchronization ---
        self.realtime = False                # Whether to run in ACT-R real-time mode (computationally heavy).
        self.actr_agent = None               # Core pyACT-R agent instance (Lisp model equivalent).
        self.actr_adapter = None             # Python adapter providing arithmetic and logical extensions.
        self.actr_agent_type_name = actr_agent_type_name
        self.actr_environment = actr_environment
        self.simulation = simulation         # High-level Simulation reference (set later if None).
        self.actr_construct = None           # Model-builder instance.
        self.uses_visual_module = True        # May be disabled by an agent model.

        # --- Metadata and runtime identifiers ---
        self.name_number = name_number       # Public GUI identifier, used to bind visuals to agents.
        self.actr_time = 0.0                 # Absolute local pyactr/SimPy time.
        self.no_increase_count = 0           # Guard for zero-time event loops.
        self.middleman = middleman
        self.los = los
        self.print_agent_actions = False     # Controlled by Simulation to enable or silence logs.

        # --- Perceptual input placeholders ---
        self.visual_stimuli = []             # Human-readable matrix around the agent.
        self.visual_metadata = {}            # Rich metadata kept outside pyactr stimuli.
        self.visual_frame_origin = (0, 0)    # World coordinate of visual_stimuli[0][0].
        self.visual_frame_valid_positions = set()  # In-bounds cells in the current frame.
        self.triggers = [set()]              # One trigger collection per visual frame.
        self.stimuli = [{}]                  # One pyactr-safe visual frame.
        self.perception_dirty = True         # Rebuild perception only after world changes.
        self._perception_revision = -1
        self._dirty_buffers: set[str] = set()
        self._known_buffer_names: set[str] = set()
        self._agent_symbol_by_identity: dict[int, str] = {}

    # ---------------------------
    # Initialization utilities
    # ---------------------------
    def set_actr_agent(self, actr_agent):
        """Assign the ACT-R agent safely to avoid circular initialization deadlocks."""
        self.actr_agent = actr_agent

    def set_actr_adapter(self, actr_adapter):
        """
        Link the ACT-R adapter to this construct.

        Ensures bidirectional reference: the adapter can reach back to the
        agent construct when performing arithmetic or logical extensions.
        """
        self.actr_adapter = actr_adapter
        actr_adapter.agent_construct = self

    def set_actr_construct(self, actr_construct):
        """Attach the model builder and read its perceptual architecture flag."""
        self.actr_construct = actr_construct
        self.uses_visual_module = bool(
            getattr(actr_construct, "uses_visual_module", True)
        )

    def set_simulation(self):
        """Initialize the ACT-R simulation and load the model's initial goal."""
        if self.actr_agent is None:
            self.simulation = None
            return

        initial_goal = getattr(self.actr_construct, "initial_goal", None)
        if initial_goal is not None:
            try:
                first_goal = next(iter(self.actr_agent.goals.values()))
                if not list(first_goal):
                    first_goal.add(initial_goal)
            except (AttributeError, StopIteration, TypeError):
                pass

        simulation_kwargs = {
            "realtime": self.realtime,
            "gui": False,
            "trace": False,
        }
        if self.uses_visual_module:
            simulation_kwargs.update(
                {
                    "environment_process": self.actr_environment.environment_process,
                    "stimuli": self.stimuli,
                    "triggers": self.triggers,
                    "times": 0.1,
                }
            )
        simulation_kwargs["initial_time"] = float(self.actr_time)
        self.simulation = self.actr_agent.simulation(**simulation_kwargs)
        self.mark_buffer_dirty(*getattr(self.actr_agent, "goals", {}).keys())

    # ---------------------------
    # Social identification
    # ---------------------------
    def set_agent_dictionary(self, agent_list):
        """
        Create a mapping of letter-coded agent identifiers (A, B, ..., Z, AA, AB, ...).

        The current agent always receives code 'A' for self-referencing convenience.
        This dictionary supports symbolic reasoning and logging consistency across agents.
        """
        agent_list = [self] + [agent for agent in agent_list if agent != self]

        def generate_letter_code(index: int) -> str:
            """Generate alphabetic sequence identifiers (A, B, ..., Z, AA, ...)."""
            letters = []
            while index >= 0:
                letters.append(chr(65 + (index % 26)))  # 65 = ASCII 'A'
                index = index // 26 - 1
            return ''.join(reversed(letters))

        self.agent_dictionary = {
            generate_letter_code(i): {"agent": agent}
            for i, agent in enumerate(agent_list)
        }
        self._agent_symbol_by_identity = {
            id(info["agent"]): symbol
            for symbol, info in self.agent_dictionary.items()
        }

    def get_agent_dictionary(self):
        """Return the dictionary of letter-coded agent references."""
        return self.agent_dictionary

    # ---------------------------
    # Perception pipeline
    # ---------------------------
    def mark_perception_dirty(self) -> None:
        """Invalidate the cached visual frame after a relevant world change."""
        self.perception_dirty = True

    def mark_buffer_dirty(self, *buffer_names: str) -> None:
        """Record buffers changed outside pyactr's visible event stream."""
        self._dirty_buffers.update(str(name) for name in buffer_names if name)

    def consume_dirty_buffers(self) -> set[str]:
        dirty = set(self._dirty_buffers)
        self._dirty_buffers.clear()
        return dirty

    def update_stimulus(self, *, publish: bool = True, force: bool = False) -> bool:
        """Refresh perception only when the world invalidated this agent's frame.

        For visual ACT-R models the cached frame is still republished before a
        step because all agents share one pyactr Environment. Non-visual models
        avoid both frame construction and publication while nothing changed.
        """
        environment = self.middleman.experiment_environment
        if environment is None:
            return False
        rebuilt = bool(force or self.perception_dirty)
        if rebuilt:
            new_triggers, new_stimuli = self.middleman.get_agent_stimulus(self)
            self.triggers = new_triggers
            self.stimuli = new_stimuli
            self.perception_dirty = False
            self._perception_revision = int(
                getattr(environment, "world_revision", self._perception_revision + 1)
            )
        if publish and self.uses_visual_module:
            pyactr_extension.publish_visual_stimulus(self)
        return rebuilt

    # ---------------------------
    # ACT-R extensions and reset
    # ---------------------------
    def actr_extension(self):
        """
        Extend pyACT-R with additional production-level capabilities.

        The adapter injects custom arithmetic, boolean, and logical operations
        into the agent’s production rules at runtime.
        """
        if self.actr_adapter is None:
            return
        self.actr_adapter.agent_construct = self
        self.actr_adapter.extending_actr()

    def reset_simulation(self, default_goal=None):
        """
        Rebuild the ACT-R simulation when the agent’s knowledge or goals change.

        Effects
        -------
        - Reinstantiates the cognitive simulation loop.
        - Preserves agent identity and adapter bindings.
        - Preserves the absolute ACT-R timeline and refreshes visual buffers.
        """
        if not default_goal:
            default_goal = self.actr_construct.initial_goal
        first_goal = next(iter(self.actr_agent.goals.values()))
        first_goal.add(default_goal)

        simulation_kwargs = {
            "realtime": self.realtime,
            "gui": False,
            "trace": False,
        }
        if self.uses_visual_module:
            simulation_kwargs.update(
                {
                    "environment_process": self.actr_environment.environment_process,
                    "stimuli": self.stimuli,
                    "triggers": self.triggers,
                    "times": 0.1,
                }
            )
        # A rebuilt pyactr simulation must continue on the existing absolute
        # model timeline.  Starting again at zero would make the clock move
        # backwards and corrupt multi-agent ordering and history exports.
        simulation_kwargs["initial_time"] = float(self.actr_time)
        self.simulation = self.actr_agent.simulation(**simulation_kwargs)
        self.mark_buffer_dirty(*getattr(self.actr_agent, "goals", {}).keys())
        self.no_increase_count = 0

    def handle_empty_schedule(self):
        """
        Recover gracefully from an EmptySchedule exception.

        Instead of halting the global simulation, the agent is reset to
        reevaluate its goals and continue independently.
        """
        self.reset_simulation()
