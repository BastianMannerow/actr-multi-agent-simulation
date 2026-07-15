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
        self.actr_time = 0.0                 # Local cognitive time, synced with simulation clock.
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
        self.simulation = self.actr_agent.simulation(**simulation_kwargs)

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

    def get_agent_dictionary(self):
        """Return the dictionary of letter-coded agent references."""
        return self.agent_dictionary

    # ---------------------------
    # Perception pipeline
    # ---------------------------
    def update_stimulus(self, *, publish: bool = True):
        """Refresh the live pyactr visual frame for this agent.

        The Middleman returns only pyactr-supported stimulus fields. Rich world
        metadata remains available separately in ``visual_metadata``. When a
        running cognitive simulation exists, the updated frame is published to
        pyactr immediately and automatic visual buffers are refreshed with
        pyactr's own chunk constructors.
        """
        if not self.middleman.experiment_environment:
            return
        new_triggers, new_stimuli = self.middleman.get_agent_stimulus(self)
        self.triggers = new_triggers
        self.stimuli = new_stimuli
        if publish and self.uses_visual_module:
            pyactr_extension.publish_visual_stimulus(self)

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
        - Resets internal timing and visual stimuli buffers.
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
        self.simulation = self.actr_agent.simulation(**simulation_kwargs)

    def handle_empty_schedule(self):
        """
        Recover gracefully from an EmptySchedule exception.

        Instead of halting the global simulation, the agent is reset to
        reevaluate its goals and continue independently.
        """
        self.reset_simulation()
