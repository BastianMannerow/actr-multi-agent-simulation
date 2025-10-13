class AgentConstruct:
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
        # --- ACT-R binding and synchronization ---
        self.realtime = False                # Whether to run in ACT-R real-time mode (computationally heavy).
        self.actr_agent = None               # Core pyACT-R agent instance (Lisp model equivalent).
        self.actr_adapter = None             # Python adapter providing arithmetic and logical extensions.
        self.actr_agent_type_name = actr_agent_type_name
        self.actr_environment = actr_environment
        self.simulation = simulation         # High-level Simulation reference (set later if None).
        self.actr_construct = None           # Placeholder for future replacement; deprecated internal reference.

        # --- Metadata and runtime identifiers ---
        self.name = name
        self.name_number = name_number       # Public GUI identifier, used to bind visuals to agents.
        self.actr_time = 0.0                 # Local cognitive time, synced with simulation clock.
        self.middleman = middleman
        self.los = los
        self.print_agent_actions = False     # Controlled by Simulation to enable or silence logs.

        # --- Perceptual input placeholders ---
        self.visual_stimuli = []             # Direct-access copy of visible objects; bypasses ACT-R VISUAL if needed.
        self.triggers = [{'S': {'text': 'S', 'position': (1, 1)}}]
        self.stimuli = ['S']                 # Initial dummy stimuli to bootstrap perception.

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
        """Attach the ACT-R construct wrapper (legacy field, reserved for compatibility)."""
        self.actr_construct = actr_construct

    def set_simulation(self):
        """
        Initialize the ACT-R internal simulation.

        Used primarily for models utilizing the VISUAL module. This creates
        the simulation loop with basic timing parameters but no GUI or tracing.
        """
        self.simulation = None if self.actr_agent is None else self.actr_agent.simulation(
            realtime=self.realtime,
            environment_process=self.actr_environment.environment_process,
            stimuli=self.stimuli,
            triggers=self.triggers,
            times=0.1,
            gui=False,
            trace=False
        )

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
    def update_stimulus(self):
        """
        Refresh the agent’s perceptual buffers from the environment via the Middleman.

        Synchronizes new stimuli (text and position) into the agent’s VISUAL buffer.
        Also triggers GUI refresh to reflect perception updates.
        """
        if self.middleman.experiment_environment:
            new_triggers, new_stimuli = self.middleman.get_agent_stimulus(self)
            self.triggers = new_triggers
            self.stimuli = new_stimuli
            self.middleman.simulation.notify_gui()

    # ---------------------------
    # ACT-R extensions and reset
    # ---------------------------
    def actr_extension(self):
        """
        Extend pyACT-R with additional production-level capabilities.

        The adapter injects custom arithmetic, boolean, and logical operations
        into the agent’s production rules at runtime.
        """
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

        self.simulation = self.actr_agent.simulation(
            realtime=self.realtime,
            environment_process=self.actr_environment.environment_process,
            stimuli=self.stimuli,
            triggers=self.triggers,
            times=0.1,
            gui=False,
            trace=False
        )

    def handle_empty_schedule(self):
        """
        Recover gracefully from an EmptySchedule exception.

        Instead of halting the global simulation, the agent is reset to
        reevaluate its goals and continue independently.
        """
        self.reset_simulation()
