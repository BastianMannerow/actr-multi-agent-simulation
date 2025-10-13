import pyactr as actr


class Example:
    """
    Minimal demonstrator agent for the ACT-R architecture.

    Purpose
    -------
    - Serves as a reference implementation for agent structure and initialization.
    - Demonstrates how to define goal states, chunk types, and productions.
    - Can be extended to create domain-specific cognitive models.

    Design
    ------
    - Each agent instance holds its own ACT-R model (`actr_agent`).
    - `goal_phases` define independent reasoning states for clarity.
    - A single phase (`"test"`) is provided as a structural example.

    Attributes
    ----------
    this_agent_key : str | None
        Symbol identifying this agent in the shared simulation dictionary.
    other_agents_key_list : list[str] | None
        Symbols identifying all other agents in the environment.
    environ : Environment
        The pyACT-R environment; currently retained for backward compatibility.
    actr_agent : actr.ACTRModel
        The ACT-R cognitive model controlling this agent.
    goal_phases : list[str]
        Logical task phases used to segment the agent’s goals.
    initial_goal : actr.Chunk
        Default goal chunk that defines the agent’s starting state.
    """

    def __init__(self, environ):
        """
        Initialize the agent with a pyACT-R environment.

        Parameters
        ----------
        environ : actr.Environment
            pyACT-R environment. Deprecated; will be removed in future versions.
        """
        self.this_agent_key = None
        self.other_agents_key_list = None
        self.environ = environ

        # Base ACT-R model setup
        self.actr_agent = actr.ACTRModel(
            environment=self.environ,
            motor_prepared=True,
            automatic_visual_search=False,
            subsymbolic=True
        )

        # Define reasoning phases
        self.goal_phases = ["test"]

        # Initial goal chunk (starting cognitive state)
        self.initial_goal = actr.chunkstring(string=f"""
            isa     {self.goal_phases[0]}
            state   {self.goal_phases[0]}Start
        """)

    # ----------------------------------------------------------------------
    # Agent construction
    # ----------------------------------------------------------------------
    def build_agent(self, agent_list):
        """
        Construct and return a fully initialized ACT-R agent.

        Parameters
        ----------
        agent_list : list[str]
            The symbolic identifiers from `AgentConstruct` used to reference
            self and other agents in the simulation.

        Returns
        -------
        actr.ACTRModel
            A configured ACT-R agent instance ready for simulation.
        """
        self.this_agent_key = agent_list[0]
        self.other_agents_key_list = agent_list[1:]

        actr_agent = self.actr_agent

        # Configure ACT-R parameters
        actr_agent.model_parameters["utility_noise"] = 5  # stochastic exploration
        actr_agent.model_parameters["baselevel_learning"] = False  # disable base-level activation

        # Define goal chunk types (optional, but clarifies agent state structure)
        for phase in self.goal_phases:
            actr.chunktype(phase, "state")

        # Add productions corresponding to the first goal phase
        self.add_productions(actr_agent, self.goal_phases[0])
        return actr_agent

    # ----------------------------------------------------------------------
    # Cognitive model definition
    # ----------------------------------------------------------------------
    def add_productions(self, actr_agent, phase):
        """
        Define production rules for the given cognitive phase.

        Parameters
        ----------
        actr_agent : actr.ACTRModel
            The agent’s ACT-R model.
        phase : str
            Label of the goal phase (e.g., "test").

        Notes
        -----
        - The example rule provided here does not change state; it is a placeholder
          to demonstrate syntax and execution.
        - Extend this method to encode actual task behavior.
        """
        actr_agent.productionstring(
            name="easteregg",
            string=f"""
                =g>
                isa     {phase}
                state   {phase}Start
                ==>
                =g>
                isa     {phase}
                state   {phase}Start
            """
        )
