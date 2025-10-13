from simulation.AgentConstruct import AgentConstruct


class Middleman:
    """
    Mediates bidirectional communication between agents and their environment.

    Purpose
    -------
    - Translate cognitive outputs (motor commands) into environment actions.
    - Convert environment states into visual stimuli accessible to agents.
    - Maintain a clean separation between cognitive logic (agents) and spatial logic (environment).

    Design
    ------
    - Created before the Environment instance; attached later via `set_game_environment()`.
    - Stateless regarding simulation progression; operates purely as a message-passing layer.
    - Used by `Simulation` to unify ACT-R interaction with a matrix world.

    Attributes
    ----------
    simulation : Simulation
        The parent simulation orchestrator.
    experiment_environment : Environment | None
        The active environment containing the world matrix.
    print_middleman : bool
        Enables optional debug logging for inspection of agent-environment exchanges.
    """

    def __init__(self, simulation, print_middleman):
        self.simulation = simulation
        self.experiment_environment = None
        self.print_middleman = print_middleman

    # ---------------------------------------------------------------------
    # Environment connection
    # ---------------------------------------------------------------------
    def set_game_environment(self, experiment_environment):
        """
        Attach the environment instance after its creation.

        Reasoning
        ---------
        The Middleman is initialized before the environment exists,
        so this delayed wiring ensures proper dependency order.

        Parameters
        ----------
        experiment_environment : Environment
            Active environment instance the agents interact with.
        """
        self.experiment_environment = experiment_environment

    # ---------------------------------------------------------------------
    # Agent → Environment: Motor interface
    # ---------------------------------------------------------------------
    def motor_input(self, key, current_agent):
        """
        Execute environment actions based on a symbolic key command.

        Purpose
        --------
        Simulates motor output by translating discrete key events into
        environment movements (e.g., "W" → move up).

        Parameters
        ----------
        key : str
            Command key (typically one of W/A/S/D).
        current_agent : AgentConstruct
            The currently active cognitive agent.
        """
        if key == "W":
            self.experiment_environment.move_agent_top(current_agent)
        elif key == "A":
            self.experiment_environment.move_agent_left(current_agent)
        elif key == "S":
            self.experiment_environment.move_agent_bottom(current_agent)
        elif key == "D":
            self.experiment_environment.move_agent_right(current_agent)

    # ---------------------------------------------------------------------
    # Environment → Agent: Perception interface
    # ---------------------------------------------------------------------
    def get_agent_stimulus(self, agent):
        """
        Generate a visual stimulus representation for a specific agent.

        Purpose
        --------
        - Samples the environment grid around the agent based on its line of sight (LoS).
        - Produces symbolic stimuli (letters) representing nearby agents or objects.
        - Updates the agent’s internal `visual_stimuli` buffer for cognitive access.

        Parameters
        ----------
        agent : AgentConstruct
            Agent requesting visual information.

        Returns
        -------
        tuple[list[str], list[dict]]
            *new_triggers* — all visible object symbols
            *stimuli* — a single-element list containing a dict mapping indices to
            `{"text": symbol, "position": (row, col)}`.
        """
        matrix = self.experiment_environment.level_matrix
        r, c = self.experiment_environment.find_agent(agent)
        if r is None:
            return None, None

        agent_map = agent.get_agent_dictionary()
        los = agent.los
        rows, cols = len(matrix), len(matrix[0])

        # Determine the visible area based on line of sight (LoS)
        if los == 0 or los > cols or los > rows:
            x_los = cols
            y_los = rows
            off_x = off_y = 0
        else:
            x_los = y_los = 2 * los + 1
            off_x = off_y = los

        new_triggers = []
        frame = {}
        visual_stimuli = [['' for _ in range(x_los)] for _ in range(y_los)]
        index = 0

        # Scan through the visible window
        for i in range(y_los):
            for j in range(x_los):
                mi = r - off_y + i
                mj = c - off_x + j

                # Out of bounds
                if mi < 0 or mi >= rows or mj < 0 or mj >= cols:
                    visual_stimuli[i][j] = '-'
                    continue

                # Inspect all elements in the cell
                for element in matrix[mi][mj]:
                    if isinstance(element, AgentConstruct):
                        # Identify other agents using symbolic letter codes
                        for sym, info in agent_map.items():
                            if info["agent"] == element:
                                new_triggers.append(sym)
                                frame[index] = {"text": sym, "position": (mi, mj)}
                                visual_stimuli[i][j] = sym
                                index += 1
                                break

                    # Example: Extend here for additional object classes
                    # e.g., Food, Wall, RewardZone, etc.
                    # elif isinstance(element, Food):
                    #     sym = 'X'
                    #     new_triggers.append(sym)
                    #     frame[index] = {"text": sym, "position": (mi, mj)}
                    #     visual_stimuli[i][j] = sym
                    #     index += 1

        agent.visual_stimuli = visual_stimuli
        stimuli = [frame]
        return new_triggers, stimuli
