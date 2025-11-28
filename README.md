# ACT-R Multi-Agent Simulation Framework (Python / pyactr)

This repository provides a multi-agent simulation framework on top of
[pyactr](https://github.com/jakdot/pyactr), allowing multiple ACT-R
agents to interact in a shared spatial environment with explicit
control over time, vision, and motor actions. Beyond purely simulated
grid-worlds, the framework is designed to interface with **real-world
and robotics environments** via adapter layers, as demonstrated in the
companion project for TurtleBot integration:
[actr-turtle-bot-simulation](https://github.com/BastianMannerow/turtle-bot-simulation).
It is intended for cognitive modeling of social interaction, norm
compliance, multi-agent decision making, and embodied cognition
scenarios, and includes an extension layer for pyactr that provides
ergonomic access to goals, imaginal buffers, declarative memory,
utilities, and other internal structures during simulation. By
separating ACT-R productions from Boolean, arithmetic, and
Python-level computations, the architecture makes it straightforward to
attach powerful external modules—such as **large language models** or
robot controllers—as explainable extensions to the core cognitive
system.

**A detailed, step-by-step tutorial for setting up and extending
the framework is available in the project Wiki.**

## Abstract

Classical ACT-R models are typically implemented and evaluated in
single-agent settings. While multi-agent scenarios can in principle be
constructed in Lisp ACT-R or pyactr, the lack of orchestration for
multiple agents, shared environments, and debugging tools makes such
simulations cumbersome and error-prone.

This framework introduces:

- a time-aware scheduler for multiple ACT-R agents,
- a shared grid-world environment with line-of-sight (LoS) based
  visual stimulation,
- a "Middleman" component that mediates motor and perceptual traffic
  between agents and the environment,
- a Tkinter-based stepper GUI for single-step execution, logging, and
  "jump to production" debugging, and
- an extension layer for pyactr that provides ergonomic access to goals, imaginal buffers, declarative memory, utilities, and other internal structures during simulation.

The goal is to make multi-agent ACT-R simulations in Python practical
and transparent enough for serious cognitive modeling, including
models of social interaction and normative behavior.

## Key Features

### Multi-Agent Scheduling and Time Control

- Support for **multiple ACT-R agents** sharing a single pyactr
  `Environment`.
- Each agent maintains its own ACT-R time (`actr_time`).
- The `Simulation` class:
  - sorts agents by next ACT-R time step,
  - schedules the earliest agent,
  - maps ACT-R time to real time via a configurable `speed_factor`,
  - supports both continuous execution and an interactive stepper mode.

### Shared Grid-World Environment with Line-of-Sight

The grid-world and its Tk-based visualization are intended as a **reference implementation** rather than a fixed choice of environment. Both the `Environment` and `Middleman` components can be replaced by custom backends, enabling alternative GUIs, different world representations, or connections to real-world and robotics platforms while preserving the same ACT-R/pyactr integration pattern.

- A 2D grid (`level_matrix`) where each cell can contain agents and
  other objects (e.g., walls, resources).
- `Environment` implements agent movement (`move_agent_top/left/...`)
  and bookkeeping for grid occupancy.
- `Middleman.get_agent_stimulus`:
  - computes a local view around each agent based on its line-of-sight
    radius,
  - converts the visible neighborhood into symbolic visual stimuli
    compatible with pyactr's vision system
    (e.g., `{ "text": "A", "position": (row, col) }`),
  - stores a 2D symbol map (`visual_stimuli`) at the agent level for
    debugging and GUI visualization.

### Motor Interface via Middleman

- `Middleman.motor_input` translates discrete key symbols (e.g., "W",
  "A", "S", "D") from the ACT-R `manual` module into environment
  actions.
- This decouples cognitive modeling from low-level environment
  control, keeping the ACT-R side purely symbolic (keypresses) while
  the environment manages spatial updates.

### pyactr Extension Layer

The module `pyactrFunctionalityExtension` provides:

- Event utilities:
  - `request_if_production_fired`, `get_production_fired`
  - `request_if_key_pressed`, `key_pressed`
- Goal and imaginal accessors:
  - `get_goal`, `set_goal`
  - `get_imaginal`, `set_imaginal`
- Production utilities:
  - `update_utility`, `get_production_utility`
  - `add_production`, `get_all_productions`
- Declarative memory utilities:
  - `get_declarative_memory`, `add_to_declarative_memory`
  - `get_declarative_chunk_type`, `delete_declarative_chunk_type`

### Lisp-like agent abstraction and Python adapter layer

In classical ACT-R and Lisp-based implementations, Boolean and
arithmetic operations are often expressed directly inside productions.
In pyactr & this framework, such operations are **deliberately factored out** of
productions to keep the core ACT-R dynamics (conditions over chunks,
buffer updates, learning mechanisms) cleanly separated from additional
functionality and to make the model more explainable.

Concretely:

- Each agent type is implemented as a **Lisp-like ACT-R model** that
  offers familiar mechanisms for retrieval, goal state transitions,
  and imaginal manipulation.
- Multiple imaginal buffers can be defined and referenced in
  productions, while productions themselves remain close to ACT-R
  theory (chunk-based preconditions and actions).
- Boolean logic, arithmetic operations, and richer computations are
  implemented in **Python functions** and are invoked from outside the
  productions based on the current cognitive state.

For each ACT-R model there is a corresponding **adapter class** on the
Python side. The adapter can:

- observe when specific goals or imaginal states become active,
- react when particular productions fire,
- execute arbitrary Python code in response.

This makes it straightforward to connect an ACT-R agent to external
components such as a large language model (LLM), a robot controller,
or a logging/analysis pipeline. After the external computation
completes, the adapter can write the result back into the cognitive
system—for example by updating chunks or switching to a new goal
state—thus enabling tight, yet conceptually separated, integration
between ACT-R theory and extended Python-based functionality.

### Citation

If you use this framework in academic work, please cite it as:

Mannerow, B. (2025). ACT-R Multi-Agent Simulation Framework (Python /
pyactr). GitHub repository.
https://github.com/BastianMannerow/actr-multi-agent-simulation
