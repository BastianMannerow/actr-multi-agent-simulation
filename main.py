from simulation.runtime.simulation import Simulation
from simulation.runtime.tracer import Tracer


def main() -> int:
    interceptor = Tracer()
    simulation = Simulation(interceptor)
    return simulation.run_simulation()


if __name__ == "__main__":
    raise SystemExit(main())
