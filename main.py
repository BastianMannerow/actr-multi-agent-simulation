from simulation.Simulation import Simulation
from simulation.Tracer import Tracer

if __name__ == '__main__':
    interceptor = Tracer()
    simulation = Simulation(interceptor)
    simulation.run_simulation()