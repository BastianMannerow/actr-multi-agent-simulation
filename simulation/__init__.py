"""ACT-R simulation runtime, world model, inspection, export, and batch tools."""

from simulation.integrations import pyactr_extension

# Existing adapters created for earlier project layouts can continue to import
# ``from simulation import pyactrFunctionalityExtension`` while new code uses
# the consistent snake_case module name.
pyactrFunctionalityExtension = pyactr_extension

__all__ = ["pyactr_extension", "pyactrFunctionalityExtension"]
