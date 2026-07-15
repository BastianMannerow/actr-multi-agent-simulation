"""Optional, reversible performance overrides for pyactr 0.3.2.

The package is intentionally separate from the simulation integration layer.
Nothing in this directory is activated unless the simulation configuration
explicitly enables ``Experimental pyactr performance boost``.
"""

from pyactr_overrides.manager import configure, is_active, status

__all__ = ["configure", "is_active", "status"]
