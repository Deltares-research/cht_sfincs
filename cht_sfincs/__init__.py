"""SFINCS coastal flood model interface for the CHT toolkit.

Provides the SFINCS domain class and XMI wrapper for programmatic access to the
SFINCS flood model kernel.
"""

from .sfincs import SFINCS  # noqa: F401
from .xmi import SfincsXmi  # noqa: F401
