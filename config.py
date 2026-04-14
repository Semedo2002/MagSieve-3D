# config.py — Data & Parameters
# Variable indices, physical constants, and simulation configuration.
# Includes Magnetic Nozzle / Sieve parameters for spatially modulated B-fields.

import numpy as np
from dataclasses import dataclass, field
from typing import List
from enum import IntEnum


# ============================================================
# Variable Indices
# ============================================================
class ConsVar(IntEnum):
    RHO = 0; MX = 1; MY = 2; MZ = 3
    BX = 4; BY = 5; BZ = 6; EN = 7
    PSI = 8; RHOC = 9


class PrimVar(IntEnum):
    RHO = 0; VX = 1; VY = 2; VZ = 3
    BX = 4; BY = 5; BZ = 6; PR = 7
    PSI = 8; CLR = 9


NVAR = 10

# Conservative variable shortcut aliases
RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC = (
    ConsVar.RHO, ConsVar.MX, ConsVar.MY, ConsVar.MZ,
    ConsVar.BX, ConsVar.BY, ConsVar.BZ, ConsVar.EN,
    ConsVar.PSI, ConsVar.RHOC)

# Primitive variable shortcut aliases
iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR = (
    PrimVar.RHO, PrimVar.VX, PrimVar.VY, PrimVar.VZ,
    PrimVar.BX, PrimVar.BY, PrimVar.BZ, PrimVar.PR,
    PrimVar.PSI, PrimVar.CLR)

# Floor values for density and pressure
FLOOR_RHO = 1e-12
FLOOR_PR = 1e-12


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    """Simulation configuration with all physical and numerical parameters.

    Magnetic Nozzle / Sieve Parameters
    -----------------------------------
    B_field_type : str
        'uniform'  -- standard uniform transverse field By = B_transverse
                      (classic suppression, the "brake")
        'striped'  -- spatially modulated field By = B_transverse * cos(k_mod * y)
                      Creates alternating "bars" (high |B|, pinned interface) and
                      "gaps" (B~0, free flow) -- the Magnetic Sieve / Nozzle.
    B_modulation_mode : int
        Fourier mode number for the striped field modulation.
        k_mod = 2*pi*B_modulation_mode / Ly.
        Controls how many bar/gap pairs span the domain.
    """
    nx: int = 400
    ny: int = 200
    x_min: float = 0.0
    x_max: float = 6.0
    y_min: float = 0.0
    y_max: float = 2.0
    t_end: float = 0.25
    cfl: float = 0.30
    max_steps: int = 200000
    gamma: float = 5.0 / 3.0
    mach: float = 10.0
    interface_x: float = 1.5
    perturbation_amp: float = 0.15
    perturbation_mode: int = 4
    density_ratio: float = 3.0
    interface_width: float = 2.0
    B_transverse: float = 0.0
    B_field_type: str = "uniform"
    B_modulation_mode: int = 4
    glm_ch: float = 0.0
    glm_alpha: float = 1.5
    powell_source: bool = True
    use_char_bc: bool = True
    bc_x_type: str = "auto"
    bc_y_type: str = "periodic"
    enable_smoothing: bool = False
    diag_interval: int = 20
    snapshot_times: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    def get_bc_x(self) -> str:
        if self.bc_x_type != "auto":
            return self.bc_x_type
        return "characteristic" if self.use_char_bc else "extrapolation"

    def is_striped(self) -> bool:
        """True if using spatially modulated (cosine) magnetic field."""
        return self.B_field_type == "striped" and self.B_transverse > 0

    def is_mhd(self) -> bool:
        """True if any magnetic field is present."""
        return self.B_transverse > 0
