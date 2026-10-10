# config.py — Structured solver configuration.
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class ConsVar(IntEnum):
    RHO = 0; MX = 1; MY = 2; MZ = 3
    BX = 4; BY = 5; BZ = 6; EN = 7
    PSI = 8; RHOC = 9


class PrimVar(IntEnum):
    RHO = 0; VX = 1; VY = 2; VZ = 3
    BX = 4; BY = 5; BZ = 6; PR = 7
    PSI = 8; CLR = 9


NVAR = len(ConsVar)
RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC = tuple(ConsVar)
iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR = tuple(PrimVar)
FLOOR_RHO = 1e-12
FLOOR_PR = 1e-12


@dataclass(frozen=True)
class GridConfig:
    nx: int = 400
    ny: int = 200
    x_min: float = 0.0
    x_max: float = 6.0
    y_min: float = 0.0
    y_max: float = 2.0
    # z is used only by solver3d (periodic)
    nz: int = 1
    z_min: float = 0.0
    z_max: float = 1.0

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / self.nz


@dataclass(frozen=True)
class PhysicsConfig:
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
    # Field component carrying the striped modulation cos(k_mod*y):
    #   "z": out-of-plane magnetic-pressure bars (exactly solenoidal in 2D)
    #   "x": shock-parallel field lines with tension (exactly solenoidal)
    #   "y": legacy By(y); div(B) = -B0*k_mod*sin(k_mod*y) != 0, kept for reproduction only
    B_stripe_component: str = "x"
    # Upstream equilibrium for "x" stripes: p_gas(y) = P_total - B(y)^2/2 with
    # P_total = p_bar + B0^2/2, so the bar centres (|B| = B0) keep (rho=1, p=p_bar)
    # and the Mach number refers to that state. equilibrium_closure picks how
    # density follows the pressure: "uniform_density" or "uniform_temperature".
    p_bar: float = 1.0
    # Field left in the gaps of "x" stripes, as a fraction eps of B0:
    #   eps = 0:  Bx = B0 cos(k y)                       (field-free gaps)
    #   eps > 0:  Bx = B0 sqrt(eps^2 + (1-eps^2) cos^2)  (tension everywhere, no nulls)
    B_gap_fraction: float = 0.0
    # Phase phi of the stripes, Bx ~ cos(k_mod y + phi), relative to the interface
    # ripple sin(k_int y) (radians).
    B_stripe_phase: float = 0.0
    # 3D seeding (solver3d): random v_z in the domain and at the inflow, as a fraction of the
    # bar sound speed, and a z-ripple of the interface; seed fixes the random pattern
    z_pert_vel: float = 0.0
    z_pert_amp: float = 0.0
    z_pert_mode: int = 4
    z_pert_seed: int = 0
    # Galilean frame (solver3d): velocity subtracted from v_x everywhere; with frame_velocity > 0 the
    # x_max boundary becomes a fixed inflow of upstream gas moving at -frame_velocity
    frame_velocity: float = 0.0
    # x_max boundary in solver3d: "auto" (upstream inflow when frame_velocity > 0, else transmissive),
    # "inflow", "transmissive" or "plateau". In the co-moving frame use "plateau": upstream inflow until
    # the transmitted shock arrives, then the shocked state behind it. Holding the upstream state
    # reflects a compression; zero-gradient lets the subsonic shocked gas drain and sends a rarefaction.
    x_max_bc: str = "auto"
    # x_min boundary in solver3d: "inflow" (fixed post-shock inflow) or "plateau" (inflow until the
    # reflected shock leaves through x_min, then the post-reflected state behind it)
    x_min_bc: str = "inflow"
    # absorbing layer (solver3d): over sponge_width at each x end, relax U toward a running time-average
    # of the local state at rate sponge_rate * (depth into the layer)^2; damps waves, not the mean state
    sponge_width: float = 0.0
    sponge_rate: float = 0.0
    sponge_tau: float = 0.5
    equilibrium_closure: str = "uniform_density"
    glm_ch: float = 0.0
    # GLM damping, psi *= exp(-glm_alpha * ch * dt / h). Values in [0, 1] keep the
    # cleaning transport-dominated (Mignone & Tzeferacos 2010); 1.5 measured worse.
    glm_alpha: float = 0.4
    powell_source: bool = True
    use_char_bc: bool = True
    # x boundaries: "inflow_outflow" (Dirichlet post-shock inflow at x_min, transmissive x_max),
    # "extrapolation" (zero-gradient both ends), "periodic", or "auto" (from use_char_bc).
    # "characteristic" is accepted as an alias for "inflow_outflow".
    bc_x_type: str = "auto"
    bc_y_type: str = "periodic"
    enable_smoothing: bool = False
    eta_resistivity: float = 0.0
    localized_rh: bool = False
    enstrophy_locking_fraction: float = 0.20

    def get_bc_x(self) -> str:
        if self.bc_x_type == "characteristic":
            return "inflow_outflow"
        if self.bc_x_type != "auto":
            return self.bc_x_type
        return "inflow_outflow" if self.use_char_bc else "extrapolation"

    def is_striped(self) -> bool:
        return self.B_field_type == "striped" and self.B_transverse > 0

    def is_mhd(self) -> bool:
        return self.B_transverse > 0

    def is_resistive(self) -> bool:
        return self.eta_resistivity > 0 and self.is_mhd()


@dataclass(frozen=True)
class IOConfig:
    diag_interval: int = 20
    snapshot_times: List[float] = field(default_factory=lambda: [
        0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
        0.18, 0.20, 0.22, 0.25,
    ])


@dataclass(init=False)
class Config:
    """Compatibility facade over independently typed configuration groups."""

    def __init__(self, **kwargs):
        known = (set(GridConfig.__dataclass_fields__) | set(PhysicsConfig.__dataclass_fields__)
                 | set(IOConfig.__dataclass_fields__))
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        self.grid = GridConfig(**{key: kwargs[key] for key in GridConfig.__dataclass_fields__ if key in kwargs})
        self.physics = PhysicsConfig(**{key: kwargs[key] for key in PhysicsConfig.__dataclass_fields__ if key in kwargs})
        self.io = IOConfig(**{key: kwargs[key] for key in IOConfig.__dataclass_fields__ if key in kwargs})

    def __getattr__(self, name):
        for group in (self.grid, self.physics, self.io):
            if name in group.__dataclass_fields__:
                return getattr(group, name)
        raise AttributeError(name)

    @property
    def dx(self) -> float:
        return self.grid.dx

    @property
    def dy(self) -> float:
        return self.grid.dy

    @property
    def dz(self) -> float:
        return self.grid.dz

    def get_bc_x(self) -> str:
        return self.physics.get_bc_x()

    def is_striped(self) -> bool:
        return self.physics.is_striped()

    def is_mhd(self) -> bool:
        return self.physics.is_mhd()

    def is_resistive(self) -> bool:
        return self.physics.is_resistive()
