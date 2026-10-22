# config.py - Structured solver configuration.
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
    # resistivity model: "constant" (eta_resistivity everywhere), "material" (eta_fuel where the
    # tracer C = 0, eta_liner where C = 1, log blend in between) or "spitzer" (eta_spitzer * T^-3/2
    # with T = p/rho floored at eta_T_min; blended toward eta_liner by C when eta_liner > 0).
    # eta_max > 0 caps eta in any model (required for spitzer). Face values use eta_face_mean.
    eta_model: str = "constant"
    eta_fuel: float = 0.0
    eta_liner: float = 0.0
    eta_spitzer: float = 0.0
    eta_T_min: float = 1e-2
    eta_max: float = 0.0
    eta_face_mean: str = "harmonic"
    # equation of state, two materials: "fuel" where the tracer is 0, "liner" where it is 1.
    # fuel: ideal or stiffened (gamma, p_inf); liner: ideal, stiffened (gamma_liner, p_inf_liner) or
    # mie_gruneisen (linear Us-Up Hugoniot reference: mg_rho0, mg_c0, mg_s, mg_gamma0, code units).
    # With more than one material the tracer becomes a volume fraction advected non-conservatively
    # (material_tracer = "volume_fraction", keeps pressure equilibrium across the interface);
    # "mass_fraction" blends by the conserved tracer instead and is kept for comparison.
    eos_fuel: str = "ideal"
    p_inf: float = 0.0
    eos_liner: str = "ideal"
    gamma_liner: Optional[float] = None
    p_inf_liner: float = 0.0
    mg_rho0: float = 0.0
    mg_c0: float = 0.0
    mg_s: float = 0.0
    mg_gamma0: float = 0.0
    material_tracer: str = "volume_fraction"
    localized_rh: bool = False
    enstrophy_locking_fraction: float = 0.20

    def __post_init__(self):
        if self.eta_model not in ("constant", "material", "spitzer"):
            raise ValueError(f"eta_model must be constant, material or spitzer, not {self.eta_model!r}")
        if self.eta_face_mean not in ("harmonic", "arithmetic"):
            raise ValueError(f"eta_face_mean must be harmonic or arithmetic, not {self.eta_face_mean!r}")
        if self.eta_model == "spitzer" and not (self.eta_T_min > 0 and self.eta_max > 0 and self.eta_spitzer > 0):
            raise ValueError("eta_model='spitzer' needs eta_spitzer > 0, eta_T_min > 0 and eta_max > 0")
        if min(self.eta_resistivity, self.eta_fuel, self.eta_liner, self.eta_max) < 0:
            raise ValueError("resistivities must be non-negative")
        if self.eos_fuel not in ("ideal", "stiffened"):
            raise ValueError(f"eos_fuel must be ideal or stiffened, not {self.eos_fuel!r}")
        if self.eos_liner not in ("ideal", "stiffened", "mie_gruneisen"):
            raise ValueError(f"eos_liner must be ideal, stiffened or mie_gruneisen, not {self.eos_liner!r}")
        if self.material_tracer not in ("volume_fraction", "mass_fraction"):
            raise ValueError(f"material_tracer must be volume_fraction or mass_fraction")
        if (self.eos_fuel == "ideal" and self.p_inf != 0) or (self.eos_liner == "ideal" and self.p_inf_liner != 0):
            raise ValueError("p_inf is only used by the stiffened EOS; set eos_fuel/eos_liner='stiffened'")
        if min(self.p_inf, self.p_inf_liner) < 0:
            raise ValueError("p_inf must be non-negative")
        if self.eos_liner == "mie_gruneisen" and min(self.mg_rho0, self.mg_c0, self.mg_s, self.mg_gamma0) <= 0:
            raise ValueError("eos_liner='mie_gruneisen' needs mg_rho0, mg_c0, mg_s, mg_gamma0 > 0")

    def get_bc_x(self) -> str:
        if self.bc_x_type == "characteristic":
            return "inflow_outflow"
        if self.bc_x_type != "auto":
            return self.bc_x_type
        return "inflow_outflow" if self.use_char_bc else "extrapolation"

    def is_striped(self) -> bool:
        return self.B_field_type == "striped" and self.B_transverse > 0

    def is_multimaterial(self) -> bool:
        gl = self.gamma if self.gamma_liner is None else self.gamma_liner
        return self.eos_fuel != "ideal" or self.eos_liner != "ideal" or gl != self.gamma

    def is_mhd(self) -> bool:
        return self.B_transverse > 0

    def is_resistive(self) -> bool:
        # independent of B_transverse: a field set up by hand (e.g. axial Bz) still diffuses
        if self.eta_model == "constant":
            return self.eta_resistivity > 0
        if self.eta_model == "material":
            return max(self.eta_fuel, self.eta_liner) > 0
        return True


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

    def is_multimaterial(self) -> bool:
        return self.physics.is_multimaterial()
