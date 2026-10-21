# units.py - SI helpers for resistivity: conductivity -> code-unit magnetic diffusivity, Spitzer.
import numpy as np

MU0 = 4e-7 * np.pi
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


def eta_code_from_sigma(sigma, L0, v0):
    """Magnetic diffusivity 1/(mu0 sigma) in code units (L0 * v0). sigma in S/m; None or inf -> 0 (ideal)."""
    if sigma is None or not np.isfinite(sigma):
        return 0.0
    return 1.0 / (MU0 * sigma * L0 * v0)


def temperature_eV_per_code(v0, ion_mass_amu=2.0, Z=1.0):
    """kT in eV for code temperature p/rho = 1 (units of v0^2), fully ionised, T_e = T_i."""
    return ion_mass_amu * AMU * v0**2 / ((1.0 + Z) * E_CHARGE)


def spitzer_coefficient(L0, v0, ion_mass_amu=2.0, Z=1.0, coulomb_log=10.0):
    """eta_spitzer for Config: code-unit diffusivity at code temperature 1, so that eta = coef * T^-3/2.
    Uses eta [Ohm m] ~ 5.2e-5 Z lnL / T_eV^(3/2) (NRL formulary, transverse ~2x parallel; order of magnitude)."""
    T1 = temperature_eV_per_code(v0, ion_mass_amu, Z)
    return 5.2e-5 * Z * coulomb_log / MU0 / (L0 * v0) / T1**1.5


def resistivity_kwargs(model, L0, v0, sigma_liner=None, sigma_fuel=None, sigma_min=1e6, T_min_eV=1.0,
                       ion_mass_amu=2.0, Z=1.0, coulomb_log=10.0):
    """Config keywords for the resistivity model chosen on an example's command line.

    model: "ideal", "material" (sigma_fuel where the tracer is 0, sigma_liner where it is 1) or
    "spitzer" (Spitzer in the fuel, sigma_liner in the liner if given). sigma_min sets the eta cap,
    which also bounds the explicit diffusion time step."""
    if model == "ideal":
        return {}
    eta_max = eta_code_from_sigma(sigma_min, L0, v0)
    kw = dict(eta_model=model, eta_liner=eta_code_from_sigma(sigma_liner, L0, v0), eta_max=eta_max)
    if model == "material":
        kw["eta_fuel"] = eta_code_from_sigma(sigma_fuel, L0, v0)
    elif model == "spitzer":
        kw["eta_spitzer"] = spitzer_coefficient(L0, v0, ion_mass_amu, Z, coulomb_log)
        kw["eta_T_min"] = T_min_eV / temperature_eV_per_code(v0, ion_mass_amu, Z)
    else:
        raise ValueError(f"unknown eta model {model!r}")
    return kw


def add_resistivity_args(ap):
    g = ap.add_argument_group("resistivity (SI)")
    g.add_argument("--eta-model", dest="eta_model", choices=["ideal", "material", "spitzer"], default=None,
                   help="default: material if a conductivity is given, else ideal")
    g.add_argument("--sigma-liner", dest="sigma_liner", type=float, default=None, help="liner conductivity, S/m (Al ~3.5e7)")
    g.add_argument("--sigma-fuel", dest="sigma_fuel", type=float, default=None, help="fuel conductivity, S/m (default: ideal)")
    g.add_argument("--sigma-min", dest="sigma_min", type=float, default=1e6,
                   help="conductivity floor = eta cap, S/m; sets the smallest diffusion time step")
    g.add_argument("--t-min-ev", dest="t_min_ev", type=float, default=1.0, help="Spitzer temperature floor, eV")
    g.add_argument("--coulomb-log", dest="coulomb_log", type=float, default=10.0)
    g.add_argument("--ion-mass-amu", dest="ion_mass_amu", type=float, default=2.0, help="fuel ion mass (D = 2)")


def resistivity_from_args(a, L0, v0):
    model = a.eta_model or ("material" if (a.sigma_liner or a.sigma_fuel) else "ideal")
    a.eta_model = model
    return resistivity_kwargs(model, L0, v0, a.sigma_liner, a.sigma_fuel, a.sigma_min, a.t_min_ev,
                              a.ion_mass_amu, 1.0, a.coulomb_log)
