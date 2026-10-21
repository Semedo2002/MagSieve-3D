# physics.py - CUDA HLLD kernel wrapper, variable conversions, MUSCL, jump conditions,
# resistive terms and the shock-parallel stripe equilibrium.

import sys
import os
import sysconfig
from pathlib import Path
import numpy as np


def _configure_cuda_dll_search_path():
    """Expose the bundled CUDA wheel DLLs to CuPy on Windows."""
    if os.name != 'nt':
        return

    candidates = []
    for base in [
        sysconfig.get_paths().get('purelib'),
        sysconfig.get_paths().get('platlib'),
        str(Path(sys.executable).resolve().parent.parent / 'Lib' / 'site-packages'),
    ]:
        if base:
            candidates.append(Path(base))

    for base in candidates:
        for wheel_root in ('nvidia/cuda_runtime', 'nvidia/cuda_nvrtc'):
            wheel_dir = base / wheel_root
            bin_dir = wheel_dir / 'bin'
            if not bin_dir.exists():
                continue
            os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
            if not os.environ.get('CUDA_PATH') or not os.path.exists(os.environ['CUDA_PATH']):
                os.environ['CUDA_PATH'] = str(wheel_dir)
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass


_configure_cuda_dll_search_path()

# ============================================================
# CuPy / CUDA setup
# ============================================================
try:
    import cupy as cp
    HAS_CUPY = True
    dev = cp.cuda.Device(0)
    dev.use()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    dev_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(f"CuPy CUDA acceleration: ENABLED  (GPU: {dev_name})")
    print(f"  Compute capability: {props['major']}.{props['minor']}")
    print(f"  Total GPU memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
except ImportError:
    print("ERROR: CuPy is required for this GPU version.")
    print("Install with: pip install cupy-cuda12x  (or appropriate CUDA version)")
    sys.exit(1)

xp = cp  # alias: all array ops go through this

print(f"NumPy {np.__version__}, CuPy {cp.__version__}, Python {sys.version.split()[0]}")

from .config import (
    NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)


# ============================================================
# CUDA Raw Kernels
# ============================================================

# CUDA source is compiled by CuPy at module import time.
_kernel_path = Path(__file__).with_name("hlld_kernel.cu")
_kernel_source = _kernel_path.read_text(encoding="utf-8-sig")
_hlld_module = cp.RawModule(code=_kernel_source)
_hlld_kernel = _hlld_module.get_function("hlld_flux_x_kernel")


# ============================================================
# GPU Vectorized Operations
# ============================================================
def cons_to_prim(U, gamma):
    """Conservative to primitive -- fully on GPU."""
    W = xp.empty_like(U)
    rho = xp.maximum(U[RHO], FLOOR_RHO)
    ri = 1.0 / rho
    W[iRHO] = rho
    W[iVX] = U[MX] * ri
    W[iVY] = U[MY] * ri
    W[iVZ] = U[MZ] * ri
    W[iBX] = U[BX]
    W[iBY] = U[BY]
    W[iBZ] = U[BZ]
    W[iPSI] = U[PSI]
    W[iCLR] = U[RHOC] * ri
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)
    W[iPR] = xp.maximum((gamma - 1) * (U[EN] - KE - ME), FLOOR_PR)
    return W


def prim_to_cons(W, gamma):
    """Primitive to conservative -- fully on GPU."""
    U = xp.empty_like(W)
    rho = xp.maximum(W[iRHO], FLOOR_RHO)
    U[RHO] = rho
    U[MX] = rho * W[iVX]
    U[MY] = rho * W[iVY]
    U[MZ] = rho * W[iVZ]
    U[BX] = W[iBX]
    U[BY] = W[iBY]
    U[BZ] = W[iBZ]
    U[PSI] = W[iPSI]
    U[RHOC] = rho * W[iCLR]
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    U[EN] = W[iPR] / (gamma - 1) + KE + ME
    return U


def muscl_x(W):
    """MUSCL reconstruction in x -- GPU vectorized with van Leer limiter."""
    dm = W[:, 1:-1, :] - W[:, :-2, :]
    dp = W[:, 2:, :] - W[:, 1:-1, :]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, 1:-2, :] + 0.5 * d[:, :-1, :]
    WR = W[:, 2:-1, :] - 0.5 * d[:, 1:, :]
    WL[iRHO] = xp.maximum(WL[iRHO], FLOOR_RHO)
    WR[iRHO] = xp.maximum(WR[iRHO], FLOOR_RHO)
    WL[iPR] = xp.maximum(WL[iPR], FLOOR_PR)
    WR[iPR] = xp.maximum(WR[iPR], FLOOR_PR)
    return WL, WR


def muscl_y(W):
    """MUSCL reconstruction in y -- GPU vectorized with van Leer limiter."""
    dm = W[:, :, 1:-1] - W[:, :, :-2]
    dp = W[:, :, 2:] - W[:, :, 1:-1]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, :, 1:-2] + 0.5 * d[:, :, :-1]
    WR = W[:, :, 2:-1] - 0.5 * d[:, :, 1:]
    WL[iRHO] = xp.maximum(WL[iRHO], FLOOR_RHO)
    WR[iRHO] = xp.maximum(WR[iRHO], FLOOR_RHO)
    WL[iPR] = xp.maximum(WL[iPR], FLOOR_PR)
    WR[iPR] = xp.maximum(WR[iPR], FLOOR_PR)
    return WL, WR


def hlld_flux_x(WL, WR, gamma, ch):
    """HLLD flux using CUDA kernel -- one thread per interface."""
    n_faces = WL.shape[1]
    F = xp.empty_like(WL)
    smax_arr = xp.empty(n_faces, dtype=xp.float64)

    block = 256
    grid = (n_faces + block - 1) // block

    _hlld_kernel(
        (grid,), (block,),
        (WL, WR, F, smax_arr,
         np.int32(n_faces), np.float64(gamma), np.float64(ch),
         np.float64(FLOOR_RHO), np.float64(FLOOR_PR))
    )

    smax = float(xp.max(smax_arr))
    return F, smax


def hlld_flux_y(WL, WR, gamma, ch):
    """HLLD flux in y by coordinate rotation."""
    perm = list(range(NVAR))
    perm[int(iVX)], perm[int(iVY)] = int(iVY), int(iVX)
    perm[int(iBX)], perm[int(iBY)] = int(iBY), int(iBX)
    Fr, sm = hlld_flux_x(WL[perm], WR[perm], gamma, ch)
    return Fr[perm], sm


# ============================================================
# MHD Rankine-Hugoniot
# ============================================================
def mhd_rankine_hugoniot(gamma, M, rho1, p1, By1):
    """Solve MHD Rankine-Hugoniot (CPU -- called once at init).
    For striped fields, use the RMS By for shock jump estimates."""
    M2 = M * M
    if abs(By1) < 1e-14:
        r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
        cs1 = np.sqrt(gamma * p1 / rho1)
        vs = M * cs1
        p2 = p1 * (2 * gamma * M2 - (gamma - 1)) / (gamma + 1)
        vx2 = vs * (1 - 1.0 / r)
        return rho1 * r, p2, vx2, 0.0, vs

    cs1_sq = gamma * p1 / rho1
    va1_sq = By1**2 / rho1
    cf1 = np.sqrt(cs1_sq + va1_sq)
    vs = M * cf1

    def get_p2(r):
        return p1 + rho1 * vs**2 * (1 - 1.0/r) + 0.5 * By1**2 * (1 - r**2)

    def energy_residual(r):
        p2_ = get_p2(r)
        lhs = 0.5*vs**2 + gamma/(gamma-1)*p1/rho1 + By1**2/rho1
        rhs = 0.5*vs**2/r**2 + gamma/(gamma-1)*p2_/(rho1*r) + r*By1**2/rho1
        return lhs - rhs

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)

    for _ in range(100):
        f = energy_residual(r)
        dr = 1e-6 * r
        fp = (energy_residual(r + dr) - energy_residual(r - dr)) / (2 * dr)
        if abs(fp) < 1e-30: break
        r_new = r - f / fp
        r_new = max(r_new, 1.001)
        r_new = min(r_new, (gamma + 1) / (gamma - 1) - 0.001)
        if abs(r_new - r) < 1e-12:
            r = r_new; break
        r = r_new

    rho2 = rho1 * r
    p2 = get_p2(r)
    By2 = r * By1
    vx2 = vs * (1 - 1.0 / r)
    return rho2, p2, vx2, By2, vs


# ============================================================
# Shock-parallel striped field: upstream pressure equilibrium
# ============================================================
def parallel_stripe_limits(gamma, M, p_bar, closure, rho_bar=1.0, gap_fraction=0.0):
    """Largest admissible B0 for Bx = B0 cos(k y) stripes in pressure balance.

    Both conditions are evaluated for a planar shock of speed vs = M c_s(bar):
      * fast shock: the hydrodynamic jump is an evolutionary MHD shock only if
        the downstream flow stays super-Alfvenic, u2 = vs/r > vA2, which binds
        at the bar centres: B0 < vs sqrt(rho_bar / r);
      * gap Mach > 1 (uniform density only): the gaps carry
        p_bar + (1 - eps^2) B0^2/2 (eps = gap_fraction), so
        M_gap = M sqrt(p_bar / (p_bar + (1 - eps^2) B0^2/2)) must exceed 1.
    """
    cs_bar = np.sqrt(gamma * p_bar / rho_bar)
    vs = M * cs_bar
    r = (gamma + 1) * M**2 / ((gamma - 1) * M**2 + 2)
    b_fast = vs * np.sqrt(rho_bar / r)
    b_gap = (np.sqrt(2 * p_bar * (M**2 - 1) / (1 - gap_fraction**2))
             if closure == "uniform_density" else np.inf)
    return {"vs": vs, "B0_max_fast_shock": b_fast, "B0_max_gap_mach": b_gap,
            "B0_max": min(b_fast, b_gap)}


def parallel_stripe_state(gamma, M, p_bar, closure, Bx, rho_bar=1.0):
    """Upstream equilibrium and per-row hydrodynamic jump for Bx(y) stripes.

    Returns upstream (rho1, p1), post-shock (rho2, p2, vx2), the shock speed,
    the local upstream Mach number and the downstream Alfven Mach number.
    """
    B0 = float(xp.max(xp.abs(Bx)))
    P_total = p_bar + 0.5 * B0**2
    p1 = P_total - 0.5 * Bx**2
    if closure == "uniform_density":
        rho1 = xp.full_like(Bx, rho_bar)
    elif closure == "uniform_temperature":
        rho1 = rho_bar * p1 / p_bar
    else:
        raise ValueError(f"equilibrium_closure must be 'uniform_density' or "
                         f"'uniform_temperature', got {closure!r}")
    vs = M * np.sqrt(gamma * p_bar / rho_bar)
    M_loc = vs / xp.sqrt(gamma * p1 / rho1)
    M2 = M_loc**2
    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    p2 = p1 * (2 * gamma * M2 - (gamma - 1)) / (gamma + 1)
    rho2 = r * rho1
    vx2 = vs * (1.0 - 1.0 / r)
    # Downstream Alfven Mach number in the shock frame (> 1 for a fast shock).
    absB = xp.abs(Bx)
    MA2 = xp.where(absB > 0, (vs / r) * xp.sqrt(rho2) / xp.maximum(absB, 1e-300), xp.inf)
    return {"P_total": P_total, "rho1": rho1, "p1": p1, "rho2": rho2, "p2": p2,
            "vx2": vx2, "vs": vs, "M_local": M_loc, "MA_downstream": MA2}


# ============================================================
# Smoothing Helper
# ============================================================
def smooth(y, n=5, enabled=True):
    """Optional smoothing filter for diagnostics."""
    if not enabled:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * n:
        return y
    kernel = np.ones(n) / n
    yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
    return np.convolve(yp, kernel, mode='same')[n:-n]


# ============================================================
# Linear Theory
# ============================================================
def richtmyer_linear_theory(t_arr, gamma, mach, rho1, rho_heavy, a0, k, By=0.0):
    """Compute linear RMI growth (CPU)."""
    M2 = mach * mach
    if abs(By) < 1e-14:
        cs1 = np.sqrt(gamma * 1.0 / rho1)
        vs = mach * cs1
    else:
        cs1_sq = gamma * 1.0 / rho1
        va1_sq = By**2 / rho1
        cf1 = np.sqrt(cs1_sq + va1_sq)
        vs = mach * cf1

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)
    a0_post = a0 / r
    rho2_light = rho1 * r
    rho2_heavy = rho_heavy * r
    A_post = (rho2_heavy - rho2_light) / (rho2_heavy + rho2_light)
    delta_v = vs * (1.0 - 1.0 / r)
    x_shock_init = 1.2
    x_interface = 1.5
    t_shock = (x_interface - x_shock_init) / vs
    da_dt = A_post * k * a0_post * delta_v

    a_linear = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t < t_shock:
            a_linear[i] = a0
        else:
            dt_val = t - t_shock
            if abs(By) < 1e-14:
                a_linear[i] = a0_post + da_dt * dt_val
            else:
                rho_avg = 0.5 * (rho2_light + rho2_heavy)
                vA = By / np.sqrt(rho_avg)
                omega_A = k * vA
                if omega_A > 1e-14:
                    a_linear[i] = a0_post * np.cos(omega_A * dt_val) + \
                                  (da_dt / omega_A) * np.sin(omega_A * dt_val)
                else:
                    a_linear[i] = a0_post + da_dt * dt_val

    info = {
        'A_post': A_post, 'da_dt': da_dt, 'delta_v': delta_v,
        'vs': vs, 't_shock': t_shock, 'a0_post': a0_post, 'r': r,
    }
    return np.abs(a_linear), t_shock, info


# ============================================================
# Localized Rankine-Hugoniot (GPU-vectorised)
# ============================================================
def mhd_rankine_hugoniot_local_gpu(gamma, M, rho1, p1, By_field):
    """Solve MHD R-H jump conditions at every grid cell on the GPU.

    Instead of using a single RMS By_eff, this computes the post-shock
    state (rho2, p2, vx2, By2, vs) at each (x,y) using the local By(y).

    Parameters
    ----------
    gamma : float
    M : float -- shock Mach number
    rho1 : float -- pre-shock density (scalar, uniform)
    p1 : float -- pre-shock pressure (scalar, uniform)
    By_field : cupy array, shape (nx_tot, ny_tot) -- local By values

    Returns
    -------
    rho2, p2, vx2, By2, vs : cupy arrays, same shape as By_field
    """
    # For cells where |By| < threshold, use pure hydro jump
    M2 = M * M
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0

    # Hydro jump (By=0 limit)
    r_hydro = gp1 * M2 / (gm1 * M2 + 2.0)
    r_hydro = min(r_hydro, gp1 / gm1 - 0.01)
    cs1 = np.sqrt(gamma * p1 / rho1)
    vs_hydro = M * cs1
    p2_hydro = p1 * (2.0 * gamma * M2 - gm1) / gp1
    vx2_hydro = vs_hydro * (1.0 - 1.0 / r_hydro)

    By_abs = xp.abs(By_field)
    is_hydro = By_abs < 1e-10

    # For MHD cells, compute per-cell jump
    # Fast magnetosonic speed: cf = sqrt(cs^2 + va^2)
    cs1_sq = gamma * p1 / rho1
    va1_sq = By_field**2 / rho1
    cf1 = xp.sqrt(cs1_sq + va1_sq)
    vs_local = M * cf1

    # Initial guess for compression ratio (hydro value)
    r_init = gp1 * M2 / (gm1 * M2 + 2.0)
    r_init = min(r_init, gp1 / gm1 - 0.01)
    r = xp.full_like(By_field, r_init)

    # Vectorized Newton iteration with a per-cell residual mask. Cells whose
    # residual is already below 1e-6 are left unchanged and do not consume
    # further GPU work in the next iteration.
    tolerance = 1e-6
    max_iterations = 50
    for _ in range(max_iterations):
        p2_iter = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / r) + 0.5 * By_field**2 * (1.0 - r**2)
        lhs = 0.5 * vs_local**2 + gamma / gm1 * p1 / rho1 + By_field**2 / rho1
        rhs = 0.5 * vs_local**2 / r**2 + gamma / gm1 * p2_iter / (rho1 * r) + r * By_field**2 / rho1
        f_val = lhs - rhs
        converged = xp.abs(f_val) < tolerance
        if bool(xp.all(converged)):
            break

        dr = 1e-6 * r
        p2_p = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / (r + dr)) + 0.5 * By_field**2 * (1.0 - (r + dr)**2)
        rhs_p = 0.5 * vs_local**2 / (r + dr)**2 + gamma / gm1 * p2_p / (rho1 * (r + dr)) + (r + dr) * By_field**2 / rho1
        p2_m = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / (r - dr)) + 0.5 * By_field**2 * (1.0 - (r - dr)**2)
        rhs_m = 0.5 * vs_local**2 / (r - dr)**2 + gamma / gm1 * p2_m / (rho1 * (r - dr)) + (r - dr) * By_field**2 / rho1
        fp_val = ((lhs - rhs_p) - (lhs - rhs_m)) / (2.0 * dr)

        fp_safe = xp.where(xp.abs(fp_val) > 1e-30, fp_val, xp.ones_like(fp_val))
        r_new = r - f_val / fp_safe
        r_new = xp.clip(r_new, 1.001, gp1 / gm1 - 0.001)
        r = xp.where(converged, r, r_new)

    # Compute post-shock state
    rho2 = rho1 * r
    p2 = p1 + rho1 * vs_local**2 * (1.0 - 1.0 / r) + 0.5 * By_field**2 * (1.0 - r**2)
    By2 = r * By_field
    vx2 = vs_local * (1.0 - 1.0 / r)

    # Fall back to hydro jump where |By| ~ 0
    rho2 = xp.where(is_hydro, rho1 * r_hydro, rho2)
    p2 = xp.where(is_hydro, p2_hydro, p2)
    vx2 = xp.where(is_hydro, vx2_hydro, vx2)
    By2 = xp.where(is_hydro, 0.0, By2)
    vs_out = xp.where(is_hydro, vs_hydro, vs_local)

    # Floor pressure
    p2 = xp.maximum(p2, FLOOR_PR)

    return rho2, p2, vx2, By2, vs_out


# ============================================================
# Resistive MHD (GPU-vectorised, conservative)
# ============================================================
def resistivity_field(W, cfg):
    """Cell-centred eta for the configured model. Returns a float for "constant" (so the
    uniform case runs exactly as before) and an array shaped like W[0] otherwise."""
    model = cfg.eta_model
    if model == "constant":
        eta = cfg.eta_resistivity
        return min(eta, cfg.eta_max) if cfg.eta_max > 0 else eta
    C = xp.clip(W[iCLR], 0.0, 1.0)
    if model == "material":
        eta = _blend(cfg.eta_fuel, cfg.eta_liner, C)
    else:
        # Spitzer: eta ~ T^-3/2, with T floored so cold cells don't blow up
        T = xp.maximum(W[iPR] / xp.maximum(W[iRHO], FLOOR_RHO), cfg.eta_T_min)
        eta = xp.minimum(cfg.eta_spitzer * T**-1.5, cfg.eta_max)
        if cfg.eta_liner > 0:
            eta = _blend(eta, cfg.eta_liner, C)
    if cfg.eta_max > 0:
        eta = xp.minimum(eta, cfg.eta_max)
    return eta


def _blend(a, b, C):
    # log blend keeps a mixed cell between two very different conductivities from being
    # dominated by the larger eta; falls back to linear if either side is zero
    if isinstance(a, float) and isinstance(b, float) and (a <= 0 or b <= 0):
        return (1 - C) * a + C * b
    a = xp.maximum(a, 1e-300); b = xp.maximum(b, 1e-300)
    return xp.exp((1 - C) * xp.log(a) + C * xp.log(b))


def eta_max_of(eta):
    return float(eta) if isinstance(eta, (int, float)) else float(xp.max(eta))


def face_eta(eta, axis, mean="harmonic"):
    """eta on the faces between neighbouring cells along `axis`. Harmonic mean by default:
    it gives the right series resistance across a conductivity jump (a face between a good
    and a bad conductor carries the current the good one allows, not the average)."""
    if isinstance(eta, (int, float)):
        return eta
    n = eta.shape[axis]
    a = xp.take(eta, xp.arange(0, n - 1), axis=axis)
    b = xp.take(eta, xp.arange(1, n), axis=axis)
    if mean == "arithmetic":
        return 0.5 * (a + b)
    return 2.0 * a * b / xp.maximum(a + b, 1e-300)


def compute_resistive_terms(U, eta, dx, dy, face_mean="harmonic"):
    """Resistive induction and energy terms in conservative form.

    Induction: dB/dt = -curl(eta J), J = curl(B), with all three current
    components of the 2.5D problem (dz = 0).
    Energy:    dE/dt = -div(eta J x B)   (resistive Poynting flux).

    eta is a float or a cell-centred array; on faces it is averaged with face_eta.
    Both terms are differences of face quantities, so the energy update telescopes
    exactly and Ohmic heating eta|J|^2 shows up in the pressure through the drop in
    magnetic energy -- it is not added to E separately.

    Returns (R, Gx, Gy, q): R has shape U.shape with nonzero BX, BY, BZ, EN
    entries on cells [1:-1, 1:-1]; Gx (nx_tot-1, ny_tot) and Gy (nx_tot, ny_tot-1)
    are the energy fluxes on x- and y-faces; q is the Ohmic heating density on cells
    [1:-1, 1:-1] (diagnostic only), built from each current component on the faces
    where its derivative is compact. In 1D it equals the discrete magnetic energy
    loss exactly (summation by parts).
    """
    R = xp.zeros_like(U)
    Gx = xp.zeros((U.shape[1] - 1, U.shape[2]), dtype=U.dtype)
    Gy = xp.zeros((U.shape[1], U.shape[2] - 1), dtype=U.dtype)
    q = xp.zeros(U.shape[1:], dtype=U.dtype)
    if eta_max_of(eta) <= 0:
        return R, Gx, Gy, q
    ex = face_eta(eta, 0, face_mean)
    ey = face_eta(eta, 1, face_mean)

    Bx, By, Bz = U[BX], U[BY], U[BZ]

    # Cell-centred transverse derivatives (zero on the outermost ring).
    dBx_dy = xp.zeros_like(Bx); dBz_dy = xp.zeros_like(Bz)
    dBy_dx = xp.zeros_like(By); dBz_dx = xp.zeros_like(Bz)
    dBx_dy[:, 1:-1] = (Bx[:, 2:] - Bx[:, :-2]) / (2.0 * dy)
    dBz_dy[:, 1:-1] = (Bz[:, 2:] - Bz[:, :-2]) / (2.0 * dy)
    dBy_dx[1:-1, :] = (By[2:, :] - By[:-2, :]) / (2.0 * dx)
    dBz_dx[1:-1, :] = (Bz[2:, :] - Bz[:-2, :]) / (2.0 * dx)

    # Currents on x-faces (i+1/2, j): compact normal derivative, averaged transverse.
    Jx_xf = 0.5 * (dBz_dy[1:, :] + dBz_dy[:-1, :])
    Jy_xf = -(Bz[1:, :] - Bz[:-1, :]) / dx
    Jz_xf = (By[1:, :] - By[:-1, :]) / dx - 0.5 * (dBx_dy[1:, :] + dBx_dy[:-1, :])

    # Currents on y-faces (i, j+1/2).
    Jx_yf = (Bz[:, 1:] - Bz[:, :-1]) / dy
    Jz_yf = 0.5 * (dBy_dx[:, 1:] + dBy_dx[:, :-1]) - (Bx[:, 1:] - Bx[:, :-1]) / dy

    # electric field E = eta J on the faces
    Ex_yf, Ez_yf = ey * Jx_yf, ey * Jz_yf
    Ey_xf, Ez_xf = ex * Jy_xf, ex * Jz_xf

    # Induction, curl form: dBx/dt = -d(Ez)/dy, dBy/dt = d(Ez)/dx,
    # dBz/dt = -d(Ey)/dx + d(Ex)/dy.
    R[BX, 1:-1, 1:-1] = -(Ez_yf[1:-1, 1:] - Ez_yf[1:-1, :-1]) / dy
    R[BY, 1:-1, 1:-1] = (Ez_xf[1:, 1:-1] - Ez_xf[:-1, 1:-1]) / dx
    R[BZ, 1:-1, 1:-1] = (-(Ey_xf[1:, 1:-1] - Ey_xf[:-1, 1:-1]) / dx
                         + (Ex_yf[1:-1, 1:] - Ex_yf[1:-1, :-1]) / dy)

    # Resistive Poynting flux E x B on faces.
    By_xf = 0.5 * (By[1:, :] + By[:-1, :]); Bz_xf = 0.5 * (Bz[1:, :] + Bz[:-1, :])
    Bx_yf = 0.5 * (Bx[:, 1:] + Bx[:, :-1]); Bz_yf = 0.5 * (Bz[:, 1:] + Bz[:, :-1])
    Gx[...] = Ey_xf * Bz_xf - Ez_xf * By_xf
    Gy[...] = Ez_yf * Bx_yf - Ex_yf * Bz_yf
    R[EN, 1:-1, 1:-1] = (-(Gx[1:, 1:-1] - Gx[:-1, 1:-1]) / dx
                         - (Gy[1:-1, 1:] - Gy[1:-1, :-1]) / dy)

    # Ohmic heating: Jy is compact on x-faces, Jx on y-faces, Jz on both (half weight each)
    qx = Ey_xf * Jy_xf + 0.5 * Ez_xf * Jz_xf
    qy = Ex_yf * Jx_yf + 0.5 * Ez_yf * Jz_yf
    q[1:-1, 1:-1] = (0.5 * (qx[1:, 1:-1] + qx[:-1, 1:-1])
                     + 0.5 * (qy[1:-1, 1:] + qy[1:-1, :-1]))
    return R, Gx, Gy, q

    Bx, By, Bz = U[BX], U[BY], U[BZ]

    # Cell-centred transverse derivatives (zero on the outermost ring).
    dBx_dy = xp.zeros_like(Bx); dBz_dy = xp.zeros_like(Bz)
    dBy_dx = xp.zeros_like(By); dBz_dx = xp.zeros_like(Bz)
    dBx_dy[:, 1:-1] = (Bx[:, 2:] - Bx[:, :-2]) / (2.0 * dy)
    dBz_dy[:, 1:-1] = (Bz[:, 2:] - Bz[:, :-2]) / (2.0 * dy)
    dBy_dx[1:-1, :] = (By[2:, :] - By[:-2, :]) / (2.0 * dx)
    dBz_dx[1:-1, :] = (Bz[2:, :] - Bz[:-2, :]) / (2.0 * dx)

    # Currents on x-faces (i+1/2, j): compact normal derivative, averaged transverse.
    Jx_xf = 0.5 * (dBz_dy[1:, :] + dBz_dy[:-1, :])
    Jy_xf = -(Bz[1:, :] - Bz[:-1, :]) / dx
    Jz_xf = (By[1:, :] - By[:-1, :]) / dx - 0.5 * (dBx_dy[1:, :] + dBx_dy[:-1, :])

    # Currents on y-faces (i, j+1/2).
    Jx_yf = (Bz[:, 1:] - Bz[:, :-1]) / dy
    Jz_yf = 0.5 * (dBy_dx[:, 1:] + dBy_dx[:, :-1]) - (Bx[:, 1:] - Bx[:, :-1]) / dy

    # Induction, curl form: dBx/dt = -d(eta Jz)/dy, dBy/dt = d(eta Jz)/dx,
    # dBz/dt = -d(eta Jy)/dx + d(eta Jx)/dy.
    R[BX, 1:-1, 1:-1] = -eta * (Jz_yf[1:-1, 1:] - Jz_yf[1:-1, :-1]) / dy
    R[BY, 1:-1, 1:-1] = eta * (Jz_xf[1:, 1:-1] - Jz_xf[:-1, 1:-1]) / dx
    R[BZ, 1:-1, 1:-1] = (-eta * (Jy_xf[1:, 1:-1] - Jy_xf[:-1, 1:-1]) / dx
                         + eta * (Jx_yf[1:-1, 1:] - Jx_yf[1:-1, :-1]) / dy)

    # Resistive Poynting flux eta (J x B) on faces.
    Bx_xf = 0.5 * (Bx[1:, :] + Bx[:-1, :]); By_xf = 0.5 * (By[1:, :] + By[:-1, :])
    Bz_xf = 0.5 * (Bz[1:, :] + Bz[:-1, :])
    Bx_yf = 0.5 * (Bx[:, 1:] + Bx[:, :-1]); Bz_yf = 0.5 * (Bz[:, 1:] + Bz[:, :-1])
    Gx[...] = eta * (Jy_xf * Bz_xf - Jz_xf * By_xf)
    Gy[...] = eta * (Jz_yf * Bx_yf - Jx_yf * Bz_yf)
    R[EN, 1:-1, 1:-1] = (-(Gx[1:, 1:-1] - Gx[:-1, 1:-1]) / dx
                         - (Gy[1:-1, 1:] - Gy[1:-1, :-1]) / dy)
    return R, Gx, Gy
