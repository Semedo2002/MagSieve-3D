# main.py — The Execution Hub
# Magnetic Sieve Scaling Law: B_critical = Psi * sqrt(rho_avg) * V_hydro
# Parametric sweeps to prove the linear relationship between B_crit and
# the impulsive hydrodynamic growth velocity.
#
# Strategy for ~15 min/run on Quadro P1000:
#   - Sweep runs use 200x100 resolution (~3-4 min each)
#   - Bisection search for B_crit (4-5 iterations per parameter point)
#   - Total: ~30 runs = ~2 hours
#   - Full-resolution showcase runs (400x200) for the 4 key cases

import time
import os
import sys
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import (
    Config, NVAR,
    iRHO, iVX, iVY, iBX, iBY, iBZ, iPR, iCLR,
)
from physics import (
    xp, cp,
    cons_to_prim, prim_to_cons,
    mhd_rankine_hugoniot,
    richtmyer_linear_theory,
)
from solver import (
    MHDSolver, PostProcessor,
    save_figure, to_numpy, OUTPUT_DIR,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Verification Tests (unchanged)
# ============================================================
def brio_wu_test(nx=800, t_end=0.1, plot=True):
    print("\n--- Brio-Wu Shock Tube Verification (GPU) ---")
    sys.stdout.flush()
    cfg = Config(
        nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.05,
        t_end=t_end, cfl=0.30, gamma=2.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=10000, snapshot_times=[t_end],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic",
    )
    solver = MHDSolver(cfg); ng = solver.ng
    W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x; left = x < 0.5
    W[iRHO] = xp.where(left[:, None], 1.0, 0.125)
    W[iPR] = xp.where(left[:, None], 1.0, 0.1)
    W[iBX] = 0.75
    W[iBY] = xp.where(left[:, None], 1.0, -1.0)
    W[iCLR] = xp.where(left[:, None], 1.0, 0.0)
    solver.U = prim_to_cons(W, cfg.gamma); solver.t = 0.0; solver.step = 0
    t0 = time.time()
    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        dt_cfl = solver.compute_dt(W_c)
        solver.dt = min(dt_cfl, cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3(); solver.t += solver.dt; solver.step += 1
    elapsed = time.time() - t0
    print(f"  Brio-Wu: {solver.step} steps, {elapsed:.1f}s")
    W_final = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = to_numpy(W_final[iRHO, ng:-ng, ng])
    By_1d = to_numpy(W_final[iBY, ng:-ng, ng])
    vx_1d = to_numpy(W_final[iVX, ng:-ng, ng])
    rho_max = float(np.max(rho_1d)); rho_min = float(np.min(rho_1d))
    n_levels = len(np.unique(np.round(rho_1d, 2)))
    check4 = bool(np.any(By_1d[:-1] * By_1d[1:] < 0))
    vx_range = float(np.max(vx_1d) - np.min(vx_1d))
    passed = (0.1 < rho_min < 0.2) and (0.9 < rho_max < 1.05) and (n_levels > 10) and check4 and (rho_min < 0.18) and (vx_range > 0.5)
    print(f"  rho: [{rho_min:.4f}, {rho_max:.4f}], levels: {n_levels}, By flip: {check4}, vx range: {vx_range:.3f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if plot:
        p_1d = to_numpy(W_final[iPR, ng:-ng, ng]); x_1d = to_numpy(x[ng:-ng])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, data, ylabel, title, color in [
            (axes[0,0], rho_1d, r'$\rho$', 'Density', 'b'),
            (axes[0,1], p_1d, '$p$', 'Pressure', 'r'),
            (axes[1,0], vx_1d, '$v_x$', 'Velocity', 'g'),
            (axes[1,1], By_1d, '$B_y$', 'Transverse B', 'm'),
        ]:
            ax.plot(x_1d, data, f'{color}-', lw=1)
            ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3)
        axes[1,0].set_xlabel('$x$'); axes[1,1].set_xlabel('$x$')
        fig.suptitle(f'Brio-Wu (GPU), t={t_end}, nx={nx}', fontsize=14, fontweight='bold')
        plt.tight_layout(); save_figure(fig, 'brio_wu_test.png')
    sys.stdout.flush()
    return passed


def linear_wave_convergence_test(plot=True):
    print("\n--- Alfven Wave Convergence Test (GPU) ---")
    sys.stdout.flush()
    resolutions = [32, 64, 128, 256]; errors = []
    rho0 = 1.0; p0 = 0.1; Bx0 = 1.0; amp = 1e-6; gamma = 5.0/3.0; Lx = 1.0
    vA = Bx0 / np.sqrt(rho0); period = Lx / vA
    print(f"  vA={vA:.4f}, period={period:.4f}")
    for nx in resolutions:
        cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=Lx, y_min=0.0, y_max=0.05,
            t_end=period, cfl=0.25, gamma=gamma, mach=1.0, B_transverse=0.0,
            interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
            diag_interval=100000, snapshot_times=[period],
            powell_source=False, use_char_bc=False, bc_x_type="periodic", bc_y_type="periodic")
        solver = MHDSolver(cfg); ng = solver.ng; x = solver.x; kx = 2.0*np.pi/Lx
        W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
        W[iRHO] = rho0; W[iPR] = p0; W[iBX] = Bx0
        W[iBY] = amp * xp.sin(kx*x)[:, None] * xp.ones(solver.ny_tot)[None, :]
        W[iVY] = -amp * xp.sin(kx*x)[:, None] * xp.ones(solver.ny_tot)[None, :] / np.sqrt(rho0)
        W_init = W.copy()
        solver.U = prim_to_cons(W, cfg.gamma); solver.t = 0.0; solver.step = 0
        while solver.t < cfg.t_end and solver.step < 100000:
            W_c = cons_to_prim(solver.U, cfg.gamma)
            solver.dt = min(solver.compute_dt(W_c), cfg.t_end - solver.t)
            if solver.dt <= 1e-16: break
            solver.step_ssprk3(); solver.t += solver.dt; solver.step += 1
        W_final = cons_to_prim(solver.U, cfg.gamma)
        L1_err = float(np.mean(np.abs(to_numpy(W_final[iBY, ng:-ng, ng]) - to_numpy(W_init[iBY, ng:-ng, ng]))))
        errors.append(L1_err); print(f"  nx={nx:4d}: L1={L1_err:.2e}, steps={solver.step}")
    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            orders.append(np.log(errors[i-1]/errors[i]) / np.log(resolutions[i]/resolutions[i-1]))
    mean_order = np.mean(orders) if orders else 0
    passed = mean_order >= 1.5
    print(f"  Mean order: {mean_order:.2f} ({'PASS' if passed else 'FAIL'})")
    if plot and len(errors) > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(resolutions, errors, 'bo-', lw=2, ms=8, label=f'Measured ({mean_order:.2f})')
        ref_x = np.array([resolutions[0], resolutions[-1]], dtype=float)
        ax.loglog(ref_x, errors[0]*(resolutions[0])**2 / ref_x**2, 'k--', lw=1, alpha=0.5, label='2nd order')
        ax.set_xlabel('$N_x$'); ax.set_ylabel('$L_1$ error'); ax.set_title('Alfven Wave Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); save_figure(fig, 'convergence_alfven.png')
    sys.stdout.flush()
    return passed


def contact_discontinuity_test(plot=True):
    print("\n--- Contact Discontinuity Test (GPU) ---")
    sys.stdout.flush()
    cfg = Config(nx=400, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.01,
        t_end=0.2, cfl=0.30, gamma=5.0/3.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=100000, snapshot_times=[0.2],
        powell_source=False, use_char_bc=False, bc_x_type="extrapolation", bc_y_type="periodic")
    solver = MHDSolver(cfg); ng = solver.ng
    W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot)); x = solver.x
    W[iRHO] = xp.where(x[:, None] < 0.5, 1.0, 3.0); W[iPR] = 1.0; W[iVX] = 1.0; W[iBX] = 0.5
    W[iCLR] = xp.where(x[:, None] < 0.5, 0.0, 1.0)
    solver.U = prim_to_cons(W, cfg.gamma); solver.t = 0.0; solver.step = 0
    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        solver.dt = min(solver.compute_dt(W_c), cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3(); solver.t += solver.dt; solver.step += 1
    W_final = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = to_numpy(W_final[iRHO, ng:-ng, ng]); p_1d = to_numpy(W_final[iPR, ng:-ng, ng])
    vx_1d = to_numpy(W_final[iVX, ng:-ng, ng])
    p_var = float(np.max(p_1d) - np.min(p_1d)) / float(np.mean(p_1d))
    v_var = float(np.max(vx_1d) - np.min(vx_1d))
    rho_min = float(np.min(rho_1d)); rho_max = float(np.max(rho_1d))
    passed = p_var < 0.05 and v_var < 0.1 and rho_min > 0.9 and rho_max < 3.1
    print(f"  p_var={p_var*100:.2f}%, v_var={v_var:.4f}, rho=[{rho_min:.3f},{rho_max:.3f}]")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if plot:
        x_1d = to_numpy(x[ng:-ng])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(x_1d, rho_1d, 'b-'); axes[0].set_title('Density')
        axes[1].plot(x_1d, p_1d, 'r-'); axes[1].set_title('Pressure')
        axes[2].plot(x_1d, vx_1d, 'g-'); axes[2].set_title('Velocity')
        for ax in axes: ax.grid(True, alpha=0.3); ax.set_xlabel('$x$')
        fig.suptitle('Contact Discontinuity Test', fontweight='bold'); plt.tight_layout()
        save_figure(fig, 'contact_test.png')
    sys.stdout.flush()
    return passed


# ============================================================
# SCALING LAW ENGINE
# ============================================================

def compute_theoretical_V_hydro(gamma, mach, rho1, density_ratio, a0,
                                perturbation_mode, Ly):
    """Compute the Richtmyer impulsive linear growth rate V_hydro.

    V_hydro = k_int * Delta_U * A+ * a0+

    where:
        k_int  = 2*pi*perturbation_mode / Ly   (interface wavenumber)
        Delta_U = post-shock interface velocity (from R-H jump)
        A+     = post-shock Atwood number
        a0+    = post-shock perturbation amplitude = a0 / r
        r      = density compression ratio
    """
    M2 = mach * mach
    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)

    cs1 = np.sqrt(gamma * 1.0 / rho1)
    vs = mach * cs1
    delta_U = vs * (1.0 - 1.0 / r)

    rho_light_post = rho1 * r
    rho_heavy_post = (rho1 * density_ratio) * r
    A_post = (rho_heavy_post - rho_light_post) / (rho_heavy_post + rho_light_post)
    a0_post = a0 / r

    k_int = 2.0 * np.pi * perturbation_mode / Ly
    V_hydro = k_int * delta_U * A_post * a0_post

    rho_avg = 0.5 * (rho_light_post + rho_heavy_post)

    return V_hydro, {
        'k_int': k_int, 'delta_U': delta_U, 'A_post': A_post,
        'a0_post': a0_post, 'r': r, 'vs': vs,
        'rho_avg': rho_avg,
        'rho_light_post': rho_light_post,
        'rho_heavy_post': rho_heavy_post,
    }


def extract_locking_metric(solver: MHDSolver, t_skip: float = 0.05) -> float:
    """Extract the locking metric from a completed solver run.

    Returns the late-time mean Gap/Bar momentum ratio.
    Locking occurs when this drops below a threshold (e.g. 0.80).
    For non-striped cases (no bar/gap masks), returns 1.0 (unlocked).
    """
    t = np.array(solver.diag_times)
    mask = t > t_skip
    if not np.any(mask):
        return 1.0

    gap = np.array(solver.diag_jet_momentum_flux_gap)[mask]
    bar = np.array(solver.diag_jet_momentum_flux_bar)[mask]

    # Use the last third of the time window for steady-state estimate
    n = len(gap)
    tail = max(n // 3, 1)
    gap_late = gap[-tail:]
    bar_late = bar[-tail:]

    mean_gap = float(np.mean(gap_late))
    mean_bar = float(np.mean(bar_late))

    if mean_bar < 1e-14:
        return 1.0

    return mean_gap / mean_bar


def run_single_sieve(base_params: dict, B0: float, quiet: bool = True) -> MHDSolver:
    """Run a single striped-field simulation and return the solver."""
    params = {**base_params, 'B_transverse': B0,
              'B_field_type': 'striped', 'B_modulation_mode': 4}
    # Suppress snapshot saving for sweep runs to save memory
    params['snapshot_times'] = [0.0, params['t_end']]
    params['diag_interval'] = 40  # coarser diagnostics for speed

    cfg = Config(**params)
    s = MHDSolver(cfg)
    s.initialize()
    s.run()
    return s


def find_B_critical(base_params: dict,
                    B_low: float, B_high: float,
                    locking_threshold: float = 0.80,
                    tol: float = 0.05,
                    max_iter: int = 6) -> Tuple[float, List[dict]]:
    """Bisection search for B_critical where gap/bar ratio crosses threshold.

    The gap/bar ratio starts > 1 for weak fields (nozzle effect) and drops
    below the threshold when the field is strong enough to lock the interface.

    Returns (B_critical, history) where history is a list of dicts recording
    each bisection step.
    """
    history = []

    print(f"\n    Bisection search: B in [{B_low:.2f}, {B_high:.2f}], "
          f"threshold={locking_threshold:.2f}")
    sys.stdout.flush()

    for iteration in range(max_iter):
        B_mid = 0.5 * (B_low + B_high)

        t0 = time.time()
        solver = run_single_sieve(base_params, B_mid)
        elapsed = time.time() - t0

        ratio = extract_locking_metric(solver, t_skip=0.05)

        step_info = {
            'iteration': iteration,
            'B0': B_mid,
            'gap_bar_ratio': ratio,
            'elapsed_s': elapsed,
        }
        history.append(step_info)

        locked = ratio < locking_threshold
        status = "LOCKED" if locked else "UNLOCKED"
        print(f"      iter {iteration}: B0={B_mid:.3f}  gap/bar={ratio:.3f}  "
              f"[{status}]  ({elapsed:.0f}s)")
        sys.stdout.flush()

        if locked:
            B_high = B_mid
        else:
            B_low = B_mid

        if (B_high - B_low) < tol:
            print(f"      Converged: B_crit = {0.5*(B_low+B_high):.3f} "
                  f"(bracket width {B_high-B_low:.3f})")
            break

        # Free GPU memory between runs
        del solver
        cp.get_default_memory_pool().free_all_blocks()

    B_crit = 0.5 * (B_low + B_high)
    return B_crit, history


# ============================================================
# PARAMETRIC SWEEP DEFINITIONS
# ============================================================

def sweep_base_params(mach: float, density_ratio: float,
                      nx: int = 200, ny: int = 100) -> dict:
    """Build base parameter dict for sweep runs at reduced resolution."""
    return dict(
        nx=nx, ny=ny,
        x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, mach=mach,
        interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, density_ratio=density_ratio,
        interface_width=2.0,
        powell_source=True, use_char_bc=True,
        bc_x_type="characteristic", bc_y_type="periodic",
        enable_smoothing=False,
    )


def run_mach_sweep(locking_threshold: float = 0.80) -> List[dict]:
    """Sweep 1: Vary Mach number to probe Delta_U dependence."""
    print("\n" + "=" * 72)
    print("  SWEEP 1: MACH NUMBER  (probing Delta_U dependence)")
    print("  Fixed: density_ratio=3.0, perturbation_mode=4, a0=0.15")
    print("=" * 72)
    sys.stdout.flush()

    mach_values = [5.0, 10.0, 15.0]
    density_ratio = 3.0
    gamma = 5.0 / 3.0
    Ly = 2.0; a0 = 0.15; pert_mode = 4

    results = []
    for M in mach_values:
        print(f"\n  --- Mach = {M:.1f} ---")

        V_hydro, theory = compute_theoretical_V_hydro(
            gamma, M, 1.0, density_ratio, a0, pert_mode, Ly)

        # Estimate B range: V_A = B/sqrt(rho), so B ~ V_hydro * sqrt(rho_avg)
        B_estimate = V_hydro * np.sqrt(theory['rho_avg'])
        B_low = max(0.2, B_estimate * 0.3)
        B_high = B_estimate * 3.0

        print(f"    V_hydro = {V_hydro:.3f}")
        print(f"    sqrt(rho_avg) = {np.sqrt(theory['rho_avg']):.3f}")
        print(f"    B_estimate ~ {B_estimate:.3f}")

        base = sweep_base_params(M, density_ratio)
        B_crit, history = find_B_critical(
            base, B_low, B_high,
            locking_threshold=locking_threshold)

        results.append({
            'sweep': 'mach',
            'mach': M,
            'density_ratio': density_ratio,
            'V_hydro': V_hydro,
            'rho_avg': theory['rho_avg'],
            'sqrt_rho_avg': np.sqrt(theory['rho_avg']),
            'B_critical': B_crit,
            'B_crit_over_sqrt_rho': B_crit / np.sqrt(theory['rho_avg']),
            'k_int': theory['k_int'],
            'delta_U': theory['delta_U'],
            'A_post': theory['A_post'],
            'a0_post': theory['a0_post'],
            'history': history,
        })

        cp.get_default_memory_pool().free_all_blocks()

    return results


def run_density_sweep(locking_threshold: float = 0.80) -> List[dict]:
    """Sweep 2: Vary density ratio to probe sqrt(rho_avg) dependence."""
    print("\n" + "=" * 72)
    print("  SWEEP 2: DENSITY RATIO  (probing sqrt(rho_avg) dependence)")
    print("  Fixed: Mach=10, perturbation_mode=4, a0=0.15")
    print("=" * 72)
    sys.stdout.flush()

    density_ratios = [1.5, 3.0, 5.0]
    M = 10.0
    gamma = 5.0 / 3.0
    Ly = 2.0; a0 = 0.15; pert_mode = 4

    results = []
    for eta in density_ratios:
        print(f"\n  --- density_ratio = {eta:.1f} ---")

        V_hydro, theory = compute_theoretical_V_hydro(
            gamma, M, 1.0, eta, a0, pert_mode, Ly)

        B_estimate = V_hydro * np.sqrt(theory['rho_avg'])
        B_low = max(0.2, B_estimate * 0.3)
        B_high = B_estimate * 3.0

        print(f"    V_hydro = {V_hydro:.3f}")
        print(f"    sqrt(rho_avg) = {np.sqrt(theory['rho_avg']):.3f}")
        print(f"    B_estimate ~ {B_estimate:.3f}")

        base = sweep_base_params(M, eta)
        B_crit, history = find_B_critical(
            base, B_low, B_high,
            locking_threshold=locking_threshold)

        results.append({
            'sweep': 'density',
            'mach': M,
            'density_ratio': eta,
            'V_hydro': V_hydro,
            'rho_avg': theory['rho_avg'],
            'sqrt_rho_avg': np.sqrt(theory['rho_avg']),
            'B_critical': B_crit,
            'B_crit_over_sqrt_rho': B_crit / np.sqrt(theory['rho_avg']),
            'k_int': theory['k_int'],
            'delta_U': theory['delta_U'],
            'A_post': theory['A_post'],
            'a0_post': theory['a0_post'],
            'history': history,
        })

        cp.get_default_memory_pool().free_all_blocks()

    return results


# ============================================================
# DATA OUTPUT & SCALING LAW PLOTS
# ============================================================

def write_csv(all_results: List[dict], filename: str = "scaling_law_data.csv"):
    """Write sweep results to CSV."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fieldnames = [
        'sweep', 'mach', 'density_ratio',
        'V_hydro', 'rho_avg', 'sqrt_rho_avg',
        'B_critical', 'B_crit_over_sqrt_rho',
        'k_int', 'delta_U', 'A_post', 'a0_post',
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in fieldnames}
            # Round floats for readability
            for k in row:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
    print(f"  CSV written: {filepath}")
    return filepath


def plot_scaling_law(all_results: List[dict]):
    """The money plot: B_crit vs V_hydro proving the linear relationship.

    Theory predicts: B_crit = Psi(k_mod/k_int) * sqrt(rho_avg) * V_hydro
    So: B_crit / sqrt(rho_avg) vs V_hydro should be a straight line
    through the origin, with slope = Psi.
    """
    mach_results = [r for r in all_results if r['sweep'] == 'mach']
    density_results = [r for r in all_results if r['sweep'] == 'density']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- Panel 1: B_crit / sqrt(rho_avg) vs V_hydro (the scaling law) ----
    ax = axes[0]

    all_V = []
    all_B_norm = []

    if mach_results:
        V = np.array([r['V_hydro'] for r in mach_results])
        B_norm = np.array([r['B_crit_over_sqrt_rho'] for r in mach_results])
        ax.plot(V, B_norm, 'rs-', ms=10, lw=2, label='Mach sweep', zorder=5)
        all_V.extend(V.tolist())
        all_B_norm.extend(B_norm.tolist())

    if density_results:
        V = np.array([r['V_hydro'] for r in density_results])
        B_norm = np.array([r['B_crit_over_sqrt_rho'] for r in density_results])
        ax.plot(V, B_norm, 'b^-', ms=10, lw=2, label='Density sweep', zorder=5)
        all_V.extend(V.tolist())
        all_B_norm.extend(B_norm.tolist())

    # Linear fit through origin: B_norm = Psi * V_hydro
    if len(all_V) >= 2:
        all_V = np.array(all_V)
        all_B_norm = np.array(all_B_norm)
        # Least squares fit through origin: Psi = sum(V*B) / sum(V^2)
        Psi = float(np.sum(all_V * all_B_norm) / np.sum(all_V**2))
        V_fit = np.linspace(0, np.max(all_V) * 1.2, 100)
        ax.plot(V_fit, Psi * V_fit, 'k--', lw=1.5, alpha=0.7,
                label=f'Linear fit: $\\Psi = {Psi:.2f}$')
        ax.fill_between(V_fit, Psi * V_fit * 0.8, Psi * V_fit * 1.2,
                         alpha=0.1, color='gray')

        # R-squared
        predicted = Psi * all_V
        ss_res = np.sum((all_B_norm - predicted) ** 2)
        ss_tot = np.sum((all_B_norm - np.mean(all_B_norm)) ** 2)
        R2 = 1 - ss_res / max(ss_tot, 1e-14)
        ax.text(0.05, 0.92, f'$R^2 = {R2:.4f}$', transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        Psi = None

    ax.set_xlabel(r'$V_{hydro} = k_{int} \cdot \Delta U \cdot A^+ \cdot a_0^+$',
                  fontsize=13)
    ax.set_ylabel(r'$B_{crit} / \sqrt{\rho_{avg}}$', fontsize=13)
    ax.set_title('THE SCALING LAW\n'
                 r'$B_{crit} = \Psi \cdot \sqrt{\rho_{avg}} \cdot V_{hydro}$',
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ---- Panel 2: B_crit vs Mach number ----
    ax = axes[1]
    if mach_results:
        machs = [r['mach'] for r in mach_results]
        B_crits = [r['B_critical'] for r in mach_results]
        V_hydros = [r['V_hydro'] for r in mach_results]
        sqrt_rhos = [r['sqrt_rho_avg'] for r in mach_results]

        ax.plot(machs, B_crits, 'rs-', ms=10, lw=2, label=r'$B_{crit}$ (measured)')

        if Psi is not None:
            B_pred = [Psi * v * sr for v, sr in zip(V_hydros, sqrt_rhos)]
            ax.plot(machs, B_pred, 'k^--', ms=8, lw=1.5, label=f'Theory ($\\Psi={Psi:.2f}$)')

    ax.set_xlabel('Mach number $M$', fontsize=13)
    ax.set_ylabel(r'$B_{critical}$', fontsize=13)
    ax.set_title(r'$B_{crit}$ vs Mach ($\propto \Delta U$)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Panel 3: B_crit vs density ratio ----
    ax = axes[2]
    if density_results:
        etas = [r['density_ratio'] for r in density_results]
        B_crits = [r['B_critical'] for r in density_results]
        V_hydros = [r['V_hydro'] for r in density_results]
        sqrt_rhos = [r['sqrt_rho_avg'] for r in density_results]

        ax.plot(etas, B_crits, 'b^-', ms=10, lw=2, label=r'$B_{crit}$ (measured)')

        if Psi is not None:
            B_pred = [Psi * v * sr for v, sr in zip(V_hydros, sqrt_rhos)]
            ax.plot(etas, B_pred, 'k^--', ms=8, lw=1.5, label=f'Theory ($\\Psi={Psi:.2f}$)')

    ax.set_xlabel(r'Density ratio $\rho_H / \rho_L$', fontsize=13)
    ax.set_ylabel(r'$B_{critical}$', fontsize=13)
    ax.set_title(r'$B_{crit}$ vs Density Ratio ($\propto \sqrt{\rho}, A^+$)',
                 fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle('Magnetic Sieve Scaling Law Verification',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'scaling_law_proof.png')
    print("  Scaling law plot saved.")

    return Psi


def plot_bisection_convergence(all_results: List[dict]):
    """Plot the bisection convergence history for all sweep points."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sweep_name, title in [
        (axes[0], 'mach', 'Mach Sweep Bisection'),
        (axes[1], 'density', 'Density Sweep Bisection'),
    ]:
        subset = [r for r in all_results if r['sweep'] == sweep_name]
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        for i, r in enumerate(subset):
            if sweep_name == 'mach':
                lbl = f"M={r['mach']:.0f}"
            else:
                lbl = f"eta={r['density_ratio']:.1f}"
            iters = [h['iteration'] for h in r['history']]
            ratios = [h['gap_bar_ratio'] for h in r['history']]
            ax.plot(iters, ratios, 'o-', color=colors[i % 5], lw=2, ms=8,
                    label=f"{lbl}: B_c={r['B_critical']:.2f}")
        ax.axhline(0.80, color='red', ls='--', lw=1.5, alpha=0.7,
                   label='Locking threshold')
        ax.set_xlabel('Bisection iteration'); ax.set_ylabel('Gap/Bar momentum ratio')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'bisection_convergence.png')


def print_scaling_equation(Psi: Optional[float], all_results: List[dict]):
    """Print the final derived scaling law equation."""
    print("\n" + "=" * 72)
    print("  DERIVED MAGNETIC SIEVE EQUATION")
    print("=" * 72)

    if Psi is None:
        print("  ERROR: Insufficient data to derive Psi.")
        return

    print(f"\n  B_critical = Psi * sqrt(rho_avg) * (k_int * Delta_U * A+ * a0+)")
    print(f"\n  GEOMETRIC FACTOR:  Psi = {Psi:.4f}")
    print(f"  (for k_mod/k_int = 1, perturbation_mode = B_modulation_mode = 4)")

    print(f"\n  --- Verification against each data point ---")
    print(f"  {'Sweep':<10s}  {'Param':>8s}  {'B_crit':>8s}  {'B_pred':>8s}  {'Error':>8s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for r in all_results:
        B_pred = Psi * r['sqrt_rho_avg'] * r['V_hydro']
        err = abs(r['B_critical'] - B_pred) / max(r['B_critical'], 1e-14) * 100
        param = f"M={r['mach']:.0f}" if r['sweep'] == 'mach' else f"eta={r['density_ratio']:.1f}"
        print(f"  {r['sweep']:<10s}  {param:>8s}  {r['B_critical']:8.3f}  {B_pred:8.3f}  {err:7.1f}%")


# ============================================================
# SHOWCASE RUNS (full resolution, 4 key cases)
# ============================================================

def run_showcase():
    """Run the 4 key showcase cases at full resolution with all plots."""
    print("\n" + "=" * 72)
    print("  SHOWCASE RUNS (400x200, full visualization)")
    print("=" * 72)

    base = dict(
        nx=400, ny=200,
        x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, mach=10.0,
        interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, density_ratio=3.0,
        interface_width=2.0,
        powell_source=True, use_char_bc=True,
        bc_x_type="characteristic", bc_y_type="periodic",
        enable_smoothing=False,
        diag_interval=20,
        snapshot_times=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    )

    cases = {
        "Hydro (B=0)": dict(B_transverse=0.0, B_field_type="uniform"),
        "Uniform (By=1.0)": dict(B_transverse=1.0, B_field_type="uniform"),
        "Striped (B0=1.5)": dict(B_transverse=1.5, B_field_type="striped", B_modulation_mode=4),
        "Striped (B0=2.0)": dict(B_transverse=2.0, B_field_type="striped", B_modulation_mode=4),
    }

    solvers: Dict[str, MHDSolver] = {}
    for label, case_params in cases.items():
        print(f"\n{'='*64}\n  CASE: {label}\n{'='*64}")
        sys.stdout.flush()
        params = {**base, **case_params}
        cfg = Config(**params)
        s = MHDSolver(cfg)
        s.initialize()
        s.run()
        solvers[label] = s

    post = PostProcessor(solvers)
    post.plot_all()
    return solvers


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  MAGNETIC SIEVE: Scaling Law Discovery Campaign")
    print("  B_crit = Psi * sqrt(rho_avg) * V_hydro")
    print("  HLLD | MUSCL | SSP-RK3 | GLM+Powell | CuPy/CUDA")
    print("=" * 72)
    sys.stdout.flush()

    total_t0 = time.time()

    # ========================================
    # Phase 1: Verification (fast, ~2 min)
    # ========================================
    print("\n" + "=" * 64)
    print("  PHASE 1: VERIFICATION SUITE")
    print("=" * 64)

    bw_passed = brio_wu_test(nx=800, t_end=0.1, plot=True)
    wave_passed = linear_wave_convergence_test(plot=True)
    contact_passed = contact_discontinuity_test(plot=True)
    all_passed = bw_passed and wave_passed and contact_passed
    print(f"\n  Verification: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # ========================================
    # Phase 2: Parametric Sweeps (the science)
    # ========================================
    print("\n" + "=" * 64)
    print("  PHASE 2: PARAMETRIC SWEEPS FOR SCALING LAW")
    print("  Resolution: 200x100 (fast sweep mode)")
    print("=" * 64)

    sweep_t0 = time.time()

    mach_results = run_mach_sweep(locking_threshold=0.80)
    density_results = run_density_sweep(locking_threshold=0.80)

    all_sweep_results = mach_results + density_results

    sweep_elapsed = time.time() - sweep_t0
    print(f"\n  Sweep phase completed in {sweep_elapsed:.0f}s "
          f"({sweep_elapsed/60:.1f} min)")

    # ========================================
    # Phase 3: Data Output & Scaling Law
    # ========================================
    print("\n" + "=" * 64)
    print("  PHASE 3: SCALING LAW ANALYSIS")
    print("=" * 64)

    csv_path = write_csv(all_sweep_results)
    Psi = plot_scaling_law(all_sweep_results)
    plot_bisection_convergence(all_sweep_results)
    print_scaling_equation(Psi, all_sweep_results)

    # ========================================
    # Phase 4: Showcase Runs (full resolution)
    # ========================================
    print("\n" + "=" * 64)
    print("  PHASE 4: SHOWCASE RUNS (full resolution)")
    print("=" * 64)

    showcase_solvers = run_showcase()

    # ========================================
    # Final Summary
    # ========================================
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 72)
    print("  CAMPAIGN COMPLETE")
    print("=" * 72)
    print(f"  Verification: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"  Sweep points: {len(all_sweep_results)} "
          f"({len(mach_results)} Mach + {len(density_results)} density)")
    if Psi is not None:
        print(f"  Geometric factor Psi = {Psi:.4f}")
        print(f"  Scaling law: B_crit = {Psi:.4f} * sqrt(rho_avg) * V_hydro")
    print(f"  Total runs: {len(all_sweep_results) * 5 + 4 + 7} approx "
          f"(sweeps + showcase + verification)")
    print(f"  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  CSV data: {csv_path}")
    print("=" * 72)
    sys.stdout.flush()
