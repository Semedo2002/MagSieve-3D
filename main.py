# main.py - entry point: verification tests, single cases, and the (legacy) campaign.

import time
import os
import sys
import csv
import io
import contextlib
import numpy as np
from typing import Dict, List, Tuple

from sieve.config import Config, NVAR, iRHO, iVX, iVY, iBX, iBY, iBZ, iPR, iCLR
from sieve.physics import xp, cp, cons_to_prim, prim_to_cons, mhd_rankine_hugoniot
from sieve.solver import MHDSolver, PostProcessor, save_figure, to_numpy, OUTPUT_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Verification Tests (unchanged from working code)
# ============================================================
def brio_wu_test(nx=800, t_end=0.1, plot=True):
    print("\n--- Brio-Wu Shock Tube ---"); sys.stdout.flush()
    cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.05,
        t_end=t_end, cfl=0.30, gamma=2.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=10000, snapshot_times=[t_end],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic")
    solver = MHDSolver(cfg); ng = solver.ng
    W = xp.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x; left = x < 0.5
    W[iRHO] = xp.where(left[:, None], 1.0, 0.125)
    W[iPR] = xp.where(left[:, None], 1.0, 0.1)
    W[iBX] = 0.75
    W[iBY] = xp.where(left[:, None], 1.0, -1.0)
    W[iCLR] = xp.where(left[:, None], 1.0, 0.0)
    solver.U = prim_to_cons(W, cfg.gamma); solver.t = 0.0; solver.step = 0
    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        solver.dt = min(solver.compute_dt(W_c), cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3(); solver.t += solver.dt; solver.step += 1
    W_f = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = to_numpy(W_f[iRHO, ng:-ng, ng]); By_1d = to_numpy(W_f[iBY, ng:-ng, ng])
    vx_1d = to_numpy(W_f[iVX, ng:-ng, ng])
    passed = (0.1 < np.min(rho_1d) < 0.2) and (0.9 < np.max(rho_1d) < 1.05) and bool(np.any(By_1d[:-1]*By_1d[1:]<0))
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if plot:
        p_1d = to_numpy(W_f[iPR, ng:-ng, ng]); x_1d = to_numpy(x[ng:-ng])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, data, yl, tl, c in [(axes[0,0],rho_1d,r'$\rho$','Density','b'),
            (axes[0,1],p_1d,'$p$','Pressure','r'), (axes[1,0],vx_1d,'$v_x$','Velocity','g'),
            (axes[1,1],By_1d,'$B_y$','Transverse B','m')]:
            ax.plot(x_1d, data, f'{c}-', lw=1); ax.set_ylabel(yl); ax.set_title(tl); ax.grid(True, alpha=0.3)
        fig.suptitle(f'Brio-Wu (GPU), t={t_end}, nx={nx}', fontweight='bold')
        plt.tight_layout(); save_figure(fig, 'brio_wu_test.png')
    return passed

def alfven_convergence_test(plot=True):
    print("\n--- Alfven Wave Convergence ---"); sys.stdout.flush()
    resolutions = [32, 64, 128, 256]; errors = []
    rho0=1.0; p0=0.1; Bx0=1.0; amp=1e-6; gamma=5.0/3.0; Lx=1.0
    vA = Bx0/np.sqrt(rho0); period = Lx/vA
    for nx in resolutions:
        cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=Lx, y_min=0.0, y_max=0.05,
            t_end=period, cfl=0.25, gamma=gamma, mach=1.0, B_transverse=0.0,
            interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
            diag_interval=100000, snapshot_times=[period],
            powell_source=False, bc_x_type="periodic", bc_y_type="periodic")
        s = MHDSolver(cfg); ng = s.ng; x = s.x; kx = 2.0*np.pi/Lx
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO]=rho0; W[iPR]=p0; W[iBX]=Bx0
        W[iBY]=amp*xp.sin(kx*x)[:,None]*xp.ones(s.ny_tot)[None,:]
        W[iVY]=-amp*xp.sin(kx*x)[:,None]*xp.ones(s.ny_tot)[None,:]/np.sqrt(rho0)
        W_init = W.copy()
        s.U = prim_to_cons(W, cfg.gamma); s.t=0.0; s.step=0
        while s.t < cfg.t_end and s.step < 100000:
            W_c = cons_to_prim(s.U, cfg.gamma)
            s.dt = min(s.compute_dt(W_c), cfg.t_end - s.t)
            if s.dt <= 1e-16: break
            s.step_ssprk3(); s.t += s.dt; s.step += 1
        W_f = cons_to_prim(s.U, cfg.gamma)
        errors.append(float(np.mean(np.abs(to_numpy(W_f[iBY,ng:-ng,ng])-to_numpy(W_init[iBY,ng:-ng,ng])))))
    orders = [np.log(errors[i-1]/errors[i])/np.log(resolutions[i]/resolutions[i-1]) for i in range(1,len(errors)) if errors[i]>0 and errors[i-1]>0]
    mean_order = np.mean(orders) if orders else 0
    passed = mean_order >= 1.5
    print(f"  Mean order: {mean_order:.2f} ({'PASS' if passed else 'FAIL'})")
    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(resolutions, errors, 'bo-', lw=2, ms=8, label=f'Measured ({mean_order:.2f})')
        ref = np.array([resolutions[0], resolutions[-1]], dtype=float)
        ax.loglog(ref, errors[0]*resolutions[0]**2/ref**2, 'k--', lw=1, alpha=0.5, label='2nd order')
        ax.set_xlabel('$N_x$'); ax.set_ylabel('$L_1$ error'); ax.set_title('Alfven Wave Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout(); save_figure(fig, 'convergence_alfven.png')
    return passed


def _advance(s, t_end, max_steps=100000):
    """Advance a solver whose state was set by hand (no diagnostics)."""
    while s.t < t_end and s.step < max_steps:
        W_c = cons_to_prim(s.U, s.cfg.gamma)
        s.dt = min(s.compute_dt(W_c), t_end - s.t)
        if s.dt <= 1e-16: break
        s.step_ssprk3(); s.t += s.dt; s.step += 1


def boundary_reflection_test(nx=200, eps=1e-4, tol=1e-2):
    """Send small right-going fast and Alfven pulses through the x_max outflow
    boundary and measure the fraction of the pulse left in the domain."""
    print("\n--- Outflow Boundary Reflection ---"); sys.stdout.flush()
    rho0, p0, B0, gamma, sigma, x0 = 1.0, 1.0, 1.0, 5.0/3.0, 0.05, 0.5
    results = {}
    for wave in ("fast", "alfven"):
        cfg = Config(nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.02,
            cfl=0.3, gamma=gamma, mach=1.0, powell_source=False,
            bc_x_type="inflow_outflow", bc_y_type="periodic")
        s = MHDSolver(cfg); x = s.x[:, None] * xp.ones(s.ny_tot)[None, :]
        f = eps * xp.exp(-0.5 * ((x - x0) / sigma)**2)
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
        W[iRHO] = rho0; W[iPR] = p0
        if wave == "fast":    # B perpendicular to x: right-going fast eigenvector
            cs2 = gamma * p0 / rho0; c = np.sqrt(cs2 + B0**2 / rho0)
            W[iBY] = B0 * (1 + f); W[iRHO] = rho0 * (1 + f)
            W[iVX] = c * f; W[iPR] = p0 + rho0 * cs2 * f
            probe = lambda Wf: xp.abs(Wf[iRHO] - rho0) / rho0
        else:                 # B along x: right-going Alfven eigenvector
            c = B0 / np.sqrt(rho0)
            W[iBX] = B0; W[iBY] = B0 * f; W[iVY] = -B0 * f / np.sqrt(rho0)
            probe = lambda Wf: xp.abs(Wf[iBY]) / B0
        s.U = prim_to_cons(W, gamma); s.t = 0.0; s.step = 0
        background = W.copy(); background[iVX] = 0; background[iVY] = 0
        background[iRHO] = rho0; background[iPR] = p0
        background[iBY] = B0 if wave == "fast" else 0.0
        s._inflow_U = prim_to_cons(background, gamma)[:, :s.ng, :].copy()
        _advance(s, (cfg.x_max - x0 + 8 * sigma) / c)
        ng = s.ng
        left = float(xp.max(probe(cons_to_prim(s.U, gamma))[ng:-ng, ng:-ng])) / eps
        results[wave] = left
        print(f"  {wave:>6s} pulse: residual amplitude after exit = {left:.2e} of incident")
    passed = all(v < tol for v in results.values())
    print(f"  Result: {'PASS' if passed else 'FAIL'} (tolerance {tol:.0e})")
    return passed


def conservation_test(nx=100, ny=50, t_end=0.05, tol=1e-10, div_tol=0.1):
    """Short sieve runs: energy must balance boundary fluxes to round-off.
    Bz stripes have no in-plane field and must keep div B = 0 exactly; in-plane
    fields are cleaned by GLM and must stay below div_tol in h max|div B|/max|B|
    (legacy By(y) stripes start at h*k_mod ~ 0.4 and fail this)."""
    print("\n--- Conservation & Solenoidality ---"); sys.stdout.flush()
    base = dict(nx=nx, ny=ny, t_end=t_end, mach=10.0, localized_rh=True,
                bc_x_type="inflow_outflow", diag_interval=10**9, snapshot_times=[])
    cases = {
        "uniform By=1 (Powell+GLM)": dict(B_transverse=1.0, B_field_type="uniform"),
        "striped Bz, eta=1e-2": dict(B_transverse=2.0, B_field_type="striped",
                                      B_stripe_component="z", eta_resistivity=1e-2),
        "striped Bx (tension)": dict(B_transverse=2.0, B_field_type="striped",
                                     B_stripe_component="x"),
    }
    passed = True
    for label, c in cases.items():
        s = MHDSolver(Config(**base, **c))
        with contextlib.redirect_stdout(io.StringIO()):
            s.initialize(); s.run()
        b = s.energy_budget()
        err = abs(b["conservation_error_rel"]); div_rel = s.diag_divB_rel_max[-1]
        exact = c.get("B_stripe_component") == "z"
        ok = err < tol and div_rel < (1e-12 if exact else div_tol)
        passed &= ok
        print(f"  {label:<26s} E change {100*b['raw_change_rel']:+7.1f}% (inflow)  "
              f"conservation err {err:.1e}  h*max|divB|/max|B| {div_rel:.1e}  [{'PASS' if ok else 'FAIL'}]")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def equilibrium_test(B0=5.0, nx=200, ny=100, t_end=0.05, tol=1e-2):
    """The gas ahead of the shock must stay at rest. The balance p + Bx^2/2 = P_total
    holds exactly in the initial data but only to O(h^2) at MUSCL-reconstructed
    faces, so the residual flow is truncation error: measured 1.6e-2, 4.0e-3,
    1.2e-3 c_s at nx = 100, 200, 400 (order ~2), against 0.30 c_s with no
    pressure balance. A well-balanced scheme would be needed for round-off."""
    print("\n--- Upstream Pressure Equilibrium ---"); sys.stdout.flush()
    passed = True
    for closure in ("uniform_density", "uniform_temperature"):
        cfg = Config(nx=nx, ny=ny, t_end=t_end, mach=10.0, B_transverse=B0,
                     B_field_type="striped", B_stripe_component="x",
                     equilibrium_closure=closure, bc_x_type="inflow_outflow",
                     diag_interval=10**9, snapshot_times=[])
        s = MHDSolver(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            s.initialize(); s.run()
        W = cons_to_prim(s.U, cfg.gamma); ng = s.ng
        far = s.x[ng:-ng] > 3.5   # shock is at x ~ 1.9 at t_end
        v = xp.sqrt(W[iVX, ng:-ng, ng:-ng]**2 + W[iVY, ng:-ng, ng:-ng]**2)[far]
        v_rel = float(xp.max(v)) / np.sqrt(cfg.gamma * cfg.p_bar)
        ok = v_rel < tol
        passed &= ok
        eq = s._equilibrium
        print(f"  {closure:<20s} B0={B0:g}: P_total={eq['P_total']:.4g}, gap Mach={eq['M_gap']:.3g}, "
              f"max|v|/c_s upstream = {v_rel:.1e}  [{'PASS' if ok else 'FAIL'}]")
    try:
        Config(nx=10, eta_resistivty=1e-2)
        print("  misspelled config key accepted  [FAIL]"); passed = False
    except ValueError as e:
        print(f"  misspelled config key rejected: {e}  [PASS]")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def z_invariance_test(tol=1e-8):
    """3D runs with a z-invariant initial state must reproduce the 2D solver (ideal and resistive)."""
    print("\n--- 3D: z-invariance against the 2D solver ---"); sys.stdout.flush()
    return all(_z_invariance_case(eta, tol) for eta in (0.0, 0.02))


def _z_invariance_case(eta, tol):
    from sieve.solver3d import MHDSolver3D
    nx, ny = 128, 32
    dx, dy = 6.0 / nx, 2.0 / ny
    common = dict(nx=nx, ny=ny, x_max=6.0, t_end=0.1, mach=10.0, B_transverse=3.0, B_field_type="striped",
                  B_gap_fraction=0.9, perturbation_amp=0.02, bc_x_type="inflow_outflow",
                  eta_resistivity=eta, diag_interval=10**9, snapshot_times=[])
    s2 = MHDSolver(Config(**common))
    with contextlib.redirect_stdout(io.StringIO()):
        s2.initialize(); s2.run()
    s3 = MHDSolver3D(Config(**common, nz=4, z_max=4 * max(dx, dy)))
    s3.initialize(); s3.run(verbose=False)
    ng = s2.ng
    r2 = cons_to_prim(s2.U, 5.0 / 3.0)[iRHO, ng:-ng, ng:-ng]
    r3 = cons_to_prim(s3.U, 5.0 / 3.0)[iRHO, ng:-ng, ng:-ng, ng:-ng]
    diff = float(xp.max(xp.abs(r3 - r2[:, :, None]) / r2[:, :, None]))
    zvar = float(xp.max(xp.abs(r3 - r3[:, :, :1])))
    err = s3.energy_error()
    ok = diff < tol and zvar == 0.0 and err < 1e-10 and s2.step == s3.step
    print(f"  eta={eta:g}: steps 2D/3D {s2.step}/{s3.step}, max rel diff in rho {diff:.1e}, z-variation {zvar:.1e}, "
          f"energy error {err:.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def alfven_z_test():
    """Alfven wave travelling along z: tests the z-face fluxes, expects ~2nd order."""
    print("\n--- 3D: Alfven wave along z ---"); sys.stdout.flush()
    from sieve.solver3d import MHDSolver3D
    errors, res = [], [16, 32, 64]
    for nz in res:
        cfg = Config(nx=4, ny=4, nz=nz, x_max=4.0 / nz, y_max=4.0 / nz, z_max=1.0, t_end=1.0, cfl=0.25,
                     mach=1.0, powell_source=False, bc_x_type="periodic", diag_interval=10**9)
        s = MHDSolver3D(cfg)
        z = s.z[None, None, :] * xp.ones((s.nx_tot, s.ny_tot, 1))
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
        amp, k = 1e-6, 2 * np.pi
        W[iRHO] = 1.0; W[iPR] = 0.1; W[iBZ] = 1.0
        W[iBX] = amp * xp.sin(k * z); W[iVX] = -amp * xp.sin(k * z)
        W0 = W.copy()
        s.U = prim_to_cons(W, cfg.gamma)
        s._inflow_U = s.U[:, :s.ng].copy()
        s.run(verbose=False)
        ng = s.ng
        Wf = cons_to_prim(s.U, cfg.gamma)
        errors.append(float(xp.mean(xp.abs(Wf[iBX, ng:-ng, ng:-ng, ng:-ng] - W0[iBX, ng:-ng, ng:-ng, ng:-ng]))))
    orders = [np.log2(errors[i - 1] / errors[i]) for i in range(1, len(errors))]
    ok = np.mean(orders) >= 1.5
    print(f"  L1 errors {', '.join(f'{e:.2e}' for e in errors)}; orders {', '.join(f'{o:.2f}' for o in orders)}"
          f"  [{'PASS' if ok else 'FAIL'}]")
    return ok


def resistive_z_test(eta=0.01, tol=0.01):
    """Weak B_x = b sin(k z) in a static plasma decays as exp(-eta k^2 t); tests the z-face
    resistive fluxes and that Ohmic heating conserves total energy."""
    print("\n--- 3D: resistive diffusion along z ---"); sys.stdout.flush()
    from sieve.solver3d import MHDSolver3D
    nz, b, k, t_end = 64, 1e-3, 2 * np.pi, 1.0
    cfg = Config(nx=4, ny=4, nz=nz, x_max=4.0 / nz, y_max=4.0 / nz, z_max=1.0, t_end=t_end, cfl=0.3,
                 mach=1.0, powell_source=False, bc_x_type="periodic", eta_resistivity=eta, diag_interval=10**9)
    s = MHDSolver3D(cfg)
    z = s.z[None, None, :] * xp.ones((s.nx_tot, s.ny_tot, 1))
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
    W[iRHO] = 1.0; W[iPR] = 1.0; W[iBX] = b * xp.sin(k * z)
    s.U = prim_to_cons(W, cfg.gamma)
    s._inflow_U = s.U[:, :s.ng].copy()
    s.run(verbose=False)
    ng = s.ng
    Bx = cons_to_prim(s.U, cfg.gamma)[iBX, ng, ng, ng:-ng]
    zc = s.z[ng:-ng]
    amp = float(2 * xp.mean(Bx * xp.sin(k * zc)))
    expected = b * np.exp(-eta * k**2 * t_end)
    rel = abs(amp - expected) / expected
    err = s.energy_error()
    ok = rel < tol and err < 1e-12
    print(f"  amplitude {amp:.4e} vs exp(-eta k^2 t) {expected:.4e} (rel. error {rel:.1e}); "
          f"energy error {err:.1e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def galilean_test(U=6.49):
    """Interface frame: the interface must stay put, energy must close, and the interface growth must match
    the lab frame (within 10%). Mixing-layer enstrophy is reported, not asserted: the scheme is not exactly
    Galilean invariant, and advecting the interface across the grid in the lab frame adds numerical
    diffusion (measured 15-36% at h = 0.042-0.083)."""
    print("\n--- 3D: Galilean frame ---"); sys.stdout.flush()
    from sieve.solver3d import MHDSolver3D
    h = 10.0 / 120
    base = dict(nx=120, ny=24, nz=4, x_max=10.0, z_max=4 * h, t_end=0.4, mach=10.0, interface_x=4.0,
                B_field_type="striped", B_transverse=6.0, eta_resistivity=0.02, B_modulation_mode=1,
                bc_x_type="inflow_outflow", diag_interval=5)
    res = {}
    for name, fv in (("lab", 0.0), ("co-moving", U)):
        s = MHDSolver3D(Config(**base, frame_velocity=fv))
        s.initialize(); res[name] = s.run(verbose=False); res[name]["err"] = s.energy_error()
    lab, cm = res["lab"], res["co-moving"]
    d_rough = abs(cm["rough"][-1] - lab["rough"][-1]) / lab["rough"][-1]
    d_emix = abs(cm["emix"][-1] - lab["emix"][-1]) / lab["emix"][-1]
    drift = cm["x_int"][-1] - cm["x_int"][len(cm["t"]) // 4]
    err = max(lab["err"], cm["err"])
    ok = d_rough < 0.10 and abs(drift) < 0.1 and err < 1e-10
    print(f"  roughness lab/co-moving {lab['rough'][-1]:.4f}/{cm['rough'][-1]:.4f} ({100*d_rough:.1f}%); "
          f"interface drift {drift:+.3f}; energy error {err:.1e}; enstrophy differs by {100*d_emix:.0f}% "
          f"(numerical diffusion, reported only)  [{'PASS' if ok else 'FAIL'}]")
    return ok


VERIFICATION_TESTS = {
    "brio_wu": brio_wu_test,
    "alfven": alfven_convergence_test,
    "reflection": boundary_reflection_test,
    "conservation": conservation_test,
    "equilibrium": equilibrium_test,
    "z_invariance": z_invariance_test,
    "alfven_z": alfven_z_test,
    "resistive_z": resistive_z_test,
    "galilean": galilean_test,
}


# ============================================================
# Theoretical V_hydro (from Richtmyer linear theory)
# ============================================================
def compute_V_hydro(gamma, mach, rho1, density_ratio, a0, pert_mode, Ly):
    M2 = mach**2
    r = min((gamma+1)*M2/((gamma-1)*M2+2), (gamma+1)/(gamma-1)-0.01)
    cs1 = np.sqrt(gamma/rho1)
    vs = mach*cs1
    delta_U = vs*(1.0-1.0/r)
    A_post = ((density_ratio*r-r)/((density_ratio*r+r)))
    a0_post = a0/r
    k_int = 2.0*np.pi*pert_mode/Ly
    V_hydro = k_int*delta_U*A_post*a0_post
    rho_avg = 0.5*(rho1*r + rho1*density_ratio*r)
    return V_hydro, {'k_int': k_int, 'delta_U': delta_U, 'A_post': A_post,
                      'a0_post': a0_post, 'r': r, 'rho_avg': rho_avg, 'vs': vs}


# ============================================================
# Enstrophy-based locking metric
# ============================================================
def extract_enstrophy_locking(solver, hydro_enstrophy, t_skip=0.05):
    """Returns True if late-time enstrophy is suppressed by > (1-fraction) of hydro."""
    t = np.array(solver.diag_times); mask = t > t_skip
    if not np.any(mask) or hydro_enstrophy < 1e-10:
        return False, 1.0
    en = np.array(solver.diag_enstrophy_local)[mask]
    tail = max(len(en)//3, 1)
    mean_en = float(np.mean(en[-tail:]))
    ratio = mean_en / hydro_enstrophy
    locked = ratio < solver.cfg.enstrophy_locking_fraction
    return locked, ratio

def extract_momentum_locking(solver, t_skip=0.05):
    """Original gap/bar momentum ratio criterion."""
    t = np.array(solver.diag_times); mask = t > t_skip
    if not np.any(mask): return 1.0
    gap = np.array(solver.diag_jet_momentum_flux_gap)[mask]
    bar = np.array(solver.diag_jet_momentum_flux_bar)[mask]
    tail = max(len(gap)//3, 1)
    mg = float(np.mean(gap[-tail:])); mb = float(np.mean(bar[-tail:]))
    return mg/max(mb, 1e-14)


# ============================================================
# Bisection search for B_crit
# ============================================================
def find_B_critical(base_params, B_low, B_high, hydro_enstrophy=None,
                     threshold_momentum=0.80, tol=0.05, max_iter=6):
    history = []
    for it in range(max_iter):
        B_mid = 0.5*(B_low + B_high)
        params = {**base_params, 'B_transverse': B_mid,
                  'B_field_type': 'striped', 'B_modulation_mode': base_params.get('B_modulation_mode', 4),
                  'snapshot_times': [0.0, base_params.get('t_end', 0.25)], 'diag_interval': 40}
        cfg = Config(**params)
        s = MHDSolver(cfg); s.initialize(); s.run()
        mom_ratio = extract_momentum_locking(s)
        # Also check enstrophy criterion if hydro baseline available
        if hydro_enstrophy is not None:
            locked_en, en_ratio = extract_enstrophy_locking(s, hydro_enstrophy)
        else:
            locked_en, en_ratio = False, 1.0
        history.append({'B0': B_mid, 'mom_ratio': mom_ratio, 'en_ratio': en_ratio})
        locked = mom_ratio < threshold_momentum
        print(f"      iter {it}: B0={B_mid:.3f}  gap/bar={mom_ratio:.3f}  "
              f"en_frac={en_ratio:.3f}  [{'LOCKED' if locked else 'UNLOCKED'}]")
        if locked: B_high = B_mid
        else: B_low = B_mid
        if (B_high - B_low) < tol: break
        del s; cp.get_default_memory_pool().free_all_blocks()
    return 0.5*(B_low+B_high), history


# ============================================================
# MAIN
# ============================================================
def run_campaign():
    """Full research campaign (verification, showcase, resistivity, Psi map, GCI, sweeps)."""
    print("\n" + "="*72)
    print("  MAGNETIC SIEVE - campaign")
    print("  Localized R-H | Resistive MHD | Psi Mapping | GCI | Enstrophy Locking")
    print("="*72); sys.stdout.flush()
    total_t0 = time.time()

    # ========== Phase 1: Verification ==========
    print("\n" + "="*64 + "\n  PHASE 1: VERIFICATION\n" + "="*64)
    bw = brio_wu_test(); af = alfven_convergence_test()
    boundary_reflection_test(); conservation_test()
    print(f"  Overall: {'ALL PASSED' if (bw and af) else 'SOME FAILED'}")

    # ========== Phase 2: Showcase runs with localized R-H ==========
    print("\n" + "="*64 + "\n  PHASE 2: SHOWCASE (localized R-H, 400x200)\n" + "="*64)

    base = dict(nx=400, ny=200, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, mach=10.0, interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, density_ratio=3.0, interface_width=2.0,
        powell_source=True, use_char_bc=True, bc_x_type="characteristic",
        bc_y_type="periodic", enable_smoothing=False, diag_interval=20,
        localized_rh=True)

    cases = {
        "Hydro (B=0)": dict(B_transverse=0.0, B_field_type="uniform", localized_rh=False),
        "Uniform (By=1.0)": dict(B_transverse=1.0, B_field_type="uniform"),
        "Striped (B0=1.5)": dict(B_transverse=1.5, B_field_type="striped", B_modulation_mode=4),
        "Striped (B0=2.0)": dict(B_transverse=2.0, B_field_type="striped", B_modulation_mode=4),
    }

    solvers: Dict[str, MHDSolver] = {}
    for label, cp_dict in cases.items():
        print(f"\n{'='*64}\n  CASE: {label}\n{'='*64}")
        params = {**base, **cp_dict}
        cfg = Config(**params)
        s = MHDSolver(cfg); s.initialize(); s.run()
        solvers[label] = s

    post = PostProcessor(solvers)
    post.plot_all()

    # Get hydro enstrophy baseline for enstrophy-based locking
    hydro_key = list(solvers.keys())[0]
    hydro_solver = solvers[hydro_key]
    t_arr = np.array(hydro_solver.diag_times)
    mask = t_arr > 0.05
    hydro_enstrophy = float(np.mean(np.array(hydro_solver.diag_enstrophy_local)[mask])) if np.any(mask) else 1.0

    # ========== Enstrophy-based locking report ==========
    print("\n  --- Enstrophy-Based Locking ---")
    print(f"  Hydro baseline enstrophy: {hydro_enstrophy:.1f}")
    for label, s in solvers.items():
        locked, ratio = extract_enstrophy_locking(s, hydro_enstrophy)
        status = "LOCKED" if locked else "unlocked"
        print(f"  {label:<25s}: en/en_hydro = {ratio:.3f}  [{status}]")

    # ========== Phase 3: Resistivity comparison ==========
    print("\n" + "="*64 + "\n  PHASE 3: RESISTIVE MHD COMPARISON\n" + "="*64)

    eta_values = [0.0, 1e-3, 1e-2]
    resist_solvers = {}
    for eta in eta_values:
        label = f"Striped B0=2.0, eta={eta:.0e}" if eta > 0 else "Striped B0=2.0, ideal"
        print(f"\n  --- {label} ---")
        params = {**base, 'nx': 200, 'ny': 100,
                  'B_transverse': 2.0, 'B_field_type': 'striped', 'B_modulation_mode': 4,
                  'eta_resistivity': eta, 'localized_rh': True,
                  'snapshot_times': [0.0, 0.10, 0.25], 'diag_interval': 40}
        cfg = Config(**params)
        s = MHDSolver(cfg); s.initialize(); s.run()
        resist_solvers[label] = s

    # Plot resistivity comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['C0', 'C1', 'C2']
    for i, (label, s) in enumerate(resist_solvers.items()):
        t = np.array(s.diag_times)
        axes[0].plot(t, s.diag_enstrophy_local, color=colors[i], lw=2, label=label)
        axes[1].plot(t, s.diag_jet_max_vx, color=colors[i], lw=2, label=label)
        axes[2].plot(t, s.diag_jet_collimation, color=colors[i], lw=2, label=label)
    for ax, yl, tl in [(axes[0], r'$\langle\rho\omega^2\rangle$', 'Enstrophy'),
                        (axes[1], '$v_x^{max}$', 'Peak Jet Velocity'),
                        (axes[2], 'Collimation', 'Gap Momentum Fraction')]:
        ax.set_xlabel('Time $t$'); ax.set_ylabel(yl); ax.set_title(tl, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Resistive MHD: Does Reconnection Break the Sieve?', fontsize=14, fontweight='bold')
    plt.tight_layout(); save_figure(fig, 'resistivity_comparison.png')

    # ========== Phase 4: Psi(k_mod/k_int) functional mapping ==========
    print("\n" + "="*64 + "\n  PHASE 4: Psi MAPPING (k_mod/k_int sweep)\n" + "="*64)

    k_ratios = [0.5, 1.0, 2.0, 4.0]
    pert_mode = 4; Ly = 2.0
    gamma = 5.0/3.0; M = 10.0; eta_r = 3.0
    V_hydro, theory = compute_V_hydro(gamma, M, 1.0, eta_r, 0.15, pert_mode, Ly)
    sqrt_rho = np.sqrt(theory['rho_avg'])

    psi_results = []
    for kr in k_ratios:
        mod_mode = int(round(pert_mode * kr))
        if mod_mode < 1: mod_mode = 1
        actual_kr = mod_mode / pert_mode
        print(f"\n  --- k_mod/k_int = {actual_kr:.2f} (mod_mode={mod_mode}) ---")
        sweep_base = dict(nx=200, ny=100, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
            t_end=0.25, cfl=0.30, mach=M, interface_x=1.5, perturbation_amp=0.15,
            perturbation_mode=pert_mode, density_ratio=eta_r, interface_width=2.0,
            powell_source=True, use_char_bc=True, bc_x_type="characteristic",
            bc_y_type="periodic", B_modulation_mode=mod_mode, localized_rh=True)

        B_est = V_hydro * sqrt_rho * 3.0  # rough estimate using Psi~3
        B_crit, hist = find_B_critical(sweep_base, max(0.5, B_est*0.3), B_est*2.0,
                                        hydro_enstrophy=hydro_enstrophy)
        Psi_measured = B_crit / (sqrt_rho * V_hydro) if V_hydro > 1e-10 else 0
        psi_results.append({'k_ratio': actual_kr, 'mod_mode': mod_mode,
                            'B_crit': B_crit, 'Psi': Psi_measured})
        print(f"    B_crit = {B_crit:.3f}, Psi = {Psi_measured:.3f}")
        cp.get_default_memory_pool().free_all_blocks()

    # Plot Psi mapping
    fig, ax = plt.subplots(figsize=(8, 5))
    krs = [r['k_ratio'] for r in psi_results]
    psis = [r['Psi'] for r in psi_results]
    ax.plot(krs, psis, 'ro-', ms=10, lw=2, label='Measured $\\Psi$')
    ax.axhline(psis[krs.index(1.0)] if 1.0 in krs else np.mean(psis),
               color='gray', ls='--', lw=1, alpha=0.5, label='$k_{mod}/k_{int}=1$ value')
    ax.set_xlabel('$k_{mod}/k_{int}$', fontsize=13); ax.set_ylabel('$\\Psi$', fontsize=13)
    ax.set_title('Geometric Nozzle Factor: The Universal Function', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    save_figure(fig, 'psi_mapping.png')

    # ========== Phase 5: Grid Convergence Index (Richardson) ==========
    print("\n" + "="*64 + "\n  PHASE 5: GRID CONVERGENCE INDEX\n" + "="*64)

    gci_resolutions = [(100, 50), (200, 100), (400, 200)]
    gci_B_crits = []

    for (nx, ny) in gci_resolutions:
        print(f"\n  --- Resolution {nx}x{ny} ---")
        gci_base = dict(nx=nx, ny=ny, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
            t_end=0.25, cfl=0.30, mach=10.0, interface_x=1.5, perturbation_amp=0.15,
            perturbation_mode=4, density_ratio=3.0, interface_width=2.0,
            powell_source=True, use_char_bc=True, bc_x_type="characteristic",
            bc_y_type="periodic", B_modulation_mode=4, localized_rh=True)
        B_crit, _ = find_B_critical(gci_base, 5.0, 40.0,
                                     hydro_enstrophy=hydro_enstrophy, max_iter=5)
        gci_B_crits.append(B_crit)
        print(f"    B_crit({nx}x{ny}) = {B_crit:.3f}")
        cp.get_default_memory_pool().free_all_blocks()

    # Richardson extrapolation
    if len(gci_B_crits) == 3:
        h = [6.0/nx for nx, _ in gci_resolutions]  # grid spacings
        f1, f2, f3 = gci_B_crits  # coarse, medium, fine
        r21 = h[0]/h[1]; r32 = h[1]/h[2]
        # Observed order
        if abs(f3-f2) > 1e-10 and abs(f2-f1) > 1e-10:
            p_obs = np.log(abs((f1-f2)/(f2-f3))) / np.log(r21)
            f_exact = f3 + (f3-f2)/(r32**p_obs - 1)
            gci_fine = 1.25 * abs(f3-f2)/(r32**p_obs - 1) / abs(f3) * 100
            print(f"\n  Richardson Extrapolation:")
            print(f"    B_crit(h->0) = {f_exact:.3f}")
            print(f"    Observed order p = {p_obs:.2f}")
            print(f"    GCI_fine = {gci_fine:.2f}%")
        else:
            print("  Grid solutions too close for Richardson extrapolation")
            f_exact = f3; p_obs = 2.0; gci_fine = 0.0

    # Plot GCI
    fig, ax = plt.subplots(figsize=(7, 5))
    h_arr = np.array([6.0/nx for nx, _ in gci_resolutions])
    ax.plot(h_arr, gci_B_crits, 'bs-', ms=10, lw=2, label='$B_{crit}(h)$')
    if len(gci_B_crits) == 3:
        ax.axhline(f_exact, color='red', ls='--', lw=1.5, label=f'$B_{{crit}}(h\\to 0) = {f_exact:.2f}$')
    ax.set_xlabel('Grid spacing $h = L_x/N_x$', fontsize=12)
    ax.set_ylabel('$B_{crit}$', fontsize=12)
    ax.set_title('Grid Convergence Index (Richardson Extrapolation)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3); ax.invert_xaxis()
    plt.tight_layout(); save_figure(fig, 'gci_convergence.png')

    # ========== Phase 6: Mach + Density sweeps (scaling law proof) ==========
    print("\n" + "="*64 + "\n  PHASE 6: SCALING LAW SWEEPS\n" + "="*64)

    sweep_base_common = dict(nx=200, ny=100, x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, interface_width=2.0,
        powell_source=True, use_char_bc=True, bc_x_type="characteristic",
        bc_y_type="periodic", B_modulation_mode=4, localized_rh=True)

    all_sweep = []
    # Mach sweep
    for M_val in [5.0, 10.0, 15.0]:
        V, th = compute_V_hydro(5.0/3.0, M_val, 1.0, 3.0, 0.15, 4, 2.0)
        B_est = V*np.sqrt(th['rho_avg'])*3.0
        sb = {**sweep_base_common, 'mach': M_val, 'density_ratio': 3.0}
        Bc, _ = find_B_critical(sb, max(0.5, B_est*0.3), B_est*2.0, hydro_enstrophy=hydro_enstrophy)
        all_sweep.append({'sweep':'mach', 'param': M_val, 'V_hydro': V,
                          'sqrt_rho': np.sqrt(th['rho_avg']), 'B_crit': Bc})
        cp.get_default_memory_pool().free_all_blocks()

    # Density sweep
    for eta_val in [1.5, 3.0, 5.0]:
        V, th = compute_V_hydro(5.0/3.0, 10.0, 1.0, eta_val, 0.15, 4, 2.0)
        B_est = V*np.sqrt(th['rho_avg'])*3.0
        sb = {**sweep_base_common, 'mach': 10.0, 'density_ratio': eta_val}
        Bc, _ = find_B_critical(sb, max(0.5, B_est*0.3), B_est*2.0, hydro_enstrophy=hydro_enstrophy)
        all_sweep.append({'sweep':'density', 'param': eta_val, 'V_hydro': V,
                          'sqrt_rho': np.sqrt(th['rho_avg']), 'B_crit': Bc})
        cp.get_default_memory_pool().free_all_blocks()

    # Scaling law plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    mach_r = [r for r in all_sweep if r['sweep']=='mach']
    dens_r = [r for r in all_sweep if r['sweep']=='density']
    all_V = np.array([r['V_hydro'] for r in all_sweep])
    all_Bn = np.array([r['B_crit']/r['sqrt_rho'] for r in all_sweep])
    Psi_fit = float(np.sum(all_V*all_Bn)/np.sum(all_V**2)) if len(all_V)>1 else 0

    ax = axes[0]
    if mach_r:
        ax.plot([r['V_hydro'] for r in mach_r], [r['B_crit']/r['sqrt_rho'] for r in mach_r],
                'rs-', ms=10, lw=2, label='Mach sweep')
    if dens_r:
        ax.plot([r['V_hydro'] for r in dens_r], [r['B_crit']/r['sqrt_rho'] for r in dens_r],
                'b^-', ms=10, lw=2, label='Density sweep')
    Vf = np.linspace(0, max(all_V)*1.2, 100)
    ax.plot(Vf, Psi_fit*Vf, 'k--', lw=1.5, label=f'$\\Psi = {Psi_fit:.2f}$')
    pred = Psi_fit*all_V; ss_res = np.sum((all_Bn-pred)**2)
    ss_tot = np.sum((all_Bn-np.mean(all_Bn))**2)
    R2 = 1-ss_res/max(ss_tot,1e-14)
    ax.text(0.05, 0.92, f'$R^2 = {R2:.4f}$', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$V_{hydro}$'); ax.set_ylabel(r'$B_{crit}/\sqrt{\bar\rho}$')
    ax.set_title('THE SCALING LAW', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    ax = axes[1]
    if mach_r:
        ax.plot([r['param'] for r in mach_r], [r['B_crit'] for r in mach_r], 'rs-', ms=10, lw=2, label='Measured')
        ax.plot([r['param'] for r in mach_r], [Psi_fit*r['V_hydro']*r['sqrt_rho'] for r in mach_r],
                'k^--', ms=8, lw=1.5, label=f'Theory ($\\Psi={Psi_fit:.2f}$)')
    ax.set_xlabel('Mach $M$'); ax.set_ylabel('$B_{crit}$')
    ax.set_title('$B_{crit}$ vs Mach', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if dens_r:
        ax.plot([r['param'] for r in dens_r], [r['B_crit'] for r in dens_r], 'b^-', ms=10, lw=2, label='Measured')
        ax.plot([r['param'] for r in dens_r], [Psi_fit*r['V_hydro']*r['sqrt_rho'] for r in dens_r],
                'k^--', ms=8, lw=1.5, label=f'Theory ($\\Psi={Psi_fit:.2f}$)')
    ax.set_xlabel(r'$\rho_H/\rho_L$'); ax.set_ylabel('$B_{crit}$')
    ax.set_title('$B_{crit}$ vs Density Ratio', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Magnetic Sieve Scaling Law (Localized R-H)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); save_figure(fig, 'scaling_law_proof.png')

    # ========== Final Summary ==========
    total_elapsed = time.time() - total_t0
    print("\n" + "="*72)
    print("  CAMPAIGN COMPLETE")
    print("="*72)
    print(f"  Verification: {'ALL PASSED' if (bw and af) else 'SOME FAILED'}")
    print(f"  Psi(k_mod/k_int=1) = {Psi_fit:.4f}, R^2 = {R2:.4f}")
    print(f"  Psi mapping: {len(psi_results)} k-ratio points")
    for r in psi_results:
        print(f"    k_mod/k_int = {r['k_ratio']:.2f} -> Psi = {r['Psi']:.3f}")
    print(f"  GCI: B_crit(h->0) = {f_exact:.3f}, GCI_fine = {gci_fine:.2f}%")
    print(f"  Resistivity: {len(resist_solvers)} eta values tested")
    print(f"  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("="*72); sys.stdout.flush()


# ============================================================
# Single showcase case
# ============================================================
SHOWCASE_CASES = {
    "hydro": dict(B_transverse=0.0, B_field_type="uniform", localized_rh=False),
    "uniform": dict(B_transverse=1.0, B_field_type="uniform"),
    "striped": dict(B_transverse=2.0, B_field_type="striped", B_modulation_mode=4),
}


def run_case(name, nx=400, ny=100, t_end=0.8, B0=None, stripe="x", eta=0.0, gap_fraction=0.0,
             a0=0.02, n_frames=61, compare_hydro=False, plot=True):
    """One showcase run on a 12 x 2 domain (long enough to keep the interface in view
    to t ~ 1). Snapshots are evenly spaced so the GIFs play smoothly for ~12 s."""
    params = dict(nx=nx, ny=ny, x_min=0.0, x_max=12.0, y_min=0.0, y_max=2.0,
        t_end=t_end, cfl=0.30, mach=10.0, interface_x=1.5, perturbation_amp=a0,
        perturbation_mode=4, density_ratio=3.0, interface_width=2.0,
        powell_source=True, bc_x_type="inflow_outflow", bc_y_type="periodic",
        diag_interval=20, localized_rh=True, B_stripe_component=stripe,
        eta_resistivity=eta, snapshot_times=list(np.linspace(0.0, t_end, n_frames)))
    runs = {}
    if compare_hydro and name != "hydro":
        runs["hydro"] = {**params, **SHOWCASE_CASES["hydro"]}
    case = {**params, **SHOWCASE_CASES[name]}
    if B0 is not None:
        case["B_transverse"] = B0
    if case.get("B_field_type") == "striped":
        case["B_gap_fraction"] = gap_fraction
    runs[name if B0 is None else f"{name} B0={B0:g}"] = case
    solvers = {}
    for label, p in runs.items():
        s = MHDSolver(Config(**p)); s.initialize(); s.run()
        solvers[label] = s
    if plot:
        PostProcessor(solvers).plot_all()
    return solvers


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Magnetic Sieve: GPU HLLD-MHD solver for the magnetised Richtmyer-Meshkov problem.",
        epilog="examples:\n"
               "  python main.py --test brio_wu          # ~10 s verification\n"
               "  python main.py --test all              # full verification suite\n"
               "  python main.py --case striped --nx 200 # one sieve run + plots\n"
               "  python main.py --campaign              # full scaling-law campaign (hours)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", choices=[*VERIFICATION_TESTS, "all"],
                      help="run a verification test (exit code 1 on failure)")
    mode.add_argument("--case", choices=list(SHOWCASE_CASES), help="run a single showcase case")
    mode.add_argument("--campaign", action="store_true", help="run the full research campaign")
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=None, help="default: nx/4")
    parser.add_argument("--t-end", type=float, default=0.8)
    parser.add_argument("--B0", type=float, default=None, help="override field strength")
    parser.add_argument("--stripe", choices=["x", "z", "y"], default="x",
                        help="striped component: x (tension, default), z (pressure bars), y (legacy, div B != 0)")
    parser.add_argument("--eta", type=float, default=0.0, help="resistivity")
    parser.add_argument("--gap-fraction", type=float, default=0.0,
                        help="field left in the stripe gaps as a fraction of B0 (x stripes)")
    parser.add_argument("--a0", type=float, default=0.02, help="initial interface amplitude")
    parser.add_argument("--frames", type=int, default=61, help="snapshots / GIF frames")
    parser.add_argument("--compare-hydro", action="store_true",
                        help="also run the hydro case and plot side by side")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    if args.test:
        names = list(VERIFICATION_TESTS) if args.test == "all" else [args.test]
        results = {}
        for n in names:
            fn = VERIFICATION_TESTS[n]
            results[n] = fn(plot=not args.no_plots) if n in ("brio_wu", "alfven") else fn()
        print("\n" + "\n".join(f"  {n:<14s} {'PASS' if ok else 'FAIL'}" for n, ok in results.items()))
        return 0 if all(results.values()) else 1
    if args.case:
        run_case(args.case, nx=args.nx, ny=args.ny or args.nx // 4, t_end=args.t_end,
                 B0=args.B0, stripe=args.stripe, eta=args.eta, gap_fraction=args.gap_fraction,
                 a0=args.a0, n_frames=args.frames, compare_hydro=args.compare_hydro,
                 plot=not args.no_plots)
        return 0
    if args.campaign:
        run_campaign()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
