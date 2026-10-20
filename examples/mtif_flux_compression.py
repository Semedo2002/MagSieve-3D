"""MTIF benchmark: a dense liner implodes onto a magnetized core and compresses its axial field.

2D cross-section of a cylinder: core (rho = 1, axial seed field B_z = B0) inside a dense liner
(rho = 10 or 20) moving inward at Mach M relative to the core sound speed. B_z is out of plane, so
div B = 0 holds exactly; flux surfaces are drawn as circles enclosing fixed fractions of the flux.

    python examples/mtif_flux_compression.py                 # 512 x 512, run + plots
    python examples/mtif_flux_compression.py --quick --check # 128 x 128 smoke test
    python examples/mtif_flux_compression.py --plot-only     # redo plots from saved data

Needs the MagSieve-3D solver: pip install git+https://github.com/Semedo2002/MagSieve-3D
"""
import os
import sys
import csv
import json
import time
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a repo checkout without installing
with contextlib.redirect_stdout(io.StringIO()):
    from sieve.config import Config, NVAR, EN, iRHO, iVX, iVY, iBZ, iPR, iCLR
    from sieve.physics import xp, prim_to_cons, cons_to_prim
    from sieve.solver import MHDSolver, to_numpy

GAMMA = 5.0 / 3.0


def setup(a):
    """Build the solver and the initial state."""
    L = 1.0
    cfg = Config(nx=a.n, ny=a.n, x_min=-L, x_max=L, y_min=-L, y_max=L, t_end=a.t_end, cfl=0.3,
                 gamma=GAMMA, B_transverse=a.B0, B_field_type="uniform", powell_source=True,
                 bc_x_type="extrapolation", bc_y_type="outflow", diag_interval=10**9, snapshot_times=[])
    s = MHDSolver(cfg)
    X, Y = s.X, s.Y
    r = xp.sqrt(X**2 + Y**2) + 1e-12
    th = xp.arctan2(Y, X)
    w = 1.5 * cfg.dx                                            # interface half-width
    R0 = a.R0 * (1 + a.amp * xp.cos(a.mode * th))               # perturbed inner liner surface
    R1 = a.R1 * (1 + a.amp * xp.cos(a.mode * th))
    step = lambda R: 0.5 * (1 + xp.tanh((r - R) / w))
    core = 1 - step(R0)
    liner = step(R0) * (1 - step(R1))
    amb = step(R1)

    cs = np.sqrt(GAMMA * a.p_core / a.rho_core)
    V = a.mach * cs                                             # liner speed
    Bz = a.B0 * core                                            # seed field in the core only
    p_tot = a.p_core + 0.5 * a.B0**2                            # total pressure balance at t = 0
    W = xp.zeros((NVAR, s.nx_tot, s.ny_tot))
    W[iRHO] = a.rho_core * core + a.rho_liner * liner + a.rho_core * amb
    W[iPR] = p_tot - 0.5 * Bz**2
    W[iVX] = -V * liner * X / r
    W[iVY] = -V * liner * Y / r
    W[iBZ] = Bz
    W[iCLR] = 1 - core                                          # 0 in the core, 1 outside
    s.U = prim_to_cons(W, GAMMA)
    s.t, s.step = 0.0, 0
    return s, V


def diag(s, a, E0=None):
    """One row of diagnostics."""
    ng = s.ng
    U = s.U[:, ng:-ng, ng:-ng]
    W = cons_to_prim(s.U, GAMMA)[:, ng:-ng, ng:-ng]
    dA = s.cfg.dx * s.cfg.dy
    rho, p, Bz, C = W[iRHO], W[iPR], W[iBZ], xp.clip(W[iCLR], 0, 1)
    v2 = W[iVX]**2 + W[iVY]**2
    B2 = W[4]**2 + W[5]**2 + Bz**2
    core_area = float(xp.sum(1 - C)) * dA
    R = np.sqrt(core_area / np.pi)                              # equivalent core radius
    flux = float(xp.sum(Bz)) * dA                               # total axial flux (conserved)
    core_flux = float(xp.sum(Bz * (1 - C))) * dA
    divB = xp.gradient(W[4], s.cfg.dx, axis=0) + xp.gradient(W[5], s.cfg.dy, axis=1)
    p_raw = (GAMMA - 1) * (U[EN] - 0.5 * (U[1]**2 + U[2]**2 + U[3]**2) / U[0] - 0.5 * (U[4]**2 + U[5]**2 + U[6]**2))
    E = float(xp.sum(U[EN])) * dA
    row = {
        "t": s.t, "step": s.step,
        "R": R, "B_max": float(xp.max(xp.abs(Bz))), "B_core_mean": core_flux / max(core_area, 1e-30),
        "flux": flux,
        "E_kin": float(xp.sum(0.5 * rho * v2)) * dA, "E_mag": float(xp.sum(0.5 * B2)) * dA,
        "E_th": float(xp.sum(p)) * dA / (GAMMA - 1), "E_tot": E,
        "divB_L2": float(xp.sqrt(xp.mean(divB**2))), "divB_max": float(xp.max(xp.abs(divB))),
        "p_min": float(xp.min(p_raw)), "rho_min": float(xp.min(U[0])),
    }
    E0 = E if E0 is None else E0
    row["energy_error"] = (E - E0 - s._cumulative_boundary_energy - s._cumulative_powell_energy) / E0
    return row, E0


def frame(s):
    ng = s.ng
    W = cons_to_prim(s.U, GAMMA)[:, ng:-ng, ng:-ng]
    return to_numpy(W[iRHO]).astype(np.float32), to_numpy(W[iBZ]).astype(np.float32)


def run(a, out):
    os.makedirs(out, exist_ok=True)
    s, V = setup(a)
    print(f"liner speed {V:.3f} (Mach {a.mach:g}), rho_liner {a.rho_liner:g}, grid {a.n}x{a.n}", flush=True)
    rows, E0 = [], None
    row, E0 = diag(s, a, E0)
    rows.append(row)
    times = list(np.linspace(0, a.t_end, a.frames))
    rho_f, bz_f, t_f = [], [], []
    t0 = time.time()
    while s.t < a.t_end:
        if times and s.t >= times[0] - 1e-12:
            times.pop(0)
            r_, b_ = frame(s); rho_f.append(r_); bz_f.append(b_); t_f.append(s.t)
        s.dt = min(s.compute_dt(cons_to_prim(s.U, GAMMA)), a.t_end - s.t)
        s.step_ssprk3()
        s.t += s.dt
        s.step += 1
        if s.step % a.diag_every == 0:
            row, _ = diag(s, a, E0)
            rows.append(row)
            if row["p_min"] <= 0 or row["rho_min"] <= 0 or not np.isfinite(row["E_tot"]):
                raise RuntimeError(f"non-physical state at t={s.t:.4f}: p_min={row['p_min']:.3g}")
    row, _ = diag(s, a, E0)
    rows.append(row)
    r_, b_ = frame(s); rho_f.append(r_); bz_f.append(b_); t_f.append(s.t)

    # save
    with open(os.path.join(out, "diagnostics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    x = to_numpy(s.x[s.ng:-s.ng])
    np.savez_compressed(os.path.join(out, "frames.npz"), x=x, t=np.array(t_f), rho=np.array(rho_f), Bz=np.array(bz_f))
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    i_turn = int(np.argmin(d["R"]))
    summary = {
        "params": vars(a), "liner_speed": V,
        "steps": s.step, "wall_s": time.time() - t0,
        "R_min": float(d["R"][i_turn]), "t_turnaround": float(d["t"][i_turn]),
        "convergence_ratio": float(a.R0 / d["R"][i_turn]),
        "B_max_over_B0": float(d["B_max"].max() / a.B0),
        "B_core_over_B0_at_turnaround": float(d["B_core_mean"][i_turn] / a.B0),
        "ideal_(R0/R)^2_at_turnaround": float((d["R"][0] / d["R"][i_turn])**2),
        "flux_drift": float(abs(d["flux"][-1] / d["flux"][0] - 1)),
        "energy_error_max": float(np.max(np.abs(d["energy_error"]))),
        "p_min": float(d["p_min"].min()), "rho_min": float(d["rho_min"].min()),
        "divB_max": float(d["divB_max"].max()),
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done: {s.step} steps, {summary['wall_s']:.0f}s | convergence ratio {summary['convergence_ratio']:.2f}, "
          f"B_max/B0 {summary['B_max_over_B0']:.1f}, turnaround t={summary['t_turnaround']:.4f} | "
          f"energy error {summary['energy_error_max']:.1e}, flux drift {summary['flux_drift']:.1e}", flush=True)
    return summary


# ---- plots

def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with open(os.path.join(out, "diagnostics.csv")) as fh:
        rows = list(csv.DictReader(fh))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    summ = json.load(open(os.path.join(out, "summary.json")))
    B0 = summ["params"]["B0"]
    R0 = d["R"][0]

    # curves
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].semilogy(d["t"], d["B_max"] / B0, color="#c0392b", lw=1.8, label=r"$B_{max}/B_0$")
    ax[0].semilogy(d["t"], d["B_core_mean"] / B0, color="#1f5f99", lw=1.8, label=r"mean core $B/B_0$")
    ax[0].semilogy(d["t"], (R0 / d["R"])**2, color="#888888", ls="--", lw=1.2, label=r"ideal $(R_0/R)^2$")
    ax[0].set_title("(a) flux amplification", loc="left"); ax[0].legend(frameon=False, fontsize=8.5)
    ax[1].plot(d["t"], d["R"] / R0, color="#222222", lw=1.8)
    ax[1].axvline(summ["t_turnaround"], color="#888888", ls=":", lw=1)
    ax[1].set_title(r"(b) core radius $R(t)/R_0$", loc="left")
    ax[1].text(summ["t_turnaround"], d["R"].min() / R0, f"  turnaround\n  CR = {summ['convergence_ratio']:.1f}",
               fontsize=8.5, va="bottom")
    Ek0 = d["E_kin"][0]
    for k, c, lab in (("E_kin", "#222222", "kinetic"), ("E_mag", "#c0392b", "magnetic"), ("E_th", "#1f5f99", "thermal")):
        ax[2].plot(d["t"], (d[k] - d[k][0]) / Ek0, color=c, lw=1.6, label=lab)
    ax[2].set_title(r"(c) energy change / $E_{kin}(0)$", loc="left"); ax[2].legend(frameon=False, fontsize=8.5)
    for x in ax:
        x.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "flux_compression_curves.png"), dpi=220)
    plt.close(fig)

    # gif: log density | |B| with flux surfaces
    fr = np.load(os.path.join(out, "frames.npz"))
    x, ts, rho, Bz = fr["x"], fr["t"], fr["rho"], fr["Bz"]
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    order = np.argsort(r.ravel())
    dA = (x[1] - x[0])**2
    flux0 = Bz[0].sum() * dA
    rl = (np.log10(rho.min()), np.log10(rho.max()))
    bl = (0, np.abs(Bz).max())
    imgs = []
    for k in range(len(ts)):
        fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.4))
        im0 = ax[0].pcolormesh(x, x, np.log10(rho[k]).T, cmap="inferno", vmin=rl[0], vmax=rl[1], shading="auto", rasterized=True)
        im1 = ax[1].pcolormesh(x, x, np.abs(Bz[k]).T, cmap="magma", vmin=bl[0], vmax=bl[1], shading="auto", rasterized=True)
        # flux surfaces: radii enclosing 10..90% of the seed flux
        cum = np.cumsum(Bz[k].ravel()[order]) * dA / flux0
        rad = r.ravel()[order]
        for f in np.arange(0.1, 1.0, 0.1):
            j = np.searchsorted(cum, f)
            if j < rad.size:
                ax[1].add_patch(plt.Circle((0, 0), rad[j], fill=False, color="white", lw=0.6, alpha=0.8))
        for a_, ttl in ((ax[0], r"$\log_{10}\rho$"), (ax[1], r"$|B_z|$, flux surfaces (10% steps)")):
            a_.set_aspect("equal"); a_.set_xlim(-0.8, 0.8); a_.set_ylim(-0.8, 0.8)
            a_.set_title(f"{ttl},  $t = {ts[k]:.3f}$", loc="left", fontsize=10)
        fig.colorbar(im0, ax=ax[0], shrink=0.85); fig.colorbar(im1, ax=ax[1], shrink=0.85)
        fig.tight_layout()
        fig.canvas.draw()
        imgs.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB"))
        plt.close(fig)
    imgs[0].save(os.path.join(out, "mtif_implosion.gif"), save_all=True, append_images=imgs[1:], duration=150, loop=0)
    imgs[-1].save(os.path.join(out, "mtif_final_frame.png"))
    print(f"plots in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=512, help="cells per side")
    ap.add_argument("--rho-liner", dest="rho_liner", type=float, default=10.0)
    ap.add_argument("--mach", type=float, default=4.0)
    ap.add_argument("--B0", type=float, default=1.0)
    ap.add_argument("--t-end", dest="t_end", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--mode", type=int, default=6, help="azimuthal mode of the liner perturbation")
    ap.add_argument("--amp", type=float, default=0.005, help="relative perturbation amplitude")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mtif_flux_compression"))
    ap.add_argument("--quick", action="store_true", help="128 x 128, fewer frames")
    ap.add_argument("--check", action="store_true", help="fail unless the run passes the smoke checks")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    a.rho_core, a.p_core, a.R0, a.R1, a.diag_every = 1.0, 1.0, 0.5, 0.7, 5
    if a.quick:
        a.n, a.frames = 128, 15
        if a.out == ap.get_default("out"):
            a.out += "_quick"
    if not a.plot_only:
        summ = run(a, a.out)
        if a.check:
            ok = (summ["p_min"] > 0 and summ["rho_min"] > 0 and summ["energy_error_max"] < 1e-10
                  and summ["flux_drift"] < 1e-10 and summ["B_max_over_B0"] > 2)
            print("smoke check:", "PASS" if ok else "FAIL")
            if not ok:
                sys.exit(1)
    if not a.no_plot:
        plot(a.out)


if __name__ == "__main__":
    main()
