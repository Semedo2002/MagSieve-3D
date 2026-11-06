"""Inverse diagnostics on a twin experiment (aegis/ml/pinn_diagnostics.py): reconstruct what no probe measured.

A straight r-z liner implosion is run with the solver. Only three outside signals are kept, the virtual probes of
aegis/validation.py with 1% noise: a B-dot coil outside the liner, VISAR on the inner surface and a framing camera. A PINN
(data + MHD residuals + known pre-shot state) reconstructs the interior. Its core field and core temperature are compared
with the solver's truth, and with the closed-form estimate from the same radius trace (flux conservation, adiabatic heating).
This is a twin test with a known answer, not an inversion of a real shot.

--ensemble K adds error bars (PINNEnsemble: K members on bootstrap resamples with fresh noise) on complete data and on
incomplete data (40% of the VISAR / framing samples lost at random and a gap in the middle third of the B-dot record),
and measures their calibration: how often the truth lies within +-1 and +-2 sigma.

    python examples/pinn_inverse_diagnostics.py
    python examples/pinn_inverse_diagnostics.py --quick
    python examples/pinn_inverse_diagnostics.py --ensemble 5
"""
import os
import sys
import json
import time
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, iRHO, iVX, iPR, iBY, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver, to_numpy
from aegis import validation as val
from aegis.ml import pinn_diagnostics as pd


def twin(t_end=0.7, nr=128, b0=0.1, p0=0.01, seed=3, noise=0.01):
    cfg = Config(nx=nr, ny=4, x_min=0.0, x_max=1.6, y_min=0.0, y_max=4 * 1.6 / nr, cfl=0.3, gamma=5 / 3, geometry="cylindrical",
                 bc_x_type="extrapolation", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    prof = 0.5 * (xp.tanh((s.X - 1.0) / 0.03) - xp.tanh((s.X - 1.2) / 0.03))
    W = xp.zeros_like(s.U)
    W[iRHO], W[iPR], W[iBY], W[iCLR] = 0.05 + prof, p0, b0, prof
    W[iVX] = -1.1 / xp.maximum(s.X, 1e-3) * prof
    s.U[...] = prim_to_cons(W, cfg.gamma)
    ng = s.ng
    r = to_numpy(s.x)[ng:-ng]
    W0 = to_numpy(cons_to_prim(s.U, s.eos))[:, ng:-ng, ng]
    init = dict(r=r, rho=W0[iRHO], v=W0[iVX], p=W0[iPR], B=W0[iBY])
    zm = 0.5 * cfg.y_max
    rp = 1.45
    D = val.DiagnosticSet([val.BdotProbe("b", (rp, zm), component=1), val.VisarProbe("v", (0.02, zm), axis=0, direction=1),
                           val.FramingCamera("f", (0.02, zm), axis=0, direction=1)])
    D.record(s)
    truth = []
    k = 0
    while s.t < t_end - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), t_end - s.t)
        s.step_ssprk3()
        s.t += s.dt
        k += 1
        D.record(s)
        Wn = to_numpy(cons_to_prim(s.U, s.eos))[:, ng:-ng, ng]
        core = r < 0.1
        truth.append((s.t, Wn[iBY][core].mean(), (Wn[iPR][core] / Wn[iRHO][core]).mean()))
    rng = np.random.default_rng(seed)
    tv, V = D.probes["v"].signal()
    tf, R = D.probes["f"].signal()
    tb, Vb = D.probes["b"].signal()
    Bp = b0 - np.concatenate([[0.0], np.cumsum(0.5 * (Vb[1:] + Vb[:-1]) * np.diff(tb))])
    sub = max(len(tv) // 40, 1)
    V_n = V + rng.normal(0, noise * np.nanmax(np.abs(V)), len(V))
    R_n = R + rng.normal(0, noise * 0.1, len(R))
    Bp_n = Bp + rng.normal(0, noise * np.max(np.abs(Bp)), len(Bp))
    # surface density: the tracer-half value, 0.05 + 0.5
    data = dict(t=tv[::sub], R=R_n[::sub], V=V_n[::sub], rho_surface=np.full(len(tv[::sub]), 0.55), t_b=tb[::sub], r_p=rp, B_p=Bp_n[::sub])
    sig = dict(V=noise * float(np.nanmax(np.abs(V))), R=noise * 0.1, B_p=noise * float(np.max(np.abs(Bp))))
    return init, data, np.array(truth), dict(b0=b0, R0=float(R[0]), theta0=p0 / 0.05, div_b=float(s.div_b_max()), steps=k,
                                             sigma=sig)


def degrade(data, seed=11, lose=0.4):
    """incomplete records: a fraction of the VISAR / framing samples lost at random, the middle third of B-dot missing"""
    rng = np.random.default_rng(seed)
    d = {k: (np.array(v, float) if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    gone = rng.random(len(d["t"])) < lose
    d["V"][gone] = np.nan
    d["R"][gone] = np.nan
    tb = d["t_b"]
    d["B_p"][(tb > tb[0] + (tb[-1] - tb[0]) / 3) & (tb < tb[0] + 2 * (tb[-1] - tb[0]) / 3)] = np.nan
    return d


def run_ensemble(K, iters, nr=128, width=64, ngrid=48, incomplete=False, log=None):
    init, data, truth, inf = twin(nr=nr)
    if incomplete:
        data = degrade(data)
    T = float(truth[-1, 0])
    ens = pd.PINNEnsemble(1.6, T, (1.0, 1.0, 0.1, 0.5), members=K, width=width)
    rg, tg = np.linspace(0.02, 1.6, ngrid), np.linspace(0.0, T, ngrid)
    sub = max(len(init["r"]) // 96, 1)
    t0 = time.time()
    ens.fit(data, rg, tg, {k: v[::sub] for k, v in init.items()}, inf["sigma"], iters=iters, log=log)
    tt = truth[:, 0]
    pr = ens.predict(np.full(len(tt), 0.05), tt)
    out = dict(seconds=time.time() - t0, members=K, incomplete=incomplete,
               used=dict(visar=int(np.sum(np.isfinite(data["V"]))), bdot=int(np.sum(np.isfinite(data["B_p"])))))
    rel = lambda a, b: float(np.sqrt(np.mean((a - b)**2)) / np.sqrt(np.mean(b**2)))
    for name, m, sd, true in (("B", pr["mean"][3], pr["sigma"][3], truth[:, 1]), ("T", pr["T_mean"], pr["T_sigma"], truth[:, 2])):
        z = np.abs(m - true) / np.maximum(sd, 1e-300)
        out[name] = dict(rel_err_mean=rel(m, true), mean_rel_sigma=float(np.mean(sd / np.maximum(np.abs(true), 1e-12))),
                         coverage_1sigma=float(np.mean(z <= 1)), coverage_2sigma=float(np.mean(z <= 2)))
    curves = dict(t=tt.tolist(), B_true=truth[:, 1].tolist(), B_mean=pr["mean"][3].tolist(), B_sigma=pr["sigma"][3].tolist(),
                  T_true=truth[:, 2].tolist(), T_mean=pr["T_mean"].tolist(), T_sigma=pr["T_sigma"].tolist())
    return out, curves


def run(iters=4000, nr=128, width=64, ngrid=48, log=None):
    init, data, truth, inf = twin(nr=nr)
    T = float(truth[-1, 0])
    scale = (1.0, 1.0, 0.1, 0.5)
    m = pd.PINN(1.6, T, scale, width=width)
    rg, tg = np.linspace(0.02, 1.6, ngrid), np.linspace(0.0, T, ngrid)
    sub = max(len(init["r"]) // 96, 1)
    ini = {k: v[::sub] for k, v in init.items()}
    t0 = time.time()
    hist = m.fit(data, rg, tg, ini, iters=iters, log=log)
    sec = time.time() - t0
    tt = truth[:, 0]
    rho, v, p, B = m.field(np.full(len(tt), 0.05), tt)
    Bpinn, Tpinn = B, p / rho
    Bt, Tt = truth[:, 1], truth[:, 2]
    Rt = np.interp(tt, data["t"], data["R"])
    Bfx, Tfx = pd.flux_estimate(Rt, inf["b0"], inf["R0"], inf["theta0"])
    rel = lambda a, b: float(np.sqrt(np.mean((a - b)**2)) / np.sqrt(np.mean(b**2)))
    last = tt > 0.7 * T
    rep = dict(B_rel_err_pinn=rel(Bpinn, Bt), B_rel_err_flux=rel(Bfx, Bt), T_rel_err_pinn=rel(Tpinn, Tt), T_rel_err_flux=rel(Tfx, Tt),
               B_late_pinn=rel(Bpinn[last], Bt[last]), B_late_flux=rel(Bfx[last], Bt[last]), T_late_pinn=rel(Tpinn[last], Tt[last]),
               T_late_flux=rel(Tfx[last], Tt[last]), B_true_final=float(Bt[-1]), B_pinn_final=float(Bpinn[-1]), B_flux_final=float(Bfx[-1]),
               T_true_final=float(Tt[-1]), T_pinn_final=float(Tpinn[-1]), T_flux_final=float(Tfx[-1]), seconds=sec, final_loss=hist[-1], div_b=inf["div_b"])
    return rep, dict(t=tt.tolist(), B_true=Bt.tolist(), B_pinn=Bpinn.tolist(), B_flux=Bfx.tolist(), T_true=Tt.tolist(), T_pinn=Tpinn.tolist(), T_flux=Tfx.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ensemble", type=int, default=0, help="members of an error-bar ensemble (0: single PINN)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pinn_diagnostics"))
    a = ap.parse_args()
    it = 800 if a.quick else a.iters
    if a.ensemble:
        os.makedirs(a.out, exist_ok=True)
        res = {}
        for inc in (False, True):
            o, c = run_ensemble(a.ensemble, it, incomplete=inc)
            res["incomplete" if inc else "complete"] = dict(report=o, curves=c)
            print(json.dumps(o))
        with open(os.path.join(a.out, "ensemble_report.json"), "w") as f:
            json.dump(dict(iters=it, **res), f, indent=1)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        for i, key in enumerate(("complete", "incomplete")):
            c = res[key]["curves"]
            for q, (k, lab) in enumerate((("B", "core B_z"), ("T", "core temperature"))):
                t_, m_, s_ = np.array(c["t"]), np.array(c[f"{k}_mean"]), np.array(c[f"{k}_sigma"])
                ax[i, q].fill_between(t_, m_ - 2 * s_, m_ + 2 * s_, color="C3", alpha=0.25, label="ensemble +-2 sigma")
                ax[i, q].plot(t_, m_, "C3", label="ensemble mean")
                ax[i, q].plot(t_, c[f"{k}_true"], "k", label="solver (truth)")
                ax[i, q].set_title(f"{lab}, {key} data"); ax[i, q].set_xlabel("t"); ax[i, q].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "pinn_ensemble.png"), dpi=130)
        return
    rep, curves = run(iters=it, log=(lambda r: print(r, flush=True)))
    for k, v in rep.items():
        print(f"{k}: {v}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(dict(report=rep, curves=curves, iters=it), f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for q, (k, lab) in enumerate((("B", "core B_z"), ("T", "core temperature"))):
        ax[q].plot(curves["t"], curves[f"{k}_true"], "k", label="solver (truth)")
        ax[q].plot(curves["t"], curves[f"{k}_pinn"], "C3", label="PINN")
        ax[q].plot(curves["t"], curves[f"{k}_flux"], "C0--", label="closed-form estimate")
        ax[q].set_xlabel("t"); ax[q].set_title(lab); ax[q].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "pinn_reconstruction.png"), dpi=130)


if __name__ == "__main__":
    main()
