"""Adjoint optimisation of a drive current waveform I(t) together with a liner shape (results/drive_waveform/).

An r-z liner with a DT fuel core is driven by a current I(t) on the outer boundary, a natural cubic spline through
knot values (aegis.autodiff.spline_current_drive). The objective is the thermonuclear DT yield of the hydrodynamic
history, in reactions, from Bosch-Hale reactivities (aegis.autodiff.fusion_rate; no burn, alpha heating or radiation).
One adjoint pass gives the gradient with respect to all knot currents and the shape modes at once.

Constraints: I(t) >= 0; a fixed drive budget int I^2 dt (a stand-in for the bank energy, equal to that of the starting
linear ramp); shape amplitudes |a_m| <= 0.05. Projected gradient ascent. Compared at the same budget with the starting
ramp and with a constant current.

    python examples/optimize_drive_waveform.py [--iters 15]
"""
import os
import sys
import io
import json
import argparse
import contextlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aegis.config import Config                # noqa: E402
from aegis.solver import MHDSolver              # noqa: E402
from aegis import autodiff as ad                # noqa: E402

UNITS = dict(L0=1e-3, v0=1e5, rho0=100.0)       # 1 mm, 100 km/s, 0.1 g/cc per code unit
N_KNOTS, T_END, N_STEPS = 6, 0.5, 80


def setup():
    cfg = Config(nx=32, ny=16, x_min=0.0, x_max=1.6, y_min=0.0, y_max=1.0, cfl=0.3, gamma=5 / 3, geometry="cylindrical",
                 bc_x_type="extrapolation", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    tk = np.linspace(0.0, T_END, N_KNOTS)
    drive = ad.spline_current_drive(tk, cfg.x_max, offset=2)
    sim = ad.DiffMHD(s, drive=drive, xp=np)
    r = np.asarray(s.rc.get() if hasattr(s.rc, "get") else s.rc)[:, None] * np.ones((1, s.ny_tot))
    zz = np.asarray(s.y.get() if hasattr(s.y, "get") else s.y)
    z = np.ones((s.nx_tot, 1)) * zz[None, :]

    def W_fn(p):
        Rz = 1.0 + p[0] * ad.const(np.cos(2 * np.pi * z)) + p[1] * ad.const(np.cos(4 * np.pi * z))
        prof = 0.5 * (ad.tanh((ad.const(r) - Rz) / 0.05) - ad.tanh((ad.const(r) - Rz - 0.25) / 0.05))
        return [0.05 + prof, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.4, 0.0, prof, 0.0]
    init = lambda p: ad.cells_to_state(sim, W_fn, p)
    tq = np.linspace(0.0, T_END, 401)
    Wq = np.array([drive.weights(t) for t in tq])                   # I(tq) = Wq @ knots
    return sim, init, tq, Wq


def budget(knots, tq, Wq):
    return float(np.trapz((Wq @ knots)**2, tq))


def project(p, tq, Wq, E_target):
    q = p.copy()
    q[:2] = np.clip(q[:2], -0.05, 0.05)
    k = np.maximum(q[2:], 0.0)
    for _ in range(50):                              # nonnegative I(t) on the quadrature grid, then the budget
        I = Wq @ k
        if I.min() >= -1e-12:
            break
        k = np.maximum(k + 0.5 * np.maximum(-I.min(), 0.0), 0.0)
    q[2:] = k * np.sqrt(E_target / budget(k, tq, Wq))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "drive_waveform"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sim, init, tq, Wq = setup()
    Y = lambda st, t, p: ad.fusion_rate(sim, st, **UNITS)
    ramp = np.concatenate([[0.0, 0.0], np.linspace(1.0, 8.0, N_KNOTS)])
    E_target = budget(ramp[2:], tq, Wq)
    const = np.concatenate([[0.0, 0.0], np.full(N_KNOTS, np.sqrt(E_target / T_END))])
    dts = sim.run(init, ramp, n_steps=N_STEPS)[1]["dts"]            # one step sequence for every evaluation

    def J(p):
        return float(sim.run(init, p, dts=dts, step_objective=Y)[0])
    y_ramp, y_const = J(ramp), J(const)
    p = ramp.copy()
    hist = []
    step = np.concatenate([[0.01, 0.01], np.full(N_KNOTS, 0.5)])
    for it in range(a.iters):
        j, g, _ = sim.value_and_grad(init, p, dts=dts, step_objective=Y)
        hist.append(dict(iter=it, yield_=float(j), params=p.tolist()))
        print(f"iter {it:2d}: yield {j:.4e} reactions, shape {p[0]:+.4f} {p[1]:+.4f}, I knots " +
              " ".join(f"{x:.2f}" for x in p[2:]))
        gn = g / (np.abs(g).max() + 1e-300)
        trial = project(p + step * gn * np.where(np.arange(len(p)) < 2, 1.0, np.abs(p).max()), tq, Wq, E_target)
        if J(trial) > j:
            p = trial
        else:
            step *= 0.5
    y_opt = J(p)
    out = dict(units=UNITS, budget_int_I2_dt=E_target, yield_ramp=y_ramp, yield_constant=y_const, yield_optimised=y_opt,
               params_optimised=p.tolist(), history=hist,
               note="DT yield of the hydrodynamic history (Bosch-Hale), no burn, alpha heating or radiation; one "
                    "configuration on a 32 x 16 grid; a demonstration of joint waveform and shape gradients, not a design")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"same budget: ramp {y_ramp:.4e}, constant {y_const:.4e}, optimised {y_opt:.4e} reactions "
          f"({y_opt / max(y_ramp, y_const):.2f}x the better baseline)")


if __name__ == "__main__":
    main()
