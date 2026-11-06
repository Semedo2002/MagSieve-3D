"""Train and test the sub-grid stress co-processor (aegis/ml/fno_subgrid.py) on filtered runs of this solver.

Reference runs: compressible 2D MHD (an Orszag-Tang vortex with random phases and amplitudes) at fine resolution, filtered
by a box of r cells. The network learns the sub-grid stress from the coarse fields. Measured on a held-out initial condition,
never seen in training:
  a priori: correlation of predicted and true stress, against a Smagorinsky eddy viscosity;
  a posteriori: coarse runs with no model, with Smagorinsky and with the network, against the filtered reference run (velocity
  and density errors), plus the exact conservation of each (the model acts after the HLLD / CT step and never replaces it).

    python examples/sgs_fno_closure.py          # 256^2 reference, 64^2 coarse
    python examples/sgs_fno_closure.py --quick
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
    from aegis.config import Config, EN, RHO, MX, MY, iRHO, iVX, iVY, iPR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver, to_numpy
from aegis.ml import fno_subgrid as fs


def make(n, seed, gamma=5 / 3):
    """random-phase Orszag-Tang-type state; the same seed gives the same flow at any resolution"""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 1.2, 4)
    ph = rng.uniform(0, 2 * np.pi, 4)
    cfg = Config(nx=n, ny=n, x_max=1.0, y_max=1.0, cfl=0.3, gamma=gamma, bc_x_type="periodic", bc_y_type="periodic",
                 diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    W = xp.zeros_like(s.U)
    W[iRHO] = 25 / (36 * np.pi)
    W[iPR] = 5 / (12 * np.pi)
    W[iVX] = -a[0] * xp.sin(2 * np.pi * s.Y + float(ph[0]))
    W[iVY] = a[1] * xp.sin(2 * np.pi * s.X + float(ph[1]))
    s.U[...] = prim_to_cons(W, gamma)
    B0 = 1 / np.sqrt(4 * np.pi)
    s.set_vector_potential(lambda x, y: B0 * (a[2] * np.cos(4 * np.pi * x + ph[2]) / (4 * np.pi)
                                              + a[3] * np.cos(2 * np.pi * y + ph[3]) / (2 * np.pi)))
    return s


def march_to(s, T):
    while s.t < T - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), T - s.t)
        s.step_ssprk3()
        s.t += s.dt
        s.step += 1


def fine_prim(s):
    ng = s.ng
    return cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng]


def dataset(seeds, n, r, times):
    X, Y = [], []
    for sd in seeds:
        s = make(n, sd)
        for T in times:
            march_to(s, T)
            f, t = fs.sgs_sample(fine_prim(s), r)
            X.append(f)
            Y.append(t)
    return xp.stack(X), xp.stack(Y)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt(float((a * a).sum() * (b * b).sum()) + 1e-300))


def coarse_run(n, seed, T, stress=None, model=None):
    s = make(n, seed)
    cop = fs.SGSCoprocessor(s, model=model, stress_fn=stress) if (stress or model) else None
    ng = s.ng
    tot0 = [float(xp.sum(s.U[v, ng:-ng, ng:-ng])) for v in (RHO, MX, MY, EN)]
    while s.t < T - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), T - s.t)
        s.step_ssprk3()
        if cop:
            cop.step(s.dt)
        s.t += s.dt
    tot1 = [float(xp.sum(s.U[v, ng:-ng, ng:-ng])) for v in (RHO, MX, MY, EN)]
    return s, cop, dict(mass=abs(tot1[0] - tot0[0]) / tot0[0], momentum=abs(tot1[1] - tot0[1]) + abs(tot1[2] - tot0[2]),
                        energy=abs(tot1[3] - tot0[3]) / tot0[3], div_b=float(s.div_b_max()))


def evaluate(n_f, r, seed, T, model):
    n_c = n_f // r
    h = 1.0 / n_c
    ref = make(n_f, seed)
    march_to(ref, T)
    Ft, Tt = fs.sgs_sample(fine_prim(ref), r)
    Tp = model.predict(Ft[None])[0]
    Ts = fs.smagorinsky_stress(Ft, h, h)
    apri = {}
    for name, P in (("network", Tp), ("smagorinsky", Ts)):
        apri[name] = [corr(to_numpy(P[k]), to_numpy(Tt[k])) for k in range(3)]
    res = {}
    sm = lambda F: fs.smagorinsky_stress(F, h, h)
    for name, kw in (("no model", {}), ("smagorinsky", dict(stress=sm)), ("network", dict(model=model))):
        s, cop, cons = coarse_run(n_c, seed, T, **kw)
        W = fine_prim(s)
        e = lambda a, b: float(xp.sqrt(xp.mean((a - b)**2)) / xp.sqrt(xp.mean(b**2)))
        res[name] = dict(err_velocity=0.5 * (e(W[iVX], Ft[1]) + e(W[iVY], Ft[2])), err_density=e(W[iRHO], Ft[0]), **cons,
                         accepted=cop.accepted if cop else None, rejected=cop.rejected if cop else None)
    return dict(apriori=apri, aposteriori=res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--train-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--t-eval", type=float, default=0.4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "sgs_fno"))
    a = ap.parse_args()
    if a.quick:
        a.n, a.train_seeds, a.epochs = 128, 3, 15
    os.makedirs(a.out, exist_ok=True)
    times = np.linspace(0.05, 0.5, 10)
    t0 = time.time()
    X, Y = dataset(range(a.train_seeds), a.n, a.r, times)
    Xv, Yv = dataset([1000], a.n, a.r, times)
    print(f"data: {X.shape[0]} train, {Xv.shape[0]} validation samples ({time.time() - t0:.0f}s)")
    m = fs.FNO(X.shape[1], Y.shape[1], width=20, modes=min(12, a.n // a.r // 2 - 1), layers=3, xp=xp)
    m.fit_normalisation(X, Y)
    t0 = time.time()
    hist = m.train(X, Y, epochs=a.epochs, batch=8, Xv=Xv, Yv=Yv, log=lambda r: print(f"  epoch {r['epoch']}: train {r['train']:.3e} val {r['val']:.3e}"))
    print(f"training {time.time() - t0:.0f}s")
    rep = evaluate(a.n, a.r, 2000, a.t_eval, m)
    print("a priori, correlation with the true stress (xx, xy, yy), held-out flow:")
    for k, v in rep["apriori"].items():
        print(f"  {k:12s} " + " ".join(f"{x:+.3f}" for x in v))
    print("a posteriori, coarse run vs filtered reference:")
    for k, v in rep["aposteriori"].items():
        print(f"  {k:12s} velocity error {v['err_velocity']:.4f}, density error {v['err_density']:.4f}; mass {v['mass']:.0e}, "
              f"momentum {v['momentum']:.0e}, energy {v['energy']:.0e}, div B {v['div_b']:.0e}; accepted {v['accepted']}, rejected {v['rejected']}")
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(dict(args=vars(a), history=hist, **rep), f, indent=1)


if __name__ == "__main__":
    main()
