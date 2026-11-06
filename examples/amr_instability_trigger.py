"""Train and measure the predictive AMR trigger (aegis/ml/amr_trigger.py).

Training flows: random-phase Orszag-Tang runs of this solver at 128^2. The label says a cell's density gradient will cross the AMR
threshold 5 to 10 steps from now. On a held-out flow, at the same extra flagged fraction, recall of the upcoming crossings is
compared with the baseline: the same gradient measure with a lowered threshold. Then a real AMR run with the trigger added
to its flags checks that conservation and div B stay at round-off, as without it.

    python examples/amr_instability_trigger.py
    python examples/amr_instability_trigger.py --quick
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
sys.path.insert(0, os.path.join(ROOT, "examples"))
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, EN, RHO, iRHO
    from aegis.physics import xp, cons_to_prim, prim_to_cons
    from aegis.solver import to_numpy
from aegis.ml import amr_trigger as at
import sgs_fno_closure as sc


def states(n, seed, n_steps, t0=0.1):
    s = sc.make(n, seed)
    sc.march_to(s, t0)
    ng = s.ng
    out = []
    for _ in range(n_steps):
        s.dt = s.compute_dt(cons_to_prim(s.U, s.eos))
        s.step_ssprk3()
        s.t += s.dt
        out.append(cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng].copy())
    return out


def data(seeds, n, n_steps, tol, stride=3):
    X, Y, C, G = [], [], [], []
    for sd in seeds:
        W = states(n, sd, n_steps)
        x, y = at.collect(W, tol, xp, stride=stride)
        X.append(x)
        Y.append(y)
        idx = range(0, len(W) - 10, stride)
        C.append(xp.stack([at.criterion(W[k], tol, xp) for k in idx]))
        G.append(xp.stack([_grad(W[k]) for k in idx]))
    return xp.concatenate(X), xp.concatenate(Y), xp.concatenate(C), xp.concatenate(G)


def _grad(W):
    gx, gy = at._grad2(W[iRHO], xp)
    return xp.sqrt(gx**2 + gy**2) / xp.maximum(W[iRHO], 1e-12)


def amr_check(model, keep, n=64, n_steps=40, tol=0.05):
    from aegis.amr import AMR, AMRParams
    out = {}
    for name, trig in (("without", None), ("with", at.Trigger(model, keep=keep))):
        s0 = sc.make(n, 3000)
        cfg = s0.cfg
        a = AMR(cfg, AMRParams(ratio=2, regrid_every=4, tol_rho=tol, tol_p=0.2, trigger=trig, graphs=False))
        a.set_state(s0.U)
        c = a.coarse
        c.Bf, c._faces_owner = s0.Bf, c.U                # the divergence-free faces of the initial state
        c._Bf1, c._dBf = [xp.empty_like(f) for f in c.Bf], [xp.empty_like(f) for f in c.Bf]
        a.regrid()
        t0 = a.totals()
        for _ in range(n_steps):
            a.advance(a.compute_dt())
        t1 = a.totals()
        out[name] = dict(mass=abs(t1["mass"] - t0["mass"]) / t0["mass"], energy=abs(t1["E"] - t0["E"] - t1["boundary"] + t0["boundary"]) / t0["E"],
                         div_b=float(a.div_b()), fine_cells=sum(q.cfg.nx * q.cfg.ny for q in a.patches), patches=len(a.patches))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "amr_trigger"))
    a = ap.parse_args()
    if a.quick:
        a.n, a.train_seeds, a.steps, a.epochs = 64, 2, 40, 5
    t0 = time.time()
    X, Y, _, _ = data(range(a.train_seeds), a.n, a.steps, a.tol)
    Xh, Yh, Ch, Gh = data([5000], a.n, a.steps, a.tol)
    print(f"{X.shape[0]} training frames, positive cells {100 * float(xp.mean(Y)):.1f}% ({time.time() - t0:.0f}s)")
    m = at.TriggerCNN(xp=xp)
    t0 = time.time()
    m.fit(X, Y, epochs=a.epochs, log=lambda e, l: print(f"  epoch {e}: {l:.4f}", flush=True))
    print(f"training {time.time() - t0:.0f}s")
    rep = dict(n=a.n, recall={})
    for frac in (0.02, 0.05, 0.1, 0.2):
        r_cnn, r_base, pos = at.scores(m, Xh, Yh, Ch, Gh, frac, xp)
        rep["recall"][str(frac)] = dict(cnn=r_cnn, baseline=r_base)
        print(f"extra flagged {100 * frac:4.0f}% of unflagged cells: recall of crossings 5-10 steps ahead, CNN {r_cnn:.3f}, lowered threshold {r_base:.3f}")
    rep["amr"] = amr_check(m, 0.05)
    for k, v in rep["amr"].items():
        print(f"AMR {k} trigger: mass {v['mass']:.0e}, energy {v['energy']:.0e}, div B {v['div_b']:.0e}, {v['patches']} patches, {v['fine_cells']} fine cells")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    m.save(os.path.join(a.out, "trigger_cnn.npz"))


if __name__ == "__main__":
    main()
