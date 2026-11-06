"""Train and measure the predictive AMR trigger (aegis/ml/amr_trigger.py).

Training flows: random-phase Orszag-Tang runs of this solver at 128^2. --families adds three more randomised families on the
same periodic box and trains on all four: an imploding dense shell with an m = 3 radius perturbation and an axial field
(wall modes), a hot spot in a magnetised plasma with Nernst advection of the field (field erosion), and a strong planar
blast (pressure ratio 300, Mach about 10) hitting a dense bubble (shock interaction). Each family is scored on held-out seeds,
also with the Orszag-Tang-only model, to show what the extra training buys. The label says a cell's density gradient will cross the AMR
threshold 5 to 10 steps from now. On a held-out flow, at the same extra flagged fraction, recall of the upcoming crossings is
compared with the baseline: the same gradient measure with a lowered threshold. Then a real AMR run with the trigger added
to its flags checks that conservation and div B stay at round-off, as without it.

    python examples/amr_instability_trigger.py
    python examples/amr_instability_trigger.py --quick
    python examples/amr_instability_trigger.py --families
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
    from aegis.config import Config, EN, RHO, iRHO, iVX, iVY, iPR, iBX, iBY, iBZ, iCLR
    from aegis.solver import MHDSolver
    from aegis.physics import xp, cons_to_prim, prim_to_cons
    from aegis.solver import to_numpy
from aegis.ml import amr_trigger as at
import sgs_fno_closure as sc


def make_family(family, n, seed):
    """a randomised periodic 2D state of one training family (see the module notes)"""
    if family == "orszag_tang":
        return sc.make(n, seed)
    rng = np.random.default_rng(seed)
    kw = dict(nx=n, ny=n, x_max=1.0, y_max=1.0, cfl=0.3, gamma=5 / 3, bc_x_type="periodic", bc_y_type="periodic",
              diag_interval=10**9, snapshot_times=[])
    if family == "nernst":
        kw.update(nernst_coeff=float(rng.uniform(0.02, 0.06)), cond_x0=1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(Config(**kw))
    X, Y = s.X - 0.5, s.Y - 0.5
    R = xp.sqrt(X**2 + Y**2)
    th = xp.arctan2(Y, X)
    W = xp.zeros_like(s.U)
    if family == "wall":                                       # imploding shell, m = 3 perturbed radius, axial field
        R0 = 0.32 * (1 + float(rng.uniform(0.02, 0.06)) * xp.cos(3 * th + float(rng.uniform(0, 2 * np.pi))))
        shell = 0.5 * (xp.tanh((R - R0) / 0.01) - xp.tanh((R - R0 - 0.06) / 0.01))
        W[iRHO] = 0.1 + 2.0 * shell
        W[iPR] = 0.05
        v = -float(rng.uniform(0.8, 1.2)) * shell
        W[iVX], W[iVY] = v * X / xp.maximum(R, 1e-6), v * Y / xp.maximum(R, 1e-6)
        W[iBZ] = float(rng.uniform(0.2, 0.5))
    elif family == "nernst":                                   # hot spot in a magnetised plasma: Nernst erodes B_z
        x0, y0 = rng.uniform(-0.1, 0.1, 2)
        hot = xp.exp(-((X - x0)**2 + (Y - y0)**2) / float(rng.uniform(0.006, 0.015)))
        W[iRHO] = 1.0 / (1.0 + 4.0 * hot)                      # pressure balance with p = 1 + small excess
        W[iPR] = 1.0 + 0.2 * hot
        W[iBZ] = float(rng.uniform(0.5, 1.0))
    else:                                                      # blast slab driving a strong shock into a dense bubble
        xs, yb = float(rng.uniform(-0.35, -0.25)), float(rng.uniform(-0.1, 0.1))
        W[iRHO] = 1.0 + float(rng.uniform(3, 8)) * (((X - 0.05)**2 + (Y - yb)**2) < 0.1**2)
        W[iPR] = 1.0 + 300.0 * (xp.abs(X - xs) < 0.04)
        W[iBX] = float(rng.uniform(0.05, 0.2))
    s.U[...] = prim_to_cons(W, s.cfg.gamma)
    s._ct_sync()
    s.apply_bc(s.U)
    return s


def states(n, seed, n_steps, t0=0.1, family="orszag_tang"):
    s = make_family(family, n, seed)
    if family == "orszag_tang":
        sc.march_to(s, t0)
    ng = s.ng
    out = []
    for _ in range(n_steps):
        s.dt = s.compute_dt(cons_to_prim(s.U, s.eos))
        s.step_ssprk3()
        s.t += s.dt
        out.append(cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng].copy())
    return out


def data(seeds, n, n_steps, tol, stride=3, family="orszag_tang"):
    X, Y, C, G = [], [], [], []
    for sd in seeds:
        W = states(n, sd, n_steps, family=family)
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
    ap.add_argument("--families", action="store_true", help="train on four flow families and score each")
    ap.add_argument("--width", type=int, default=24, help="CNN width for --families")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "amr_trigger"))
    a = ap.parse_args()
    if a.families:
        return families(a)
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


FAMILIES = ("orszag_tang", "wall", "nernst", "shock")


def families(a):
    t0 = time.time()
    train, held = {}, {}
    for f in FAMILIES:
        train[f] = data(range(a.train_seeds), a.n, a.steps, a.tol, family=f)[:2]
        held[f] = data([5000, 5001], a.n, a.steps, a.tol, family=f)
        print(f"{f}: {train[f][0].shape[0]} training frames, positive cells {100 * float(xp.mean(train[f][1])):.2f}%, "
              f"held-out {100 * float(xp.mean(held[f][1])):.2f}% ({time.time() - t0:.0f}s)", flush=True)
    X = xp.concatenate([train[f][0] for f in FAMILIES])
    Y = xp.concatenate([train[f][1] for f in FAMILIES])
    m = at.TriggerCNN(width=a.width, xp=xp)
    m.fit(X, Y, epochs=a.epochs, log=lambda e, l: print(f"  epoch {e}: {l:.4f}", flush=True))
    m_ot = at.TriggerCNN(xp=xp)                                 # the same network trained on Orszag-Tang only
    m_ot.fit(train["orszag_tang"][0], train["orszag_tang"][1], epochs=a.epochs)
    rep = dict(n=a.n, train_seeds=a.train_seeds, width=a.width, epochs=a.epochs, recall={})
    for f in FAMILIES:
        Xh, Yh, Ch, Gh = held[f]
        rep["recall"][f] = {}
        for frac in (0.05, 0.1):
            r_all, r_base, _ = at.scores(m, Xh, Yh, Ch, Gh, frac, xp)
            r_ot, _, _ = at.scores(m_ot, Xh, Yh, Ch, Gh, frac, xp)
            rep["recall"][f][str(frac)] = dict(cnn_all_families=r_all, cnn_orszag_tang_only=r_ot, lowered_threshold=r_base)
            print(f"{f:12s} extra {100 * frac:3.0f}%: recall, CNN (all families) {r_all:.3f}, CNN (Orszag-Tang only) {r_ot:.3f}, "
                  f"lowered threshold {r_base:.3f}", flush=True)
    rep["amr"] = amr_check(m, 0.05)
    for k, v in rep["amr"].items():
        print(f"AMR {k} trigger: mass {v['mass']:.0e}, energy {v['energy']:.0e}, div B {v['div_b']:.0e}, {v['fine_cells']} fine cells")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report_families.json"), "w") as f:
        json.dump(rep, f, indent=1)
    m.save(os.path.join(a.out, "trigger_cnn_families.npz"))


if __name__ == "__main__":
    main()
