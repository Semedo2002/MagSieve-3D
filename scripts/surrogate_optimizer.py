"""Active-learning design optimisation on top of AEGIS-MHD: a Gaussian process decides which solver runs to spend.

The surrogate never stands in for the solver in a result: every value it is trained on, and the design it finally reports, comes
from an actual run. It only chooses the next run, by expected improvement (high value, high uncertainty), so a design space
can be explored with fewer runs than a blind scan.

  - GP: anisotropic squared-exponential kernel on the unit cube, length scales and noise by marginal likelihood on a random
    search; NumPy only.
  - Design problem (LinerDesign): a 3D (x, y, z) corrugated liner on a uniform B_z, three parameters: wall thickness, seed
    field, and implosion speed (the drive, here the speed the pulse would have given the liner; a current waveform I(t) is
    available in 2D r-z only, through DiffMHD's circuit). Objective: the peak mean B_z in a small core region, divided by the
    implosion kinetic energy so that a faster implosion is not simply better.
  - Compared with a Latin-hypercube scan of the same budget: on the design problem and on the Branin test function.

    python scripts/surrogate_optimizer.py --runs 16         # 3D liner design, about 10 s per run on a P1000
    python scripts/surrogate_optimizer.py --branin          # test function only
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


class GaussianProcess:
    """zero-mean GP on standardised y over the unit cube; fit() picks the hyperparameters"""

    def __init__(self, dim, seed=0):
        self.dim, self.rng = dim, np.random.default_rng(seed)
        self.ls, self.sf2, self.sn2 = np.full(dim, 0.3), 1.0, 1e-4

    def _k(self, A, B, ls):
        d = (A[:, None, :] - B[None, :, :]) / ls
        return np.exp(-0.5 * np.sum(d * d, axis=-1))

    def _nll(self, X, y, ls, sn2):
        K = self._k(X, X, ls) + (sn2 + 1e-10) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return 0.5 * y @ a + np.sum(np.log(np.diag(L)))

    def fit(self, X, y, n_try=300):
        self.X = np.asarray(X, float)
        self.mu, self.sd = float(np.mean(y)), float(np.std(y)) + 1e-12
        self.y = (np.asarray(y, float) - self.mu) / self.sd
        best, arg = np.inf, (self.ls, self.sn2)
        for k in range(n_try):
            ls = np.exp(self.rng.uniform(np.log(0.05), np.log(3.0), self.dim))
            sn2 = np.exp(self.rng.uniform(np.log(1e-6), np.log(1e-1)))
            v = self._nll(self.X, self.y, ls, sn2)
            if v < best:
                best, arg = v, (ls, sn2)
        self.ls, self.sn2 = arg
        K = self._k(self.X, self.X, self.ls) + (self.sn2 + 1e-10) * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs):
        """posterior mean and standard deviation, in the units of y"""
        Ks = self._k(np.asarray(Xs, float), self.X, self.ls)
        m = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        s2 = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
        return m * self.sd + self.mu, np.sqrt(s2) * self.sd


def _erf(x):
    """error function (Abramowitz & Stegun 7.1.26), vectorised"""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def expected_improvement(mean, std, best, xi=0.01):
    z = (mean - best - xi) / np.maximum(std, 1e-12)
    cdf = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return (mean - best - xi) * cdf + std * pdf


def latin_hypercube(n, dim, rng):
    return np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(dim)], axis=1)


def optimise(f, dim, budget, n_init=None, seed=0, log=None, n_cand=4000):
    """maximise f over the unit cube: initial Latin hypercube, then one run at a time at the largest expected improvement.
    Returns (X, y, history of the best value, predicted std at each chosen point)"""
    rng = np.random.default_rng(seed)
    n_init = n_init or max(2 * dim + 1, 5)
    X = list(latin_hypercube(n_init, dim, rng))
    y = [f(x) for x in X]
    stds = []
    while len(y) < budget:
        gp = GaussianProcess(dim, seed=int(rng.integers(1 << 30))).fit(np.array(X), np.array(y))
        best_x = X[int(np.argmax(y))]
        cand = np.concatenate([rng.random((n_cand, dim)), np.clip(best_x + rng.normal(0, 0.05, (n_cand // 4, dim)), 0, 1)])
        # a run repeated at the same point adds nothing (the solver is deterministic)
        near = np.min(np.sum((cand[:, None, :] - np.array(X)[None, :, :])**2, axis=-1), axis=1) < 0.02**2
        cand = cand[~near]
        m, s = gp.predict(cand)
        ei = expected_improvement(m, s, max(y), xi=0.01 * gp.sd)
        k = int(np.argmax(ei))
        X.append(cand[k])
        y.append(f(cand[k]))                         # always an actual evaluation
        stds.append(float(s[k]))
        if log:
            log(len(y), cand[k], y[-1], m[k], s[k], max(y))
    return np.array(X), np.array(y), np.maximum.accumulate(y), stds


def branin(x):
    """Branin-Hoo on the unit cube, negated for maximisation; optimum -0.397887"""
    a, b = 15 * x[0] - 5, 15 * x[1]
    return -((b - 5.1 / (4 * np.pi**2) * a**2 + 5 / np.pi * a - 6)**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10)


class LinerDesign:
    """3D corrugated liner run through MHDSolver3D; __call__(x in unit cube) is one solver run"""
    bounds = dict(thickness=(0.1, 0.4), b0=(0.03, 0.3), v0=(0.6, 1.4))

    def __init__(self, n=32, nz=8, t_end=1.3, eps=0.02):
        self.n, self.nz, self.t_end, self.eps = n, nz, t_end, eps
        self.runs = []

    def params(self, x):
        return {k: lo + float(u) * (hi - lo) for u, (k, (lo, hi)) in zip(x, self.bounds.items())}

    def run(self, thickness, b0, v0):
        with contextlib.redirect_stdout(io.StringIO()):
            from aegis.config import Config, NVAR, iRHO, iVX, iVY, iPR, iBZ, iCLR, EN
            from aegis.physics import xp, prim_to_cons, cons_to_prim
            from aegis.solver3d import MHDSolver3D
        H, L, n = 1.8, 1.0, self.n
        cfg = Config(nx=n, ny=n, nz=self.nz, x_min=-H, x_max=H, y_min=-H, y_max=H, z_min=0.0, z_max=L, cfl=0.3, gamma=5 / 3,
                     bc_x_type="extrapolation", bc_y_type="extrapolation", powell_source=False, diag_interval=10**9,
                     snapshot_times=[])
        with contextlib.redirect_stdout(io.StringIO()):
            s = MHDSolver3D(cfg)
        X = s.x[:, None, None] * xp.ones((1, s.ny_tot, s.nz_tot))
        Y = s.y[None, :, None] * xp.ones((s.nx_tot, 1, s.nz_tot))
        Z = s.z[None, None, :] * xp.ones((s.nx_tot, s.ny_tot, 1))
        R = xp.sqrt(X**2 + Y**2) + 1e-12
        th = xp.arctan2(Y, X)
        Rin = 1.0 + self.eps * xp.cos(4 * th) * xp.cos(2 * np.pi * Z / L)
        w = 0.06
        ring = 0.5 * (xp.tanh((R - Rin) / w) - xp.tanh((R - Rin - thickness) / w))
        W = xp.zeros((NVAR, s.nx_tot, s.ny_tot, s.nz_tot))
        W[iRHO] = 0.05 + ring
        vr = -v0 * (1.0 + 0.5 * thickness) / R * ring
        W[iVX], W[iVY] = vr * X / R, vr * Y / R
        W[iPR], W[iBZ], W[iCLR] = 0.01, b0, ring
        s.U = prim_to_cons(W, cfg.gamma)
        ke0 = float(xp.sum(0.5 * W[iRHO] * (W[iVX]**2 + W[iVY]**2)))
        core = R[s.ng:-s.ng, s.ng:-s.ng, s.ng:-s.ng] < 0.25
        peak, steps, div = 0.0, 0, 0.0
        E0 = s.total_energy()
        while s.t < self.t_end - 1e-12:
            s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), self.t_end - s.t)
            s.step_ssprk3()
            s.t += s.dt
            steps += 1
            if steps % 4 == 0:
                b = cons_to_prim(s.U, s.eos)[iBZ][s.ng:-s.ng, s.ng:-s.ng, s.ng:-s.ng]
                peak = max(peak, float(xp.mean(b[core])))
        div = float(s.div_b_max())
        return dict(gain=peak / b0, steps=steps, div_b=div, ke0=ke0 * cfg.dx * cfg.dy * cfg.dz)

    def __call__(self, x):
        p = self.params(x)
        r = self.run(**p)
        # compression per unit of log implosion energy: rewards using the energy well, not just adding more of it
        val = float(np.log(max(r["gain"], 1e-3)) - 0.5 * np.log(r["ke0"]))
        self.runs.append(dict(p, **r, objective=val))
        return val


def compare_branin(budget=30, seeds=5):
    gp, lhs = [], []
    for sd in range(seeds):
        _, _, hist, _ = optimise(branin, 2, budget, seed=sd, n_cand=2000)
        gp.append(hist[-1])
        rng = np.random.default_rng(1000 + sd)
        lhs.append(max(branin(x) for x in latin_hypercube(budget, 2, rng)))
    return float(np.mean(gp)), float(np.mean(lhs)), -0.397887


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--branin", action="store_true")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "surrogate_optimizer"))
    a = ap.parse_args()
    g, l, opt = compare_branin()
    print(f"Branin, 30 evaluations, mean of 5 seeds: active {g:.4f}, Latin hypercube {l:.4f}, optimum {opt:.4f}")
    if a.branin:
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = LinerDesign(n=a.n)

    def log(n, x, y, m, s, best):
        print(f"run {n:2d}: {d.params(x)} -> {y:.3f} (predicted {m:.3f} +- {s:.3f}), best {best:.3f}", flush=True)
    X, y, hist, _ = optimise(d, 3, a.runs, seed=1, log=log)
    best_active = d.runs[int(np.argmax(y))]
    print(f"active: {a.runs} solver runs in {time.time() - t0:.0f}s, best objective {y.max():.3f}: {best_active}")
    t0 = time.time()
    d2 = LinerDesign(n=a.n)
    Xl = latin_hypercube(a.runs, 3, np.random.default_rng(99))
    yl = np.array([d2(x) for x in Xl])
    print(f"Latin hypercube, same budget: best objective {yl.max():.3f}: {d2.runs[int(np.argmax(yl))]} ({time.time() - t0:.0f}s)")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(active=d.runs, scan=d2.runs, best_active=float(y.max()), best_scan=float(yl.max()), history=hist.tolist()),
                  f, indent=1, default=float)


if __name__ == "__main__":
    main()
