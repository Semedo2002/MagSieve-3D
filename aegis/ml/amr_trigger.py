# amr_trigger.py - a small convolutional classifier that predicts where the AMR criterion will fire a few steps ahead.
#
# Additive: AMR flags cells from local gradients (AMR.flags). With AMRParams.trigger set, the cells this model marks are
# added to those flags, so a patch can exist before a steep gradient arrives. It can only add refinement: nothing it does
# touches the fluxes, the refluxing or the CT update, and regridding is the conservative one the solver always uses.
#
# Input per cell: log |B|, log |grad p|, log |J| and log |grad rho| of the coarse state, normalised. Label: the gradient
# criterion is false in this cell now and becomes true within `horizon` steps (default 5..10). Training data come from runs of
# this solver. The honest comparison is a baseline with the same flagged fraction: lowering the current gradient threshold until
# it flags as many cells. fit_report() measures recall of future crossings for both on a held-out run. The model knows nothing
# specific about wall modes or Nernst erosion: it learns gradient arrival in the flows it was shown.
import json

import numpy as np

from .. import autodiff as ad
from ..config import iRHO, iPR, iBX, iBY, iBZ


def _grad2(f, xp):
    gx = 0.5 * (xp.roll(f, -1, 0) - xp.roll(f, 1, 0))
    gy = 0.5 * (xp.roll(f, -1, 1) - xp.roll(f, 1, 1))
    return gx, gy


def features(W, xp):
    """(4, nx, ny) on the cells of W (primitive, var first): log10 of |B|, |grad p| / p, |J|, |grad rho| / rho per cell"""
    rho, p = W[iRHO], W[iPR]
    bx, by, bz = W[iBX], W[iBY], W[iBZ]
    tiny = 1e-8
    gpx, gpy = _grad2(p, xp)
    grx, gry = _grad2(rho, xp)
    bzx, bzy = _grad2(bz, xp)
    bxx, bxy = _grad2(bx, xp)
    byx, byy = _grad2(by, xp)
    J = xp.sqrt((byx - bxy)**2 + bzx**2 + bzy**2)
    return xp.stack([xp.log10(xp.sqrt(bx**2 + by**2 + bz**2) + tiny), xp.log10(xp.sqrt(gpx**2 + gpy**2) / xp.maximum(p, tiny) + tiny),
                     xp.log10(J + tiny), xp.log10(xp.sqrt(grx**2 + gry**2) / xp.maximum(rho, tiny) + tiny)])


def criterion(W, tol_rho, xp):
    """the density-gradient part of AMR.flags on the same cells (periodic differences)"""
    grx, gry = _grad2(W[iRHO], xp)
    return xp.sqrt(grx**2 + gry**2) / xp.maximum(W[iRHO], 1e-12) > tol_rho


def _roll(a, s, axis, xp):
    return ad._node(xp.roll(ad.val(a), s, axis), ((a, lambda g: xp.roll(g, -s, axis)),))


class TriggerCNN:
    """two 3x3 convolution layers (periodic) and a per-cell output; logistic loss"""

    def __init__(self, cin=4, width=12, seed=0, xp=np):
        rng = np.random.default_rng(seed)
        self.xp = xp
        s = 1.0 / np.sqrt(9 * width)
        P = dict(k1=rng.normal(0, 1 / np.sqrt(9 * cin), (cin, width, 3, 3)), b1=np.zeros(width),
                 k2=rng.normal(0, s, (width, width, 3, 3)), b2=np.zeros(width), w=rng.normal(0, s, (width, 1)), c=np.zeros(1))
        self.params = {k: xp.asarray(v) for k, v in P.items()}
        self.mu, self.sd = np.zeros(cin), np.ones(cin)
        self.threshold = 0.5

    def _conv(self, x, K):
        xp = self.xp
        out = None
        for a in range(3):
            for b in range(3):
                y = x
                if a != 1:
                    y = _roll(y, a - 1, 2, xp)
                if b != 1:
                    y = _roll(y, b - 1, 3, xp)
                t = ad.einsum("bcij,cd->bdij", y, K[:, :, a, b])
                out = t if out is None else out + t
        return out

    def logits(self, x, P):
        rs = lambda v: ad.reshape(v, (1, -1, 1, 1))
        h = ad.gelu(self._conv(x, P["k1"]) + rs(P["b1"]))
        h = ad.gelu(self._conv(h, P["k2"]) + rs(P["b2"]))
        return ad.einsum("bcij,cd->bdij", h, P["w"]) + rs(P["c"])

    def _norm(self, F):
        xp = self.xp
        return (F - xp.asarray(self.mu).reshape(1, -1, 1, 1)) / xp.asarray(self.sd).reshape(1, -1, 1, 1)

    def probability(self, F):
        """probability (nx, ny) of an upcoming crossing, from features (4, nx, ny)"""
        xp = self.xp
        z = self.logits(ad.const(self._norm(F[None])), {k: ad.const(v) for k, v in self.params.items()}).v[0, 0]
        return 1.0 / (1.0 + xp.exp(-z))

    def fit(self, X, Y, epochs=30, batch=8, lr=3e-3, seed=0, log=None):
        """X (N, 4, nx, ny), Y (N, nx, ny) booleans; balanced logistic loss with Adam"""
        xp = self.xp
        get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
        xn = get(X)
        self.mu, self.sd = xn.mean(axis=(0, 2, 3)), xn.std(axis=(0, 2, 3)) + 1e-8
        Xn = self._norm(xp.asarray(X))
        Yf = xp.asarray(Y).astype(float)
        pos = float(xp.mean(Yf))
        w1 = 0.5 / max(pos, 1e-6)
        w0 = 0.5 / max(1 - pos, 1e-6)
        rng = np.random.default_rng(seed)
        m = {k: xp.zeros_like(v) for k, v in self.params.items()}
        v2 = {k: xp.zeros_like(v) for k, v in self.params.items()}
        it, hist = 0, []
        for ep in range(epochs):
            perm = rng.permutation(Xn.shape[0])
            tot = 0.0
            for s in range(0, len(perm), batch):
                idx = xp.asarray(perm[s:s + batch])
                with ad.record():
                    P = {k: ad.leaf(v) for k, v in self.params.items()}
                    z = self.logits(ad.const(Xn[idx]), P)
                    y = Yf[idx][:, None]
                    # log(1 + e^(-s z)) with s = +-1, balanced: softplus by log(1 + exp)
                    sg = 2.0 * y - 1.0
                    loss_c = ad.log(1.0 + ad.exp(ad.minimum(z * (-sg), 30.0)))
                    wt = w1 * y + w0 * (1.0 - y)
                    loss = ad.tsum(loss_c * wt) * (1.0 / y.size)
                    ad.backward(loss)
                it += 1
                for k, p in P.items():
                    g = p.g if p.g is not None else xp.zeros_like(p.v)
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v2[k] = 0.999 * v2[k] + 0.001 * g * g
                    self.params[k] = self.params[k] - lr * (m[k] / (1 - 0.9**it)) / (xp.sqrt(v2[k] / (1 - 0.999**it)) + 1e-8)
                tot += float(loss.v) * len(idx)
            hist.append(tot / len(perm))
            if log:
                log(ep, hist[-1])
        return hist

    def save(self, path):
        get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
        np.savez(path, mu=self.mu, sd=self.sd, threshold=self.threshold, **{f"p_{k}": get(v) for k, v in self.params.items()})

    @classmethod
    def load(cls, path, xp=np):
        z = np.load(path)
        m = cls(xp=xp, width=z["p_b1"].shape[0])
        m.params = {k[2:]: xp.asarray(z[k]) for k in z.files if k.startswith("p_")}
        m.mu, m.sd, m.threshold = z["mu"], z["sd"], float(z["threshold"])
        return m


def collect(run_states, tol_rho, xp, horizon=(5, 10), stride=1):
    """(X, Y) from a list of primitive states one step apart (interior cells, periodic): X features at step n, Y = criterion
    false at n and true at any of n + horizon[0] .. n + horizon[1]"""
    X, Y = [], []
    crit = [criterion(W, tol_rho, xp) for W in run_states]
    for n in range(0, len(run_states) - horizon[1], stride):
        fut = xp.zeros_like(crit[n])
        for k in range(horizon[0], horizon[1] + 1):
            fut = fut | crit[n + k]
        X.append(features(run_states[n], xp))
        Y.append(fut & ~crit[n])
    return xp.stack(X), xp.stack(Y)


def scores(model, X, Y, crit_now, grad_now, flag_fraction, xp):
    """recall of upcoming crossings at a fixed extra flagged fraction: the CNN, and the baseline (a lowered gradient
    threshold) flagging the same number of currently unflagged cells; X (N, 4, ...), Y (N, ...)"""
    n, got, base, pos = X.shape[0], 0.0, 0.0, 0.0
    for t in range(n):
        free = ~crit_now[t]
        k = max(int(flag_fraction * float(xp.sum(free))), 1)
        p = model.probability(X[t])
        for score, slot in ((p, 0), (grad_now[t], 1)):
            s = xp.where(free, score, -xp.inf).ravel()
            idx = xp.argsort(-s)[:k]
            mask = xp.zeros(s.shape, dtype=bool)
            mask[idx] = True
            hit = float(xp.sum(mask & Y[t].ravel()))
            if slot == 0:
                got += hit
            else:
                base += hit
        pos += float(xp.sum(Y[t]))
    return got / max(pos, 1.0), base / max(pos, 1.0), pos / n


class Trigger:
    """the AMRParams.trigger callable: coarse solver -> host bool mask of interior cells to add to the flags. keep: the
    flagged fraction of unflagged cells, as used in the measurement"""

    def __init__(self, model, keep=0.05):
        self.model, self.keep = model, keep

    def __call__(self, s):
        from ..physics import xp, cons_to_prim
        ng = s.ng
        W = cons_to_prim(s.U, s.eos)[:, ng:-ng, ng:-ng]
        p = self.model.probability(features(W, xp))
        k = max(int(self.keep * p.size), 1)
        thr = xp.sort(p.ravel())[-k]
        m = p >= thr
        return m.get() if hasattr(m, "get") else m
