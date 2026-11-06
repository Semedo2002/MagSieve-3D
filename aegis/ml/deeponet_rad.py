# deeponet_rad.py - a deep operator network for emergent line-of-sight spectra (a synthetic-spectrometer surrogate).
#
# Additive and diagnostic: it reads plasma profiles and predicts what a spectrometer would see; it never feeds back into the
# solver. The operator: a profile along a chord (log T(s), log rho(s), N zones) -> the emergent multigroup intensity I_g.
#
# Ground truth (formal_solution): the exact 1D formal solution of the transfer equation along the chord for each group,
#     I_g = sum_s S_g(T_s) (1 - exp(-dtau_s)) exp(-tau_s->observer),   dtau = rho kappa_g ds,
# with group opacities from a multigroup table (tabular.build_atomic_table: NIST energies, screened-hydrogenic line model,
# collisional-radiative ion balance, so the ionization is non-LTE) and a Planck group source S_g = B_g(T): the populations
# within an ion are not solved for, so it is not a full non-LTE line-transfer calculation. Self-absorption of the lines is
# exact in 1D (no escape-probability approximation is needed along a single chord).
#
# DeepONet (Lu et al., Nat. Mach. Intell. 3, 218, 2021): a branch net encodes the profile into p coefficients, a trunk net
# maps the photon energy to p basis functions, I(E) = sum_k b_k t_k(E) + b0, here for log10 I. Trained on random smooth
# profiles; measured on held-out ones against the exact solution and against the optically thin sum (no absorption).
import numpy as np

from .. import autodiff as ad


def planck_groups(edges_eV, T_eV, xp, n=24):
    """group-integrated Planck intensity B_g(T) (relative units: x^3 / (e^x - 1) dx integrated, times T^4), (G, ...)"""
    E = xp.asarray(edges_eV, float)
    out = []
    for g in range(len(E) - 1):
        x = xp.exp(xp.linspace(float(np.log(E[g])), float(np.log(E[g + 1])), n))       # log-spaced points in the group
        u = x[:, None] / T_eV.ravel()[None, :]
        f = xp.where(u < 300, u**3 / xp.expm1(xp.minimum(u, 300.0)), 0.0) * u            # x^3/(e^x-1) d(ln E) weight * E/T
        w = xp.log(x[1:] / x[:-1])
        integ = (0.5 * (f[1:] + f[:-1]) * w[:, None]).sum(axis=0)
        out.append((integ * T_eV.ravel()**4).reshape(T_eV.shape))
    return xp.stack(out)


def formal_solution(rho, T, ds, kappa_fn, edges_eV, xp):
    """exact emergent intensity per group at the observer (end of the chord) for profiles rho, T (B, N): (B, G)"""
    kR, kP = kappa_fn(rho, T)                               # (G, B, N), absorption ~ Planck-mean group opacity
    dtau = kP * rho[None] * ds
    S = planck_groups(edges_eV, T, xp)                      # (G, B, N)
    tau_after = xp.cumsum(dtau[..., ::-1], axis=-1)[..., ::-1] - dtau            # optical depth from zone to observer
    I = (S * (-xp.expm1(-dtau)) * xp.exp(-tau_after)).sum(axis=-1)
    thin = (S * dtau).sum(axis=-1)
    return I.T, thin.T


def random_profiles(B, N, rng, T_range=(5.0, 1500.0), rho_range=(1e-3, 1.0)):
    """smooth random chords: one or two hot spots in T over a background, an anti-correlated density (rough pressure
    balance) with a random overall level; log T, log rho returned too"""
    s = np.linspace(0, 1, N)[None, :]
    lt0, lt1 = np.log(T_range)
    lr0, lr1 = np.log(rho_range)
    T = np.exp(lt0 + 0.3 * (lt1 - lt0) * rng.random((B, 1)))
    for _ in range(2):
        c, w = rng.random((B, 1)), 0.05 + 0.25 * rng.random((B, 1))
        T = T + np.exp(lt0 + (lt1 - lt0) * rng.random((B, 1))) * np.exp(-0.5 * ((s - c) / w)**2) * (rng.random((B, 1)) < 0.8)
    T = np.minimum(T, T_range[1])
    level = np.exp(lr0 + (lr1 - lr0) * rng.random((B, 1)))
    rho = np.clip(level * (T[:, :1] / T)**rng.uniform(0.3, 1.0, (B, 1)), rho_range[0], rho_range[1])
    return rho, T


class DeepONet:
    def __init__(self, n_in, p=64, width=128, depth=3, seed=0, xp=np):
        rng = np.random.default_rng(seed)
        self.xp, self.p, self.depth = xp, p, depth
        P = {}
        n0 = n_in
        for k in range(depth):
            P[f"bw{k}"] = rng.normal(0, 1 / np.sqrt(n0), (n0, width)); P[f"bb{k}"] = np.zeros(width); n0 = width
        P["bwo"] = rng.normal(0, 1 / np.sqrt(width), (width, p)); P["bbo"] = np.zeros(p)
        n0 = 1
        for k in range(depth):
            P[f"tw{k}"] = rng.normal(0, 1 / np.sqrt(max(n0, 1)), (n0, width)); P[f"tb{k}"] = rng.normal(0, 1, width) * (k == 0)
            n0 = width
        P["two"] = rng.normal(0, 1 / np.sqrt(width), (width, p)); P["tbo"] = np.zeros(p)
        P["b0"] = np.zeros(1)
        self.P = {k: xp.asarray(v) for k, v in P.items()}
        self.mu_x = self.sd_x = self.mu_y = self.sd_y = None

    def forward(self, x, e, P):
        """x: (B, n_in) normalised branch inputs; e: (G, 1) normalised trunk inputs; -> (B, G) normalised log10 I"""
        h = ad.const(x)
        for k in range(self.depth):
            h = ad.gelu(ad.einsum("bi,ij->bj", h, P[f"bw{k}"]) + P[f"bb{k}"])
        b = ad.einsum("bi,ij->bj", h, P["bwo"]) + P["bbo"]
        t = ad.const(e)
        for k in range(self.depth):
            t = ad.gelu(ad.einsum("gi,ij->gj", t, P[f"tw{k}"]) + P[f"tb{k}"])
        t = ad.einsum("gi,ij->gj", t, P["two"]) + P["tbo"]
        return ad.einsum("bk,gk->bg", b, t) * (1.0 / np.sqrt(self.p)) + P["b0"]

    @staticmethod
    def inputs(rho, T, xp):
        return xp.concatenate([xp.log10(T), xp.log10(rho)], axis=1)

    def fit(self, X, E, Y, epochs=200, batch=64, lr=1e-3, seed=0, log=None, Xv=None, Yv=None):
        """X (B, n_in) raw branch inputs, E (G,) log10 photon energies, Y (B, G) log10 intensities"""
        xp = self.xp
        self.mu_x, self.sd_x = X.mean(axis=0), X.std(axis=0) + 1e-6
        self.mu_y, self.sd_y = float(Y.mean()), float(Y.std()) + 1e-12
        self.e = ((E - E.mean()) / (E.std() + 1e-12))[:, None]
        Xn, Yn = (X - self.mu_x) / self.sd_x, (Y - self.mu_y) / self.sd_y
        m = {k: xp.zeros_like(v) for k, v in self.P.items()}
        v2 = {k: xp.zeros_like(v) for k, v in self.P.items()}
        rng = np.random.default_rng(seed)
        n, it, hist = X.shape[0], 0, []
        steps = epochs * max(n // batch, 1)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n - batch + 1, batch):
                it += 1
                idx = xp.asarray(perm[i:i + batch])
                with ad.record():
                    P = {k: ad.leaf(v) for k, v in self.P.items()}
                    d = self.forward(Xn[idx], self.e, P) - Yn[idx]
                    loss = ad.tsum(d * d) * (1.0 / d.v.size)
                    ad.backward(loss)
                lr_k = lr * 0.5 * (1 + np.cos(np.pi * it / steps))
                for k, p in P.items():
                    g = p.g
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v2[k] = 0.999 * v2[k] + 0.001 * g * g
                    self.P[k] = self.P[k] - lr_k * (m[k] / (1 - 0.9**it)) / (xp.sqrt(v2[k] / (1 - 0.999**it)) + 1e-8)
            rec = dict(epoch=ep, train=float(loss.v))
            if Xv is not None:
                rec["val"] = float(xp.mean((self.predict(Xv) - Yv)**2)) / self.sd_y**2
            hist.append(rec)
            if log and (ep % max(epochs // 10, 1) == 0 or ep == epochs - 1):
                log(rec)
        return hist

    def predict(self, X):
        """log10 I (B, G) for raw branch inputs X"""
        P = {k: ad.const(v) for k, v in self.P.items()}
        return self.forward((X - self.mu_x) / self.sd_x, self.e, P).v * self.sd_y + self.mu_y
