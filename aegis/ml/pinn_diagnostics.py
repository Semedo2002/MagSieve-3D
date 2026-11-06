# pinn_diagnostics.py - physics-informed inverse diagnostics: unmeasured internal states from outside probes.
#
# A diagnostic that reads measurements, never part of the solver. A coordinate network maps (r, t) to (rho, v_r, p, B_z) for
# a straight axisymmetric implosion and minimises
#     L = w_data L_data + w_phys L_physics + w_init L_init
#   L_data: an outer B-dot coil (integrated to B_z(r_p, t) from the known seed), VISAR (velocity at the liner surface), a framing
#     camera (surface position R(t), where the network's density must equal the surface value);
#   L_physics: mean squared residuals of ideal MHD, 1D cylindrical, on a collocation grid (finite differences of the network):
#     d_t rho + (1/r) d_r (r rho v), rho (d_t v + v d_r v) + d_r (p + B^2/2), d_t p + v d_r p + gamma p (1/r) d_r (r v),
#     d_t B + (1/r) d_r (r v B);
#   L_init: the pre-shot state, assumed known (liner geometry, fill, seed field).
# Output: the reconstructed fields, e.g. the core field and temperature nobody measured. This is an estimate with no error bar
# and not an exact inversion: the physics term is only as good as the assumed model (ideal, 1D), and a coordinate network
# represents steep fronts poorly. Always compare with the closed-form estimate from the same data (flux_estimate): flux
# conservation B = B0 (R0 / R)^2 and adiabatic heating of the fill.
import numpy as np

from .. import autodiff as ad


def flux_estimate(R, b0, R0, theta0, gamma=5 / 3):
    """the closed-form baseline from a measured radius: (core B, core temperature) under flux conservation and adiabatic
    compression, no neural network"""
    c = (R0 / np.maximum(R, 1e-6))**2
    return b0 * c, theta0 * c**(gamma - 1)


class PINN:
    def __init__(self, r_max, t_max, scale, width=64, depth=3, n_freq=6, gamma=5 / 3, seed=0):
        rng = np.random.default_rng(seed)
        self.r_max, self.t_max, self.scale, self.gamma = r_max, t_max, np.asarray(scale, float), gamma
        self.freq = 2.0 ** np.arange(n_freq) * np.pi
        cin = 2 + 2 * 2 * n_freq
        self.P = {}
        n0 = cin
        for k in range(depth):
            self.P[f"w{k}"] = rng.normal(0, 1 / np.sqrt(n0), (n0, width))
            self.P[f"b{k}"] = np.zeros(width)
            n0 = width
        self.P["wo"] = rng.normal(0, 0.1 / np.sqrt(width), (width, 4))
        self.P["bo"] = np.zeros(4)
        self.depth = depth

    def _embed(self, r, t):
        x, y = r / self.r_max, t / self.t_max
        parts = [x, y]
        for f in self.freq:
            parts += [np.sin(f * x), np.cos(f * x), np.sin(f * y), np.cos(f * y)]
        return np.stack(parts, axis=1)

    def net(self, r, t, P):
        """rho, v, p, B (N,) Tensors at points: log outputs for rho and p, so they stay positive"""
        h = ad.const(self._embed(r, t))
        for k in range(self.depth):
            h = ad.gelu(ad.einsum("ni,ij->nj", h, P[f"w{k}"]) + P[f"b{k}"])
        o = ad.einsum("ni,ij->nj", h, P["wo"]) + P["bo"]
        rho = ad.exp(o[:, 0]) * self.scale[0]
        v = o[:, 1] * self.scale[1]
        p = ad.exp(o[:, 2]) * self.scale[2]
        B = o[:, 3] * self.scale[3]
        return rho, v, p, B

    def residuals(self, P, rg, tg):
        """physics residuals on the grid rg x tg (central differences), each normalised by its scales"""
        nr, nt = len(rg), len(tg)
        R, T = np.meshgrid(rg, tg, indexing="ij")
        rho, v, p, B = self.net(R.ravel(), T.ravel(), P)
        sh = lambda a: ad.reshape(a, (nr, nt))
        rho, v, p, B = sh(rho), sh(v), sh(p), sh(B)
        dr, dt = rg[1] - rg[0], tg[1] - tg[0]
        rc = ad.const(R[1:-1, 1:-1])
        Dr = lambda f: (f[2:, 1:-1] - f[:-2, 1:-1]) * (0.5 / dr)
        Dt = lambda f: (f[1:-1, 2:] - f[1:-1, :-2]) * (0.5 / dt)
        C = lambda f: f[1:-1, 1:-1]
        rr = ad.const(R)
        g = self.gamma
        r_rho = Dt(rho) + Dr(rr * rho * v) / rc
        r_v = C(rho) * (Dt(v) + C(v) * Dr(v)) + Dr(p + 0.5 * B * B)
        r_p = Dt(p) + C(v) * Dr(p) + g * C(p) * Dr(rr * v) / rc
        r_B = Dt(B) + Dr(rr * v * B) / rc
        s = self.scale
        L0 = self.r_max / max(abs(s[1]), 1e-12)
        return [r_rho * (L0 / s[0]), r_v * (L0 / (s[0] * s[1])), r_p * (L0 / s[2]), r_B * (L0 / s[3])]

    def fit(self, data, rg, tg, init, iters=3000, lr=2e-3, w_data=10.0, w_phys=1.0, w_init=10.0, log=None):
        """data: dict(t, R, V, rho_surface, t_b, r_p, B_p); init: dict(r, rho, v, p, B) at t = 0. Returns the loss history"""
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.P.items()}
        s = self.scale
        hist = []
        for it in range(1, iters + 1):
            with ad.record():
                P = {k: ad.leaf(v) for k, v in self.P.items()}
                res = self.residuals(P, rg, tg)
                Lp = sum(ad.tsum(q * q) * (1.0 / q.v.size) for q in res)
                rho, v, _, _ = self.net(data["R"], data["t"], P)
                Ld = ad.tsum(((v - data["V"]) * (1.0 / s[1]))**2) * (1.0 / len(data["t"]))
                Ld = Ld + ad.tsum(((rho - data["rho_surface"]) * (1.0 / s[0]))**2) * (1.0 / len(data["t"]))
                _, _, _, Bp = self.net(np.full(len(data["t_b"]), data["r_p"]), data["t_b"], P)
                Ld = Ld + ad.tsum(((Bp - data["B_p"]) * (1.0 / s[3]))**2) * (1.0 / len(data["t_b"]))
                t0 = np.zeros(len(init["r"]))
                r0, v0, p0, b0 = self.net(init["r"], t0, P)
                Li = 0.0
                for got, want, sc in ((r0, init["rho"], s[0]), (v0, init["v"], s[1]), (p0, init["p"], s[2]), (b0, init["B"], s[3])):
                    Li = Li + ad.tsum(((got - want) * (1.0 / sc))**2) * (1.0 / len(t0))
                loss = w_data * Ld + w_phys * Lp + w_init * Li
                ad.backward(loss)
            lr_k = lr * 0.5 * (1 + np.cos(np.pi * it / iters))
            for k, p in P.items():
                g = p.g if p.g is not None else 0.0
                m[k] = 0.9 * m[k] + 0.1 * g
                v2[k] = 0.999 * v2[k] + 0.001 * g * g
                self.P[k] = self.P[k] - lr_k * (m[k] / (1 - 0.9**it)) / (np.sqrt(v2[k] / (1 - 0.999**it)) + 1e-8)
            if it % max(iters // 20, 1) == 0 or it == 1:
                rec = dict(it=it, loss=float(loss.v), data=float(ad.val(Ld)), physics=float(ad.val(Lp)), init=float(ad.val(Li)))
                hist.append(rec)
                if log:
                    log(rec)
        return hist

    def field(self, r, t):
        """reconstructed (rho, v, p, B) as arrays at points"""
        P = {k: ad.const(v) for k, v in self.P.items()}
        return tuple(a.v for a in self.net(np.asarray(r, float), np.asarray(t, float), P))
