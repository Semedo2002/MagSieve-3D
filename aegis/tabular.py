# tabular.py - tabulated equation of state and opacities: SESAME ASCII (EOS table 301, opacity tables 502 and
# 505) or an SI .npz table, interpolated on the GPU.
#
# On load a table is resampled onto a uniform grid in log10(rho) and log10(T) (linear interpolation in those
# coordinates). p, e and the opacities are interpolated in log space when they are positive everywhere (power
# laws, e.g. an ideal gas, are then exact) and linearly otherwise. The hydrodynamics asks for p(rho, e) and
# e(rho, p): both invert the table along T at fixed rho (binary search, then bilinear interpolation), in one
# CUDA kernel per call. c^2 comes from the table when it has one, otherwise from its derivatives,
#     c^2 = p_rho - p_T e_rho / e_T + p p_T / (rho^2 e_T)       (subscripts: partial derivatives).
# States beyond the table (in rho, or in e / p along T) are extrapolated linearly from the edge interval and
# blended into an analytic fallback (Mie-Gruneisen or stiffened gas) over `margin` grid cells, so the EOS stays
# continuous. Code units are set by (rho_unit, v_unit): rho / rho_unit, p / (rho_unit v_unit^2), e / v_unit^2.
#
# In the two-material solver a tabulated liner is used in pure liner cells; mixed cells (liner fraction below
# 0.98) keep the tangent stiffened gas of the volume-fraction closure, blended over 0.98 ... 1 as for the
# Mie-Gruneisen liner (TabularEOS).
import re

import numpy as np

from .physics import xp

_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


# ------------------------------------------------------------------ reading

def read_sesame(path, table, material=None):
    """words of one SESAME ASCII table (e.g. 301, 502): a header line '<flag> <material> <table> <nwords> ...'
    followed by the numbers, five per line. Returns (material, np.ndarray of words)."""
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if len(tok) >= 4 and all(t.lstrip("-").isdigit() for t in tok[:4]) and int(tok[0]) in (0, 1, 2):
            mat, tab, nw = int(tok[1]), int(tok[2]), int(tok[3])
            words, j = [], i + 1
            while len(words) < nw and j < len(lines):
                words += [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT.findall(lines[j])]
                j += 1
            if tab == table and (material is None or mat == material):
                return mat, np.array(words[:nw])
            i = j
        else:
            i += 1
    raise ValueError(f"table {table}" + (f" for material {material}" if material else "") + f" not in {path}")


def write_sesame(path, material, tables):
    """write SESAME ASCII: tables = {table id: words}; five E22.15 numbers per line"""
    with open(path, "w") as fh:
        for tab, words in tables.items():
            words = np.asarray(words, dtype=float)
            fh.write(f" 0 {material:6d} {tab:6d} {len(words):8d}   010126   1\n")
            for k in range(0, len(words), 5):
                fh.write("".join(f"{w:22.15E}" for w in words[k:k + 5]) + "\n")


def sesame_301(words):
    """total EOS: NR, NT, rho (g/cc), T (K), P (GPa), E (MJ/kg) with rho fastest -> SI arrays (nr, nt)"""
    nr, nt = int(words[0]), int(words[1])
    o = 2
    rho = words[o:o + nr] * 1e3; o += nr
    T = words[o:o + nt]; o += nt
    P = words[o:o + nr * nt].reshape(nt, nr).T * 1e9; o += nr * nt
    E = words[o:o + nr * nt].reshape(nt, nr).T * 1e6
    return rho, T, P, E


def sesame_opacity(words):
    """500-series opacity table: NR, NT, log10 rho (g/cc), log10 T (eV), log10 kappa (cm^2/g), rho fastest
    -> SI (rho kg/m^3, T K, kappa m^2/kg). Check that your library uses this convention."""
    nr, nt = int(words[0]), int(words[1])
    o = 2
    rho = 10**words[o:o + nr] * 1e3; o += nr
    T = 10**words[o:o + nt] * 11604.518; o += nt
    k = 10**words[o:o + nr * nt].reshape(nt, nr).T * 0.1
    return rho, T, k


def load(path, material=None):
    """dict of SI arrays from a SESAME file (tables 301 and, if present, 502 / 505) or an .npz with rho, T,
    p, e and optionally cs, kappa_R, kappa_P ((nr, nt) arrays, rho [kg/m^3], T [K], SI values)"""
    if path.endswith(".npz"):
        d = np.load(path)
        return {k: np.asarray(d[k], dtype=float) for k in d.files}
    out = {}
    try:
        _, w = read_sesame(path, 301, material)
        out["rho"], out["T"], out["p"], out["e"] = sesame_301(w)
    except ValueError:
        pass
    for tab, name in ((502, "kappa_R"), (505, "kappa_P")):
        try:
            _, w = read_sesame(path, tab, material)
            r, T, k = sesame_opacity(w)
            out.setdefault("rho_k", r); out.setdefault("T_k", T)
            out[name] = k
        except ValueError:
            pass
    if not out:
        raise ValueError(f"no 301 / 502 / 505 table in {path}")
    return out


# ------------------------------------------------------------------ uniform log grid

def _resample(lr, lt, f, LR, LT):
    """bilinear resampling of f(lr, lt) (nr, nt) onto the grid LR x LT (all log10)"""
    fr = np.interp(LR, lr, np.arange(len(lr)))
    ft = np.interp(LT, lt, np.arange(len(lt)))
    i = np.clip(np.floor(fr).astype(int), 0, len(lr) - 2); wr = (fr - i)[:, None]
    j = np.clip(np.floor(ft).astype(int), 0, len(lt) - 2); wt = (ft - j)[None, :]
    i, j = i[:, None], j[None, :]
    return ((1 - wr) * (1 - wt) * f[i, j] + wr * (1 - wt) * f[i + 1, j] + (1 - wr) * wt * f[i, j + 1]
            + wr * wt * f[i + 1, j + 1])


class Grid:
    """a quantity set on a uniform log10 rho x log10 T grid (code units), stored flat on the device"""

    def __init__(self, rho, T, fields, refine=2):
        lr, lt = np.log10(rho), np.log10(T)
        self.nr, self.nt = refine * (len(lr) - 1) + 1, refine * (len(lt) - 1) + 1
        LR, LT = np.linspace(lr[0], lr[-1], self.nr), np.linspace(lt[0], lt[-1], self.nt)
        self.lr0, self.dlr, self.lt0, self.dlt = LR[0], LR[1] - LR[0], LT[0], LT[1] - LT[0]
        self.logs, self.host, self.dev = {}, {}, {}
        for name, f in fields.items():
            lg = bool(np.all(f > 0))
            g = _resample(lr, lt, np.log10(f) if lg else f, LR, LT)
            self.logs[name] = lg
            self.host[name] = g
            self.dev[name] = xp.asarray(np.ascontiguousarray(g).ravel())
        self.LR, self.LT = LR, LT


_KERNEL = None


def _kernel():
    """CUDA elementwise kernel: invert table Q along T at fixed rho (mode 0) or evaluate at given log T
    (mode 1), returning the other tables bilinearly; plus the out-of-table distance in grid cells"""
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        _KERNEL = cp.ElementwiseKernel(
            "float64 lrho, float64 q, raw float64 Q, raw float64 A, raw float64 B, raw float64 C, "
            "int32 nr, int32 nt, float64 lr0, float64 dlr, int32 mode",
            "float64 a, float64 b, float64 c, float64 lt, float64 out",
            r"""
            double fr = (lrho - lr0) / dlr;
            int ir = (int)floor(fr);
            ir = ir < 0 ? 0 : (ir > nr - 2 ? nr - 2 : ir);
            double wr = fr - ir;                                  // outside [0, 1]: extrapolated in rho
            double dr = wr < 0 ? -wr : (wr > 1 ? wr - 1 : 0.0);
            int jt; double wt;
            if (mode == 0) {
                #define QV(jj) ((1 - wr) * Q[ir * nt + (jj)] + wr * Q[(ir + 1) * nt + (jj)])
                int lo = 0, hi = nt - 1;
                double qlo = QV(0), qhi = QV(nt - 1);
                if (q <= qlo) { jt = 0; }
                else if (q >= qhi) { jt = nt - 2; }
                else {
                    while (hi - lo > 1) { int m = (lo + hi) >> 1; if (QV(m) <= q) lo = m; else hi = m; }
                    jt = lo;
                }
                double q0 = QV(jt), q1 = QV(jt + 1);
                wt = (q - q0) / (q1 - q0 != 0 ? q1 - q0 : 1e-300);
                #undef QV
            } else {
                double ft = q;                                   // q is the fractional T index
                jt = (int)floor(ft); jt = jt < 0 ? 0 : (jt > nt - 2 ? nt - 2 : jt);
                wt = ft - jt;
            }
            double dt = wt < 0 ? -wt : (wt > 1 ? wt - 1 : 0.0);
            int k00 = ir * nt + jt, k10 = (ir + 1) * nt + jt;
            double w00 = (1 - wr) * (1 - wt), w10 = wr * (1 - wt), w01 = (1 - wr) * wt, w11 = wr * wt;
            a = w00 * A[k00] + w10 * A[k10] + w01 * A[k00 + 1] + w11 * A[k10 + 1];
            b = w00 * B[k00] + w10 * B[k10] + w01 * B[k00 + 1] + w11 * B[k10 + 1];
            c = w00 * C[k00] + w10 * C[k10] + w01 * C[k00 + 1] + w11 * C[k10 + 1];
            lt = jt + wt;
            out = dr > dt ? dr : dt;
            """, "aegis_tab_lookup")
    return _KERNEL


def _lookup(g, lrho, q, Q, A, B, C, mode):
    if xp.__name__ == "cupy":
        return _kernel()(lrho, q, g.dev[Q], g.dev[A], g.dev[B], g.dev[C], np.int32(g.nr), np.int32(g.nt),
                         float(g.lr0), float(g.dlr), np.int32(mode))
    # NumPy path: the same algorithm, vectorised (bisection over the T index)
    H = {k: g.host[k] for k in (Q, A, B, C)}
    fr = (lrho - g.lr0) / g.dlr
    i = np.clip(np.floor(fr).astype(int), 0, g.nr - 2)
    wr = fr - i
    dr = np.maximum(np.maximum(-wr, wr - 1), 0)
    if mode == 0:
        QV = lambda jj: (1 - wr) * H[Q][i, jj] + wr * H[Q][i + 1, jj]
        lo, hi = np.zeros_like(i), np.full_like(i, g.nt - 1)
        for _ in range(int(np.ceil(np.log2(g.nt))) + 1):
            m = (lo + hi) // 2
            up = QV(m) <= q
            lo, hi = np.where(up & (hi - lo > 1), m, lo), np.where(~up & (hi - lo > 1), m, hi)
        j = np.where(q <= QV(np.zeros_like(i)), 0, np.where(q >= QV(np.full_like(i, g.nt - 1)), g.nt - 2, lo))
        q0, q1 = QV(j), QV(j + 1)
        wt = (q - q0) / np.where(q1 - q0 != 0, q1 - q0, 1e-300)
    else:
        j = np.clip(np.floor(q).astype(int), 0, g.nt - 2)
        wt = q - j
    dt = np.maximum(np.maximum(-wt, wt - 1), 0)
    bl = lambda f: ((1 - wr) * (1 - wt) * f[i, j] + wr * (1 - wt) * f[i + 1, j] + (1 - wr) * wt * f[i, j + 1]
                    + wr * wt * f[i + 1, j + 1])
    return bl(H[A]), bl(H[B]), bl(H[C]), j + wt, np.maximum(dr, dt)


# ------------------------------------------------------------------ EOS

class TabularMaterial:
    """p(rho, e), e(rho, p), c^2 from a table, in code units; fallback: an eos.Material for the states beyond it"""

    def __init__(self, data, rho_unit, v_unit, fallback, margin=2.0, refine=2):
        pu = rho_unit * v_unit**2
        rho = data["rho"] / rho_unit
        p, e = data["p"] / pu, data["e"] / v_unit**2
        if "cs" in data:
            c2 = (data["cs"] / v_unit)**2
        else:                                        # thermodynamic sound speed from the table's derivatives
            lr, lt = np.log10(data["rho"]), np.log10(data["T"])
            R, T = np.meshgrid(rho, data["T"], indexing="ij")

            def d(f, ax):                            # df/drho or df/dT; through log f where f > 0 (power laws exact)
                l, X = (lr, R) if ax == 0 else (lt, T)
                if np.all(f > 0):
                    return f / X * np.gradient(np.log10(f), l, axis=ax, edge_order=2)
                return np.gradient(f, l, axis=ax, edge_order=2) / (X * np.log(10))
            p_r, p_T, e_r, e_T = d(p, 0), d(p, 1), d(e, 0), d(e, 1)
            e_T = np.where(np.abs(e_T) > 0, e_T, 1e-300)
            c2 = np.maximum(p_r - p_T * e_r / e_T + p * p_T / (R**2 * e_T), 1e-30)
        if not (np.all(np.diff(e, axis=1) > 0) and np.all(np.diff(p, axis=1) > 0)):
            raise ValueError("table: e and p must increase with T at every density")
        self.g = Grid(rho, data["T"], {"p": p, "e": e, "c2": c2}, refine)
        self.fallback, self.margin = fallback, margin
        self.kind = "tabular"
        self.rho_range = (float(rho.min()), float(rho.max()))

    def _fb(self, rho):
        return self.fallback.phis(rho, xp)

    def _blend(self, out, tab, fb_fn):
        """table values, blended into the fallback (evaluated only if some state is off the table)"""
        if not bool(xp.any(out > 0)):
            return tab
        w = xp.clip(1.0 - out / self.margin, 0.0, 1.0)
        return w * tab + (1 - w) * fb_fn()

    def _decode(self, name, v):
        return 10**v if self.g.logs[name] else v

    def _encode(self, name, v):
        return xp.log10(xp.maximum(v, 1e-300)) if self.g.logs[name] else v

    def pressure(self, rho, e):
        """p from density and specific internal energy"""
        lr = xp.log10(xp.maximum(rho, 1e-300))
        p, _, c2, lt, out = _lookup(self.g, lr, self._encode("e", e), "e", "p", "e", "c2", 0)

        def fb():
            f1, f2, _, _ = self._fb(rho)
            return (rho * e - f2) / f1
        return self._blend(out, self._decode("p", p), fb)

    def energy(self, rho, p):
        """specific internal energy from density and pressure"""
        lr = xp.log10(xp.maximum(rho, 1e-300))
        _, e, _, lt, out = _lookup(self.g, lr, self._encode("p", p), "p", "p", "e", "c2", 0)

        def fb():
            f1, f2, _, _ = self._fb(rho)
            return (f1 * p + f2) / rho
        return self._blend(out, self._decode("e", e), fb)

    def c2(self, rho, p):
        lr = xp.log10(xp.maximum(rho, 1e-300))
        _, _, c2, _, out = _lookup(self.g, lr, self._encode("p", p), "p", "p", "e", "c2", 0)

        def fb():
            f1, f2, d1, d2 = self._fb(rho)
            return ((f1 * p + f2 + p) / rho - d2 - p * d1) / f1
        return self._blend(out, self._decode("c2", c2), fb)

    def temperature(self, rho, e):
        """T (K) from density and specific internal energy (inside the table)"""
        lr = xp.log10(xp.maximum(rho, 1e-300))
        _, _, _, lt, _ = _lookup(self.g, lr, self._encode("e", e), "e", "p", "e", "c2", 0)
        return 10**(self.g.lt0 + lt * self.g.dlt)

    def describe(self):
        r = self.rho_range
        return f"tabular ({self.g.nr} x {self.g.nt} log grid, rho {r[0]:.3g}-{r[1]:.3g} code) -> {self.fallback.describe()}"


class TabularEOS:
    """two-material EOS (same interface as eos.EOS) whose liner is a TabularMaterial in pure liner cells; mixed
    cells use the mixture of the fuel and the fallback's tangent stiffened gas, blended in over a = 0.98 ... 1"""

    @property
    def no_graph(self):                              # the array eps() branches on the host: no CUDA-graph
        from .eos_fused import device                # capture unless the device EOS (eos_fused.cu) covers it
        return device(self) is None

    def __init__(self, base, liner_table):
        self.base, self.tab = base, liner_table
        self.fuel, self.liner = base.fuel, base.liner
        self.volume_fraction = base.volume_fraction
        tangent = liner_table.fallback.tangent_stiffened() if liner_table.fallback.kind == "mie_gruneisen" \
            else liner_table.fallback
        from .eos import EOS
        self._mix = EOS(base.fuel, tangent, base.volume_fraction)

    def _w(self, a):
        w = xp.clip((xp.clip(a, 0, 1) - 0.98) / 0.02, 0.0, 1.0)
        return w * w * (3 - 2 * w)

    def coeffs(self, rho, a):                        # linear-form coefficients of the mixed-cell closure
        return self._mix.coeffs(rho, a)

    def pressure(self, rho, eps, a):
        from .eos import _dev
        r = _dev(self, "pressure", rho, eps, a)
        if r is not None:
            return r
        w = self._w(a)
        p_mix = self._mix.pressure(rho, eps, a)
        p_tab = self.tab.pressure(rho, eps / xp.maximum(rho, 1e-300))
        return (1 - w) * p_mix + w * p_tab

    def eps(self, rho, p, a):
        from .eos import _dev
        r = _dev(self, "eps", rho, p, a)
        if r is not None:
            return r
        w = self._w(a)
        e_mix = self._mix.eps(rho, p, a)
        e_tab = rho * self.tab.energy(rho, p)
        e = (1 - w) * e_mix + w * e_tab
        idx = xp.nonzero(((w > 0) & (w < 1)).ravel())[0]
        if idx.size:                                 # exact inverse of pressure() in the blend band: Newton,
            r_, p_, a_ = (q.ravel()[idx] for q in (rho, p, a))   # on those cells only
            e0 = e.ravel()[idx]                      # secant from the blended guess and a 0.1% step
            e1 = e0 * 1.001 + 1e-12
            f0 = self.pressure(r_, e0, a_) - p_
            for _ in range(4):
                f1 = self.pressure(r_, e1, a_) - p_
                d = f1 - f0
                e2 = e1 - f1 * (e1 - e0) / xp.where(d != 0, d, 1e-300)
                e0, f0, e1 = e1, f1, xp.where(d != 0, e2, e1)
            e_ = e1
            e = e.copy()
            e.ravel()[idx] = e_
        return e

    def c2(self, rho, p, a):
        from .eos import _dev
        r = _dev(self, "c2", rho, p, a)
        if r is not None:
            return r
        w = self._w(a)
        return (1 - w) * self._mix.c2(rho, p, a) + w * self.tab.c2(rho, p)

    def p_min(self, rho, a):
        return self._mix.p_min(rho, a)

    def p_floor(self, rho, a):
        return self._mix.p_floor(rho, a)

    def describe(self):
        return f"fuel: {self.fuel.describe()}; liner: {self.tab.describe()}; mixed cells: tangent stiffened"


# ------------------------------------------------------------------ opacities

class OpacityTable:
    """kappa_R(rho, T), kappa_P(rho, T) in code units (kappa rho_unit L0) at code temperature theta = T / T1"""

    def __init__(self, data, rho_unit, L0, T1, refine=1):
        rho = data.get("rho_k", data["rho"]) / rho_unit
        T = data.get("T_k", data["T"])
        f = {}
        for k in ("kappa_R", "kappa_P"):
            if k in data:
                f[k] = np.maximum(data[k], 1e-300) * rho_unit * L0
        if "kappa_R" not in f:
            raise ValueError("opacity table needs kappa_R")
        f.setdefault("kappa_P", f["kappa_R"])
        self.g = Grid(rho, T, {**f, "zero": np.ones_like(f["kappa_R"])}, refine)
        self.T1 = T1

    def __call__(self, rho, theta):
        """(kappa_R, kappa_P) per unit mass, code units; clamped to the table edges"""
        lr = xp.clip(xp.log10(xp.maximum(rho, 1e-300)), self.g.LR[0], self.g.LR[-1])
        lt = xp.clip(xp.log10(xp.maximum(theta * self.T1, 1e-300)), self.g.LT[0], self.g.LT[-1])
        ft = (lt - self.g.lt0) / self.g.dlt
        kr, kp, _, _, _ = _lookup(self.g, lr, ft, "kappa_R", "kappa_R", "kappa_P", "zero", 1)
        return self._d("kappa_R", kr), self._d("kappa_P", kp)

    def _d(self, name, v):
        return 10**v if self.g.logs[name] else v


# ------------------------------------------------------------------ open atomic data: ion balance and opacities
# Built from measured or evaluated ionization energies (NIST ASD, tools/atomic_importer.py). Temperatures in eV, density
# in kg/m^3 on the host; tables go to GroupOpacityTable / IonizationTable for the GPU.
#   Ion balance per stage, LTE: Saha with continuum lowering (ion-sphere) and unit partition-function ratios (ground
#   states only); solved for n_e by bisection on charge neutrality.
#   Non-LTE (collisional-radiative steady state): electron-impact ionization by Lotz's formula, radiative recombination
#   2.6e-13 z^2 sqrt(13.6 / T) cm^3/s, three-body recombination by detailed balance with ionization, so the balance
#   reduces to Saha at high density and to a coronal, density-independent one at low density. Dielectronic
#   recombination optional (dr_rates, below); no excitation-autoionization, no radiation field: an optically thin
#   estimate, not FLYCHK.
#   Opacity on a fine photon grid: free-free (Kramers, Gaunt 1.2, with the stage-averaged Z^2), bound-free per subshell
#   (hydrogenic nu^-3 from an edge at the NIST ionization energy of the ion in which that subshell is the outermost),
#   Thomson scattering in the Rosseland mean, and lines: tabulated (NIST, CHIANTI; f values, Boltzmann lower levels)
#   and/or the screened-hydrogenic line model below for every stage without tabulated lines, with Voigt-type profiles.
#   Against Los Alamos TOPS (--test atomic, results/atomic_lines/) Z_bar agrees within about 2% for Al and Cu; with
#   lines the Planck mean comes within about 2x (from 20-150x low without lines), but the Rosseland mean stays low
#   by about 5x (Al) to 25x (Cu): the model has no term structure, so the windows between lines stay open. Treat the
#   Rosseland means as lower bounds and, for Z <= 30, prefer tables imported from TOPS itself (tops_groups_to_table).
_ME_CGS, _E2_CGS, _KB_EV, _AMU = 9.1093837e-28, 2.30707755e-19, 1.602176634e-12, 1.66053907e-27
_SUBSHELLS = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (4, 3), (5, 2), (6, 0),
              (6, 1), (5, 3), (6, 2), (7, 0)]


def _shells_of(N):
    """(n, occupation, electrons up to and including this subshell) of an ion with N bound electrons"""
    out, c = [], 0
    for n, l in _SUBSHELLS:
        if c >= N:
            break
        k = min(2 * (2 * l + 1), N - c)
        c += k
        out.append((n, k, c))
    return out


def fill_ionization(I):
    """ionization energies with gaps filled (log-linear in charge) and made non-decreasing"""
    I = np.array(I, float)
    ok = np.isfinite(I)
    x = np.arange(len(I))
    I = np.exp(np.interp(x, x[ok], np.log(I[ok])))
    return np.maximum.accumulate(I)


def _ion_sphere_lowering(z, ni_cgs):
    """continuum lowering (eV) of an ion of charge z + 1 in the ion sphere of radius (3 / 4 pi n_i)^(1/3)"""
    return 1.5 * (z + 1.0) * 1.43996e-7 / np.maximum((3.0 / (4 * np.pi * ni_cgs))**(1.0 / 3.0), 1e-30)


def _balance_from_ratios(ratio_of_ne, nz, ni, zmax):
    """fractions from n_(z+1) / n_z = ratio(ne), neutrality solved for ne by bisection in log ne"""
    lo, hi = np.full_like(ni, 1e-8) * ni, np.maximum(zmax * ni, 1e-300)

    def fractions(ne):
        lr = np.log(np.maximum(ratio_of_ne(ne), 1e-290))             # (nz - 1, ...)
        lf = np.concatenate([np.zeros((1,) + ne.shape), np.cumsum(lr, axis=0)], axis=0)
        lf -= lf.max(axis=0, keepdims=True)
        f = np.exp(lf)
        return f / f.sum(axis=0, keepdims=True)
    zs = np.arange(nz).reshape((-1,) + (1,) * ni.ndim)
    for _ in range(70):
        mid = np.sqrt(lo * hi)
        f = fractions(mid)
        excess = (f * zs).sum(axis=0) * ni - mid
        lo, hi = np.where(excess > 0, mid, lo), np.where(excess > 0, hi, mid)
    ne = np.sqrt(lo * hi)
    return fractions(ne), ne


def saha_balance(I_eV, A, rho, T_eV):
    """LTE ion fractions (Z + 1, ...), n_e (cm^-3), Z_bar. rho in kg/m^3, T in eV (broadcast together)."""
    I = fill_ionization(I_eV)
    rho, T = np.broadcast_arrays(np.asarray(rho, float), np.asarray(T_eV, float))
    ni = rho * 1e-3 / (A * _AMU * 1e3)
    nz = len(I) + 1
    zz = np.arange(len(I)).reshape((-1,) + (1,) * ni.ndim)
    Ik = I.reshape(zz.shape)
    S0 = 6.0359e21 * T**1.5                          # 2 (2 pi m_e k T / h^2)^(3/2), cm^-3, T in eV

    def ratio(ne):
        return S0[None] / ne[None] * np.exp(-np.maximum(Ik - _ion_sphere_lowering(zz, ni[None]), 0.0) / T[None])
    f, ne = _balance_from_ratios(ratio, nz, ni, len(I))
    return f, ne, ne / ni


def _e1(x):
    """exponential integral E1(x) for x > 0 (Abramowitz & Stegun 5.1.53 / 5.1.56)"""
    x = np.asarray(x, float)
    small = -np.log(np.maximum(x, 1e-300)) - 0.57721566 + x * (0.99999193 + x * (-0.24991055 + x * (0.05519968 + x * (
        -0.00976004 + x * 0.00107857))))
    y = np.maximum(x, 1e-300)
    large = (y**2 + 2.334733 * y + 0.250621) / (y**2 + 3.330657 * y + 1.681534) * np.exp(-np.minimum(y, 700.0)) / y
    return np.where(x < 1.0, small, large)


def lotz_rate(I_eV, z, Z, T_eV):
    """electron-impact ionization rate coefficient (cm^3/s) of the ion of charge z, Lotz (1968), over its subshells"""
    Ifill = fill_ionization(I_eV)
    N = Z - z
    S = 0.0
    for n, k, c in _shells_of(N):
        Ie = Ifill[Z - c]
        S = S + 3.0e-6 * k / (np.sqrt(T_eV) * Ie) * _e1(Ie / T_eV)
    return S


def dr_rates(I_eV, Z, T_eV, dr="burgess", dr_table=None):
    """dielectronic rates (Z, ...) cm^3/s, row z for the recombination z + 1 -> z: 'burgess' (General Formula over the
    model core transitions), 'chianti' (fitted rates in dr_table {recombining charge: parsed .drparams}, Burgess where
    the table has none) or None"""
    T = np.asarray(T_eV, float)
    if dr is None:
        return np.zeros((Z,) + T.shape)
    out = []
    for z in range(Z):
        q = z + 1
        if dr == "chianti" and dr_table and q in dr_table:
            out.append(chianti_dr_rate(dr_table[q], T))
        else:
            out.append(burgess_dr(I_eV, Z, q, T))
    return np.stack([np.broadcast_to(o, T.shape) for o in out])


def cr_balance(I_eV, Z, A, rho, T_eV, dr=None, dr_table=None):
    """collisional-radiative steady-state ion fractions (see the notes above): (fractions, n_e cm^-3, Z_bar). dr: the
    dielectronic recombination model of dr_rates ('burgess', 'chianti' with dr_table, or None)"""
    I = fill_ionization(I_eV)
    rho, T = np.broadcast_arrays(np.asarray(rho, float), np.asarray(T_eV, float))
    ni = rho * 1e-3 / (A * _AMU * 1e3)
    nz = len(I) + 1
    shape = (-1,) + (1,) * ni.ndim
    zz = np.arange(len(I)).reshape(shape)
    Ik = I.reshape(shape)
    S = np.stack([lotz_rate(I_eV, z, Z, T) for z in range(len(I))])
    alpha_r = 2.6e-13 * (zz + 1.0)**2 * np.sqrt(13.6 / T[None]) + dr_rates(I_eV, Z, T, dr, dr_table)
    S0 = 6.0359e21 * T**1.5

    def ratio(ne):
        saha = S0[None] / ne[None] * np.exp(-np.maximum(Ik - _ion_sphere_lowering(zz, ni[None]), 0.0) / T[None])
        alpha_3 = S / np.maximum(saha, 1e-300)           # three-body rate per unit volume, detailed balance
        return S / (alpha_r + alpha_3)
    f, ne = _balance_from_ratios(ratio, nz, ni, len(I))
    return f, ne, ne / ni


def _bf_cross_sections(I, Z, E):
    """sigma_bf (Z, len(E)) cm^2 of each ion stage with bound electrons: hydrogenic nu^-3 per subshell"""
    out = np.zeros((Z, len(E)))
    for z in range(Z):
        for n, k, c in _shells_of(Z - z):
            Ie = I[Z - c]
            Zeff2 = Ie * n * n / 13.6057
            out[z] += k * 7.91e-18 * n / Zeff2 * (Ie / E)**3 * (E >= Ie)
    return out


# ------------------------------------------------------------------ bound-bound lines and dielectronic recombination
# Screened-hydrogenic line model for any element and ion stage, from the ground configuration (aufbau order) and the
# NIST ionization energies: a subshell's binding energy is the ionization energy of the ion in which it is outermost
# (as for the bound-free edges); an empty upper level of principal number m is hydrogenic in the screened charge z + 1.
#   dn >= 1 (n -> m): hydrogenic oscillator strengths, Kramers with the Johnson (1972, ApJ 174, 227) bound-bound Gaunt
#     factor, per electron of the lower subshell and times the vacancy of shell m.
#   dn = 0 (nl -> n l+1, n >= 2): f = (2/3) (dE / Ry) (l + 1) / (2l + 1) R^2 with the hydrogenic radial integral
#     R^2 = (9/4) n^2 (n^2 - (l+1)^2) / Z_eff^2 a0^2.
#   Upper levels whose radius m^2 a0 / (z + 1) exceeds the ion-sphere radius are dissolved (no line).
# Profiles: Doppler (Gaussian) and electron-impact (Lorentzian) widths combined into a pseudo-Voigt (Thompson, Cox and
# Hastings 1987, J. Appl. Cryst. 20, 79), integrated exactly over each photon bin so line strength is conserved on any grid.
# The impact width is Griem's semi-empirical estimate (Phys. Rev. 165, 258, 1968) with an effective Gaunt factor of
# max(0.2, sqrt(3)/(2 pi) ln(kT / dE_neighbour)): an order-of-magnitude width, not a line-shape calculation.
# Missing: fine structure and term splitting (each line here is a whole transition array at one energy), dn = 0 lines
# (their energies would need the subshell energies of the same ion; with the ionization-energy convention above they
# come out 2-5x too high, so dn0=True is off by default), satellites, occupation probabilities. excited_bf=True adds
# photoionisation and lines from hydrogenic excited levels, but with the hard ion-sphere cutoff it puts too much of the
# population in high levels (Planck mean 1.9x -> 5x off TOPS for Al), so it is off by default.
# Measured against TOPS (examples/atomic_vs_tops.py, results/atomic_lines/): with model lines the Planck mean comes from
# 12x (Al) and 57x (Cu) low to within about 2x; the Rosseland mean only from 8x to 4x (Al) and 36x to 19x (Cu) low.
# Tabulated lines do not close the Rosseland gap either (CHIANTI Al, 72k lines: 4.8x), so most of what is missing is
# continuum and broadening physics, not line lists. Nothing here comes within 5% of TOPS.
# Dielectronic recombination (cr_balance dr=...): the Burgess General Formula (verified against ADAS's worked Ti 21+
# values, --test atomic) over valence dn >= 1 core transitions of the model, or CHIANTI's fitted rates for Z <= 30
# (dr="chianti" with dr_table from tools/atomic_importer.fetch_chianti_dr; Al and Cu bundled). Against CHIANTI's
# coronal ion balance the fitted rates improve the mean charge (mean |dZ| 0.32 -> 0.22 for Al, 1.09 -> 0.82 for Cu), but
# the General Formula with model core transitions makes it worse (0.38, 1.38): its core energies are too crude. So the
# default is no DR; use the CHIANTI fits where they exist. Zero-density rates: density suppression is not modelled.
_RY, _A0_CM = 13.605693, 0.529177e-8


def _subshells(N):
    """(n, l, occupation, electrons up to and including this subshell) of the ground configuration with N electrons"""
    out, c = [], 0
    for n, l in _SUBSHELLS:
        if c >= N:
            break
        k = min(2 * (2 * l + 1), N - c)
        c += k
        out.append((n, l, k, c))
    return out


def _johnson_gaunt(n, x):
    if n == 1:
        g0, g1, g2 = 1.1330, -0.4059, 0.07014
    elif n == 2:
        g0, g1, g2 = 1.0785, -0.2319, 0.02947
    else:
        g0 = 0.9935 + 0.2328 / n - 0.1296 / n**2
        g1 = -(0.6282 - 0.5598 / n + 0.5299 / n**2) / n
        g2 = (0.3887 - 1.181 / n + 1.470 / n**2) / n**2
    return g0 + g1 / x + g2 / x**2


def hydrogenic_f(n, m):
    """absorption oscillator strength n -> m of hydrogen averaged over the l states of n: Kramers times the Johnson
    (1972) Gaunt factor (0.4162, 0.0791, 0.6407 exact for 1-2, 1-3, 2-3)"""
    n, m = float(n), float(m)                        # integer powers overflow int32 (the NumPy default on Windows)
    x = 1.0 - (n / m)**2
    return 32 / (3 * np.sqrt(3) * np.pi) / (n**5 * m**3) * (1 / n**2 - 1 / m**2)**-3 * _johnson_gaunt(n, x)


def model_lines(I_eV, Z, z, ni_cgs=None, n_max=10, dn0=False):
    """screened-hydrogenic absorption lines of the ion of charge z from its ground configuration (see the notes above):
    rows (z, E_lower = 0, E_upper eV, f, g = 1, m, n), the NIST line layout plus the upper and lower principal numbers,
    f summed over the electrons of the lower subshell"""
    I = fill_ionization(I_eV)
    N = Z - z
    if N <= 0:
        return np.zeros((0, 7))
    sub = _subshells(N)
    zc = z + 1.0
    occ_n, occ_nl, bind = {}, {}, {}
    for n, l, k, c in sub:
        occ_n[n] = occ_n.get(n, 0) + k
        occ_nl[(n, l)] = k
        bind[(n, l)] = I[Z - c]
    if ni_cgs is not None:                           # dissolved levels: hydrogenic radius beyond the ion sphere
        r_is = (3.0 / (4 * np.pi * ni_cgs))**(1.0 / 3.0) / _A0_CM
        n_max = int(min(n_max, max(np.sqrt(r_is * zc), 1.0)))

    def upper(m):                                    # binding of shell m: its outermost occupied subshell, else hydrogenic
        b = [bind[(n, l)] for n, l, k, c in sub if n == m]
        return min(b) if b else _RY * zc**2 / m**2
    rows = []
    for n, l, k, c in sub:
        B = bind[(n, l)]
        if dn0 and l + 1 < n:                        # dn = 0 (off by default: energies not reliable, see the notes)
            vac = 1.0 - occ_nl.get((n, l + 1), 0) / (2.0 * (2 * l + 3))
            Bu = bind.get((n, l + 1), _RY * zc**2 / n**2)
            dE = B - Bu
            if vac > 0 and dE > 0:
                R2 = 2.25 * n * n * (n * n - (l + 1)**2) / (B * n * n / _RY)
                rows.append((z, 0.0, dE, (2.0 / 3.0) * dE / _RY * (l + 1) / (2 * l + 1) * R2 * k * vac, 1.0, n, n))
        for m in range(n + 1, n_max + 1):           # dn >= 1
            vac = 1.0 - occ_n.get(m, 0) / (2.0 * m * m)
            dE = B - upper(m)
            if vac > 0 and dE > 0:
                rows.append((z, 0.0, dE, k * hydrogenic_f(n, m) * vac, 1.0, m, n))
    return np.array(rows, float).reshape(-1, 7)


def _erf(x):
    """erf to 1.5e-7 (Abramowitz & Stegun 7.1.26)"""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def pseudo_voigt_cdf(x, fwhm_g, fwhm_l):
    """cumulative pseudo-Voigt profile at offset x from line centre (same units as the widths), Thompson-Cox-Hastings"""
    g, lo = np.maximum(fwhm_g, 1e-300), fwhm_l
    G = (g**5 + 2.69269 * g**4 * lo + 2.42843 * g**3 * lo**2 + 4.47163 * g**2 * lo**3 + 0.07842 * g * lo**4 + lo**5)**0.2
    r = lo / G
    eta = 1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3
    cg = 0.5 * (1.0 + _erf(x * (2.0 * np.sqrt(np.log(2.0)) / G)))
    cl = 0.5 + np.arctan(2.0 * x / G) / np.pi
    return eta * cl + (1.0 - eta) * cg


def line_widths(E0, z, m, A, T_eV, ne):
    """(Doppler FWHM, electron-impact FWHM) in eV of a line at E0 (eV) to an upper level of principal number m in an ion
    of charge z (see the notes above)"""
    zc = z + 1.0
    dop = E0 * np.sqrt(8 * np.log(2.0) * T_eV / (A * 931.494e6))
    r2 = m * m * (5.0 * m * m + 1.0 - 6.0) / (2.0 * zc * zc)          # <r^2> / a0^2 of an upper p-like level
    dE_nb = 2.0 * _RY * zc * zc / np.maximum(m, 1.0)**3
    gauntf = np.maximum(0.2, np.sqrt(3.0) / (2 * np.pi) * np.log(np.maximum(T_eV / dE_nb, 1e-30)))
    hw = 3.46e-23 * ne * np.sqrt(_RY / T_eV) * np.maximum(r2, 0.0) * gauntf     # half width, eV (Griem 1968)
    return dop, 2.0 * hw


def deposit_lines(E, E0, strength, fwhm_g, fwhm_l, reach=30.0):
    """absorption (cm^-1) on the photon grid E (log-spaced, eV) of lines at E0 with integrated strengths in cm^-1 Hz: each
    profile is integrated over the bins within reach widths of its centre and renormalised there"""
    edges = np.sqrt(E[1:] * E[:-1])
    edges = np.concatenate([[E[0]**2 / edges[0]], edges, [E[-1]**2 / edges[-1]]])
    out = np.zeros(len(E))
    if not len(E0):
        return out
    k0 = np.clip(np.searchsorted(edges, E0) - 1, 0, len(E) - 1)
    width = np.maximum(fwhm_g, fwhm_l)
    K = np.clip(np.ceil(reach * width / (edges[k0 + 1] - edges[k0])), 1, 512).astype(int)
    for kk in np.unique(2**np.ceil(np.log2(K)).astype(int)):     # group lines by window size
        sel = (2**np.ceil(np.log2(K)).astype(int)) == kk
        off = np.arange(-kk, kk + 1)
        idx = np.clip(k0[sel, None] + off[None], 0, len(E) - 1)
        lo, hi = edges[idx] - E0[sel, None], edges[idx + 1] - E0[sel, None]
        c = pseudo_voigt_cdf(hi, fwhm_g[sel, None], fwhm_l[sel, None]) - pseudo_voigt_cdf(lo, fwhm_g[sel, None], fwhm_l[sel, None])
        c = np.where(np.diff(np.concatenate([idx[:, :1] - 1, idx], axis=1), axis=1) > 0, c, 0.0)   # clipped duplicates
        c /= np.maximum(c.sum(axis=1, keepdims=True), 1e-300)
        np.add.at(out, idx.ravel(), (strength[sel, None] * c).ravel())
    return out / (np.diff(edges) * 2.417989e14)


def burgess_dr(I_eV, Z, q, T_eV, core=None):
    """zero-density dielectronic recombination rate coefficient (cm^3/s) of the ion of charge q into q - 1, Burgess
    General Formula over the dipole core transitions of the recombining ion (model_lines unless core rows are given)"""
    if q >= Z or q < 1:
        return np.zeros_like(np.asarray(T_eV, float))
    if core is None:                                 # core transitions of the valence (outermost) shell only
        core = model_lines(I_eV, Z, q)
        n_out = max(n for n, l, k, c in _subshells(Z - q))
        core = core[(core[:, 6] == n_out) & (core[:, 5] > core[:, 6])]
    T_K = np.asarray(T_eV, float) * 11604.518
    B = np.sqrt(q) * (q + 1)**2.5 / np.sqrt(q * q + 13.4)
    a = 1.0 + 0.015 * q**3 / (q + 1)**2
    out = 0.0
    for row in core:
        El, Eu, f = row[1], row[2], row[3]
        e = (Eu - El) / _RY
        x = e / (q + 1)
        Ax = np.sqrt(x) / (1.0 + 0.105 * x + 0.015 * x * x)
        out = out + f * Ax * np.exp(-e * 157887.5 / (a * T_K))
    return 3.0e-3 * T_K**-1.5 * B * out


def chianti_dr_rate(rows, T_eV):
    """dielectronic rate (cm^3/s) from a parsed CHIANTI .drparams entry: type 1, T^-3/2 sum c_i exp(-E_i / T), or
    type 2, A T^-3/2 exp(-T0 / T) (1 + B exp(-T1 / T)); T in K inside"""
    T = np.asarray(T_eV, float) * 11604.518
    if rows["type"] == 1:
        return T**-1.5 * sum(c * np.exp(-e / T) for e, c in zip(rows["E"], rows["c"]) if c != 0)
    A, B, T0, T1 = rows["c"][:4]
    return A * T**-1.5 * np.exp(-T0 / T) * (1 + B * np.exp(-T1 / T))


def atomic_opacity(I_eV, Z, A, rho, T_eV, fractions, ne, E, lines=None, sigma_bf=None, line_model=False, dn0=False,
                   excited_bf=False):
    """absorption alpha / rho (E grid) and Thomson kappa per unit mass, cm^2/g, at one density and temperature. lines:
    tabulated lines (NIST layout); line_model: add screened-hydrogenic lines for the stages without tabulated ones"""
    I = fill_ionization(I_eV)
    sigma_bf = _bf_cross_sections(I, Z, E) if sigma_bf is None else sigma_bf
    rho_cgs = rho * 1e-3
    ni = rho_cgs / (A * 1.66053907e-24)
    stim = -np.expm1(-E / T_eV)
    zs = np.arange(Z + 1)
    z2 = float(np.sum(fractions * zs**2))
    nu = E * 2.417989e14
    T_K = T_eV * 11604.518
    a_ff = 3.692e8 * 1.2 * stim * z2 * ne * ni / np.sqrt(T_K) / nu**3
    U_ground = np.ones(Z + 1)                        # stage population over its ground-configuration share
    excited_lines = []
    if excited_bf:                                   # photoionisation from excited (hydrogenic, Boltzmann) levels too
        a_bf = np.zeros_like(E)
        for z in range(Z):
            if fractions[z] < 1e-8:
                continue
            m0 = max(n for n, l, k, c in _subshells(Z - z)) + 1
            m_hi = int(max(np.sqrt((3.0 / (4 * np.pi * ni))**(1.0 / 3.0) / _A0_CM * (z + 1.0)), 1.0))
            ms = np.arange(m0, max(m_hi, m0 - 1) + 1)
            Bm = _RY * (z + 1.0)**2 / ms**2
            wm = 2.0 * ms**2 * np.exp(-np.maximum(I[z] - Bm, 0.0) / T_eV)
            U = 1.0 + wm.sum()
            U_ground[z] = U
            sig = sigma_bf[z] / U
            for m, B, w in zip(ms, Bm, wm):
                sig = sig + (w / U) * 7.91e-18 * m / (z + 1.0)**2 * (B / E)**3 * (E >= B)
                for m2, B2 in zip(ms, Bm):           # and its bound-bound lines to higher excited levels
                    if m2 > m:
                        El = max(I[z] - B, 0.0)
                        excited_lines.append((z, El, El + B - B2, hydrogenic_f(m, m2), 2.0 * m * m, m2))
            a_bf += ni * fractions[z] * sig * stim
    else:
        a_bf = ni * (fractions[:Z] @ sigma_bf) * stim
    a = a_ff + a_bf
    tab = np.zeros((0, 5)) if lines is None else np.asarray(lines, float)
    tab = tab[tab[:, 0].astype(int) <= Z] if len(tab) else tab
    have = set(tab[:, 0].astype(int)) if len(tab) else set()
    tab = tab[:, :5]
    if len(tab):                                     # bound lower levels only (lists include autoionising levels)
        tab = tab[tab[:, 1] < I[tab[:, 0].astype(int)]]
    parts, ms = [tab], [np.full(len(tab), 3.0)]     # tabulated lines: upper principal number 3 for the impact width
    if line_model and excited_lines:
        X = np.array(excited_lines, float)
        parts.append(X[:, :5])
        ms.append(X[:, 5])
    if line_model:
        for z in range(Z):
            if z not in have and fractions[z] > 1e-8:
                L = model_lines(I, Z, z, ni, dn0=dn0)
                parts.append(L[:, :5])
                ms.append(L[:, 5])
    L = np.concatenate(parts) if len(parts) > 1 else tab
    if len(L):
        m_up = np.concatenate(ms)
        qi = L[:, 0].astype(int)
        El, Eu, fl, gl = L[:, 1], L[:, 2], L[:, 3], L[:, 4]
        U = U_ground[qi].copy()                      # model lines start in the ground configuration
        if len(tab):
            U[:len(tab)] = _partition_of(qi[:len(tab)], tab, T_eV)
        # lower-level populations: Boltzmann within the stage over the partition function of its listed lower levels
        pop = fractions[qi] * ni * gl / U * np.exp(-El / T_eV)
        dE = Eu - El
        strength = 2.6540e-2 * fl * pop * (-np.expm1(-dE / T_eV))      # pi e^2 / (m c) f n, cm^-1 Hz
        gD, gL = line_widths(dE, qi, m_up, A, T_eV, ne)
        a = a + deposit_lines(E, dE, strength, gD, gL)
    k_es = 0.6652e-24 * ne / rho_cgs
    return a / rho_cgs, k_es


def _partition_of(q, lines, T_eV):
    """partition function of each stage at T: sum of g exp(-E / T) over the distinct lower levels in its line list
    (CHIANTI and NIST list lines from many excited levels; their populations must sum to the stage's, so the ground
    weight alone is not enough). Levels that appear only as upper levels are left out, a small underestimate."""
    out = np.ones(int(lines[:, 0].max()) + 1)
    for z in np.unique(lines[:, 0].astype(int)):
        s = lines[lines[:, 0].astype(int) == z]
        lev = np.unique(np.round(s[:, [1, 4]], 6), axis=0)
        out[z] = max(float(np.sum(lev[:, 1] * np.exp(-lev[:, 0] / T_eV))), 1.0)
    return out[q]


def group_means(k_abs, k_es, E, T_eV, edges):
    """Planck and Rosseland means per photon group on a log grid E (eV), cm^2/g; arrays (G,)"""
    u = E / T_eV
    w = E                                            # d(ln E) measure
    with np.errstate(over="ignore", under="ignore"):
        wP = np.where(u < 500, u**3 / np.maximum(np.expm1(np.minimum(u, 500)), 1e-300), 0.0)
        wR = np.where(u < 500, u**4 * np.exp(np.minimum(u, 500)) / np.maximum(np.expm1(np.minimum(u, 500)), 1e-300)**2, 0.0)
    G = len(edges) - 1
    kP, kR = np.zeros(G), np.zeros(G)
    for g in range(G):
        m = (E >= edges[g]) & (E < edges[g + 1])
        if not m.any():
            continue
        dP, dR = (wP * w)[m].sum(), (wR * w)[m].sum()
        kP[g] = (wP * w * k_abs)[m].sum() / dP if dP > 0 else float(k_abs[m][0])
        kR[g] = dR / (wR * w / (k_abs + k_es))[m].sum() if dR > 0 else float(k_abs[m][0] + k_es)
    return kP, kR


def build_atomic_table(Z, A, I_eV, edges_eV, rho=None, T_eV=None, lines=None, nonlte=False, n_E=6000, line_model=True,
                       dr=None, dr_table=None, line_source="NIST"):
    """multigroup opacity and ionization table from NIST-type atomic data, in the format of fld_mg_table: rho (kg/m^3),
    T (K), kappa_R_g, kappa_P_g (G, nr, nt) in m^2/kg, group_edges_eV, plus Zbar_lte, Zbar_nlte (nr, nt). lines:
    tabulated lines (NIST layout, named by line_source); line_model: screened-hydrogenic lines for the other stages;
    dr, dr_table: dielectronic recombination in the non-LTE balance (dr_rates)"""
    rho = np.geomspace(1e-3, 1e4, 15) if rho is None else np.asarray(rho, float)
    T_eV = np.geomspace(1.0, 3e3, 36) if T_eV is None else np.asarray(T_eV, float)
    E = np.geomspace(0.05, 2e5, n_E)
    edges = np.asarray(edges_eV, float)
    G = len(edges) - 1
    I = fill_ionization(I_eV)
    sbf = _bf_cross_sections(I, Z, E)
    R, TT = np.meshgrid(rho, T_eV, indexing="ij")
    f_l, ne_l, zb_l = saha_balance(I, A, R, TT)
    f_n, ne_n, zb_n = cr_balance(I, Z, A, R, TT, dr=dr, dr_table=dr_table)
    f, ne = (f_n, ne_n) if nonlte else (f_l, ne_l)
    kR, kP = np.zeros((G, len(rho), len(T_eV))), np.zeros((G, len(rho), len(T_eV)))
    for i in range(len(rho)):
        for j in range(len(T_eV)):
            ka, ke = atomic_opacity(I, Z, A, rho[i], T_eV[j], f[:, i, j], ne[i, j], E, lines, sbf, line_model=line_model)
            kP[:, i, j], kR[:, i, j] = group_means(ka, ke, E, T_eV[j], edges)
    src = ("NIST ASD ionization energies" + (f"; {line_source} lines" if lines is not None else "")
           + ("; screened-hydrogenic model lines" if line_model else "") + "; "
           + ("collisional-radiative" + (f" with {dr} dielectronic recombination" if dr else "") if nonlte else "Saha")
           + " ion balance; screened-hydrogenic bound-free, Kramers free-free")
    return dict(rho=rho, T=T_eV * 11604.518, kappa_R_g=np.maximum(kR, 1e-300) * 0.1, kappa_P_g=np.maximum(kP, 1e-300) * 0.1,
                group_edges_eV=edges, Zbar_lte=zb_l, Zbar_nlte=zb_n, Z=Z, A=A, source=src)


def tops_groups_to_table(rows, edges_eV, n_rho=None, n_T=None):
    """a table in the fld_mg_table format from parsed TOPS group output (tools/atomic_importer.parse_tops_groups),
    resampled log-linearly onto a log-uniform rho x T grid for GroupOpacityTable"""
    T0, r0 = np.array(sorted({x["T_keV"] for x in rows})), np.array(sorted({x["rho"] for x in rows}))
    G = len(rows[0]["kR_g"])
    q = {k: np.zeros((len(r0), len(T0))) for k in ("Z",)}
    kR, kP = np.zeros((G, len(r0), len(T0))), np.zeros((G, len(r0), len(T0)))
    for x in rows:
        i, j = np.searchsorted(r0, x["rho"]), np.searchsorted(T0, x["T_keV"])
        kR[:, i, j], kP[:, i, j], q["Z"][i, j] = x["kR_g"], x["kP_g"], x["Zfree"]
    nr, nt = n_rho or len(r0), n_T or 2 * len(T0)
    lr0, lt0 = np.log10(r0), np.log10(T0)
    lr, lt = np.linspace(lr0[0], lr0[-1], nr), np.linspace(lt0[0], lt0[-1], nt)

    def resample(f):
        a = np.stack([np.interp(lt, lt0, f[i]) for i in range(len(r0))])
        return np.stack([np.interp(lr, lr0, a[:, j]) for j in range(nt)], axis=1)
    fR = np.stack([resample(np.log10(np.maximum(kR[g], 1e-300))) for g in range(G)])
    fP = np.stack([resample(np.log10(np.maximum(kP[g], 1e-300))) for g in range(G)])
    # TOPS: cm^2/g, g/cm^3, keV -> m^2/kg, kg/m^3, K
    return dict(rho=10**lr * 1e3, T=10**lt * 1e3 * 11604.518, kappa_R_g=10**fR * 0.1, kappa_P_g=10**fP * 0.1,
                group_edges_eV=np.asarray(edges_eV, float), Zbar_lte=resample(q["Z"]),
                source="Los Alamos TOPS multigroup opacities (Magee et al., ASP Conf. Ser. 78, 51, 1995)")


_GROUP_SRC = r"""
extern "C" __global__ void group_lookup(const double* __restrict__ rho, const double* __restrict__ th,
                                        const double* __restrict__ KR, const double* __restrict__ KP,
                                        double* __restrict__ oR, double* __restrict__ oP, const long N, const int G,
                                        const int nr, const int nt, const double lr0, const double dlr, const double lt0,
                                        const double dlt, const double T1) {
    long c = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= N) return;
    double fr = (log10(fmax(rho[c], 1e-300)) - lr0) / dlr, ft = (log10(fmax(th[c] * T1, 1e-300)) - lt0) / dlt;
    fr = fmin(fmax(fr, 0.0), nr - 1 - 1e-9);
    ft = fmin(fmax(ft, 0.0), nt - 1 - 1e-9);
    long i = (long)fr, j = (long)ft;
    double a = fr - i, b = ft - j;
    double w00 = (1 - a) * (1 - b), w10 = a * (1 - b), w01 = (1 - a) * b, w11 = a * b;
    for (long g = 0; g < G; g++) {
        long o = g * nr * nt + i * nt + j;
        oR[g * N + c] = pow(10.0, w00 * KR[o] + w10 * KR[o + nt] + w01 * KR[o + 1] + w11 * KR[o + nt + 1]);
        oP[g * N + c] = pow(10.0, w00 * KP[o] + w10 * KP[o + nt] + w01 * KP[o + 1] + w11 * KP[o + nt + 1]);
    }
}
"""
_GROUP_K = []


def _group_kernel():
    if not _GROUP_K:
        import cupy as cp
        _GROUP_K.append(cp.RawKernel(_GROUP_SRC, "group_lookup"))
    return _GROUP_K[0]


class GroupOpacityTable:
    """every group's (kappa_P, kappa_R) at once from one device array: bilinear in (log rho, log T), clamped to the
    table, in code units as OpacityTable. One vectorised lookup instead of one table lookup per group."""

    def __init__(self, data, rho_unit, L0, T1):
        rho, T = np.asarray(data["rho"], float) / rho_unit, np.asarray(data["T"], float)
        self.lr, self.lt = np.log10(rho), np.log10(T)
        self.nr, self.nt = len(rho), len(T)
        if not (np.allclose(np.diff(self.lr), self.lr[1] - self.lr[0]) and np.allclose(np.diff(self.lt), self.lt[1] - self.lt[0])):
            raise ValueError("GroupOpacityTable needs log-uniform rho and T")
        self.dlr, self.dlt = self.lr[1] - self.lr[0], self.lt[1] - self.lt[0]
        s = rho_unit * L0
        self.kR = xp.ascontiguousarray(xp.asarray(np.log10(np.maximum(data["kappa_R_g"], 1e-300) * s)))
        self.kP = xp.ascontiguousarray(xp.asarray(np.log10(np.maximum(data["kappa_P_g"], 1e-300) * s)))
        self.T1 = T1

    def __call__(self, rho, theta):
        """(kappa_R, kappa_P), arrays (G, ...), one kernel launch"""
        rho, theta = xp.ascontiguousarray(rho), xp.ascontiguousarray(theta)
        G, N = self.kR.shape[0], rho.size
        oR, oP = xp.empty((G,) + rho.shape), xp.empty((G,) + rho.shape)
        _group_kernel()(((N + 255) // 256,), (256,),
                        (rho, theta, self.kR, self.kP, oR, oP, np.int64(N), np.int32(G), np.int32(self.nr), np.int32(self.nt),
                         float(self.lr[0]), float(self.dlr), float(self.lt[0]), float(self.dlt), float(self.T1)))
        return oR, oP


class IonizationTable:
    """Z_bar(rho, T) from a table of build_atomic_table on the device: bilinear in (log rho, log T), clamped"""

    def __init__(self, data, rho_unit, T1, key="Zbar_lte"):
        self._g = GroupOpacityTable(dict(rho=data["rho"], T=data["T"], kappa_R_g=np.asarray(data[key])[None],
                                         kappa_P_g=np.asarray(data[key])[None]), rho_unit, 1.0 / rho_unit, T1)

    def __call__(self, rho, theta):
        return self._g(rho, theta)[0][0]


# ------------------------------------------------------------------ synthetic tables (tests, examples)

def mie_gruneisen_table(rho0, c0, s, gamma0, cv, rho_range, T_range, nr=80, nt=80, kappa=None):
    """SI table of a Mie-Gruneisen solid with a constant heat capacity cv (J/kg/K): e = e_c(rho) + cv (T - T0),
    p = p_c(rho) + rho0 gamma0 cv (T - T0), cold curve from the linear Us-Up Hugoniot reference as in eos.py
    (T0 = 300 K). kappa: optional (kR0, kP0, m, n) power laws in SI. For tests and as a format example."""
    from .eos import Material
    m = Material("mie_gruneisen", rho0=rho0, c0=c0, s=s, gamma0=gamma0)
    rho = np.geomspace(*rho_range, nr)
    T = np.geomspace(*T_range, nt)
    phi1, phi2, _, _ = m.phis(rho, np)               # eps = phi1 p + phi2 on the reference (T0) state
    R, TT = np.meshgrid(rho, T, indexing="ij")
    T0 = 300.0
    p_ref = np.zeros_like(rho)                       # reference: the MG curve at p such that eps(p) = phi2 + ...
    e_ref = (phi1 * p_ref + phi2) / rho
    e = e_ref[:, None] + cv * (TT - T0)
    p = (R * e - phi2[:, None]) / phi1[:, None]      # MG: p = (eps - phi2) / phi1
    out = {"rho": rho, "T": T, "p": p, "e": e}
    if kappa is not None:
        kR0, kP0, mm, nn = kappa
        out["kappa_R"] = kR0 * R**mm * TT**nn
        out["kappa_P"] = kP0 * R**mm * TT**nn
    return out


def ideal_gas_table(gamma, R_gas, rho_range, T_range, nr=60, nt=60):
    """SI table of an ideal gas: p = rho R T, e = R T / (gamma - 1)"""
    rho = np.geomspace(*rho_range, nr)
    T = np.geomspace(*T_range, nt)
    Rr, TT = np.meshgrid(rho, T, indexing="ij")
    return {"rho": rho, "T": T, "p": Rr * R_gas * TT, "e": R_gas * TT / (gamma - 1) + 0 * Rr}
