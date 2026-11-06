# validation.py - comparing simulated observables with shot data: traces, goodness of fit, calibration.
#
# A Trace is a measured signal y(t) with 1-sigma uncertainties (VISAR free-surface velocity, a B-dot probe
# voltage, radiography radii, ...). compare() samples a simulated signal at the data times and returns
#     RMS error, RMS / peak |y|, chi^2 = sum ((y - f) / sigma)^2, chi^2_nu = chi^2 / (N - n_params),
#     peak offset = delay of the simulation against the data that maximises their cross-correlation,
# optionally after fitting a timing shift (fiducial uncertainty) that minimises chi^2. calibrate() fits a
# parabola to chi^2 over a scan of one model parameter: the minimum is the calibrated value and its 1-sigma
# range is where chi^2 rises by 1 above it (the usual Delta chi^2 = 1 rule, valid when the model is close to
# linear in the parameter there and the sigmas are right). Observables for a pulsed-power liner (FCG with the
# RLC drive) are in observables_from_fcg(). load_digitized() reads provenance-checked traces digitized from the
# literature (benchmarks/literature/), and BdotProbe / VisarProbe / FramingCamera are virtual diagnostics at
# physical probe locations inside a running solver.
import csv
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    name: str
    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    units: str = ""
    kind: str = ""                                   # visar, bdot, radiography, ...
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t, self.y = np.asarray(self.t, float), np.asarray(self.y, float)
        s = np.asarray(self.sigma, float)
        self.sigma = np.broadcast_to(s, self.y.shape).copy()


def load_trace(path, name=None, kind="", units="", sigma=None, t_col=0, y_col=1, s_col=2, t_scale=1.0, y_scale=1.0):
    """CSV with a header row: time, value[, sigma]. Without a sigma column: `sigma` (absolute, or a string like
    '3%' of the peak |value|)."""
    rows = [r for r in csv.reader(open(path)) if r and not r[0].lstrip().startswith("#")]
    head, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
    t, y = data[:, t_col] * t_scale, data[:, y_col] * y_scale
    if data.shape[1] > s_col:
        s = data[:, s_col] * y_scale
    elif isinstance(sigma, str) and sigma.endswith("%"):
        s = float(sigma[:-1]) / 100 * np.max(np.abs(y))
    elif sigma is not None:
        s = float(sigma)
    else:
        raise ValueError(f"{path}: no sigma column and no sigma given")
    return Trace(name or head[y_col].strip(), t, y, s, units, kind)


def save_trace(path, tr):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        fh.write(f"# {tr.name} ({tr.kind}), {tr.units}" + (f"; {tr.meta}" if tr.meta else "") + "\n")
        w.writerow(["t", tr.name, "sigma"])
        w.writerows(zip(tr.t, tr.y, tr.sigma))


def sample(t_sim, y_sim, t, shift=0.0):
    """simulated signal at the data times (linear interpolation; NaN outside the simulated span)"""
    return np.interp(t - shift, t_sim, y_sim, left=np.nan, right=np.nan)


def _peak_time(t, y):
    """time of the largest |y|, refined by a parabola through the sample and its neighbours"""
    k = int(np.nanargmax(np.abs(y)))
    if 0 < k < len(t) - 1:
        a, b, c = np.abs(y[k - 1:k + 2])
        den = a - 2 * b + c
        if den != 0:
            return t[k] + 0.5 * (a - c) / den * 0.5 * (t[k + 1] - t[k - 1])
    return t[k]


def compare(tr, t_sim, y_sim, n_params=0, fit_shift=False, max_shift=None, shift_sigma=None):
    """goodness of fit of a simulated signal to a trace (see module notes); with fit_shift the timing shift
    that minimises chi^2 (grid then golden section within +-max_shift) is applied first and counts as one
    parameter; shift_sigma (the timing-fiducial uncertainty) adds the prior (shift / shift_sigma)^2, reported
    as chi2_prior and included in chi2_total"""
    def chi2(shift, prior=True):
        f = sample(t_sim, y_sim, tr.t, shift)
        ok = np.isfinite(f)
        c = float(np.sum(((tr.y[ok] - f[ok]) / tr.sigma[ok])**2))
        if prior and shift_sigma:
            c += (shift / shift_sigma)**2
        return c, ok
    shift = 0.0
    if fit_shift:
        L = max_shift if max_shift is not None else 0.1 * (tr.t[-1] - tr.t[0])
        grid = np.linspace(-L, L, 121)               # dense search first: chi^2(shift) can be multimodal
        k = int(np.argmin([chi2(x)[0] for x in grid]))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, 120)]
        g = (np.sqrt(5) - 1) / 2
        x1, x2 = b - g * (b - a), a + g * (b - a)
        f1, f2 = chi2(x1)[0], chi2(x2)[0]
        for _ in range(60):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - g * (b - a); f1 = chi2(x1)[0]
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + g * (b - a); f2 = chi2(x2)[0]
        shift = 0.5 * (a + b)
        n_params += 1
    c2, ok = chi2(shift, prior=False)
    cp = (shift / shift_sigma)**2 if (fit_shift and shift_sigma) else 0.0
    f = sample(t_sim, y_sim, tr.t, shift)
    r = tr.y[ok] - f[ok]
    n = int(ok.sum())
    dof = max(n - n_params, 1)
    tp_sim = _peak_time(t_sim + shift, y_sim)
    return {"name": tr.name, "n": n, "rms": float(np.sqrt(np.mean(r**2))),
            "nrms": float(np.sqrt(np.mean(r**2)) / max(np.max(np.abs(tr.y)), 1e-300)),
            "chi2": c2, "chi2_prior": cp, "chi2_total": c2 + cp, "dof": dof, "chi2_nu": c2 / dof, "shift": shift,
            "peak_offset": xcorr_lag(tr, t_sim, y_sim, max_shift if max_shift is not None else 0.1 * np.ptp(tr.t)),
            "peak_time_offset": float(tp_sim - _peak_time(tr.t, tr.y)), "coverage": n / len(tr.t)}


def xcorr_lag(tr, t_sim, y_sim, max_lag, n=401):
    """peak alignment offset: the delay of the simulation relative to the data that maximises their
    cross-correlation over the data window (parabolic refinement); robust to noise and to oscillating
    signals, unlike the time of the largest sample"""
    lags = np.linspace(-max_lag, max_lag, n)
    C = []
    for L in lags:
        f = sample(t_sim, y_sim, tr.t + L)
        ok = np.isfinite(f)
        C.append(np.sum(tr.y[ok] * f[ok]) / max(np.sqrt(np.sum(f[ok]**2) * np.sum(tr.y[ok]**2)), 1e-300))
    C = np.array(C)
    k = int(np.argmax(C))
    if 0 < k < n - 1:
        a, b, c = C[k - 1:k + 2]
        den = a - 2 * b + c
        if den != 0:
            return float(lags[k] + 0.5 * (a - c) / den * (lags[1] - lags[0]))
    return float(lags[k])


def calibrate(values, chi2s):
    """parabola through chi^2(parameter) around its minimum: (best, sigma, chi2_min); sigma from
    chi^2(best +- sigma) = chi2_min + 1. Uses the lowest point and its neighbours (all points if <= 3)."""
    v, c = np.asarray(values, float), np.asarray(chi2s, float)
    order = np.argsort(v)
    v, c = v[order], c[order]
    k = int(np.argmin(c))
    lo, hi = max(k - 1, 0), min(k + 2, len(v))
    if hi - lo < 3:
        lo, hi = max(0, hi - 3), max(3, hi)
    A, B, C = np.polyfit(v[lo:hi], c[lo:hi], 2)
    if A <= 0:                                       # no curvature: the minimum is at the scan edge
        return float(v[k]), float("inf"), float(c[k])
    best = -B / (2 * A)
    return float(best), float(1 / np.sqrt(A)), float(C - B * B / (4 * A))


def calibrate_valley(values, chi2s, dof, window=None):
    """least-squares parabola through every point of the chi^2 valley (chi^2 <= chi2_min + window, default
    max(100, chi2_min)): robust where chi^2 is jagged on small parameter steps (numerical noise of the model).
    Returns (best, sigma, chi2_min_fit, scale): sigma from Delta chi^2 = 1 on the parabola, multiplied by
    scale = sqrt(chi2_min_fit / dof) when that exceeds 1 (the fit is worse than the stated errors)."""
    v, c = np.asarray(values, float), np.asarray(chi2s, float)
    cmin = c.min()
    w = window if window is not None else max(100.0, cmin)
    sel = c <= cmin + w
    if sel.sum() < 3:
        sel = np.argsort(c)[:3]
    A, B, C = np.polyfit(v[sel], c[sel], 2)
    if A <= 0:
        k = int(np.argmin(c))
        return float(v[k]), float("inf"), float(cmin), 1.0
    best = -B / (2 * A)
    cfit = C - B * B / (4 * A)
    scale = max(1.0, np.sqrt(max(cfit, 0.0) / max(dof, 1)))
    return float(best), float(scale / np.sqrt(A)), float(cfit), float(scale)


def band(curves, t):
    """ensemble band on a common time grid: (min, max, mean) over curves = [(t_i, y_i), ...]"""
    Y = np.array([np.interp(t, ti, yi, left=np.nan, right=np.nan) for ti, yi in curves])
    return np.nanmin(Y, axis=0), np.nanmax(Y, axis=0), np.nanmean(Y, axis=0)


# ------------------------------------------------------------------ pulsed-power liner observables

def instrument(t, y, fwhm):
    """signal seen through a Gaussian instrument response of the given FWHM (same time units): resampled onto
    a uniform grid, convolved, returned on that grid"""
    if fwhm <= 0:
        return t, y
    dt = min(np.median(np.diff(t)), fwhm / 10)
    tu = np.arange(t[0], t[-1], dt)
    yu = np.interp(tu, t, y)
    s = fwhm / 2.3548 / dt
    k = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
    w = np.exp(-0.5 * (k / s)**2)
    w /= w.sum()
    pad = len(k) // 2
    yp = np.concatenate([np.full(pad, yu[0]), yu, np.full(pad, yu[-1])])
    return tu, np.convolve(yp, w, mode="valid")


def observables_from_fcg(d, probe_r_mm=20.0, probe_area_mm2=1.0, fwhm_us=0.01):
    """simulated diagnostics from an FCG run with the RLC drive (columns of its diagnostics.csv):
    visar: free-surface velocity of the liner (inward positive), v_r at its outer surface when the run
    logged v_surface_kms, else -d r_out/dt (km/s); bdot: a one-turn probe of area A at radius
    r_p in the return gap sees B_theta = mu0 I / (2 pi r_p), so V = A mu0 / (2 pi r_p) dI/dt (V);
    radiography: the cavity radius (mm). Times in us. Each goes through a Gaussian instrument response of
    FWHM fwhm_us (10 ns), the instruments' bandwidth, which also damps the step-to-step noise of the logged
    signals (differences between runs at that level would otherwise swamp chi^2)."""
    mu0 = 4e-7 * np.pi
    t = np.asarray(d["t_us"], float)
    k = np.unique(t, return_index=True)[1]
    t = t[k]
    I = np.asarray(d["I_MA"], float)[k] * 1e6
    r_out = np.asarray(d["r_out_mm"], float)[k]
    dIdt = np.gradient(I, t * 1e-6)
    vis = -np.asarray(d["v_surface_kms"], float)[k] if "v_surface_kms" in d else -np.gradient(r_out, t)
    raw = {"visar": vis, "bdot": probe_area_mm2 * 1e-6 * mu0 / (2 * np.pi * probe_r_mm * 1e-3) * dIdt,
           "radiography": np.asarray(d["R_cavity_mm"], float)[k]}
    return {key: instrument(t, y, fwhm_us) for key, y in raw.items()}


# ============================================================
# digitized literature traces: provenance-checked CSV / HDF5
# ============================================================
# A CSV starts with "# key: value" lines, then a header row and x, y[, sigma] rows. Required: name, kind, units,
# provenance. provenance is digitized (read off a published figure), measured (published or facility numbers) or
# synthetic (made by a simulation or by hand; never evidence about an experiment). digitized and measured traces
# must name their source (citation) and figure or table; goodness-of-fit output carries the provenance, so a
# synthetic row cannot pass for a comparison with an experiment.
PROVENANCE = ("digitized", "measured", "synthetic")
_REQUIRED = ("name", "kind", "units", "provenance")


def _check_meta(meta, path):
    miss = [k for k in _REQUIRED if not meta.get(k)]
    if miss:
        raise ValueError(f"{path}: missing header fields {miss}")
    if meta["provenance"] not in PROVENANCE:
        raise ValueError(f"{path}: provenance must be one of {PROVENANCE}, not {meta['provenance']!r}")
    if meta["provenance"] != "synthetic" and not (meta.get("citation") and meta.get("figure")):
        raise ValueError(f"{path}: {meta['provenance']} data must give citation and figure (or table)")


def _finish_trace(meta, t, y, s, path):
    _check_meta(meta, path)
    t, y = np.asarray(t, float), np.asarray(y, float)
    if s is None:
        if not meta.get("sigma"):
            raise ValueError(f"{path}: no sigma column and no sigma header (absolute, or a percentage of peak |y|)")
        v = str(meta["sigma"]).strip()
        s = float(v[:-1]) / 100 * np.max(np.abs(y)) if v.endswith("%") else float(v)
    s = np.broadcast_to(np.asarray(s, float), y.shape).copy()
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y)) and np.all(s > 0)):
        raise ValueError(f"{path}: non-finite values or non-positive sigma")
    o = np.argsort(t, kind="stable")                  # digitizers export points in click order
    t, y, s = t[o], y[o], s[o]
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: repeated time values")
    ts, ys = float(meta.get("t_scale", 1.0)), float(meta.get("y_scale", 1.0))
    return Trace(meta["name"], t * ts, y * ys, s * ys, meta["units"], meta["kind"], dict(meta, path=str(path)))


def load_digitized(path):
    """one trace from a CSV with a provenance header (see above), or every trace of an HDF5 file (a list); an HDF5
    group holds datasets t, y and optionally sigma, its attributes are the header fields"""
    if str(path).endswith((".h5", ".hdf5")):
        import h5py
        out = []
        with h5py.File(path, "r") as f:
            for key in f:
                g = f[key]
                meta = {k: (v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v))
                        for k, v in g.attrs.items()}
                meta.setdefault("name", key)
                out.append(_finish_trace(meta, g["t"][()], g["y"][()], g["sigma"][()] if "sigma" in g else None,
                                         f"{path}:{key}"))
        return out
    meta, rows = {}, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if ":" in line:
                k, v = line[1:].split(":", 1)
                meta[k.strip()] = v.strip()
            continue
        rows.append(line)
    data = np.array([[float(x) for x in r.split(",")] for r in rows[1:]], ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: need x, y columns")
    return _finish_trace(meta, data[:, 0], data[:, 1], data[:, 2] if data.shape[1] > 2 else None, path)


def save_digitized(path, tr):
    """write a Trace with its meta as a provenance-header CSV"""
    meta = dict(tr.meta, name=tr.name, kind=tr.kind, units=tr.units)
    meta.pop("path", None)
    _check_meta(meta, path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        fh.write("t,y,sigma\n")
        for row in zip(tr.t, tr.y, tr.sigma):
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")


def scan_library(root):
    """every provenance-header trace under a directory: {name: Trace}; files that fail the checks are reported in
    the second return value instead of being used"""
    import os
    good, bad = {}, {}
    for d, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith((".csv", ".h5", ".hdf5")):
                continue
            p = os.path.join(d, fn)
            try:
                got = load_digitized(p)
                for tr in (got if isinstance(got, list) else [got]):
                    good[tr.name] = tr
            except (ValueError, OSError) as e:
                bad[p] = str(e)
    return good, bad


# ============================================================
# synthetic diagnostics at physical probe locations
# ============================================================
def _axes_of(s):
    """interior cell-centre coordinates of a solver (2D or 3D)"""
    get = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    ng = s.ng
    ax = [get(s.x)[ng:-ng], get(s.y)[ng:-ng]]
    if hasattr(s, "nz_tot"):
        ax.append(get(s.z)[ng:-ng])
    return ax


def _interp_point(f, ax, pos):
    """multilinear value of an interior cell array at a point (held constant beyond the outermost cell centres)"""
    idx, wt = [], []
    for a, p in zip(ax, pos):
        u = float(np.clip((p - a[0]) / (a[1] - a[0]), 0.0, len(a) - 1 - 1e-12))
        i = int(u)
        idx.append(i)
        wt.append(u - i)
    v = 0.0
    for c in range(2**len(ax)):
        w, id_ = 1.0, []
        for d in range(len(ax)):
            b = (c >> d) & 1
            w *= wt[d] if b else 1.0 - wt[d]
            id_.append(idx[d] + b)
        if w:
            v += w * float(f[tuple(id_)])
    return v


def _prim_interior(s):
    from .physics import cons_to_prim
    I = (slice(None),) + (slice(s.ng, -s.ng),) * (s.U.ndim - 1)
    return cons_to_prim(s.U, s.eos)[I]


class BdotProbe:
    """virtual B-dot coil at a point: B_c(t) from the cell field, V = -n_turns A dB_c/dt from the recorded history
    (finite differences of the solver's own steps). component: index 0, 1, 2 of the B components in solver order"""
    kind = "bdot"

    def __init__(self, name, pos, component, area=1.0, n_turns=1):
        self.name, self.pos, self.c, self.area, self.n = name, tuple(pos), component, area, n_turns
        self.t, self.B = [], []

    def record(self, s, W=None, ax=None):
        from .config import iBX
        W = _prim_interior(s) if W is None else W
        ax = _axes_of(s) if ax is None else ax
        self.t.append(float(s.t))
        self.B.append(_interp_point(W[iBX + self.c], ax, self.pos))

    def signal(self):
        t = np.asarray(self.t)
        return t, -self.n * self.area * np.gradient(np.asarray(self.B), t)


class VisarProbe:
    """VISAR line cut: along axis `axis` from `start` toward `direction` (+1 / -1), the first crossing of the
    material tracer through `threshold` (or of density, with field="rho") is the reflecting surface; the signal is
    the velocity along the line at it, positive along `direction`"""
    kind = "visar"

    def __init__(self, name, start, axis, direction=1, threshold=0.5, field="tracer"):
        self.name, self.start, self.axis, self.dir = name, tuple(start), axis, direction
        self.thr, self.field = threshold, field
        self.t, self.v, self.pos = [], [], []

    def surface(self, s, W=None, ax=None):
        """(position, velocity along direction) of the first crossing; NaNs if there is none"""
        from .config import iCLR, iRHO, iVX
        W = _prim_interior(s) if W is None else W
        ax = _axes_of(s) if ax is None else ax
        idx = tuple(slice(None) if d == self.axis else int(np.argmin(np.abs(a - self.start[d])))
                    for d, a in enumerate(ax))
        get = lambda q: q.get() if hasattr(q, "get") else np.asarray(q)
        x = ax[self.axis]
        f = get(W[iCLR if self.field == "tracer" else iRHO][idx])
        v = get(W[iVX + self.axis][idx])
        i0 = int(np.argmin(np.abs(x - self.start[self.axis])))
        order = np.arange(i0, len(x)) if self.dir > 0 else np.arange(i0, -1, -1)
        fo = f[order]
        hit = np.nonzero(fo >= self.thr)[0]
        if hit.size == 0 or hit[0] == 0:
            return np.nan, np.nan
        k = hit[0]
        a = (self.thr - fo[k - 1]) / max(fo[k] - fo[k - 1], 1e-300)
        i, j = order[k - 1], order[k]
        return float(x[i] + a * (x[j] - x[i])), float(self.dir * (v[i] + a * (v[j] - v[i])))

    def record(self, s, W=None, ax=None):
        p, v = self.surface(s, W, ax)
        self.t.append(float(s.t))
        self.pos.append(p)
        self.v.append(v)

    def signal(self):
        return np.asarray(self.t), np.asarray(self.v)


class FramingCamera(VisarProbe):
    """radial framing camera: the position of the same surface along the line cut (e.g. the inner liner radius on a
    slice); average_over: an axis to average the radius over (several lines, like a slit along that axis)"""
    kind = "framing"

    def __init__(self, name, start, axis, direction=1, threshold=0.5, field="tracer", average_over=None):
        super().__init__(name, start, axis, direction, threshold, field)
        self.avg = average_over

    def record(self, s, W=None, ax=None):
        W = _prim_interior(s) if W is None else W
        ax = _axes_of(s) if ax is None else ax
        self.t.append(float(s.t))
        if self.avg is None:
            self.pos.append(self.surface(s, W, ax)[0])
            return
        st0, ps = self.start, []
        for q in ax[self.avg]:
            st = list(st0)
            st[self.avg] = float(q)
            self.start = tuple(st)
            ps.append(self.surface(s, W, ax)[0])
        self.start = st0
        self.pos.append(float(np.nanmean(ps)))

    def signal(self):
        return np.asarray(self.t), np.asarray(self.pos)


class DiagnosticSet:
    """a group of virtual probes, recorded together: record(solver) after a step, signals() -> {name: (t, y)}"""

    def __init__(self, probes):
        self.probes = {p.name: p for p in probes}

    def record(self, s):
        if getattr(self, "_t_last", None) == float(s.t):   # one sample per time (dB/dt differences its history)
            return
        self._t_last = float(s.t)
        W, ax = _prim_interior(s), _axes_of(s)
        for p in self.probes.values():
            p.record(s, W, ax)

    def signals(self):
        return {k: p.signal() for k, p in self.probes.items()}


def goodness_matrix(traces, runs, n_params=0):
    """chi^2_nu and RMS of every run against every trace it has a signal for. traces: {name: Trace}, runs: {run
    name: {trace name: (t, y)}}. Returns {run: {trace: compare() dict + provenance}}"""
    M = {}
    for run, sig in runs.items():
        M[run] = {}
        for name, tr in traces.items():
            if name in sig:
                ok = np.isfinite(sig[name][1])
                c = compare(tr, sig[name][0][ok], sig[name][1][ok], n_params=n_params)
                c["provenance"] = tr.meta.get("provenance", "unspecified")
                M[run][name] = c
    return M


def format_matrix(M, key="chi2_nu"):
    """a fixed-width text table of one metric; synthetic columns are marked [syn], so they read as self-tests"""
    runs = list(M)
    cols = []
    for r in runs:
        cols += [c for c in M[r] if c not in cols]
    prov = {c: next(M[r][c]["provenance"] for r in runs if c in M[r]) for c in cols}
    head = [f"{c}{' [syn]' if prov[c] == 'synthetic' else ''}" for c in cols]
    w0 = max([len(r) for r in runs] + [len(key)])
    w = [max(len(h), 10) for h in head]
    lines = [key.ljust(w0) + "  " + "  ".join(h.rjust(x) for h, x in zip(head, w))]
    for r in runs:
        cells = [(f"{M[r][c][key]:.3g}" if c in M[r] else "-").rjust(x) for c, x in zip(cols, w)]
        lines.append(r.ljust(w0) + "  " + "  ".join(cells))
    return "\n".join(lines)


def pinn_inversion(data, rg, tg, init, r_max, t_max, scale, **kw):
    """estimate unmeasured interior fields from outside probes with a physics-informed network: a fitted
    aegis.ml.pinn_diagnostics.PINN (see there, and compare with flux_estimate: it is an estimate, not an exact inversion)"""
    from .ml.pinn_diagnostics import PINN
    fit_kw = {k: kw.pop(k) for k in ("iters", "lr", "w_data", "w_phys", "w_init", "log") if k in kw}
    m = PINN(r_max, t_max, scale, **kw)
    m.fit(data, rg, tg, init, **fit_kw)
    return m


UNITS = {"visar": "km/s", "bdot": "V", "radiography": "mm"}
LABELS = {"visar": "VISAR free-surface velocity", "bdot": "B-dot probe voltage", "radiography": "radiography core radius"}


def report(path, traces, best, ens=None, metrics=None, title=""):
    """shot_calibration_fit.png: data with error bars, the best-fit simulation, the ensemble band and residuals"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    n = len(traces)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 5.6), sharex="col", gridspec_kw={"height_ratios": [3, 1]},
                           squeeze=False)
    for j, tr in enumerate(traces):
        a, r = ax[0, j], ax[1, j]
        tb, yb = best[tr.kind]
        m = (metrics or {}).get(tr.kind, {})
        sh = m.get("shift", 0.0)
        if ens:
            tg = np.linspace(tr.t.min(), tr.t.max(), 400)
            lo, hi, _ = band([(ti + sh, yi) for ti, yi in ens[tr.kind]], tg)
            a.fill_between(tg, lo, hi, color="#7fb3d5", alpha=0.35, lw=0, label="simulation ensemble (1 sigma)")
        a.plot(tb + sh, yb, color="#1f5f99", lw=1.4, label="best-fit simulation")
        a.errorbar(tr.t, tr.y, yerr=tr.sigma, fmt="o", ms=2.5, color="#c0392b", ecolor="#e6a39a", elinewidth=0.8,
                   label="shot data")
        a.set_title(LABELS.get(tr.kind, tr.name), loc="left")
        a.set_ylabel(tr.units)
        if m:
            a.text(0.02, 0.97, f"RMS {m['rms']:.3g} {tr.units}\n$\\chi^2_\\nu$ = {m['chi2_nu']:.2f}\n"
                   f"peak offset {m['peak_offset'] * 1e3:+.0f} ns", transform=a.transAxes, va="top", fontsize=8)
        f = sample(tb, yb, tr.t, sh)
        r.axhline(0, color="#999999", lw=0.7)
        r.errorbar(tr.t, (tr.y - f) / tr.sigma, yerr=1.0, fmt="o", ms=2, color="#555555", elinewidth=0.6)
        r.set_ylabel("residual / $\\sigma$")
        r.set_xlabel("t ($\\mu$s)")
        x0, x1 = tr.t.min() - 0.05 * np.ptp(tr.t), tr.t.max() + 0.05 * np.ptp(tr.t)
        a.set_xlim(x0, x1)
        inwin = (tb + sh >= x0) & (tb + sh <= x1)
        lo_ = min(float(np.min(tr.y - tr.sigma)), float(np.min(yb[inwin])))
        hi_ = max(float(np.max(tr.y + tr.sigma)), float(np.max(yb[inwin])))
        a.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
    ax[0, -1].legend(*ax[0, 0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
