"""Automated validation sweep: virtual diagnostics against trace libraries, with goodness-of-fit matrices.

A parameter sweep of an axisymmetric (r, z) liner implosion records three virtual probes inside the solver
(aegis/validation.py): a B-dot coil on the axis, a VISAR line cut on the liner's inner surface, and a radial framing
camera averaged along z. Each run is compared with every trace of the library that has a matching simulated signal,
and chi^2_nu and RMS are printed as matrices; the scan is then calibrated (minimum of the total chi^2 and its 1-sigma
range).

The library is benchmarks/literature/. Provenance is checked on load and printed in the matrix:
  - synthetic traces (benchmarks/literature/synthetic/, made by --make-synthetic: a simulation at a hidden implosion
    speed, sampled like the instruments, with Gaussian noise) test the chain end to end: the sweep must recover the
    hidden value. They say nothing about any experiment;
  - digitized / measured traces only count as comparisons if their experiment is actually what the sweep simulates.
    Here the dimensionless test liner matches no published shot, so traces of real shots are listed but not compared
    (--case maps a trace onto this run only if you know that it should). To validate against a shot, set up its
    geometry, drive and units as a sweep case.
The bundled real data (Marsh 1980 Hugoniots) are a different kind of comparison, plate impacts rather than time
series: --hugoniot adds aegis.experimental_suite.hugoniot_benchmark to the report.

    python examples/validate_experimental_shots.py                  # sweep of 5 runs against the library
    python examples/validate_experimental_shots.py --make-synthetic # (re)write the synthetic self-test traces
    python examples/validate_experimental_shots.py --hugoniot       # plus the Marsh copper Hugoniot benchmark
"""
import os
import sys
import json
import argparse
import contextlib
import io

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from aegis.config import Config, iRHO, iVX, iPR, iBY, iCLR
    from aegis.physics import xp, prim_to_cons, cons_to_prim
    from aegis.solver import MHDSolver
from aegis import validation as val

LIB = os.path.join(ROOT, "benchmarks", "literature")
HIDDEN_V0 = 1.04
TRACES = {"liner_bdot_axis": ("bdot", "B-dot coil on the axis, mid-plane"),
          "liner_visar_inner": ("visar", "VISAR on the inner liner surface"),
          "liner_radius_framing": ("framing", "framing camera: mean inner radius")}


def run_case(v0, nr=64, nz=8, t_end=0.75, every=2, b0=0.1):
    """a coasting liner of inner radius 1, thickness 0.2, on a seed B_z (dimensionless); returns the probes' signals,
    max |div B| and the relative energy budget error"""
    cfg = Config(nx=nr, ny=nz, x_min=0.0, x_max=1.6, y_min=0.0, y_max=0.25, cfl=0.3, gamma=5 / 3, geometry="cylindrical",
                 bc_x_type="extrapolation", bc_y_type="periodic", diag_interval=10**9, snapshot_times=[])
    with contextlib.redirect_stdout(io.StringIO()):
        s = MHDSolver(cfg)
    prof = 0.5 * (xp.tanh((s.X - 1.0) / 0.03) - xp.tanh((s.X - 1.2) / 0.03))
    W = xp.zeros_like(s.U)
    W[iRHO] = 0.05 + prof
    W[iVX] = -v0 * 1.1 / xp.maximum(s.X, 1e-3) * prof
    W[iPR] = 0.01
    W[iBY] = b0
    W[iCLR] = prof
    s.U[...] = prim_to_cons(W, cfg.gamma)
    zm = 0.125
    D = val.DiagnosticSet([
        val.BdotProbe("liner_bdot_axis", (0.02, zm), component=1, area=1.0),
        val.VisarProbe("liner_visar_inner", (0.02, zm), axis=0, direction=1),
        val.FramingCamera("liner_radius_framing", (0.02, zm), axis=0, direction=1, average_over=1)])
    D.record(s)
    n, e0, bmax = 0, None, 0.0
    while s.t < t_end - 1e-12:
        s.dt = min(s.compute_dt(cons_to_prim(s.U, s.eos)), t_end - s.t)
        s.step_ssprk3()
        s.t += s.dt
        n += 1
        if n % every == 0:
            D.record(s)
            bmax = max(bmax, float(s.div_b_max()))
    D.record(s)
    return D.signals(), dict(steps=n, div_b=bmax)


def make_synthetic(out, seed=7, n_samples=40):
    """hidden-parameter 'experiment': clearly synthetic, sampled and noised like instruments"""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    sig, _ = run_case(HIDDEN_V0)
    for name, (kind, title) in TRACES.items():
        t, y = sig[name]
        ok = np.isfinite(y)
        ts = np.linspace(t[ok][1], t[ok][-2], n_samples)
        ys = np.interp(ts, t[ok], y[ok])
        s_abs = 0.02 * np.max(np.abs(ys))
        tr = val.Trace(name, ts, ys + rng.normal(0, s_abs, n_samples), s_abs, "code units", kind,
                       dict(provenance="synthetic", note=f"AEGIS-MHD run at a hidden liner speed {HIDDEN_V0}, 2% Gaussian "
                                                         f"noise, seed {seed}; a self-test, not an experiment", title=title))
        val.save_digitized(os.path.join(out, name + ".csv"), tr)
    print(f"wrote synthetic self-test traces to {out}")


def sweep(values, traces):
    runs, info = {}, {}
    for v in values:
        sig, inf = run_case(v)
        runs[f"v0={v:.3f}"] = sig
        info[f"v0={v:.3f}"] = inf
    return runs, info, val.goodness_matrix(traces, runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-synthetic", action="store_true")
    ap.add_argument("--values", type=float, nargs="+", default=[0.96, 1.0, 1.04, 1.08, 1.12], help="liner speeds to run")
    ap.add_argument("--hugoniot", action="store_true", help="add the Marsh copper Hugoniot benchmark")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "literature_validation"))
    a = ap.parse_args()
    syn = os.path.join(LIB, "synthetic")
    if a.make_synthetic or not os.path.isdir(syn):
        make_synthetic(syn)
    lib, bad = val.scan_library(LIB)
    for p, why in bad.items():
        print(f"REFUSED {p}: {why}")
    prov = {k: t.meta["provenance"] for k, t in lib.items()}
    print(f"library: {len(lib)} traces " + str({v: sum(1 for x in prov.values() if x == v) for v in set(prov.values())}))
    usable = {k: t for k, t in lib.items() if k in TRACES}
    for k in lib:
        if k not in TRACES:
            print(f"  {k} ({prov[k]}): not compared, this sweep does not simulate that experiment")
    runs, info, M = sweep(a.values, usable)
    print("\nreduced chi-squared ([syn]: synthetic self-test, no statement about an experiment)")
    print(val.format_matrix(M, "chi2_nu"))
    print("\nRMS error")
    print(val.format_matrix(M, "rms"))
    tot = [sum(M[r][k]["chi2"] for k in M[r]) for r in runs]
    report = dict(matrix=M, info=info, library=prov)
    if len(a.values) >= 3:
        best, sig, cmin = val.calibrate(np.array(a.values), np.array(tot))
        print(f"\ncalibrated liner speed {best:.4f} +- {sig:.4f}"
              + (f" (hidden value {HIDDEN_V0})" if all(p == "synthetic" for p in prov.values()) else ""))
        report["calibration"] = dict(best=best, sigma=sig, chi2_min=cmin)
    print(f"max |div B| over the sweep: {max(i['div_b'] for i in info.values()):.1e}")
    if a.hugoniot:
        from aegis import experimental_suite as ex
        r = ex.hugoniot_benchmark("copper_hugoniot", ups=[0.3, 0.8, 1.4, 2.0, 2.8], n=400)
        print(f"\nMarsh (1980) copper Hugoniot [real data]: RMS {r['rms_km_s']:.3f} km/s, chi2_nu {r['chi2_nu']:.2f} "
              f"(sigma 2% assumed), {r['n_points']} shots")
        report["hugoniot"] = {k: r[k] for k in ("rms_km_s", "chi2_nu", "n_points")}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "goodness_of_fit.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"wrote {os.path.join(a.out, 'goodness_of_fit.json')}")


if __name__ == "__main__":
    main()
