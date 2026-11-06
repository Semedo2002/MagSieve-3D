"""Train and measure the DeepONet spectrometer surrogate (aegis/ml/deeponet_rad.py; results/deeponet_spectra/).

Al plasma along a 1 mm chord, 48 zones; 48 photon groups from 10 eV to 5 keV. Group opacities from the open atomic pipeline
(NIST energies, screened-hydrogenic line model, collisional-radiative ion balance). Ground truth: the exact formal solution
of the transfer equation along the chord. The DeepONet maps the (log T, log rho) profile to log10 of the emergent spectrum.
Measured on held-out random profiles against the exact solution and against the optically thin estimate, and timed.
Spectra are floored at 1e-6 of their peak (far-Wien tails no spectrometer sees); errors are scored on the groups within
1e-3 of the peak, and the total emitted power over all groups.

    python examples/deeponet_spectra.py [--train 4000 --epochs 150]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import atomic_importer as ai                                       # noqa: E402
from aegis import tabular as tb                                    # noqa: E402
from aegis.physics import xp                                       # noqa: E402
from aegis.ml import deeponet_rad as dr                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--zones", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "deeponet_spectra"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    edges = np.geomspace(10.0, 5000.0, 49)
    tab_path = os.path.join(a.out, "al_cr_lines_48groups.npz")
    t0 = time.time()
    if os.path.exists(tab_path):
        tab = dict(np.load(tab_path))
    else:
        tab = tb.build_atomic_table(13, ai.A_OF["Al"], ai.fetch_nist_ie("Al"), edges, rho=np.geomspace(0.5, 2000.0, 10),
                                    T_eV=np.geomspace(3.0, 2000.0, 24), nonlte=True, line_model=True, n_E=4000)
        tab = {k: v for k, v in tab.items() if k != "source"}
        np.savez(tab_path, **tab)
    print(f"opacity table {tab['kappa_R_g'].shape} ({time.time() - t0:.0f}s)", flush=True)
    G = tb.GroupOpacityTable(tab, 1.0, 1.0, 11604.518)              # rho kg/m^3, kappa m^2/kg, theta in eV
    kappa = lambda rho_gcc, T: G(rho_gcc * 1e3, T)
    ds = 1e-3 / a.zones                                            # m
    Eg = xp.asarray(np.log10(np.sqrt(edges[1:] * edges[:-1])))

    def make(n, seed):
        rng = np.random.default_rng(seed)
        rho, T = (xp.asarray(q) for q in dr.random_profiles(n, a.zones, rng))
        I, thin = dr.formal_solution(rho * 1e3, T, ds, lambda r, t: G(r, t), edges, xp)
        floor = 1e-6 * I.max(axis=1, keepdims=True)                # far-Wien tails below 1e-6 of the peak: not measurable
        return (dr.DeepONet.inputs(rho, T, xp), xp.log10(xp.maximum(I, floor)), xp.log10(xp.maximum(thin, floor)), rho, T)
    X, Y, _, _, _ = make(a.train, 1)
    Xv, Yv, _, _, _ = make(500, 2)
    Xt, Yt, Ythin, rt, Tt = make(1000, 3)
    m = dr.DeepONet(X.shape[1], xp=xp)
    t0 = time.time()
    hist = m.fit(X, Eg, Y, epochs=a.epochs, batch=128, Xv=Xv, Yv=Yv, log=lambda r: print(r, flush=True))
    train_s = time.time() - t0
    P = m.predict(Xt)
    sig = Yt > Yt.max(axis=1, keepdims=True) - 3.0                 # groups within 1e-3 of the spectrum's peak
    err = xp.abs(P - Yt)[sig]                                      # in log10
    errthin = xp.abs(Ythin - Yt)[sig]
    f = lambda e: dict(median_factor=float(10**xp.median(e)), p95_factor=float(10**xp.percentile(e, 95)),
                       within_10pct=float(xp.mean(e < np.log10(1.1))), within_2x=float(xp.mean(e < np.log10(2.0))))
    tot = lambda L: (10**L).sum(axis=1)
    pw = lambda L: float(xp.median(xp.abs(tot(L) / tot(Yt) - 1)))
    # timing on the same 1000 held-out chords
    xp.cuda.Device().synchronize(); t0 = time.time()
    for _ in range(5):
        dr.formal_solution(rt * 1e3, Tt, ds, lambda r, t: G(r, t), edges, xp)
    xp.cuda.Device().synchronize(); t_exact = (time.time() - t0) / 5
    t0 = time.time()
    for _ in range(5):
        m.predict(Xt)
    xp.cuda.Device().synchronize(); t_net = (time.time() - t0) / 5
    rep = dict(train=a.train, epochs=a.epochs, train_seconds=train_s, deeponet=f(err), optically_thin=f(errthin),
               power_error_median=dict(deeponet=pw(P), optically_thin=pw(Ythin)),
               ms_per_1000_chords=dict(exact_formal_solution=1e3 * t_exact, deeponet=1e3 * t_net), history=hist)
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}, indent=1))
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
