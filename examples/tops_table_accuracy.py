"""How closely the solver's opacities follow Los Alamos TOPS when the bundled TOPS tables are the source.

Tables: aegis/data/atomic/tops_<el>_table.npz (python tools/atomic_importer.py --tops Al). Each holds TOPS's grey
Rosseland and Planck means and Z_free at TOPS's own temperatures (TOPS computes only there) and on a density grid
refined until log-log interpolation reproduces TOPS at every interval midpoint (held out, stored in the file).

Measured here, all through the solver's lookup (OpacityTable / IonizationTable, the GPU kernel when there is a GPU):
  1. at the table's nodes: the lookup returns TOPS's printed values;
  2. at the held-out midpoints (TOPS values never put in the table), at every TOPS temperature;
  3. at separate TOPS requests: 37 densities off the table's grid at every TOPS temperature
     (aegis/data/atomic/tops_<El>_check.csv), and 96 points at table densities (tops_<El>.csv, a repeat request);
  4. between TOPS temperatures, where no TOPS value exists: an estimate from dropping every other temperature
     (doubled spacing, so an overestimate for the actual spacing).

    python examples/tops_table_accuracy.py            # writes results/tops_tables/summary.{json,md}
"""
import os
import sys
import json

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from aegis import tabular                         # noqa: E402
from aegis.physics import xp                      # noqa: E402
import atomic_importer as ai                      # noqa: E402

DATA = os.path.join(ROOT, "aegis", "data", "atomic")
OUT = os.path.join(ROOT, "results", "tops_tables")
K_PER_KEV = 1e3 * 11604.518


def stats(e):
    e = np.abs(np.asarray(e, float)).ravel()
    return dict(n=int(e.size), median=float(np.median(e)), p99=float(np.percentile(e, 99)), max=float(e.max()),
                within_5pct=float(np.mean(e <= 0.05)))


def lookup(d, rho, T):
    """(kappa_R, kappa_P, Zbar) through the solver's tables, SI in and out"""
    op = tabular.OpacityTable(d, 1.0, 1.0, 1.0)
    zt = tabular.IonizationTable(d, 1.0, 1.0)
    r, t = xp.asarray(np.asarray(rho, float)), xp.asarray(np.asarray(T, float))
    kR, kP = op(r, t)
    z = zt(r, t)
    get = (lambda a: a.get()) if hasattr(kR, "get") else np.asarray
    return get(kR), get(kP), get(z)


def evaluate(el):
    path = os.path.join(DATA, f"tops_{el.lower()}_table.npz")
    raw = np.load(path)
    d = tabular.load(path)
    rho, T = d["rho"], d["T"]
    R, TT = np.meshgrid(rho, T, indexing="ij")
    out = dict(table=os.path.relpath(path, ROOT).replace("\\", "/"), densities=len(rho), temperatures=len(T),
               rho_range_gcc=[float(rho[0] / 1e3), float(rho[-1] / 1e3)],
               T_range_keV=[float(T[0] / K_PER_KEV), float(T[-1] / K_PER_KEV)],
               offtable_points=int(raw["offtable"].sum()), tol=float(raw["tol"]), source=str(raw["source"]))
    kR, kP, z = lookup(d, R, TT)                      # 1. nodes
    out["nodes"] = {q: stats(a / d[k] - 1) for q, a, k in
                    (("kappa_R", kR, "kappa_R"), ("kappa_P", kP, "kappa_P"), ("Zbar", z, "Zbar_lte"))}
    hr = d["heldout_rho"]                             # 2. held-out midpoints at every TOPS temperature
    HR, HT = np.meshgrid(hr, T, indexing="ij")
    kR, kP, z = lookup(d, HR, HT)
    out["heldout_midpoints"] = {q: stats(a / d[k] - 1) for q, a, k in
                                (("kappa_R", kR, "heldout_kappa_R"), ("kappa_P", kP, "heldout_kappa_P"),
                                 ("Zbar", z, "heldout_Zbar"))}
    for key, name in (("offgrid_request", f"tops_{el}_check.csv"), ("repeat_request", f"tops_{el}.csv")):
        csv = os.path.join(DATA, name)                # 3. separate TOPS requests: off the table's densities, and a
        if not os.path.exists(csv):                   #    repeat at table densities (same values from a new request)
            continue
        a = ai.load_tops_csv(csv)                     # T_keV, rho_gcc, kR, kP, Z_free (cm^2/g)
        kR, kP, z = lookup(d, a[:, 1] * 1e3, a[:, 0] * K_PER_KEV)
        out[key] = dict(file=os.path.relpath(csv, ROOT).replace("\\", "/"), kappa_R=stats(kR / (a[:, 2] * 0.1) - 1),
                        kappa_P=stats(kP / (a[:, 3] * 0.1) - 1), Zbar=stats(z / a[:, 4] - 1))
    lt = np.log10(T)                                  # 4. between TOPS temperatures: an estimate only
    keep, test = np.arange(0, len(T), 2), np.arange(1, len(T) - 1, 2)
    bands = [(0, 3e-3), (3e-3, 3e-2), (3e-2, 0.3), (0.3, 3), (3, 101)]
    est = {}
    for q in ("kappa_R", "kappa_P", "Zbar_lte"):
        A = np.log10(d[q])
        P = np.array([np.interp(lt[test], lt[keep], A[i, keep]) for i in range(len(rho))])
        e = np.abs(10 ** (P - A[:, test]) - 1)
        Tt = T[test] / K_PER_KEV
        est[q] = {f"{lo:g}-{hi:g} keV": dict(median=float(np.median(e[:, (Tt >= lo) & (Tt < hi)])),
                                             p95=float(np.percentile(e[:, (Tt >= lo) & (Tt < hi)], 95)))
                  for lo, hi in bands}
    out["between_temperatures_estimate_at_double_spacing"] = est
    return out


def markdown(res):
    L = ["# Solver opacities against Los Alamos TOPS", "",
         "Source tables: TOPS grey means (Magee et al. 1995; aphysics2.lanl.gov/apps) at TOPS's own temperatures on an",
         "adaptive density grid. Errors are |lookup / TOPS - 1| through the solver's table lookup.", ""]
    for el, r in res.items():
        L += [f"## {el}", "",
              f"{r['densities']} densities ({r['rho_range_gcc'][0]:g} - {r['rho_range_gcc'][1]:g} g/cc) x "
              f"{r['temperatures']} TOPS temperatures ({r['T_range_keV'][0]:g} - {r['T_range_keV'][1]:g} keV); "
              f"{r['offtable_points']} points beyond TOPS's tables.", "",
              "| check | quantity | points | median | 99th pct | max | within 5% |", "|---|---|---|---|---|---|---|"]
        for name, key in (("table nodes", "nodes"), ("held-out midpoints", "heldout_midpoints"),
                          ("separate request, off-grid densities", "offgrid_request"),
                          ("separate request, table densities", "repeat_request")):
            if key not in r:
                continue
            for q in ("kappa_R", "kappa_P", "Zbar"):
                s = r[key][q]
                L.append(f"| {name} | {q} | {s['n']} | {s['median']:.1e} | {s['p99']:.1e} | {s['max']:.2e} | "
                         f"{100 * s['within_5pct']:.2f}% |")
        L += ["", "Between TOPS temperatures (no TOPS value exists there): error when every other temperature is "
              "dropped, i.e. at twice the real spacing, so larger than the real interpolation error:", "",
              "| T band | kappa_R median / 95th | kappa_P median / 95th | Z median / 95th |", "|---|---|---|---|"]
        e = r["between_temperatures_estimate_at_double_spacing"]
        for band in e["kappa_R"]:
            L.append(f"| {band} | " + " | ".join(f"{e[q][band]['median']:.3f} / {e[q][band]['p95']:.3f}"
                                                 for q in ("kappa_R", "kappa_P", "Zbar_lte")) + " |")
        L.append("")
    return "\n".join(L)


def main():
    res = {}
    for el in ("Al", "Cu"):
        if os.path.exists(os.path.join(DATA, f"tops_{el.lower()}_table.npz")):
            res[el] = evaluate(el)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    md = markdown(res)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
