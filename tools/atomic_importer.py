"""Open atomic data for AEGIS-MHD: fetch or parse it, then build multigroup opacity / ionization tables.

Sources, all public:
  - NIST Atomic Spectra Database (physics.nist.gov/asd): ionization energies and spectral lines, for any element.
    Kramida, Ralchenko, Reader and NIST ASD Team, NIST Atomic Spectra Database, https://physics.nist.gov/asd.
  - Los Alamos TOPS (aphysics2.lanl.gov/apps): Rosseland / Planck opacities and free-electron counts, Z <= 30 only (so
    Al and Cu, not Au or Pb). Magee et al., ASP Conf. Ser. 78, 51 (1995). Used directly or as a benchmark.
  - CHIANTI (chiantidatabase.org): .elvlc level files, parsed here from a local copy (the database is a 600 MB
    download and covers Z <= 30). Dere et al., A&AS 125, 149 (1997) and later papers.
FAC output is not read: FAC is a code you run yourself, and its files depend on how it was run.

Raw responses are cached in aegis/data/atomic/ with their source and retrieval date in the header. The conversion to
tables is in aegis/tabular.py (build_atomic_table).

    python tools/atomic_importer.py --fetch-nist Al Cu Au Pb            # refresh the bundled NIST files
    python tools/atomic_importer.py --fetch-tops Al --t-kev 0.01 1 --rho 0.001 0.1
    python tools/atomic_importer.py --build Al --groups 10 100 1000 --out tables/al_atomic.npz
    python tools/atomic_importer.py --build Au --groups 10 100 1000 --lines --out tables/au_atomic.npz
"""
import os
import re
import sys
import csv
import html
import time
import argparse
import urllib.parse
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "aegis", "data", "atomic")
UA = {"User-Agent": "aegis-mhd-atomic-importer"}
NIST = "https://physics.nist.gov/cgi-bin/ASD/"
TOPS = "https://aphysics2.lanl.gov/"
Z_OF = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "Al": 13, "Fe": 26, "Cu": 29, "Mo": 42, "W": 74, "Au": 79, "Pb": 82}
A_OF = {"H": 1.008, "He": 4.003, "C": 12.011, "N": 14.007, "O": 15.999, "Al": 26.982, "Fe": 55.845, "Cu": 63.546,
        "Mo": 95.95, "W": 183.84, "Au": 196.967, "Pb": 207.2}


def _get(url, data=None, timeout=120):
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _cell(x):
    """NIST CSV cells look like ="1.23"; strip that, brackets and parentheses (which mark derived values)"""
    return x.strip().strip('=').strip('"').strip("[]()? ")


# ============================================================
# NIST ASD
# ============================================================
def parse_nist_ie(text):
    """ionization energies from the NIST CSV: I_eV (len Z), index = ion charge; missing values are NaN"""
    rows = list(csv.reader(l for l in text.splitlines() if l.strip() and not l.startswith("#")))
    head = [h.strip() for h in rows[0]]
    if "Ionization Energy (eV)" not in head:
        raise ValueError("not a NIST ionization-energy CSV (" + text[:80] + ")")
    ic, ie = head.index("Ion Charge"), head.index("Ionization Energy (eV)")
    out = {}
    for r in rows[1:]:
        if len(r) <= ie:
            continue
        q, v = _cell(r[ic]).lstrip("+"), _cell(r[ie])
        try:
            out[int(q)] = float(v)
        except ValueError:
            continue
    if not out:
        raise ValueError("no ionization energies in the response")
    I = np.full(max(out) + 1, np.nan)
    for q, v in out.items():
        I[q] = v
    return I


def fetch_nist_ie(element, cache=True):
    """I_eV of every ion stage of an element (from the bundled copy if there is one, else from NIST)"""
    path = os.path.join(DATA, f"nist_ie_{element}.csv")
    if cache and os.path.exists(path):
        return parse_nist_ie(open(path, encoding="utf-8").read())
    q = urllib.parse.urlencode(dict(spectra=element, units=1, format=2, order=0, at_num_out="on", sp_name_out="on",
                                    ion_charge_out="on", e_out=0, unc_out="on", submit="Retrieve Data"))
    url = NIST + "ie.pl?" + q
    text = _get(url)
    I = parse_nist_ie(text)
    os.makedirs(DATA, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# NIST Atomic Spectra Database, ionization energies of {element}; https://physics.nist.gov/asd\n"
                f"# retrieved {time.strftime('%Y-%m-%d')}: {url}\n" + text.replace("\r\n", "\n"))
    return I


def parse_nist_lines(text):
    """(ion charge, E_lower eV, E_upper eV, f_lu, g_lower) arrays from a NIST lines CSV (rows without an f value or
    level energies are skipped)"""
    rows = list(csv.reader(l for l in text.splitlines() if l.strip() and not l.startswith("#")))
    head = [h.strip() for h in rows[0]]
    need = ("sp_num", "fik", "Ei(eV)", "Ek(eV)", "g_i")
    if any(k not in head for k in need):
        raise ValueError("not a NIST lines CSV with " + ", ".join(need) + ": " + ",".join(head)[:120])
    ix = [head.index(k) for k in need]
    out = []
    for r in rows[1:]:
        try:
            out.append([int(_cell(r[ix[0]])) - 1, float(_cell(r[ix[2]])), float(_cell(r[ix[3]])), float(_cell(r[ix[1]])),
                        float(_cell(r[ix[4]]))])
        except (ValueError, IndexError):
            continue
    a = np.array(out, float).reshape(-1, 5)
    return a[(a[:, 3] > 0) & (a[:, 2] > a[:, 1])]


def fetch_nist_lines(element, cache=True):
    """spectral lines with f values, all ion stages NIST has them for"""
    path = os.path.join(DATA, f"nist_lines_{element}.csv")
    if cache and os.path.exists(path):
        return parse_nist_lines(open(path, encoding="utf-8").read())
    q = urllib.parse.urlencode(dict(spectra=element, limits_type=0, low_w="", upp_w="", unit=1, submit="Retrieve Data",
                                    format=2, line_out=1, remove_js="on", en_unit=1, output=0, page_size=15, show_av=2,
                                    A_out=0, f_out="on", allowed_out=1, forbid_out=1, enrg_out="on", g_out="on",
                                    ))
    url = NIST + "lines1.pl?" + q
    text = _get(url, timeout=300)
    a = parse_nist_lines(text)
    os.makedirs(DATA, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# NIST Atomic Spectra Database, lines with f values; https://physics.nist.gov/asd\n"
                f"# retrieved {time.strftime('%Y-%m-%d')}: {url}\n" + text.replace("\r\n", "\n"))
    return a


# ============================================================
# Los Alamos TOPS
# ============================================================
def _form_fields(page, action):
    i = page.index(action)
    j = page.index("</form>", i)
    return {m.group(1): html.unescape(m.group(2))
            for m in re.finditer(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page[i:j], re.I)}


def parse_tops(text):
    """a TOPS result page (or its text): list of {T_keV, rho, kR, kP, Zfree, Z2} in cm^2/g, grey tables"""
    t = html.unescape(re.sub(r"<[^>]+>", "\n", text)).replace("\xa0", " ")
    out, T = [], None
    num = r"[-+]?\d\.\d+E[-+]\d+"
    for line in t.splitlines():
        m = re.search(r"T=\s*(" + num + ")", line)
        if m and "Ross" in line:
            T = float(m.group(1))
            continue
        v = re.findall(num, line)
        if T is not None and len(v) == 5:
            r, kr, kp, z, z2 = map(float, v)
            out.append(dict(T_keV=T, rho=r, kR=kr, kP=kp, Zfree=z, Z2=z2))
    if not out:
        raise ValueError("no opacity rows in the TOPS page")
    return out


def fetch_tops(element, t_kev, rho_gcc, nr=None):
    """grey Rosseland / Planck opacities from the TOPS web form (two requests: summary, then results). t_kev: (low,
    high) on the TOPS temperature grid; rho_gcc: (low, high), nr log-spaced. A public service: keep requests small."""
    page = _get(TOPS + "apps/")
    form = page[page.index('<form method="POST"'):]
    d = {}
    for m in re.finditer(r"<input\b[^>]*>", form, re.I):
        t = m.group(0)
        n, v, ty = (re.search(rf'{k}="([^"]*)"', t, re.I) for k in ("name", "value", "type"))
        if n and not (ty and ty.group(1).lower() == "radio" and "checked" not in t.lower()):
            d[n.group(1)] = v.group(1) if v else ""
    d.update(mixture=f"1. {element.lower()}", tgrid="range", tlow=str(t_kev[0]), tup=str(t_kev[1]), rgrid="range",
             rlow=str(rho_gcc[0]), rup=str(rho_gcc[1]), nr=str(nr or 3), rspace="log", datype="gray", nthin="1")
    s = _get(TOPS + "submit", urllib.parse.urlencode(d).encode(), timeout=180)
    res = _form_fields(s, 'action="./results"')
    r = _get(TOPS + "results", urllib.parse.urlencode(res).encode(), timeout=300)
    return parse_tops(r), r


def parse_tops_groups(text):
    """a TOPS result page with multigroup output: list of {T_keV, rho, kR_g, kP_g, Zfree} (cm^2/g); group rows start
    at each group's lower photon energy"""
    t = html.unescape(re.sub(r"<[^>]+>", "\n", text)).replace("\xa0", " ")
    grey = {(r["T_keV"], r["rho"]): r for r in parse_tops(text)}
    num = r"[-+]?\d\.\d+E[-+]\d+"
    out, cur = [], None
    for line in t.splitlines():
        m = re.search(r"for T, density =\s*(" + num + r")\s+(" + num + ")", line)
        if m:
            cur = dict(T_keV=float(m.group(1)), rho=float(m.group(2)), kR_g=[], kP_g=[], E_keV=[])
            cur["Zfree"] = grey[(cur["T_keV"], cur["rho"])]["Zfree"]
            out.append(cur)
            continue
        v = re.findall(num, line)
        if cur is not None and len(v) == 3:
            cur["E_keV"].append(float(v[0]))
            cur["kR_g"].append(float(v[1]))
            cur["kP_g"].append(float(v[2]))
        elif cur is not None and line.strip() and not v:
            cur = None
    if not out:
        raise ValueError("no multigroup rows in the TOPS page")
    return out


def fetch_tops_groups(element, t_kev, rho_gcc, nr, edges_kev):
    """multigroup opacities from the TOPS form; edges_kev: photon-group boundaries (len G + 1)"""
    page = _get(TOPS + "apps/")
    form = page[page.index('<form method="POST"'):]
    d = {}
    for m in re.finditer(r"<input\b[^>]*>", form, re.I):
        t = m.group(0)
        n, v, ty = (re.search(rf'{k}="([^"]*)"', t, re.I) for k in ("name", "value", "type"))
        if n and not (ty and ty.group(1).lower() == "radio" and "checked" not in t.lower()):
            d[n.group(1)] = v.group(1) if v else ""
    d.update(mixture=f"1. {element.lower()}", tgrid="range", tlow=str(t_kev[0]), tup=str(t_kev[1]), rgrid="range",
             rlow=str(rho_gcc[0]), rup=str(rho_gcc[1]), nr=str(nr), rspace="log", datype="groups", nthin="1",
             egrid="specific", energies=" ".join(str(e) for e in edges_kev))
    s = _get(TOPS + "submit", urllib.parse.urlencode(d).encode(), timeout=180)
    res = _form_fields(s, 'action="./results"')
    return _get(TOPS + "results", urllib.parse.urlencode(res).encode(), timeout=600)


def save_tops_csv(path, element, rows, retrieved=None):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Los Alamos TOPS grey opacities of {element} (cm^2/g), https://aphysics2.lanl.gov/apps/\n"
                "# Magee et al., ASP Conf. Ser. 78, 51 (1995); output of the public TOPS form\n"
                f"# retrieved {retrieved or time.strftime('%Y-%m-%d')}\nT_keV,rho_gcc,kappa_R,kappa_P,Z_free,Z2_mean\n")
        for r in rows:
            f.write(f"{r['T_keV']:.5g},{r['rho']:.5g},{r['kR']:.5g},{r['kP']:.5g},{r['Zfree']:.5g},{r['Z2']:.5g}\n")


def load_tops_csv(path):
    rows = [l for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]
    return np.array([[float(x) for x in l.split(",")] for l in rows[1:]])


# ============================================================
# CHIANTI
# ============================================================
def parse_chianti_elvlc(path):
    """levels of a CHIANTI .elvlc file (version 8+ layout: index, configuration, term, multiplicity, L, J, observed
    and theoretical energies in cm^-1): {index, config, g, E_eV}; the observed energy is used where there is one
    (-1 marks none). Reading stops at the -1 end marker, before the references."""
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.strip().startswith("-1"):
            break
        m = re.match(r"\s*(\d+)\s+(.*?)\s+(\d+)\s+([A-Za-z])\s*(\d+\.\d)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line)
        if not m:
            raise ValueError(f"{path}: unexpected .elvlc line: {line[:70]!r}")
        E = float(m.group(6)) if float(m.group(6)) >= 0 else float(m.group(7))
        out.append(dict(index=int(m.group(1)), config=m.group(2).strip(), g=2 * float(m.group(5)) + 1,
                        E_eV=E * 1.23984198e-4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch-nist", nargs="*", default=[])
    ap.add_argument("--fetch-lines", nargs="*", default=[])
    ap.add_argument("--fetch-tops")
    ap.add_argument("--t-kev", type=float, nargs=2, default=[0.01, 1.0])
    ap.add_argument("--rho", type=float, nargs=2, default=[1e-3, 1e-1])
    ap.add_argument("--nr", type=int, default=3)
    ap.add_argument("--tops-table", help="Z <= 30: multigroup table imported from TOPS, written to aegis/data/atomic")
    ap.add_argument("--build", help="element to build a model table for (NIST energies; opacities are lower bounds)")
    ap.add_argument("--groups", type=float, nargs="+", default=[10, 100, 1000], help="inner group edges, eV")
    ap.add_argument("--lines", action="store_true", help="include NIST lines in the opacity")
    ap.add_argument("--nonlte", action="store_true", help="opacity from the collisional-radiative ion balance")
    ap.add_argument("--out")
    a = ap.parse_args()
    for el in a.fetch_nist:
        I = fetch_nist_ie(el, cache=False)
        print(f"{el}: {np.sum(np.isfinite(I))} of {Z_OF.get(el, len(I))} ionization energies, I_0 = {I[0]} eV")
    for el in a.fetch_lines:
        L = fetch_nist_lines(el, cache=False)
        print(f"{el}: {len(L)} lines with f values, ion charges {sorted(set(L[:, 0].astype(int)))[:12]} ...")
    if a.fetch_tops:
        rows, _ = fetch_tops(a.fetch_tops, a.t_kev, a.rho, a.nr)
        p = os.path.join(DATA, f"tops_{a.fetch_tops}.csv")
        save_tops_csv(p, a.fetch_tops, rows)
        print(f"wrote {p}: {len(rows)} rows")
    if a.tops_table:
        sys.path.insert(0, ROOT)
        from aegis import tabular
        inner = [1e-3 * g for g in a.groups]
        ek = [1e-3] + inner + [100.0]
        rows = parse_tops_groups(fetch_tops_groups(a.tops_table, a.t_kev, a.rho, a.nr, ek))
        ev = np.array([0.0] + list(a.groups) + [np.inf])
        tab = tabular.tops_groups_to_table(rows, ev)
        p = a.out or os.path.join(DATA, f"tops_{a.tops_table}_groups.npz")
        np.savez_compressed(p, **tab)
        print(f"wrote {p}: {tab['kappa_R_g'].shape} from {len(rows)} TOPS rows")
    if a.build:
        sys.path.insert(0, ROOT)
        from aegis import tabular
        I = fetch_nist_ie(a.build)
        lines = fetch_nist_lines(a.build) if a.lines else None
        edges = np.array([0.0] + list(a.groups) + [np.inf])
        tab = tabular.build_atomic_table(Z_OF[a.build], A_OF[a.build], I, edges, lines=lines, nonlte=a.nonlte)
        out = a.out or f"{a.build.lower()}_atomic.npz"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez(out, **tab)
        print(f"wrote {out}: {tab['kappa_R_g'].shape}, source: {tab['source']}")


if __name__ == "__main__":
    main()
