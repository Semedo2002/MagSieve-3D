"""Open atomic data for AEGIS-MHD: fetch or parse it, then build multigroup opacity / ionization tables.

Sources, all public:
  - NIST Atomic Spectra Database (physics.nist.gov/asd): ionization energies and spectral lines, for any element.
    Kramida, Ralchenko, Reader and NIST ASD Team, NIST Atomic Spectra Database, https://physics.nist.gov/asd.
  - Los Alamos TOPS (aphysics2.lanl.gov/apps): Rosseland / Planck opacities and free-electron counts, Z <= 30 only (so
    Al and Cu, not Au or Pb). Magee et al., ASP Conf. Ser. 78, 51 (1995). Used directly or as a benchmark.
  - CHIANTI (chiantidatabase.org, files from the SolarSoft mirror), Z <= 30: .elvlc levels and .wgfa lines (downloaded
    on demand into aegis/data/atomic/chianti/, MB per ion, not bundled), .drparams dielectronic recombination fits
    (bundled for Al and Cu) and the coronal ion fractions of chianti.ioneq (Al and Cu rows bundled, as a benchmark).
    Dere et al., A&AS 125, 149 (1997); Del Zanna et al., ApJ 909, 38 (2021); DR fits by Badnell and co-workers.
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


def _tops_form():
    """the TOPS form's default fields (radio buttons at their checked values)"""
    page = _get(TOPS + "apps/")
    form = page[page.index('<form method="POST"'):]
    d = {}
    for m in re.finditer(r"<input\b[^>]*>", form, re.I):
        t = m.group(0)
        n, v, ty = (re.search(rf'{k}="([^"]*)"', t, re.I) for k in ("name", "value", "type"))
        if n and not (ty and ty.group(1).lower() == "radio" and "checked" not in t.lower()):
            d[n.group(1)] = v.group(1) if v else ""
    return d


def parse_tops_offtable(text):
    """(T_keV, rho) pairs TOPS flags as beyond its tables ("WARNING - Requested densities ... went off table
    boundaries"): there it reuses the cross sections of the highest tabulated density, so these values are not data"""
    t = html.unescape(re.sub(r"<[^>]+>", "\n", text)).replace("\xa0", " ")
    num = r"[-+]?\d\.\d+E[-+]\d+"
    out, on = set(), False
    for line in t.splitlines():
        if "off table boundaries" in line:
            on = True
            continue
        if on:
            v = re.findall(num, line)
            if len(v) == 3 and line.split()[0] == v[0]:
                out.add((float(v[0]), float(v[1])))
            elif line.strip():
                on = False
    return out


def tops_query(mixture, t_kev, rho_gcc, nr, edges_kev=None, fractype="atomic", isotope=False, retries=3):
    """one TOPS request: grey means (and group means for edges_kev) at every TOPS temperature in t_kev = (low, high)
    and nr log-spaced densities in rho_gcc = (low, high). mixture in TOPS syntax ("1. al", "1 c 1 h", with isotope=True
    "1 h 2.0141 1 h 3.0160"). Returns (grey rows, group rows or None, off-table set, raw page)."""
    d = _tops_form()
    d.update(mixture=mixture, fractype=fractype, isotope="isotope" if isotope else "noniso", tgrid="range",
             tlow=str(t_kev[0]), tup=str(t_kev[1]), rgrid="range", rlow="%.6g" % rho_gcc[0], rup="%.6g" % rho_gcc[1],
             nr=str(nr), rspace="log", datype="gray", nthin="1")
    if edges_kev is not None:
        d.update(datype="groups", egrid="specific", energies=" ".join("%.6g" % e for e in edges_kev))
    for k in range(retries):
        try:
            s = _get(TOPS + "submit", urllib.parse.urlencode(d).encode(), timeout=300)
            res = _form_fields(s, 'action="./results"')
            r = _get(TOPS + "results", urllib.parse.urlencode(res).encode(), timeout=900)
            return parse_tops(r), (parse_tops_groups(r) if edges_kev is not None else None), parse_tops_offtable(r), r
        except (OSError, ValueError):
            if k == retries - 1:
                raise
            time.sleep(10 * (k + 1))


def fetch_tops_dense(mixture, rho_gcc=(1e-6, 1e2), per_decade=32, t_kev=(0.0005, 100.0), edges_kev=None,
                     fractype="atomic", isotope=False, pause=2.0, log=print):
    """a dense TOPS table: every TOPS temperature in t_kev, densities log-spaced at per_decade points per decade.
    Requests are split so each stays within TOPS's limits (100 densities; T x rho x group bounds <= 50000).
    Returns dict(T_keV (nt,), rho_gcc (nr,), kR, kP, Zfree (nr, nt) in cm^2/g, offtable (nr, nt) bool, and kR_g, kP_g
    (G, nr, nt) with edges_keV when edges_kev is given)."""
    lo, hi = np.log10(rho_gcc[0]), np.log10(rho_gcc[1])
    n = int(round((hi - lo) * per_decade)) + 1
    nb = 1 if edges_kev is None else len(edges_kev)
    chunk = min(100, max(2, 50000 // (74 * nb)))     # densities per request (74 TOPS temperatures at most)
    grey, groups, off, k = {}, {}, set(), 0
    while k < n - 1:
        m = min(chunk, n - k)
        a, b = 10 ** (lo + (hi - lo) * k / (n - 1)), 10 ** (lo + (hi - lo) * (k + m - 1) / (n - 1))
        g, gg, o, _ = tops_query(mixture, t_kev, (a, b), m, edges_kev, fractype, isotope)
        for r in g:
            grey[(r["T_keV"], r["rho"])] = r
        for r in gg or []:
            groups[(r["T_keV"], r["rho"])] = r
        off |= o
        log(f"  TOPS {mixture!r}: densities {k + 1}-{k + m} of {n} ({a:.3g}-{b:.3g} g/cc), {len(g)} rows")
        k += m - 1
        time.sleep(pause)
    T = np.array(sorted({t for t, _ in grey}))
    rho = np.array(sorted({r for _, r in grey}))
    it, ir = {t: j for j, t in enumerate(T)}, {r: i for i, r in enumerate(rho)}
    out = dict(T_keV=T, rho_gcc=rho, mixture=mixture, fractype=fractype)
    for q in ("kR", "kP", "Zfree"):
        out[q] = np.full((len(rho), len(T)), np.nan)
    out["offtable"] = np.zeros((len(rho), len(T)), bool)
    for (t, r), x in grey.items():
        for q in ("kR", "kP", "Zfree"):
            out[q][ir[r], it[t]] = x[q]
    for t, r in off:                                 # the warning lists T and rho rounded alike
        i = np.argmin(abs(np.log(rho / r))) if len(rho) else None
        j = np.argmin(abs(np.log(T / t)))
        if i is not None and abs(np.log(rho[i] / r)) < 1e-3 and abs(np.log(T[j] / t)) < 1e-3:
            out["offtable"][i, j] = True
    if np.isnan(out["kR"]).any():
        raise ValueError(f"TOPS table for {mixture!r} has {int(np.isnan(out['kR']).sum())} missing points")
    if edges_kev is not None:
        G = len(edges_kev) - 1
        out["kR_g"], out["kP_g"] = np.zeros((G, len(rho), len(T))), np.zeros((G, len(rho), len(T)))
        for (t, r), x in groups.items():
            out["kR_g"][:, ir[r], it[t]], out["kP_g"][:, ir[r], it[t]] = x["kR_g"][:G], x["kP_g"][:G]
        out["edges_keV"] = np.asarray(edges_kev, float)
    return out


def _tops_range(mixture, t_kev, a, b, n, edges_kev=None, fractype="atomic", isotope=False):
    """one TOPS range request (n log-spaced densities from a to b). TOPS's 'specific densities' input is not honoured
    by the server (it answers with a fixed default list), so every density request goes through a range. Rows come
    back in order of the n intended densities: {(T_keV, i): grey}, {(T_keV, i): groups}, {(T_keV, i)} off table."""
    g, gg, off, _ = tops_query(mixture, t_kev, (a, b), n, edges_kev, fractype, isotope)
    lv = np.linspace(np.log(a), np.log(b), n)

    def idx(rho):
        i = int(np.argmin(abs(lv - np.log(rho))))
        if abs(lv[i] - np.log(rho)) > 2e-4:
            raise ValueError(f"TOPS returned density {rho}, not in the requested range grid")
        return i
    return ({(x["T_keV"], idx(x["rho"])): x for x in g}, {(x["T_keV"], idx(x["rho"])): x for x in gg or []},
            {(t, idx(r)) for t, r in off})


def _runs(idx, step, limit):
    """split sorted integer grid indices into runs of consecutive indices step apart, at most limit per run"""
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[j] == step and j + 1 - i < limit - 1:
            j += 1
        out.append((idx[i], idx[j]))
        i = j + 1
    return out


def fetch_tops_adaptive(mixture, rho_gcc=(1e-6, 1e2), per_decade=32, tol=0.01, max_split=5, t_kev=(0.0005, 100.0),
                        fractype="atomic", isotope=False, pause=2.0, log=print):
    """a TOPS grey table on TOPS's temperatures and an adaptive density grid. Start from per_decade log-spaced
    densities; fetch the midpoint (in log rho) of every interval; where log-log linear interpolation misses TOPS at that
    midpoint by more than tol in kappa_R, kappa_P or Z_free at any temperature, the midpoint becomes a node and both
    halves are tested again (up to max_split bisections). Intervals that pass keep their midpoint out of the table as
    held-out evidence. Linear interpolation across a single slope kink errs at most twice as much as at the midpoint,
    so the table holds TOPS to about 2 tol between its nodes, at TOPS's temperatures (nothing tests between those:
    TOPS does not compute there). Densities are integer indices on the finest grid, so requests that overlap agree.
    Returns the table dict of fetch_tops_dense plus the held-out midpoints."""
    lo, hi = np.log10(rho_gcc[0]), np.log10(rho_gcc[1])
    n0 = int(round((hi - lo) * per_decade))
    F = n0 * 2 ** (max_split + 1)                    # finest grid: F intervals

    def rho(m):
        return 10 ** (lo + (hi - lo) * m / F)
    cols = {}                                        # finest-grid index -> {T: row}

    def get(a, b, n, step):
        if n == 1:                                   # one density: the 3-point range centred on it
            a, b, n = a - step, a + step, 3
        g, gg, off = _tops_range(mixture, t_kev, rho(a), rho(b), n, None, fractype, isotope)
        return a, step, g, off

    def fetch(need, step):
        """fetch grid indices need (multiples of step apart), filling gaps of up to 5 steps to save requests"""
        need = sorted(set(need) - set(cols))
        if not need:
            return
        full = []
        for a, b in zip(need[:-1], need[1:]):
            full.append(a)
            if b - a <= 5 * step:
                full += list(range(a + step, b, step))
        full.append(need[-1])
        for a, b in _runs(sorted(set(full)), step, 100):
            n = (b - a) // step + 1
            a0, s0, g, off = get(a, b, n, step)
            for (t, i), x in g.items():
                cols.setdefault(a0 + i * s0, {})[t] = dict(x, off=(t, i) in off)
            log(f"  TOPS {mixture!r}: {n} densities {rho(a):.4g}-{rho(b):.4g} g/cc ({len(cols)} columns)")
            time.sleep(pause)
    step0 = F // n0
    fetch(range(0, F + 1, step0 // 2), step0 // 2)  # nodes and every first-level midpoint
    T = np.array(sorted({t for c in cols.values() for t in c}))
    Q = ("kR", "kP", "Zfree")

    def vec(m):
        c = cols[m]
        return np.log10(np.array([[c[t][q] for t in T] for q in Q]))
    nodes = set(range(0, F + 1, step0))
    todo, held = [(a, a + step0, 0) for a in range(0, F, step0)], []
    while todo:
        w = todo[0][1] - todo[0][0]
        fetch([(a + b) // 2 for a, b, _ in todo], w // 2)
        nxt = []
        for a, b, lev in todo:
            m = (a + b) // 2
            err = float(np.max(np.abs(10 ** (0.5 * (vec(a) + vec(b)) - vec(m)) - 1)))
            if err <= tol or lev >= max_split:
                held.append((m, err, lev >= max_split and err > tol))
            else:
                nodes.add(m)
                nxt += [(a, m, lev + 1), (m, b, lev + 1)]
        todo = nxt
        log(f"  {len(nodes)} nodes, {len(todo)} intervals to test at the next level")
    nodes = sorted(nodes)
    out = dict(T_keV=T, rho_gcc=np.array([rho(m) for m in nodes]), mixture=mixture, fractype=fractype, tol=tol,
               node_index=np.array(nodes), finest=F, log_range=(lo, hi))
    for q in Q:
        out[q] = np.array([[cols[m][t][q] for t in T] for m in nodes])
    out["offtable"] = np.array([[cols[m][t]["off"] for t in T] for m in nodes])
    hm = [h[0] for h in held]
    out["heldout_rho_gcc"] = np.array([rho(m) for m in hm])
    for q in Q:
        out["heldout_" + q] = np.array([[cols[m][t][q] for t in T] for m in hm])
    out["heldout_split_capped"] = np.array([h[2] for h in held])
    return out


def fetch_tops_groups_at(tab, edges_kev, t_kev=(0.0005, 100.0), isotope=False, pause=2.0, log=print):
    """group means (G, nr, nt) at the nodes of a fetch_tops_adaptive table, every TOPS temperature, within TOPS's
    request limit (T x rho x group bounds <= 50000)"""
    lo, hi = tab["log_range"]
    F, nodes = int(tab["finest"]), [int(m) for m in tab["node_index"]]

    def rho(m):
        return 10 ** (lo + (hi - lo) * m / F)
    per = max(2, min(100, 50000 // (74 * len(edges_kev))))
    pos = {m: i for i, m in enumerate(nodes)}
    G = len(edges_kev) - 1
    kR, kP = np.zeros((G, len(nodes), len(tab["T_keV"]))), np.zeros((G, len(nodes), len(tab["T_keV"])))
    it = {t: j for j, t in enumerate(tab["T_keV"])}
    got, i = set(), 0
    while i < len(nodes):                            # runs of equally spaced nodes
        j = i
        while j + 1 < len(nodes) and j + 1 - i < per - 1 and (j == i or nodes[j + 1] - nodes[j] == nodes[i + 1] - nodes[i]):
            j += 1
        a, b, n = nodes[i], nodes[j], j - i + 1
        step = (b - a) // (n - 1) if n > 1 else 1
        if n == 1:                                   # a lone node: a 3-point range centred on it, keep the middle
            a, b, n = a - 1, a + 1, 3
        _, gg, _ = _tops_range(tab["mixture"], t_kev, rho(a), rho(b), n, edges_kev, tab["fractype"], isotope)
        for (t, k), x in gg.items():
            if j == i and k != 1:
                continue
            if len(x["kR_g"]) < G:
                raise ValueError(f"TOPS returned {len(x['kR_g'])} groups, expected {G}")
            m = a + k * step
            kR[:, pos[m], it[t]], kP[:, pos[m], it[t]] = x["kR_g"][:G], x["kP_g"][:G]
            got.add((m, t))
        log(f"  TOPS groups {tab['mixture']!r}: {n} densities {rho(a):.4g}-{rho(b):.4g} g/cc")
        time.sleep(pause)
        i = j + 1
    if len(got) != len(nodes) * len(tab["T_keV"]):
        raise ValueError(f"group table incomplete: {len(got)} of {len(nodes) * len(tab['T_keV'])} points")
    return kR, kP


def tops_dense_to_npz(tab, path, note=""):
    """write a dense TOPS table in SI for the solver: rho (kg/m^3), T (K), kappa_R, kappa_P (m^2/kg), Zbar_lte, and
    the group means (kappa_R_g, kappa_P_g, group_edges_eV) when present. TOPS's own (rho, T) points are kept as the
    grid (no resampling), so a lookup at a TOPS point returns TOPS's value."""
    d = dict(rho=tab["rho_gcc"] * 1e3, T=tab["T_keV"] * 1e3 * 11604.518, kappa_R=tab["kR"] * 0.1,
             kappa_P=tab["kP"] * 0.1, Zbar_lte=tab["Zfree"], offtable=tab["offtable"],
             source=np.array("Los Alamos TOPS (aphysics2.lanl.gov/apps; Magee et al., ASP Conf. Ser. 78, 51, 1995), "
                             f"mixture {tab['mixture']!r} by {tab['fractype']} fraction, retrieved "
                             f"{time.strftime('%Y-%m-%d')}. {note}".strip()))
    if "kR_g" in tab:
        d.update(kappa_R_g=tab["kR_g"] * 0.1, kappa_P_g=tab["kP_g"] * 0.1, group_edges_eV=tab["edges_keV"] * 1e3)
    if "heldout_rho_gcc" in tab:                     # TOPS at the midpoints between nodes, not in the table
        d.update(heldout_rho=tab["heldout_rho_gcc"] * 1e3, heldout_kappa_R=tab["heldout_kR"] * 0.1,
                 heldout_kappa_P=tab["heldout_kP"] * 0.1, heldout_Zbar=tab["heldout_Zfree"], tol=tab["tol"])
    np.savez_compressed(path, **d)
    return d


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


CHIANTI = "https://soho.nascom.nasa.gov/solarsoft/packages/chianti/dbase/"


def parse_chianti_drparams(text):
    """one CHIANTI .drparams file: {'type': 1, 'E': [...], 'c': [...]} (rate T^-3/2 sum c exp(-E / T), T in K) or
    {'type': 2, 'c': [A, B, T0, T1]}; trailing zero pairs are kept (harmless)"""
    rows = []
    for line in text.splitlines():
        if line.strip().startswith("-1"):
            break
        if line.strip():
            rows.append([float(x) for x in line.split()])
    typ = int(rows[0][0])
    if typ == 1:
        return {"type": 1, "E": rows[1][2:], "c": rows[2][2:]}
    if typ == 2:
        return {"type": 2, "c": rows[1][2:6]}
    raise ValueError(f"CHIANTI drparams type {typ} not supported")


def fetch_chianti_dr(element, cache=True):
    """CHIANTI dielectronic recombination fits for every ion of the element (Z <= 30), bundled as one text file:
    {recombining ion charge: parsed drparams}. The fits are Badnell and co-workers' DR calculations."""
    el = element.lower()
    path = os.path.join(DATA, f"chianti_dr_{element}.txt")
    if not (cache and os.path.exists(path)):
        parts = []
        for k in range(2, Z_OF[element] + 2):               # al_2 (Al+) ... al_13 (H-like); al_k has charge k - 1
            try:
                parts.append(f"### {el}_{k}\n" + _get(f"{CHIANTI}{el}/{el}_{k}/{el}_{k}.drparams").replace("\r\n", "\n"))
            except Exception:
                continue
        os.makedirs(DATA, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# CHIANTI database (Dere et al. 1997; Del Zanna et al. 2021), .drparams of {element} ions\n"
                    f"# {CHIANTI}{el}/ retrieved {time.strftime('%Y-%m-%d')}; each block keeps its own references\n"
                    + "".join(parts))
    out = {}
    for block in open(path, encoding="utf-8").read().split("### ")[1:]:
        name, body = block.split("\n", 1)
        out[int(name.split("_")[1]) - 1] = parse_chianti_drparams(body)
    return out


def load_chianti_ioneq(Z):
    """CHIANTI coronal ion fractions bundled for Al and Cu (chianti_ioneq_Al_Cu.csv): (T in eV, fractions (Z + 1, nT))"""
    rows = [l for l in open(os.path.join(DATA, "chianti_ioneq_Al_Cu.csv"), encoding="utf-8") if not l.startswith("#")]
    lt = np.array([float(x) for x in rows[0].split(",")[2:]])
    F = np.zeros((Z + 1, len(lt)))
    for l in rows[1:]:
        p = l.split(",")
        if int(p[0]) == Z:
            F[int(p[1]) - 1] = [float(x) for x in p[2:]]
    if not F.any():
        raise ValueError(f"no CHIANTI ion fractions bundled for Z = {Z}")
    return 10**lt / 11604.518, F


def parse_chianti_wgfa(path, levels):
    """lines (ion charge placeholder 0, E_lower eV, E_upper eV, f_lu, g_lower) of a CHIANTI .wgfa file with levels from
    parse_chianti_elvlc; zero-wavelength rows (two-photon, autoionisation) are skipped. f is computed from the radiative
    rate, f_lu = 1.4992e-16 lambda^2 (g_u / g_l) A (lambda in Angstrom from the level energies): the file's gf column
    equals this for allowed lines but not for some forbidden ones (1e5 too large for Al XII 1s2 - 1s3d)"""
    E = {L["index"]: L["E_eV"] for L in levels}
    g = {L["index"]: L["g"] for L in levels}
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.strip().startswith("-1"):
            break
        p = line.split()
        try:
            i, j, wvl, A = int(p[0]), int(p[1]), float(p[2]), float(p[4])
        except (ValueError, IndexError):
            continue
        if wvl == 0 or A <= 0 or i not in E or j not in E:
            continue
        lo, up = (i, j) if E[i] <= E[j] else (j, i)
        if E[up] > E[lo]:
            lam = 12398.4198 / (E[up] - E[lo])
            out.append([0.0, E[lo], E[up], 1.4992e-16 * lam * lam * g[up] / g[lo] * A, g[lo]])
    return np.array(out, float).reshape(-1, 5)


def fetch_chianti_lines(element, cache_dir=None):
    """lines of every ion of the element that CHIANTI has .elvlc and .wgfa files for, in the NIST layout (charge in
    column 0). The files are large (MB per ion), so they are cached outside the package (aegis/data/atomic/chianti/,
    not bundled); for Z <= 30 only."""
    el = element.lower()
    d = cache_dir or os.path.join(DATA, "chianti")
    os.makedirs(d, exist_ok=True)
    out = []
    for k in range(1, Z_OF[element] + 1):
        files = []
        for ext in ("elvlc", "wgfa"):
            p = os.path.join(d, f"{el}_{k}.{ext}")
            if not os.path.exists(p):
                try:
                    txt = _get(f"{CHIANTI}{el}/{el}_{k}/{el}_{k}.{ext}", timeout=300)
                except Exception:
                    break
                open(p, "w", encoding="utf-8", newline="\n").write(txt.replace("\r\n", "\n"))
            files.append(p)
        if len(files) == 2:
            L = parse_chianti_wgfa(files[1], parse_chianti_elvlc(files[0]))
            L[:, 0] = k - 1
            out.append(L)
    return np.concatenate(out) if out else np.zeros((0, 5))


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
    ap.add_argument("--chianti-lines", action="store_true", help="include CHIANTI lines (Z <= 30; downloaded, MB per ion)")
    ap.add_argument("--no-line-model", action="store_true", help="no screened-hydrogenic lines for stages without data")
    ap.add_argument("--dr", choices=["chianti", "burgess"], help="dielectronic recombination in the non-LTE balance "
                    "(chianti: fitted rates, Z <= 30, recommended; burgess: General Formula, see aegis/tabular.py)")
    ap.add_argument("--fetch-chianti-dr", nargs="*", default=[], help="refresh the bundled CHIANTI DR fits")
    ap.add_argument("--nonlte", action="store_true", help="opacity from the collisional-radiative ion balance")
    ap.add_argument("--tops", help="Z <= 30: TOPS table on TOPS's temperatures and an adaptive density grid, for the "
                    "solver (FLD fld_opacity_table, multigroup fld_mg_table with --tops-groups). An element (Al) or a "
                    "TOPS mixture string ('1 c 1 h')")
    ap.add_argument("--tops-rho", type=float, nargs=2, default=[1e-6, 1e2], help="density range, g/cc")
    ap.add_argument("--tops-tol", type=float, default=0.01, help="bisect where midpoint interpolation misses by more")
    ap.add_argument("--tops-groups", type=float, nargs="*", default=None,
                    help="inner group edges in eV for group means as well (outer groups run to 0 and infinity)")
    ap.add_argument("--mass-fraction", action="store_true", help="the --tops mixture is by mass, not by number")
    ap.add_argument("--isotope", action="store_true", help="the --tops mixture gives isotopic weights")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.tops:
        mix = a.tops if " " in a.tops.strip() else f"1. {a.tops.lower()}"
        tab = fetch_tops_adaptive(mix, tuple(a.tops_rho), tol=a.tops_tol,
                                  fractype="mass" if a.mass_fraction else "atomic", isotope=a.isotope)
        note = (f"Adaptive density grid: every interval's midpoint is within {a.tops_tol:.0%} of TOPS in kappa_R, "
                "kappa_P and Z_free (held-out midpoints stored as heldout_*).")
        if a.tops_groups:
            ek = np.array([1e-5] + [1e-3 * g for g in a.tops_groups] + [3000.0])
            tab["kR_g"], tab["kP_g"] = fetch_tops_groups_at(tab, ek, isotope=a.isotope)
            tab["edges_keV"] = np.concatenate([[0.0], ek[1:-1], [np.inf]])
            note += " Group means from TOPS with outer edges 0.01 eV and 3 MeV, used as 0 and infinity."
        name = re.sub(r"[^a-z0-9]+", "", a.tops.lower())
        p = a.out or os.path.join(DATA, f"tops_{name}_table.npz")
        tops_dense_to_npz(tab, p, note)
        print(f"wrote {p}: {len(tab['rho_gcc'])} densities x {len(tab['T_keV'])} TOPS temperatures, "
              f"{len(tab['heldout_rho_gcc'])} held-out midpoints, {int(tab['offtable'].sum())} points off TOPS's tables")
    for el in a.fetch_nist:
        I = fetch_nist_ie(el, cache=False)
        print(f"{el}: {np.sum(np.isfinite(I))} of {Z_OF.get(el, len(I))} ionization energies, I_0 = {I[0]} eV")
    for el in a.fetch_chianti_dr:
        print(f"{el}: CHIANTI DR fits for {len(fetch_chianti_dr(el, cache=False))} recombining ions")
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
        lines, src = None, "NIST"
        if a.chianti_lines:
            lines, src = fetch_chianti_lines(a.build), "CHIANTI"
        elif a.lines:
            lines = fetch_nist_lines(a.build)
        dr_table = fetch_chianti_dr(a.build) if a.dr == "chianti" else None
        edges = np.array([0.0] + list(a.groups) + [np.inf])
        tab = tabular.build_atomic_table(Z_OF[a.build], A_OF[a.build], I, edges, lines=lines, nonlte=a.nonlte,
                                         line_model=not a.no_line_model, dr=a.dr, dr_table=dr_table, line_source=src)
        out = a.out or f"{a.build.lower()}_atomic.npz"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez(out, **tab)
        print(f"wrote {out}: {tab['kappa_R_g'].shape}, source: {tab['source']}")


if __name__ == "__main__":
    main()
