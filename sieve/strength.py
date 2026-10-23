# strength.py - elastic-perfectly-plastic liner strength (2D, Cartesian or cylindrical r-z).
#
# Deviatoric stress S (S_xx, S_yy, S_xy; S_zz = -(S_xx + S_yy), in r-z S_zz is the hoop stress S_phiphi) and a
# damage flag ride along with the material (advected with the HLLD mass flux, upwind). Hypoelastic Jaumann rate:
#     dS/dt = 2 G (D - tr(D)/3 I) + S W - W S,   D, W = symmetric / antisymmetric parts of grad v
# (in r-z D_phiphi = v_r / r), then radial return to the von Mises surface sqrt(3 J2) <= Y. G and Y scale with
# the liner fraction. Spall: a liner cell whose pressure falls below -spall is damaged for good: no strength,
# and no tension (its pressure is floored at 0). The stress enters momentum and energy as face fluxes
# (div S, div(S . v)), plus the hoop term -S_phiphi / r in r-z, so total energy stays conserved exactly.
from .config import iRHO, iVX, iVY, iCLR, RHO, MX, MY, EN
from .physics import xp

SXX, SYY, SXY, DMG = 0, 1, 2, 3


def bc(S, ng, nx, ny, periodic, cyl):
    """ghost cells: periodic wrap, zero gradient, or the axis mirror (S_xy odd) in r-z"""
    for ax, (per, n) in enumerate(zip(periodic, (nx, ny))):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        if per:
            lo[1 + ax], hi[1 + ax] = slice(0, ng), slice(n, n + ng)
            S[tuple(lo)] = S[tuple(hi)]
            lo[1 + ax], hi[1 + ax] = slice(ng + n, n + 2 * ng), slice(ng, 2 * ng)
            S[tuple(lo)] = S[tuple(hi)]
        else:
            for k in range(ng):
                lo[1 + ax], hi[1 + ax] = k, ng
                S[tuple(lo)] = S[tuple(hi)]
                lo[1 + ax], hi[1 + ax] = ng + n + k, ng + n - 1
                S[tuple(lo)] = S[tuple(hi)]
    if cyl:
        for k in range(ng):
            S[:, ng - 1 - k] = S[:, ng + k]
        S[SXY, :ng] = -S[SXY, :ng]


def _advect(q, F0, WL, WR, axis, h, ng, n, wf=None, wc=None):
    """-u . grad q along one axis (non-conservative, upwind face values by the sign of the mass flux)"""
    rho_up = xp.where(F0 >= 0, WL[iRHO], WR[iRHO])
    u = F0 / rho_up
    if axis == 0:
        qf = xp.where(F0 >= 0, q[1:n + 2], q[2:n + 3])       # faces 2..n+2 lie between cells (k+1, k+2)
        qc = q[ng:ng + n]
        hi, lo = slice(1, n + 1), slice(0, n)
        uq, uu = u * qf, u
        if wf is not None:
            uq, uu = uq * wf, uu * wf
            return -((uq[hi] - uq[lo]) - qc * (uu[hi] - uu[lo])) / (h * wc)
        return -((uq[hi] - uq[lo]) - qc * (uu[hi] - uu[lo])) / h
    qf = xp.where(F0 >= 0, q[:, 1:n + 2], q[:, 2:n + 3])
    qc = q[:, ng:ng + n]
    uq = u * qf
    return -((uq[:, 1:n + 1] - uq[:, :n]) - qc * (u[:, 1:n + 1] - u[:, :n])) / h


def rates(S, W, Fx, Fy, WLx, WRx, WLy, WRy, G, h, ng, n, rc=None, rf=None):
    """dS/dt on interior cells (advection + elastic rate), and the stress fluxes of momentum and energy:
    returns (dS, Pmx, Pmy, PE) with P* the x- and y-face fluxes [(Fx_mx, Fx_my, Fx_E), (Fy_mx, Fy_my, Fy_E)]
    to be subtracted like the hydro fluxes, and the hoop source for r-z (or None)."""
    nx, ny = n
    dx, dy = h
    cyl = rc is not None
    I = (slice(ng, ng + nx), slice(ng, ng + ny))
    u, v = W[iVX], W[iVY]
    dudx = (u[ng + 1:ng + nx + 1, ng:ng + ny] - u[ng - 1:ng + nx - 1, ng:ng + ny]) / (2 * dx)
    dvdx = (v[ng + 1:ng + nx + 1, ng:ng + ny] - v[ng - 1:ng + nx - 1, ng:ng + ny]) / (2 * dx)
    dudy = (u[ng:ng + nx, ng + 1:ng + ny + 1] - u[ng:ng + nx, ng - 1:ng + ny - 1]) / (2 * dy)
    dvdy = (v[ng:ng + nx, ng + 1:ng + ny + 1] - v[ng:ng + nx, ng - 1:ng + ny - 1]) / (2 * dy)
    Dzz = u[I] / rc[ng:ng + nx, None] if cyl else 0.0
    trD = dudx + dvdy + Dzz
    Wxy = 0.5 * (dudy - dvdx)
    Sxx, Syy, Sxy = S[SXX][I], S[SYY][I], S[SXY][I]
    dS = xp.zeros((4, nx, ny))
    dS[SXX] = 2 * G[I] * (dudx - trD / 3) - 2 * Sxy * Wxy
    dS[SYY] = 2 * G[I] * (dvdy - trD / 3) + 2 * Sxy * Wxy
    dS[SXY] = 2 * G[I] * 0.5 * (dudy + dvdx) + Wxy * (Sxx - Syy)
    wf = rf[2:nx + 3][:, None] if cyl else None
    wc = rc[ng:ng + nx][:, None] if cyl else None
    for c in range(4):
        dS[c] += _advect(S[c], Fx[RHO], WLx, WRx, 0, dx, ng, nx, wf, wc)[:, ng:ng + ny]
        dS[c] += _advect(S[c], Fy[RHO], WLy, WRy, 1, dy, ng, ny)[ng:ng + nx, :]
    # momentum flux -S and energy flux -(S . v) on the faces (cell averages)
    avgx = lambda f: 0.5 * (f[1:nx + 2] + f[2:nx + 3])      # x-faces 2..nx+2, all rows
    avgy = lambda f: 0.5 * (f[:, 1:ny + 2] + f[:, 2:ny + 3])
    fx = (-avgx(S[SXX]), -avgx(S[SXY]), -avgx(S[SXX] * u + S[SXY] * v))
    fy = (-avgy(S[SXY]), -avgy(S[SYY]), -avgy(S[SXY] * u + S[SYY] * v))
    hoop = (Sxx + Syy) / rc[ng:ng + nx, None] if cyl else None   # -S_phiphi / r
    return dS, fx, fy, hoop


def plastic(S, a, p, Y, spall, ng):
    """radial return to sqrt(3 J2) <= Y a (1 - damage); spall damage where p < -spall in the liner"""
    dmg = S[DMG]
    if spall > 0:
        dmg[...] = xp.where((p < -spall) & (a > 0.5), 1.0, xp.clip(dmg, 0.0, 1.0))
    Sxx, Syy, Sxy = S[SXX], S[SYY], S[SXY]
    Szz = -(Sxx + Syy)
    seq = xp.sqrt(1.5 * (Sxx**2 + Syy**2 + Szz**2 + 2 * Sxy**2))
    Ye = Y * a * (1 - dmg)
    f = xp.where(seq > Ye, Ye / xp.maximum(seq, 1e-300), 1.0)
    for c in (SXX, SYY, SXY):
        S[c] *= f
