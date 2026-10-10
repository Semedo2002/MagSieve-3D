# solver3d.py - 3D extension of MHDSolver: unsplit MUSCL-HLLD in x, y, z with GLM + Powell,
# conservative resistive terms (constant eta),
# the same driven-shock boundaries (z periodic) and the pressure-balanced stripe set-up.
import io
import time
import contextlib

import numpy as np

from .config import (
    NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)
from .physics import xp, cons_to_prim, prim_to_cons, hlld_flux_x, parallel_stripe_state
from .solver import MHDSolver, to_numpy

# variable permutations that rotate the y or z face normal onto x for the HLLD kernel
_PERM = {1: list(range(NVAR)), 2: list(range(NVAR)), 3: list(range(NVAR))}
_PERM[2][iVX], _PERM[2][iVY] = iVY, iVX
_PERM[2][iBX], _PERM[2][iBY] = iBY, iBX
_PERM[3][iVX], _PERM[3][iVZ] = iVZ, iVX
_PERM[3][iBX], _PERM[3][iBZ] = iBZ, iBX


def _sl(ax, s):
    idx = [slice(None)] * 4
    idx[ax] = s
    return tuple(idx)


def muscl(W, ax):
    """van Leer MUSCL along array axis ax (1, 2, 3); face k lies between cells k+1 and k+2."""
    dm = W[_sl(ax, slice(1, -1))] - W[_sl(ax, slice(None, -2))]
    dp = W[_sl(ax, slice(2, None))] - W[_sl(ax, slice(1, -1))]
    ab = dm * dp
    d = xp.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[_sl(ax, slice(1, -2))] + 0.5 * d[_sl(ax, slice(None, -1))]
    WR = W[_sl(ax, slice(2, -1))] - 0.5 * d[_sl(ax, slice(1, None))]
    for Wf in (WL, WR):
        Wf[iRHO] = xp.maximum(Wf[iRHO], FLOOR_RHO)
        Wf[iPR] = xp.maximum(Wf[iPR], FLOOR_PR)
    return WL, WR


def hlld_flux(WL, WR, ax, gamma, ch):
    """HLLD flux through faces normal to axis ax, via variable rotation onto x."""
    perm = _PERM[ax]
    shape = WL.shape
    F, _ = hlld_flux_x(xp.ascontiguousarray(WL[perm].reshape(NVAR, -1)),
                       xp.ascontiguousarray(WR[perm].reshape(NVAR, -1)), gamma, ch)
    return F.reshape(shape)[perm]


def _levi(i, j, k):
    return 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0


def resistive_terms_3d(U, eta, h, ng, n):
    """Resistive induction and energy terms as differences of face fluxes.

    Induction -curl(eta J) is the divergence of F_j(B_i) = eps_ijk eta J_k; energy uses the
    resistive Poynting flux eta (J x B). J on a face: compact difference along the face normal,
    averaged centred differences across it (reduces to the 2D form when d/dz = 0).
    Returns (R over all cells, net resistive energy inflow through the domain faces).
    """
    B = [U[BX], U[BY], U[BZ]]
    D = [[None] * 3 for _ in range(3)]          # cell-centred dB_c/dx_d, zero on the outer ring
    for c in range(3):
        for d in range(3):
            a = xp.zeros_like(B[c])
            mid = [slice(None)] * 3; mid[d] = slice(1, -1)
            up = [slice(None)] * 3; up[d] = slice(2, None)
            dn = [slice(None)] * 3; dn[d] = slice(None, -2)
            a[tuple(mid)] = (B[c][tuple(up)] - B[c][tuple(dn)]) / (2 * h[d])
            D[c][d] = a
    R = xp.zeros_like(U)
    rate = xp.zeros(())
    inner = tuple(slice(ng, ng + m) for m in n)
    for ax in range(3):
        lo = [slice(None)] * 3; lo[ax] = slice(None, -1); lo = tuple(lo)
        hi = [slice(None)] * 3; hi[ax] = slice(1, None); hi = tuple(hi)

        def dface(c, d):
            if d == ax:
                return (B[c][hi] - B[c][lo]) / h[ax]
            return 0.5 * (D[c][d][lo] + D[c][d][hi])

        J = (dface(2, 1) - dface(1, 2), dface(0, 2) - dface(2, 0), dface(1, 0) - dface(0, 1))
        Bf = [0.5 * (B[c][lo] + B[c][hi]) for c in range(3)]
        # face index k lies between cells k and k+1: interior cells use F[c] - F[c-1]
        a_hi = list(inner); a_hi[ax] = slice(ng, ng + n[ax]); a_hi = tuple(a_hi)
        a_lo = list(inner); a_lo[ax] = slice(ng - 1, ng + n[ax] - 1); a_lo = tuple(a_lo)
        for i in range(3):
            if i == ax:
                continue
            k = 3 - i - ax
            F = _levi(i, ax, k) * eta * J[k]
            R[(BX + i,) + inner] -= (F[a_hi] - F[a_lo]) / h[ax]
        G = eta * (J[(ax + 1) % 3] * Bf[(ax + 2) % 3] - J[(ax + 2) % 3] * Bf[(ax + 1) % 3])
        R[(EN,) + inner] -= (G[a_hi] - G[a_lo]) / h[ax]
        f_lo = list(inner); f_lo[ax] = ng - 1
        f_hi = list(inner); f_hi[ax] = ng + n[ax] - 1
        area = np.prod([h[j] for j in range(3) if j != ax])
        rate = rate + xp.sum(G[tuple(f_lo)] - G[tuple(f_hi)]) * area
    return R, rate


def _curl(ax_, ay_, az_, dx, dy, dz):
    """curl of a cell-centred vector on interior arrays (central differences)."""
    g = lambda f, h, a: xp.gradient(f, h, axis=a)
    cx = g(az_, dy, 1) - g(ay_, dz, 2)
    cy = g(ax_, dz, 2) - g(az_, dx, 0)
    cz = g(ay_, dx, 0) - g(ax_, dy, 1)
    return cx, cy, cz


class MHDSolver3D:
    """3D MHD with HLLD fluxes on all three face orientations (unsplit, SSP-RK3), optional constant eta."""

    def __init__(self, config):
        cfg = config
        if cfg.nz < 4:
            raise ValueError("solver3d needs nz >= 4")
        self.cfg = cfg
        self.ng = ng = 2
        self.nx_tot, self.ny_tot, self.nz_tot = cfg.nx + 2 * ng, cfg.ny + 2 * ng, cfg.nz + 2 * ng
        self.x = xp.asarray(cfg.x_min + (np.arange(self.nx_tot) - ng + 0.5) * cfg.dx)
        self.y = xp.asarray(cfg.y_min + (np.arange(self.ny_tot) - ng + 0.5) * cfg.dy)
        self.z = xp.asarray(cfg.z_min + (np.arange(self.nz_tot) - ng + 0.5) * cfg.dz)
        shape = (NVAR, self.nx_tot, self.ny_tot, self.nz_tot)
        self.U = xp.zeros(shape)
        self.U_temp = xp.empty(shape)
        self.U_stage = xp.empty(shape)
        self.dU = xp.empty(shape)
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self._ch = 0.0
        self._inflow_U = None
        self._inflow_R = None
        self._acc_boundary = xp.zeros(())
        self._acc_powell = xp.zeros(())
        self._rate_boundary = xp.zeros(())
        self._rate_powell = xp.zeros(())
        self.diag = {}
        self.snapshots = {}

    # ------------------------------------------------------------ initial condition

    def initialize(self):
        """2D pressure-balanced set-up extruded in z, plus optional 3D seeding."""
        cfg = self.cfg
        if cfg.is_striped() and cfg.B_stripe_component != "x":
            raise ValueError("solver3d supports hydro or x stripes (pressure-balanced)")
        if cfg.is_mhd() and not cfg.is_striped():
            raise ValueError("solver3d supports hydro or x stripes (pressure-balanced)")
        s2 = MHDSolver(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            s2.initialize()
        W2 = cons_to_prim(s2.U, cfg.gamma)
        W = xp.repeat(W2[:, :, :, None], self.nz_tot, axis=3)
        self.equilibrium = getattr(s2, "_equilibrium", None)
        self._bar_mask_y, self._gap_mask_y = s2._bar_mask_y, s2._gap_mask_y

        rng = np.random.default_rng(cfg.z_pert_seed)
        X = self.x[:, None, None]
        Y = self.y[None, :, None]
        Z = self.z[None, None, :]
        Ly, Lz = cfg.y_max - cfg.y_min, cfg.z_max - cfg.z_min

        if cfg.z_pert_amp > 0:
            # interface ripple in z: rebuild density and tracer ahead of the shock
            ky = 2 * np.pi * cfg.perturbation_mode / Ly
            kz = 2 * np.pi * cfg.z_pert_mode / Lz
            x_if = cfg.interface_x + cfg.perturbation_amp * xp.sin(ky * Y) + cfg.z_pert_amp * xp.sin(kz * Z)
            phi = 0.5 * (1.0 + xp.tanh((X - x_if) / max(cfg.interface_width * cfg.dx, 1e-10)))
            ahead = (X >= cfg.interface_x - 0.3) * xp.ones_like(phi)
            if cfg.is_striped():
                st = parallel_stripe_state(cfg.gamma, cfg.mach, cfg.p_bar, cfg.equilibrium_closure,
                                           W2[iBX, 0, :])
                rho_l = st["rho1"][None, :, None]
            else:
                rho_l = xp.ones((1, self.ny_tot, 1))
            rho_up = rho_l * (1 - phi) + cfg.density_ratio * rho_l * phi
            W[iRHO] = xp.where(ahead > 0, rho_up, W[iRHO])
            W[iCLR] = xp.where(ahead > 0, phi, W[iCLR])

        self._vz_pattern = None
        if cfg.z_pert_vel > 0:
            cs_bar = np.sqrt(cfg.gamma * cfg.p_bar)
            amp = cfg.z_pert_vel * cs_bar
            W[iVZ] = W[iVZ] + amp * xp.asarray(self._smooth_noise(rng, (self.nx_tot, self.ny_tot, self.nz_tot)))
            self._vz_pattern = amp * xp.asarray(self._smooth_noise(rng, (self.ng, self.ny_tot, self.nz_tot)))

        if cfg.frame_velocity != 0.0:
            W[iVX] = W[iVX] - cfg.frame_velocity      # Galilean shift to the interface frame
        self.U = prim_to_cons(W, cfg.gamma)
        self._inflow_U = self.U[:, :self.ng, :, :].copy()
        # in the shifted frame the upstream gas enters at x_max (supersonic for U > c_s upstream)
        self._inflow_R = self.U[:, -self.ng:, :, :].copy() if cfg.frame_velocity > 0 else None
        if self._vz_pattern is not None:
            Win = cons_to_prim(self._inflow_U, cfg.gamma)
            Win[iVZ] = self._vz_pattern
            self._inflow_U = prim_to_cons(Win, cfg.gamma)
        self.t, self.step = 0.0, 0
        self._acc_boundary = xp.zeros(())
        self._acc_powell = xp.zeros(())

    @staticmethod
    def _smooth_noise(rng, shape, kmax=6):
        """unit-rms random field keeping the lowest kmax modes in every direction"""
        a = rng.standard_normal(shape)
        f = np.fft.rfftn(a)
        for ax, n in enumerate(shape):
            k = np.abs(np.fft.fftfreq(n) * n) if ax < len(shape) - 1 else np.fft.rfftfreq(n) * n
            mask = (k <= kmax).reshape([-1 if i == ax else 1 for i in range(len(shape))])
            f = f * mask
        a = np.fft.irfftn(f, s=shape)
        return a / max(a.std(), 1e-30)

    # ------------------------------------------------------------ boundaries and time step

    def apply_bc(self, U):
        ng = self.ng
        if self.cfg.get_bc_x() == "periodic":
            U[:, :ng] = U[:, -2 * ng:-ng]
            U[:, -ng:] = U[:, ng:2 * ng]
        else:
            U[:, :ng] = self._inflow_U                   # driven inflow at x_min
            if self._inflow_R is not None:
                U[:, -ng:] = self._inflow_R              # upstream inflow at x_max (co-moving frame)
            else:
                for i in range(ng):
                    U[:, -(i + 1)] = U[:, -(ng + 1)]     # transmissive outflow
        U[:, :, :ng] = U[:, :, -2 * ng:-ng]              # periodic y
        U[:, :, -ng:] = U[:, :, ng:2 * ng]
        U[:, :, :, :ng] = U[:, :, :, -2 * ng:-ng]        # periodic z
        U[:, :, :, -ng:] = U[:, :, :, ng:2 * ng]
        return U

    def compute_dt(self, W):
        cfg = self.cfg
        rho = xp.maximum(W[iRHO], FLOOR_RHO)
        p = xp.maximum(W[iPR], FLOOR_PR)
        cf = xp.sqrt(cfg.gamma * p / rho + (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2) / rho)
        sx, sy, sz = xp.abs(W[iVX]) + cf, xp.abs(W[iVY]) + cf, xp.abs(W[iVZ]) + cf
        self._ch = max(float(xp.max(xp.maximum(xp.maximum(sx, sy), sz))), 1e-12)
        h = min(cfg.dx, cfg.dy, cfg.dz)
        inv = max(float(xp.max(xp.maximum(xp.maximum(sx / cfg.dx, sy / cfg.dy), sz / cfg.dz))), self._ch / h)
        dt = cfg.cfl / inv
        if cfg.eta_resistivity > 0:
            # explicit diffusion: dt <= 1 / (2 eta (1/dx^2 + 1/dy^2 + 1/dz^2))
            dt = min(dt, 0.5 / (cfg.eta_resistivity * (cfg.dx**-2 + cfg.dy**-2 + cfg.dz**-2)))
        return dt

    # ------------------------------------------------------------ right-hand side

    def compute_rhs(self, U, out):
        cfg = self.cfg
        ng, g, ch = self.ng, cfg.gamma, self._ch
        n = (cfg.nx, cfg.ny, cfg.nz)
        h = (cfg.dx, cfg.dy, cfg.dz)
        U = self.apply_bc(U)
        W = cons_to_prim(U, g)
        out[...] = 0.0
        inner = (slice(None),) + tuple(slice(ng, ng + m) for m in n)
        rate = xp.zeros(())
        for ax in (1, 2, 3):
            WL, WR = muscl(W, ax)
            F = hlld_flux(WL, WR, ax, g, ch)
            m = n[ax - 1]
            dF = F[_sl(ax, slice(1, 1 + m))] - F[_sl(ax, slice(0, m))]
            sel = list(inner)
            sel[ax] = slice(None)
            out[inner] -= dF[tuple(sel)] / h[ax - 1]
            # net energy inflow through the two domain faces normal to ax
            face_area = np.prod([h[i] for i in range(3) if i != ax - 1])
            lo = list(sel); lo[0] = EN; lo[ax] = 0
            hi = list(sel); hi[0] = EN; hi[ax] = m
            rate = rate + xp.sum(F[tuple(lo)] - F[tuple(hi)]) * face_area
        self._rate_boundary = rate
        self._rate_powell = xp.zeros(())
        if cfg.powell_source and cfg.is_mhd():
            divB = xp.zeros_like(W[iBX])
            c = (slice(1, -1),) * 3
            divB[c] = ((W[iBX][2:, 1:-1, 1:-1] - W[iBX][:-2, 1:-1, 1:-1]) / (2 * cfg.dx)
                       + (W[iBY][1:-1, 2:, 1:-1] - W[iBY][1:-1, :-2, 1:-1]) / (2 * cfg.dy)
                       + (W[iBZ][1:-1, 1:-1, 2:] - W[iBZ][1:-1, 1:-1, :-2]) / (2 * cfg.dz))
            vB = W[iVX] * W[iBX] + W[iVY] * W[iBY] + W[iVZ] * W[iBZ]
            S = xp.zeros_like(U)
            S[MX], S[MY], S[MZ] = -divB * W[iBX], -divB * W[iBY], -divB * W[iBZ]
            S[BX], S[BY], S[BZ] = -divB * W[iVX], -divB * W[iVY], -divB * W[iVZ]
            S[EN] = -divB * vB
            out[inner] += S[inner]
            self._rate_powell = xp.sum(S[EN][inner[1:]]) * cfg.dx * cfg.dy * cfg.dz
        if cfg.eta_resistivity > 0:
            Rr, rate_r = resistive_terms_3d(U, cfg.eta_resistivity, h, ng, n)
            out[inner] += Rr[inner]
            self._rate_boundary = self._rate_boundary + rate_r
        return out

    def _acc(self, w):
        self._acc_boundary += w * self._rate_boundary
        self._acc_powell += w * self._rate_powell

    def step_ssprk3(self):
        dt = self.dt
        U0 = self.U
        self.compute_rhs(U0, self.dU); self._acc(dt / 6)
        xp.add(U0, dt * self.dU, out=self.U_temp)
        self.compute_rhs(self.U_temp, self.dU); self._acc(dt / 6)
        self.U_stage[...] = 0.75 * U0 + 0.25 * (self.U_temp + dt * self.dU)
        self.compute_rhs(self.U_stage, self.dU); self._acc(2 * dt / 3)
        self.U[...] = U0 / 3 + 2.0 / 3.0 * (self.U_stage + dt * self.dU)
        rho = xp.maximum(self.U[RHO], FLOOR_RHO)
        self.U[RHOC] = rho * xp.clip(self.U[RHOC] / rho, 0.0, 1.0)
        self.apply_bc(self.U)
        if self._ch > 0:
            h = min(self.cfg.dx, self.cfg.dy, self.cfg.dz)
            self.U[PSI] *= np.exp(-self.cfg.glm_alpha * self._ch * dt / h)

    # ------------------------------------------------------------ diagnostics

    def diagnostics(self):
        cfg = self.cfg
        ng = self.ng
        I = (slice(ng, -ng),) * 3
        W = cons_to_prim(self.U, cfg.gamma)
        rho, C = W[iRHO][I], xp.clip(W[iCLR][I], 0, 1)
        vx, vy, vz = W[iVX][I], W[iVY][I], W[iVZ][I]
        bx, by, bz = W[iBX][I], W[iBY][I], W[iBZ][I]
        dx, dy, dz = cfg.dx, cfg.dy, cfg.dz
        dV = dx * dy * dz
        dA = dV / (cfg.z_max - cfg.z_min)    # per unit depth: comparable across z-extents and with 2D
        mix = (C > 0.05) & (C < 0.95)

        # interface displacement eta(y, z) = x_min + int (1 - C) dx
        eta = cfg.x_min + xp.sum(1.0 - C, axis=0) * dx
        eta_f = eta - xp.mean(eta)
        eta_zmean = xp.mean(eta, axis=1)
        prof = to_numpy(eta_zmean)
        a = np.abs(np.fft.rfft(prof - prof.mean())) * 2 / cfg.ny   # host FFT: cuFFT not required
        k4 = cfg.perturbation_mode

        wx, wy, wz = _curl(vx, vy, vz, dx, dy, dz)
        jx, jy, jz = _curl(bx, by, bz, dx, dy, dz)
        J = xp.sqrt(jx**2 + jy**2 + jz**2)
        J_rms = float(xp.sqrt(xp.mean(J**2)))
        divB = (xp.gradient(bx, dx, axis=0) + xp.gradient(by, dy, axis=1) + xp.gradient(bz, dz, axis=2))
        bmax = float(xp.max(xp.sqrt(bx**2 + by**2 + bz**2)))
        E = float(xp.sum(self.U[EN][I]) * dV)
        return {
            "t": self.t,
            "a4": float(a[k4]) if k4 < a.size else 0.0,
            "x_int": float(xp.mean(eta)),
            "rough": float(xp.sqrt(xp.mean(eta_f**2))),
            "rough_z": float(xp.sqrt(xp.mean((eta - eta_zmean[:, None])**2))),   # z-variation of the interface
            "emix": float(xp.sum(xp.where(mix, rho * (wx**2 + wy**2 + wz**2), 0.0)) * dA),
            "Ekz": float(xp.sum(0.5 * rho * vz**2) * dA),
            "Eperp": float(xp.sum(0.5 * (by**2 + bz**2)) * dA),           # field generated across B_x
            "Emag": float(xp.sum(0.5 * (bx**2 + by**2 + bz**2)) * dA),
            "J_max": float(xp.max(J)),
            "J_rms": J_rms,
            "J_mix_rms": float(xp.sqrt(xp.sum(xp.where(mix, J**2, 0.0)) / max(int(xp.sum(mix)), 1))),
            "sheet_frac": float(xp.mean(J > 5 * J_rms)) if J_rms > 0 else 0.0,
            "J_xmod": self._sheet_xmod(J, C),
            "J_zvar": float(xp.sqrt(xp.mean((J - xp.mean(J, axis=2, keepdims=True))**2))) / J_rms if J_rms > 0 else 0.0,
            "divB_rel": float(xp.max(xp.abs(divB))) * min(dx, dy, dz) / bmax if bmax > 0 else 0.0,
            "E": E,
            "inflow": float(self._acc_boundary),
            "powell": float(self._acc_powell),
        }

    def _sheet_xmod(self, J, C):
        """rms x-variation / mean of z-averaged |J| along the current-sheet rows, in the shocked heavy gas
        more than 0.5 behind the interface (tearing breaks a sheet into islands along x = along B)"""
        Jm = xp.mean(J, axis=2)                       # (nx, ny)
        heavy = xp.mean(C, axis=(1, 2)) > 0.95
        if int(xp.sum(heavy)) < 8:
            return 0.0
        prof = xp.mean(Jm[heavy], axis=0)
        rows = prof > 0.5 * xp.max(prof)
        seg = Jm[heavy][:, rows]
        return float(xp.mean(xp.std(seg, axis=0) / xp.maximum(xp.mean(seg, axis=0), 1e-30)))

    def save_snapshot(self, label):
        cfg = self.cfg
        ng = self.ng
        I = (slice(ng, -ng),) * 3
        W = cons_to_prim(self.U, cfg.gamma)
        bx, by, bz = W[iBX][I], W[iBY][I], W[iBZ][I]
        vx, vy, vz = W[iVX][I], W[iVY][I], W[iVZ][I]
        jx, jy, jz = _curl(bx, by, bz, cfg.dx, cfg.dy, cfg.dz)
        wx, wy, wz = _curl(vx, vy, vz, cfg.dx, cfg.dy, cfg.dz)
        k = cfg.nz // 2
        self.snapshots[label] = {
            "t": self.t,
            "rho_mid": to_numpy(W[iRHO][I][:, :, k]),
            "C_mid": to_numpy(W[iCLR][I][:, :, k]),
            "J_mid": to_numpy(xp.sqrt(jx**2 + jy**2 + jz**2)[:, :, k]),
            "w_mid": to_numpy(xp.sqrt(wx**2 + wy**2 + wz**2)[:, :, k]),
            "vz_mid": to_numpy(vz[:, :, k]),
            # y-z cut through the mixing layer at the mean interface position
            "J_yz": to_numpy(xp.sqrt(jx**2 + jy**2 + jz**2)[int(xp.argmax(xp.mean(
                xp.abs(xp.gradient(W[iCLR][I], axis=0)), axis=(1, 2))))]),
        }

    # ------------------------------------------------------------ run

    def run(self, snapshot_times=(), verbose=True):
        cfg = self.cfg
        t0 = time.time()
        self.diag = {}

        def record():
            for k, v in self.diagnostics().items():
                self.diag.setdefault(k, []).append(v)

        record()
        pending = sorted(snapshot_times)
        while self.t < cfg.t_end and self.step < cfg.max_steps:
            self.dt = min(self.compute_dt(cons_to_prim(self.U, cfg.gamma)), cfg.t_end - self.t)
            if self.dt <= 1e-16:
                break
            self.step_ssprk3()
            self.t += self.dt
            self.step += 1
            if self.step % cfg.diag_interval == 0:
                record()
            while pending and self.t >= pending[0] - 0.5 * self.dt:
                self.save_snapshot(f"t={pending.pop(0):.3f}")
            if verbose and self.step % 200 == 0:
                d = self.diag
                print(f"  step {self.step:5d} t={self.t:.4f} dt={self.dt:.2e} rough={d['rough'][-1]:.4f} "
                      f"Ekz={d['Ekz'][-1]:.3e} Eperp={d['Eperp'][-1]:.3e}", flush=True)
        record()
        self.wall = time.time() - t0
        self.diag = {k: np.array(v) for k, v in self.diag.items()}
        return self.diag

    def energy_error(self):
        d = self.diag
        return abs(d["E"][-1] - d["E"][0] - d["inflow"][-1] - d["powell"][-1]) / abs(d["E"][0])
