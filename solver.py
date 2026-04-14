# solver.py — The Core Machine
# MHDSolver class with Magnetic Nozzle / Sieve physics,
# PostProcessor with comprehensive visualization suite.

import time
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm

matplotlib.rcParams.update({
    "font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
    "figure.dpi": 140, "savefig.dpi": 180,
    "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 9,
    "figure.facecolor": "white",
})

from config import (
    Config, NVAR, FLOOR_RHO, FLOOR_PR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
)
from physics import (
    xp, cp,
    cons_to_prim, prim_to_cons,
    muscl_x, muscl_y,
    hlld_flux_x, hlld_flux_y,
    mhd_rankine_hugoniot,
    richtmyer_linear_theory,
    smooth,
)

OUTPUT_DIR = "rmi_output_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


def save_figure(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=180)
    plt.close(fig)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        return True, filepath, size_kb
    return False, filepath, 0


def to_numpy(a):
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


# ============================================================
# MHD Solver Class
# ============================================================
class MHDSolver:
    """2D Ideal MHD solver with HLLD, MUSCL, SSP-RK3, GLM+Powell.
    Supports uniform and spatially modulated (striped) magnetic fields."""

    def __init__(self, config):
        self.cfg = config
        self.ng = 2
        self.nx_tot = config.nx + 2 * self.ng
        self.ny_tot = config.ny + 2 * self.ng

        self.x = xp.asarray(config.x_min + (np.arange(self.nx_tot) - self.ng + 0.5) * config.dx)
        self.y = xp.asarray(config.y_min + (np.arange(self.ny_tot) - self.ng + 0.5) * config.dy)
        self.X, self.Y = xp.meshgrid(self.x, self.y, indexing='ij')

        self.U = xp.zeros((NVAR, self.nx_tot, self.ny_tot))
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self._glm_ch_frozen = 0.0

        self.diag_times = []
        self.diag_mixing_width_integral = []
        self.diag_mixing_width_thresh = []
        self.diag_mixedness = []
        self.diag_enstrophy = []
        self.diag_enstrophy_local = []
        self.diag_perturbation_amp = []
        self.diag_mode_amps = []
        self.diag_stag_pressure = []
        self.diag_divB_max = []
        self.diag_divB_L2 = []
        self.diag_energy_total = []
        self.diag_boundary_flux_cumulative = []
        # Nozzle / jet diagnostics
        self.diag_jet_max_vx = []
        self.diag_jet_collimation = []
        self.diag_jet_momentum_flux_gap = []
        self.diag_jet_momentum_flux_bar = []
        self._cumulative_boundary_energy = 0.0
        self.snapshots = {}
        self._rho_light = 1.0
        self._rho_heavy = 1.0
        self._rh_results = None
        self._bar_mask_y = None
        self._gap_mask_y = None

    def _build_bar_gap_masks(self):
        """Build y-index masks for bar (high |B|) and gap (low |B|) regions."""
        cfg = self.cfg
        ng = self.ng
        y_arr = to_numpy(self.y[ng:-ng])
        Ly = cfg.y_max - cfg.y_min
        k_mod = 2 * np.pi * cfg.B_modulation_mode / Ly
        By_profile = np.cos(k_mod * y_arr)
        self._bar_mask_y = np.abs(By_profile) > 0.5
        self._gap_mask_y = np.abs(By_profile) <= 0.5

    def initialize(self):
        """Set up initial conditions with uniform or striped magnetic field."""
        cfg = self.cfg
        g = cfg.gamma
        M = cfg.mach
        rho1 = 1.0; p1 = 1.0
        By_pre_eff = cfg.B_transverse
        # For striped fields, use RMS value for R-H jump estimates
        if cfg.is_striped():
            By_pre_eff = cfg.B_transverse / np.sqrt(2.0)

        rho2, p2, vx2, By2, vs = mhd_rankine_hugoniot(g, M, rho1, p1, By_pre_eff)
        self._rh_results = {'rho2': rho2, 'p2': p2, 'vx2': vx2, 'By2': By2, 'vs': vs}

        rho_h = rho1 * cfg.density_ratio
        x_shock = cfg.interface_x - 0.3
        Ly = cfg.y_max - cfg.y_min
        ky = 2 * np.pi * cfg.perturbation_mode / Ly

        x_if = cfg.interface_x + cfg.perturbation_amp * xp.sin(ky * self.Y)
        interface_delta = cfg.interface_width * cfg.dx
        phi = 0.5 * (1.0 + xp.tanh((self.X - x_if) / max(interface_delta, 1e-10)))

        W = xp.zeros((NVAR, self.nx_tot, self.ny_tot))
        post = self.X < x_shock
        rho_pre = rho1 * (1 - phi) + rho_h * phi

        W[iRHO] = xp.where(post, rho2, rho_pre)
        W[iVX] = xp.where(post, vx2, 0.0)
        W[iPR] = xp.where(post, p2, p1)
        W[iCLR] = xp.where(post, 0.0, phi)

        # --- Magnetic field setup ---
        if cfg.is_striped():
            k_mod = 2 * np.pi * cfg.B_modulation_mode / Ly
            By_spatial = cfg.B_transverse * xp.cos(k_mod * self.Y)
            By2_spatial = By_spatial * (rho2 / rho1)
            W[iBY] = xp.where(post, By2_spatial, By_spatial)
            self._build_bar_gap_masks()
            print(f"  STRIPED B-FIELD: By = {cfg.B_transverse:.2f} * cos({cfg.B_modulation_mode}*k*y)")
            print(f"  Modulation mode: {cfg.B_modulation_mode}, bars/gaps: {cfg.B_modulation_mode} pairs")
        elif cfg.is_mhd():
            W[iBY] = xp.where(post, By2, cfg.B_transverse)
            print(f"  UNIFORM B-FIELD: By = {cfg.B_transverse:.3f}")
        else:
            print(f"  B_y=0 (pure hydro)")

        self.U = prim_to_cons(W, g)
        self.t = 0.0; self.step = 0
        self._cumulative_boundary_energy = 0.0
        self._rho_light = rho1; self._rho_heavy = rho_h

        print(f"  Mach={M:.1f}, Vs={vs:.3f}, rho2={rho2:.3f}, p2={p2:.3f}, vx2={vx2:.3f}")
        if cfg.is_mhd():
            beta = 2*p1/max(By_pre_eff**2, 1e-14)
            va = By_pre_eff/np.sqrt(rho1)
            print(f"  By_eff={By_pre_eff:.3f}, beta={beta:.2f}, v_A={va:.3f}")
        bc_x = cfg.get_bc_x()
        print(f"  Grid: {cfg.nx}x{cfg.ny}, dx={cfg.dx:.4f}")
        if cfg.powell_source:
            print(f"  Powell source terms: ENABLED")
        print(f"  BC x: {bc_x}, BC y: {cfg.bc_y_type}")
        sys.stdout.flush()

    def apply_bc(self, U):
        ng = self.ng; cfg = self.cfg; bc_x = cfg.get_bc_x()
        if bc_x == "periodic":
            U[:, :ng, :] = U[:, -2*ng:-ng, :]
            U[:, -ng:, :] = U[:, ng:2*ng, :]
        elif bc_x == "characteristic":
            self._apply_characteristic_bc_x(U)
        else:
            for i in range(ng):
                U[:, i, :] = U[:, ng, :]
                U[:, -(i+1), :] = U[:, -(ng+1), :]
        if bc_x != "periodic":
            U[PSI, :ng, :] = 0; U[PSI, -ng:, :] = 0
        if cfg.bc_y_type == "periodic":
            U[:, :, :ng] = U[:, :, -2*ng:-ng]
            U[:, :, -ng:] = U[:, :, ng:2*ng]
        else:
            for j in range(ng):
                U[:, :, j] = U[:, :, ng]
                U[:, :, -(j+1)] = U[:, :, -(ng+1)]
        return U

    def _apply_characteristic_bc_x(self, U):
        ng = self.ng
        for i in range(ng):
            U[:, i, :] = U[:, ng, :]
        for i in range(ng):
            U[:, -(i+1), :] = U[:, -(ng+1), :]
        for i in range(ng):
            U[PSI, -(i+1), :] *= 0.1
            U[PSI, i, :] *= 0.1

    def enforce_scalar_bounds(self, U):
        rho = xp.maximum(U[RHO], FLOOR_RHO)
        C = xp.clip(U[RHOC] / rho, 0.0, 1.0)
        U[RHOC] = rho * C
        return U

    def compute_dt(self, W):
        cfg = self.cfg
        rho = xp.maximum(W[iRHO], FLOOR_RHO)
        p = xp.maximum(W[iPR], FLOOR_PR)
        B2 = W[iBX]**2 + W[iBY]**2 + W[iBZ]**2
        cf = xp.sqrt(cfg.gamma * p / rho + B2 / rho)
        cf_max = float(xp.max(cf))
        v_abs_max = float(xp.max(xp.abs(W[iVX]) + xp.abs(W[iVY])))
        cfg.glm_ch = max(cf_max + v_abs_max, 1.0) * 1.5
        ch = cfg.glm_ch
        sx = xp.abs(W[iVX]) + xp.maximum(cf, ch)
        sy = xp.abs(W[iVY]) + xp.maximum(cf, ch)
        inv_dt = xp.maximum(sx / cfg.dx, sy / cfg.dy)
        sm = float(xp.max(inv_dt))
        if sm < 1e-14:
            return cfg.cfl * min(cfg.dx, cfg.dy)
        return cfg.cfl / sm

    def _compute_boundary_energy_flux(self, U):
        ng = self.ng; cfg = self.cfg
        if cfg.get_bc_x() == "periodic":
            return 0.0
        W = cons_to_prim(U, cfg.gamma)
        def boundary_flux(idx):
            rho_b = W[iRHO, idx, ng:-ng]; vx_b = W[iVX, idx, ng:-ng]
            p_b = W[iPR, idx, ng:-ng]; Bx_b = W[iBX, idx, ng:-ng]
            By_b = W[iBY, idx, ng:-ng]; Bz_b = W[iBZ, idx, ng:-ng]
            B2_b = Bx_b**2 + By_b**2 + Bz_b**2; pt_b = p_b + 0.5*B2_b
            E_b = U[EN, idx, ng:-ng]
            vB_b = vx_b*Bx_b + W[iVY, idx, ng:-ng]*By_b + W[iVZ, idx, ng:-ng]*Bz_b
            return float(xp.sum(((E_b + pt_b)*vx_b - Bx_b*vB_b) * cfg.dy))
        return boundary_flux(ng) - boundary_flux(-(ng+1))

    def _compute_powell_source(self, U, W):
        cfg = self.cfg
        S = xp.zeros_like(U)
        Bx = W[iBX]; By_f = W[iBY]
        divB = xp.zeros_like(Bx)
        divB[1:-1, 1:-1] = (
            (0.5*(Bx[1:-1, 1:-1] + Bx[2:, 1:-1]) -
             0.5*(Bx[:-2, 1:-1] + Bx[1:-1, 1:-1])) / cfg.dx +
            (0.5*(By_f[1:-1, 1:-1] + By_f[1:-1, 2:]) -
             0.5*(By_f[1:-1, :-2] + By_f[1:-1, 1:-1])) / cfg.dy
        )
        vB = W[iVX]*W[iBX] + W[iVY]*W[iBY] + W[iVZ]*W[iBZ]
        S[MX] = -divB * W[iBX]; S[MY] = -divB * W[iBY]; S[MZ] = -divB * W[iBZ]
        S[BX] = -divB * W[iVX]; S[BY] = -divB * W[iVY]; S[BZ] = -divB * W[iVZ]
        S[EN] = -divB * vB
        return S

    def compute_rhs(self, U):
        cfg = self.cfg; ng = self.ng; nx = cfg.nx; ny = cfg.ny
        g = cfg.gamma; ch = self._glm_ch_frozen
        U = self.apply_bc(U)
        W = cons_to_prim(U, g)
        WLx, WRx = muscl_x(W); s = WLx.shape
        Fx, _ = hlld_flux_x(xp.ascontiguousarray(WLx.reshape(NVAR, -1)),
                             xp.ascontiguousarray(WRx.reshape(NVAR, -1)), g, ch)
        Fx = Fx.reshape(s)
        WLy, WRy = muscl_y(W); s2 = WLy.shape
        Fy, _ = hlld_flux_y(xp.ascontiguousarray(WLy.reshape(NVAR, -1)),
                             xp.ascontiguousarray(WRy.reshape(NVAR, -1)), g, ch)
        Fy = Fy.reshape(s2)
        dFx = Fx[:, 1:1+nx, :] - Fx[:, 0:nx, :]
        dFy = Fy[:, :, 1:1+ny] - Fy[:, :, 0:ny]
        R = xp.zeros_like(U)
        R[:, ng:ng+nx, ng:ng+ny] -= dFx[:, :, ng:ng+ny] / cfg.dx
        R[:, ng:ng+nx, ng:ng+ny] -= dFy[:, ng:ng+nx, :] / cfg.dy
        # Powell source for ANY MHD case (uniform or striped)
        if cfg.powell_source and cfg.is_mhd():
            S = self._compute_powell_source(U, W)
            R[:, ng:ng+nx, ng:ng+ny] += S[:, ng:ng+nx, ng:ng+ny]
        return R

    def step_ssprk3(self):
        dt = self.dt; self._glm_ch_frozen = self.cfg.glm_ch
        U0 = self.U.copy()
        bflux = self._compute_boundary_energy_flux(U0)
        self._cumulative_boundary_energy += bflux * dt
        U1 = U0 + dt * self.compute_rhs(U0)
        U1 = self.enforce_scalar_bounds(U1); U1 = self.apply_bc(U1)
        U2 = 0.75*U0 + 0.25*(U1 + dt*self.compute_rhs(U1))
        U2 = self.enforce_scalar_bounds(U2); U2 = self.apply_bc(U2)
        self.U = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*self.compute_rhs(U2))
        self.U = self.enforce_scalar_bounds(self.U); self.U = self.apply_bc(self.U)
        if self._glm_ch_frozen > 0:
            decay = np.exp(-self.cfg.glm_alpha * self._glm_ch_frozen * dt / min(self.cfg.dx, self.cfg.dy))
            self.U[PSI] *= decay

    def compute_diagnostics(self):
        ng = self.ng; cfg = self.cfg
        W = cons_to_prim(self.U, cfg.gamma)
        rho = to_numpy(W[iRHO, ng:-ng, ng:-ng])
        p = to_numpy(W[iPR, ng:-ng, ng:-ng])
        vx = to_numpy(W[iVX, ng:-ng, ng:-ng])
        vy = to_numpy(W[iVY, ng:-ng, ng:-ng])
        Bx_f = to_numpy(W[iBX, ng:-ng, ng:-ng])
        By_f = to_numpy(W[iBY, ng:-ng, ng:-ng])
        Bz_f = to_numpy(W[iBZ, ng:-ng, ng:-ng])
        C = to_numpy(W[iCLR, ng:-ng, ng:-ng])
        x_arr = to_numpy(self.x[ng:-ng]); y_arr = to_numpy(self.y[ng:-ng])
        nx_int, ny_int = rho.shape

        C_clamped = np.clip(C, 0.0, 1.0)
        C_bar = np.mean(C_clamped, axis=1)
        integrand_mw = C_bar * (1.0 - C_bar)
        mw_integral = float(np.trapezoid(integrand_mw, x_arr))

        mixed = np.where((C_bar > 0.01) & (C_bar < 0.99))[0]
        mw_thresh = float(x_arr[mixed[-1]] - x_arr[mixed[0]]) if len(mixed) > 1 else 0.0

        C_bar_2d = C_bar[:, None]
        C_prime_sq = np.mean((C_clamped - C_bar_2d)**2, axis=1)
        denom_mix = np.maximum(C_bar * (1.0 - C_bar), 1e-14)
        if len(mixed) > 1:
            num = float(np.trapezoid(C_prime_sq[mixed], x_arr[mixed]))
            den = float(np.trapezoid(denom_mix[mixed], x_arr[mixed]))
            mixedness = float(np.clip(1.0 - num / max(den, 1e-14), 0.0, 1.0))
        else:
            mixedness = 0.0

        drho_dx = np.abs(np.gradient(C_clamped, cfg.dx, axis=0))
        interface_pos = np.full(ny_int, np.nan)
        for j in range(ny_int):
            grad_col = drho_dx[:, j]
            if np.max(grad_col) > 1e-6:
                ix_peak = np.argmax(grad_col)
                hw = min(10, ix_peak, nx_int - ix_peak - 1)
                if hw > 0:
                    sl = slice(ix_peak-hw, ix_peak+hw+1)
                    weights = grad_col[sl]; ws = np.sum(weights)
                    if ws > 1e-12:
                        interface_pos[j] = np.sum(x_arr[sl] * weights) / ws

        valid = ~np.isnan(interface_pos); n_valid = np.sum(valid)
        if n_valid > ny_int // 2:
            pos_valid = interface_pos[valid]; y_valid = y_arr[valid]
            pos_interp = np.interp(y_arr, y_valid, pos_valid) if n_valid < ny_int else pos_valid
            pos_fluct = pos_interp - np.mean(pos_interp)
            modes = np.fft.rfft(pos_fluct)
            mode_amps = 2.0 * np.abs(modes) / ny_int
            pert_amp = float(mode_amps[cfg.perturbation_mode]) if cfg.perturbation_mode < len(mode_amps) else 0.0
        else:
            mode_amps = np.zeros(ny_int // 2 + 1); pert_amp = 0.0

        dvydx = np.gradient(vy, cfg.dx, axis=0)
        dvxdy = np.gradient(vx, cfg.dy, axis=1)
        omega = dvydx - dvxdy
        enstrophy_global = float(np.mean(rho * omega**2))

        if n_valid > ny_int // 2:
            x_contact = float(np.mean(interface_pos[valid]))
            ix_lo = max(np.searchsorted(x_arr, x_contact - 0.5), 0)
            ix_hi = min(np.searchsorted(x_arr, x_contact + 0.5), nx_int)
            if ix_hi - ix_lo > 3:
                enstrophy_local = float(np.mean(rho[ix_lo:ix_hi,:] * omega[ix_lo:ix_hi,:]**2))
            else:
                enstrophy_local = enstrophy_global
        else:
            enstrophy_local = enstrophy_global

        v2 = vx**2 + vy**2
        B2 = Bx_f**2 + By_f**2 + Bz_f**2
        stag = float(np.max(p + 0.5*rho*v2 + 0.5*B2))

        divB_field = np.zeros_like(Bx_f)
        if Bx_f.shape[0] > 2 and Bx_f.shape[1] > 2:
            divB_field[1:-1, 1:-1] = (
                (0.5*(Bx_f[1:-1, 1:-1] + Bx_f[2:, 1:-1]) -
                 0.5*(Bx_f[:-2, 1:-1] + Bx_f[1:-1, 1:-1])) / cfg.dx +
                (0.5*(By_f[1:-1, 1:-1] + By_f[1:-1, 2:]) -
                 0.5*(By_f[1:-1, :-2] + By_f[1:-1, 1:-1])) / cfg.dy
            )
        divB_max = float(np.max(np.abs(divB_field)))
        divB_L2 = float(np.sqrt(np.mean(divB_field**2)))
        energy_total = float(to_numpy(xp.sum(self.U[EN, ng:-ng, ng:-ng])) * cfg.dx * cfg.dy)

        # --- Nozzle / jet diagnostics ---
        # Measure downstream region (x > interface + 0.3)
        x_downstream = cfg.interface_x + 0.3
        ix_ds = max(np.searchsorted(x_arr, x_downstream), 0)
        vx_ds = vx[ix_ds:, :]
        rho_ds = rho[ix_ds:, :]
        jet_max_vx = float(np.max(vx_ds)) if vx_ds.size > 0 else 0.0

        # Momentum flux and collimation at bar vs gap locations
        mom_flux = rho_ds * vx_ds**2  # rho*vx^2
        if self._bar_mask_y is not None and self._gap_mask_y is not None:
            bm = self._bar_mask_y; gm = self._gap_mask_y
            n_bar = max(np.sum(bm), 1); n_gap = max(np.sum(gm), 1)
            mom_bar = float(np.mean(mom_flux[:, bm])) if np.any(bm) else 0.0
            mom_gap = float(np.mean(mom_flux[:, gm])) if np.any(gm) else 0.0
            # Collimation: ratio of gap momentum to total
            total_mom = mom_bar * n_bar + mom_gap * n_gap
            collimation = mom_gap * n_gap / max(total_mom, 1e-14)
        else:
            mom_bar = float(np.mean(mom_flux)) if mom_flux.size > 0 else 0.0
            mom_gap = mom_bar
            collimation = 0.5

        self.diag_times.append(self.t)
        self.diag_mixing_width_integral.append(mw_integral)
        self.diag_mixing_width_thresh.append(mw_thresh)
        self.diag_mixedness.append(mixedness)
        self.diag_enstrophy.append(enstrophy_global)
        self.diag_enstrophy_local.append(enstrophy_local)
        self.diag_perturbation_amp.append(pert_amp)
        self.diag_mode_amps.append(mode_amps.copy())
        self.diag_stag_pressure.append(stag)
        self.diag_divB_max.append(divB_max)
        self.diag_divB_L2.append(divB_L2)
        self.diag_energy_total.append(energy_total)
        self.diag_boundary_flux_cumulative.append(self._cumulative_boundary_energy)
        self.diag_jet_max_vx.append(jet_max_vx)
        self.diag_jet_collimation.append(collimation)
        self.diag_jet_momentum_flux_gap.append(mom_gap)
        self.diag_jet_momentum_flux_bar.append(mom_bar)

    def save_snapshot(self, label=None):
        ng = self.ng
        W = cons_to_prim(self.U, self.cfg.gamma)
        key = label or f"t={self.t:.4f}"
        self.snapshots[key] = {
            'rho': to_numpy(W[iRHO, ng:-ng, ng:-ng]),
            'p': to_numpy(W[iPR, ng:-ng, ng:-ng]),
            'vx': to_numpy(W[iVX, ng:-ng, ng:-ng]),
            'vy': to_numpy(W[iVY, ng:-ng, ng:-ng]),
            'Bx': to_numpy(W[iBX, ng:-ng, ng:-ng]),
            'By': to_numpy(W[iBY, ng:-ng, ng:-ng]),
            'Bz': to_numpy(W[iBZ, ng:-ng, ng:-ng]),
            'C': to_numpy(W[iCLR, ng:-ng, ng:-ng]),
            't': self.t,
            'x': to_numpy(self.x[ng:-ng]),
            'y': to_numpy(self.y[ng:-ng]),
        }

    def run(self):
        cfg = self.cfg; g = cfg.gamma
        field_desc = "STRIPED" if cfg.is_striped() else ("UNIFORM" if cfg.is_mhd() else "HYDRO")
        print(f"\n--- Running (GPU) [{field_desc}]: B0 = {cfg.B_transverse} ---")
        sys.stdout.flush()
        t0 = time.time()
        self.compute_diagnostics()
        self.save_snapshot(label="initial")
        while self.t < cfg.t_end and self.step < cfg.max_steps:
            W = cons_to_prim(self.U, g)
            dt_cfl = self.compute_dt(W)
            self.dt = min(dt_cfl, cfg.t_end - self.t)
            if self.dt <= 1e-16: break
            self.step_ssprk3()
            self.t += self.dt; self.step += 1
            if self.step % cfg.diag_interval == 0:
                self.compute_diagnostics()
            for st in cfg.snapshot_times:
                lbl = f"t~{st:.2f}"
                if abs(self.t - st) < 1.5*self.dt and lbl not in self.snapshots:
                    self.save_snapshot(label=lbl)
            if self.step % 200 == 0:
                mwi = self.diag_mixing_width_integral[-1] if self.diag_mixing_width_integral else 0
                theta = self.diag_mixedness[-1] if self.diag_mixedness else 0
                jvx = self.diag_jet_max_vx[-1] if self.diag_jet_max_vx else 0
                col = self.diag_jet_collimation[-1] if self.diag_jet_collimation else 0
                print(f"  Step {self.step:5d}  t={self.t:.4f}  dt={self.dt:.2e}  "
                      f"MW={mwi:.4f}  theta={theta:.3f}  jet_vx={jvx:.2f}  collim={col:.3f}")
                sys.stdout.flush()
        self.compute_diagnostics()
        self.save_snapshot(label="final")
        elapsed = time.time() - t0
        if len(self.diag_energy_total) >= 2:
            e0 = self.diag_energy_total[0]; ef = self.diag_energy_total[-1]
            bf = self._cumulative_boundary_energy
            de_raw = abs(ef-e0)/max(abs(e0),1e-14)*100
            de_corr = abs(ef-bf-e0)/max(abs(e0),1e-14)*100
            print(f"  Energy drift: raw={de_raw:.2f}%, corrected={de_corr:.2f}%")
        if self.diag_divB_max:
            print(f"  Final max|divB|: {self.diag_divB_max[-1]:.2f}, L2: {self.diag_divB_L2[-1]:.4f}")
        print(f"  Done: {self.step} steps, {elapsed:.1f}s")
        sys.stdout.flush()


print("GPU Solver OK")


# ============================================================
# PostProcessor — with Magnetic Nozzle visualization suite
# ============================================================
class PostProcessor:
    """Generate all analysis plots including Magnetic Nozzle / Sieve visualizations."""

    def __init__(self, solvers_dict):
        self.solvers = solvers_dict
        self._colors = ['C3', 'C2', 'C0', 'C4', 'C5', 'C6', 'C7']
        self._styles = ['-', '--', '-.', ':', '-', '--', '-.']
        self._saved_files = []

    @staticmethod
    def _final(s):
        return s.snapshots.get('final', list(s.snapshots.values())[-1])

    def _save(self, fig, filename):
        ok, path, sz = save_figure(fig, filename)
        self._saved_files.append((filename, ok, sz))
        return ok, path, sz

    def _get_smoothing(self):
        for s in self.solvers.values():
            return s.cfg.enable_smoothing
        return False

    # ---- ORIGINAL PLOTS (preserved) ----

    def plot_density_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); rho = snap['rho']
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto',
                               vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Density Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'density_comparison.png')

    def plot_passive_scalar(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); C = np.clip(snap['C'], 0, 1)
            im = ax.pcolormesh(snap['x'], snap['y'], C.T, cmap='coolwarm', shading='auto', vmin=0, vmax=1)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label='$C$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Passive Scalar (Mixing Tracer)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'passive_scalar.png')

    def plot_schlieren(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); rho = snap['rho']; dx, dy = solver.cfg.dx, solver.cfg.dy
            grad_rho = np.sqrt(np.gradient(rho, dx, axis=0)**2 + np.gradient(rho, dy, axis=1)**2)
            schlieren = np.log10(grad_rho / np.maximum(rho, 1e-10) + 1e-10)
            vmin_s = np.percentile(schlieren, 3); vmax_s = np.percentile(schlieren, 99)
            im = ax.pcolormesh(snap['x'], snap['y'], schlieren.T, cmap='gray_r', shading='auto', vmin=vmin_s, vmax=vmax_s)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: Schlieren", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82)
        axes[0].set_ylabel('$y$')
        fig.suptitle('Numerical Schlieren', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'schlieren.png')

    def plot_vorticity_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]; omegas, snaps = [], []
        for label, solver in self.solvers.items():
            snap = self._final(solver); snaps.append(snap)
            omegas.append(np.gradient(snap['vy'], solver.cfg.dx, axis=0) - np.gradient(snap['vx'], solver.cfg.dy, axis=1))
        vmax_om = max(np.percentile(np.abs(om), 99) for om in omegas)
        vmax_om = max(vmax_om, 1e-10)
        for ax, (label, _), snap, omega in zip(axes, self.solvers.items(), snaps, omegas):
            im = ax.pcolormesh(snap['x'], snap['y'], omega.T, cmap='RdBu_r', shading='auto', vmin=-vmax_om, vmax=vmax_om)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\omega_z$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Vorticity Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'vorticity.png')

    def plot_magnetic_pressure_with_fieldlines(self):
        mhd = {k: v for k, v in self.solvers.items() if v.cfg.is_mhd()}
        if not mhd: return
        n = len(mhd)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, mhd.items()):
            snap = self._final(solver)
            Pmag = 0.5*(snap['Bx']**2 + snap['By']**2 + snap['Bz']**2)
            im = ax.pcolormesh(snap['x'], snap['y'], Pmag.T, cmap='plasma', shading='auto')
            try:
                ax.streamplot(snap['x'], snap['y'], snap['Bx'].T, snap['By'].T,
                              color='white', linewidth=0.5, density=1.2, arrowsize=0.5)
            except Exception: pass
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$P_B$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Magnetic Pressure & Field Lines', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'mag_pressure.png')

    def plot_diagnostics(self):
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        colors, styles = self._colors, self._styles
        sm_en = self._get_smoothing(); ns = 5; pairs = list(self.solvers.items())

        for ax_idx, (diag_key, ylabel, title) in enumerate([
            ('diag_mixing_width_integral', r'$W$', 'Integral Mixing Width'),
            ('diag_mixing_width_thresh', 'Threshold Width', 'Threshold MW'),
            ('diag_mixedness', r'$\theta$', 'Mixedness'),
            ('diag_perturbation_amp', r'$a_k$', 'Perturbation Amplitude'),
        ]):
            ax = axes[0, ax_idx]
            for i, (label, s) in enumerate(pairs):
                t = np.array(s.diag_times); val = np.array(getattr(s, diag_key))
                ax.plot(t, smooth(val, ns, sm_en), color=colors[i%7], ls=styles[i%7], lw=2, label=label)
            ax.set_xlabel('$t$'); ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
            if 'mixedness' in diag_key: ax.set_ylim(0, 1)
            else: ax.set_ylim(bottom=0)

        for ax_idx, (diag_key, ylabel, title) in enumerate([
            ('diag_enstrophy_local', r'$\langle\rho\omega_z^2\rangle$', 'Local Enstrophy'),
            ('diag_jet_max_vx', r'$v_x^{max}$', 'Peak Jet Velocity'),
            ('diag_jet_collimation', 'Collimation', 'Gap Momentum Fraction'),
            ('diag_energy_total', '$E_{tot}$', 'Total Energy'),
        ]):
            ax = axes[1, ax_idx]
            for i, (label, s) in enumerate(pairs):
                t = np.array(s.diag_times); val = np.array(getattr(s, diag_key))
                ax.plot(t, val, color=colors[i%7], ls=styles[i%7], lw=2, label=label)
            ax.set_xlabel('$t$'); ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        plt.tight_layout()
        self._save(fig, 'diagnostics.png')

    def plot_evolution(self):
        n_cases = len(self.solvers); n_cols = 4
        fig, axes = plt.subplots(n_cases, n_cols, figsize=(4*n_cols, 3*n_cases), sharex=True, sharey=True, squeeze=False)
        for row, (label, solver) in enumerate(self.solvers.items()):
            keys = sorted(solver.snapshots.keys(), key=lambda k: solver.snapshots[k]['t'])
            if len(keys) > n_cols:
                idx = np.linspace(0, len(keys)-1, n_cols, dtype=int)
                keys = [keys[i] for i in idx]
            for col in range(min(n_cols, len(keys))):
                ax = axes[row, col]; snap = solver.snapshots[keys[col]]; rho = snap['rho']
                ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto', vmin=0.5, vmax=rho.max()*0.95)
                ax.set_title(f"$t={snap['t']:.3f}$", fontsize=10); ax.set_aspect('equal')
                if col == 0: ax.set_ylabel(f"{label}\n$y$")
                if row == n_cases-1: ax.set_xlabel('$x$')
        fig.suptitle('Density Evolution', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'evolution.png')

    def plot_interface_shape(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (label, solver) in enumerate(self.solvers.items()):
            snap = self._final(solver); C = np.clip(snap['C'], 0, 1)
            ax.contour(snap['x'], snap['y'], C.T, levels=[0.5],
                       colors=[self._colors[i%7]], linewidths=2)
            ax.plot([], [], color=self._colors[i%7], lw=2, label=label)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('Interface Shape ($C=0.5$ contour)', fontweight='bold')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save(fig, 'interface_shape.png')

    # ---- NEW: MAGNETIC NOZZLE / SIEVE PLOTS ----

    def plot_nozzle_velocity_map(self):
        """Downstream x-velocity map showing jet formation at gaps."""
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver); vx = snap['vx']
            im = ax.pcolormesh(snap['x'], snap['y'], vx.T, cmap='hot', shading='auto',
                               vmin=0, vmax=np.percentile(vx, 99))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$v_x$')
            # Overlay bar/gap regions for striped cases
            if solver.cfg.is_striped():
                Ly = solver.cfg.y_max - solver.cfg.y_min
                k_mod = 2*np.pi*solver.cfg.B_modulation_mode / Ly
                y_arr = snap['y']
                for yn in y_arr[::1]:
                    if abs(np.cos(k_mod * yn)) < 0.1:
                        ax.axhline(yn, color='cyan', lw=0.5, alpha=0.4)
        axes[0].set_ylabel('$y$')
        fig.suptitle('X-Velocity: Jet Formation Map', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_velocity_map.png')

    def plot_momentum_flux_comparison(self):
        """Bar chart comparing momentum flux at bars vs gaps for all cases."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = self._colors; pairs = list(self.solvers.items())

        ax = axes[0]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            ax.plot(t, s.diag_jet_momentum_flux_gap, color=colors[i%7], ls='-', lw=2, label=f'{label} (gap)')
            ax.plot(t, s.diag_jet_momentum_flux_bar, color=colors[i%7], ls=':', lw=1.5, alpha=0.6, label=f'{label} (bar)')
        ax.set_xlabel('Time $t$'); ax.set_ylabel(r'$\langle \rho v_x^2 \rangle$')
        ax.set_title('Momentum Flux: Gaps vs Bars', fontweight='bold')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            gap = np.array(s.diag_jet_momentum_flux_gap)
            bar = np.array(s.diag_jet_momentum_flux_bar)
            ratio = gap / np.maximum(bar, 1e-14)
            ax.plot(t, ratio, color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
        ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Time $t$'); ax.set_ylabel('Gap/Bar Momentum Ratio')
        ax.set_title('Nozzle Effectiveness (ratio > 1 = jet focusing)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        plt.tight_layout()
        self._save(fig, 'nozzle_momentum_flux.png')

    def plot_collimation_timeseries(self):
        """Collimation ratio over time for all cases."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (label, s) in enumerate(self.solvers.items()):
            t = np.array(s.diag_times)
            ax.plot(t, s.diag_jet_collimation, color=self._colors[i%7],
                    ls=self._styles[i%7], lw=2.5, label=label)
        ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='Isotropic (0.5)')
        ax.set_xlabel('Time $t$', fontsize=13); ax.set_ylabel('Gap Momentum Fraction', fontsize=12)
        ax.set_title('Jet Collimation: Splash vs Pressure Washer', fontweight='bold', fontsize=14)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(0, 1)
        plt.tight_layout()
        self._save(fig, 'nozzle_collimation.png')

    def plot_jet_velocity_profiles(self):
        """Transverse (y) profiles of vx at a downstream slice for each case."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pairs = list(self.solvers.items()); colors = self._colors

        # Profile at x = interface + 1.0
        for ax, x_offset, title in [
            (axes[0], 0.5, 'Near-field ($x = x_{if} + 0.5$)'),
            (axes[1], 1.5, 'Far-field ($x = x_{if} + 1.5$)'),
        ]:
            for i, (label, s) in enumerate(pairs):
                snap = self._final(s)
                x_target = s.cfg.interface_x + x_offset
                ix = np.argmin(np.abs(snap['x'] - x_target))
                vx_profile = snap['vx'][ix, :]
                ax.plot(snap['y'], vx_profile, color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
                # Mark bar/gap boundaries for striped cases
                if s.cfg.is_striped():
                    Ly = s.cfg.y_max - s.cfg.y_min
                    k_mod = 2*np.pi*s.cfg.B_modulation_mode / Ly
                    for yn in snap['y']:
                        if abs(np.cos(k_mod * yn)) < 0.05:
                            ax.axvline(yn, color=colors[i%7], lw=0.3, alpha=0.3)
            ax.set_xlabel('$y$'); ax.set_ylabel('$v_x$')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig.suptitle('Jet Velocity Profiles (transverse cuts)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_jet_profiles.png')

    def plot_magnetic_tension_map(self):
        """Map of magnetic tension magnitude |(B.grad)B| showing bars vs gaps."""
        mhd = {k: v for k, v in self.solvers.items() if v.cfg.is_mhd()}
        if not mhd: return
        n = len(mhd)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, mhd.items()):
            snap = self._final(solver)
            Bx, By_f = snap['Bx'], snap['By']
            dx, dy = solver.cfg.dx, solver.cfg.dy
            # (B.grad)B_x = Bx*dBx/dx + By*dBx/dy
            dBxdx = np.gradient(Bx, dx, axis=0); dBxdy = np.gradient(Bx, dy, axis=1)
            dBydx = np.gradient(By_f, dx, axis=0); dBydy = np.gradient(By_f, dy, axis=1)
            tension_x = Bx*dBxdx + By_f*dBxdy
            tension_y = Bx*dBydx + By_f*dBydy
            tension_mag = np.sqrt(tension_x**2 + tension_y**2)
            vmax_t = np.percentile(tension_mag, 98)
            im = ax.pcolormesh(snap['x'], snap['y'], tension_mag.T, cmap='magma',
                               shading='auto', vmin=0, vmax=max(vmax_t, 1e-10))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$|(\mathbf{B}\cdot\nabla)\mathbf{B}|$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Magnetic Tension: The Bars That Pin the Interface', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_tension_map.png')

    def plot_nozzle_hero(self):
        """Hero comparison: hydro splash vs nozzle jets (density + velocity overlay)."""
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, axes = plt.subplots(2, len(keys), figsize=(5.5*len(keys), 10), squeeze=False)

        for col, key in enumerate(keys):
            solver = self.solvers[key]; snap = self._final(solver)
            rho = snap['rho']; vx = snap['vx']

            # Top row: density
            ax = axes[0, col]
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno', shading='auto',
                               vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_aspect('equal'); ax.set_title(f"{key}", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')
            if col == 0: ax.set_ylabel('$y$')
            ax.set_xlabel('$x$')

            # Bottom row: x-momentum (rho*vx) showing directed jets
            ax = axes[1, col]
            mom = rho * vx
            vmax_m = np.percentile(np.abs(mom), 98)
            im = ax.pcolormesh(snap['x'], snap['y'], mom.T, cmap='RdYlBu_r', shading='auto',
                               vmin=-vmax_m, vmax=vmax_m)
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho v_x$')
            if col == 0: ax.set_ylabel('$y$')
            ax.set_xlabel('$x$')

            # Overlay bar/gap for striped
            if solver.cfg.is_striped():
                Ly = solver.cfg.y_max - solver.cfg.y_min
                k_mod = 2*np.pi*solver.cfg.B_modulation_mode / Ly
                for yn in snap['y']:
                    if abs(np.cos(k_mod * yn)) < 0.05:
                        for r in range(2):
                            axes[r, col].axhline(yn, color='lime', lw=0.5, alpha=0.4)

        fig.suptitle('Magnetic Nozzle: Splash vs Focused Jets\n(top: density, bottom: x-momentum)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_hero.png')

    def plot_y_striped_momentum(self):
        """x-momentum as function of y, averaged over downstream region."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (label, s) in enumerate(self.solvers.items()):
            snap = self._final(s)
            x_arr = snap['x']; y_arr = snap['y']
            x_ds = s.cfg.interface_x + 0.3
            ix_ds = np.argmin(np.abs(x_arr - x_ds))
            rho_ds = snap['rho'][ix_ds:, :]; vx_ds = snap['vx'][ix_ds:, :]
            mom_y_profile = np.mean(rho_ds * vx_ds, axis=0)
            ax.plot(y_arr, mom_y_profile, color=self._colors[i%7], ls=self._styles[i%7], lw=2, label=label)

        # Mark gap locations for striped cases
        for label, s in self.solvers.items():
            if s.cfg.is_striped():
                snap = self._final(s)
                Ly = s.cfg.y_max - s.cfg.y_min
                k_mod = 2*np.pi*s.cfg.B_modulation_mode / Ly
                for yn in snap['y']:
                    if abs(np.cos(k_mod * yn)) < 0.05:
                        ax.axvline(yn, color='green', lw=0.5, alpha=0.3)
                break  # only need one set of markers

        ax.set_xlabel('$y$', fontsize=13); ax.set_ylabel(r'$\langle \rho v_x \rangle_{downstream}$', fontsize=12)
        ax.set_title('Downstream Momentum Profile: Where Does the Plasma Go?', fontweight='bold', fontsize=14)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save(fig, 'nozzle_y_momentum.png')

    def plot_sieve_initial_condition(self):
        """Visualize the initial striped B-field setup: the sieve before the shock hits."""
        striped = {k: v for k, v in self.solvers.items() if v.cfg.is_striped()}
        if not striped: return
        label, solver = next(iter(striped.items()))
        snap = solver.snapshots.get('initial', list(solver.snapshots.values())[0])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        im = ax.pcolormesh(snap['x'], snap['y'], snap['rho'].T, cmap='inferno', shading='auto')
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_aspect('equal')
        ax.set_title('Initial Density', fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\rho$')

        ax = axes[1]
        By_init = snap['By']
        vmax_b = np.max(np.abs(By_init))
        im = ax.pcolormesh(snap['x'], snap['y'], By_init.T, cmap='RdBu_r', shading='auto',
                           vmin=-vmax_b, vmax=vmax_b)
        ax.set_xlabel('$x$'); ax.set_aspect('equal')
        ax.set_title(r'Initial $B_y$ (Striped)', fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.82, label=r'$B_y$')

        ax = axes[2]
        Pmag = 0.5*(snap['Bx']**2 + snap['By']**2)
        im = ax.pcolormesh(snap['x'], snap['y'], Pmag.T, cmap='plasma', shading='auto')
        ax.set_xlabel('$x$'); ax.set_aspect('equal')
        ax.set_title(r'Initial $P_{mag}$ (bars = high, gaps = zero)', fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.82, label=r'$B^2/2$')

        fig.suptitle(f'Magnetic Sieve Setup: {label}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'sieve_initial_condition.png')

    def plot_bar_vs_gap_density_profiles(self):
        """Compare density profiles at bar and gap y-locations."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pairs = list(self.solvers.items()); colors = self._colors

        for ax, region, title in [
            (axes[0], 'bar', 'Bar Region (high |B|)'),
            (axes[1], 'gap', 'Gap Region (B ~ 0)'),
        ]:
            for i, (label, s) in enumerate(pairs):
                snap = self._final(s)
                if s._bar_mask_y is not None:
                    mask = s._bar_mask_y if region == 'bar' else s._gap_mask_y
                    rho_region = np.mean(snap['rho'][:, mask], axis=1)
                else:
                    rho_region = np.mean(snap['rho'], axis=1)
                ax.plot(snap['x'], rho_region, color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
            ax.set_xlabel('$x$'); ax.set_ylabel(r'$\langle \rho \rangle_y$')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle('Density Profiles: Bar vs Gap Regions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'nozzle_bar_gap_density.png')

    def plot_summary_bars(self):
        t_skip = 0.03
        labels_list = []; mwi_peaks = []; enst_means = []; col_means = []; jet_peaks = []
        sm_en = self._get_smoothing()
        for label, s in self.solvers.items():
            t = np.array(s.diag_times); mask = t > t_skip
            labels_list.append(label)
            mwi = smooth(np.array(s.diag_mixing_width_integral)[mask], 7, sm_en)
            mwi_peaks.append(float(np.max(mwi)) if len(mwi) > 0 else 0)
            en = np.array(s.diag_enstrophy_local)[mask]
            enst_means.append(float(np.mean(en)) if len(en) > 0 else 0)
            col = np.array(s.diag_jet_collimation)[mask]
            col_means.append(float(np.mean(col)) if len(col) > 0 else 0)
            jvx = np.array(s.diag_jet_max_vx)[mask]
            jet_peaks.append(float(np.max(jvx)) if len(jvx) > 0 else 0)

        x_pos = np.arange(len(labels_list)); bar_colors = self._colors[:len(labels_list)]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 4.5))
        for ax, vals, ylabel, title in [
            (ax1, mwi_peaks, 'Peak MW', 'Mixing Width'),
            (ax2, enst_means, 'Mean Enstrophy', 'Enstrophy'),
            (ax3, col_means, 'Mean Collimation', 'Jet Collimation'),
            (ax4, jet_peaks, r'Peak $v_x$', 'Peak Jet Velocity'),
        ]:
            bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor='k')
            ax.set_xticks(x_pos); ax.set_xticklabels(labels_list, fontsize=7, rotation=15)
            ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
            fmt = '.4f' if max(vals) < 1 else '.1f'
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        f"{val:{fmt}}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        self._save(fig, 'summary_bars.png')

    def plot_divB_comparison(self):
        mhd_cases = {k: v for k, v in self.solvers.items() if v.cfg.is_mhd()}
        if not mhd_cases: return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)); colors = self._colors
        for ax, diag, ylabel, title in [
            (axes[0], 'diag_divB_max', r'max $|\nabla\cdot\mathbf{B}|$', 'Max norm'),
            (axes[1], 'diag_divB_L2', r'$L_2$ norm', '$L_2$ norm'),
        ]:
            for i, (label, s) in enumerate(mhd_cases.items()):
                ax.plot(s.diag_times, getattr(s, diag), color=colors[i%7], ls=self._styles[i%7], lw=2, label=label)
            ax.set_xlabel('Time $t$'); ax.set_ylabel(ylabel)
            ax.set_title(f'div(B) Control ({title})', fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save(fig, 'divB.png')

    def plot_all(self):
        print("\n=== Generating figures ==="); sys.stdout.flush()
        # Core comparison plots
        self.plot_density_comparison()
        self.plot_passive_scalar()
        self.plot_schlieren()
        self.plot_vorticity_comparison()
        self.plot_magnetic_pressure_with_fieldlines()
        self.plot_diagnostics()
        self.plot_evolution()
        self.plot_interface_shape()
        self.plot_summary_bars()
        self.plot_divB_comparison()
        # Magnetic Nozzle / Sieve specific plots
        self.plot_sieve_initial_condition()
        self.plot_nozzle_hero()
        self.plot_nozzle_velocity_map()
        self.plot_jet_velocity_profiles()
        self.plot_momentum_flux_comparison()
        self.plot_collimation_timeseries()
        self.plot_magnetic_tension_map()
        self.plot_y_striped_momentum()
        self.plot_bar_vs_gap_density_profiles()

        print(f"\n=== Figure manifest ({len(self._saved_files)} files) ===")
        total_kb = 0
        for fname, ok, sz in self._saved_files:
            print(f"  {'OK' if ok else 'FAIL'} {fname:45s}  {sz:7.0f} KB")
            total_kb += sz
        print(f"  {'Total':>47s}  {total_kb:7.0f} KB")
        print(f"  Directory: {os.path.abspath(OUTPUT_DIR)}")
        print("=== All figures done ===\n"); sys.stdout.flush()


print("PostProcessor OK")
