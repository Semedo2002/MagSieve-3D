"""GPU-aware diagnostics for the MHD solver."""

import numpy as np

from .config import EN, iRHO, iPR, iVX, iVY, iBX, iBY, iBZ, iCLR
from .physics import xp, cons_to_prim


def _scalar(value):
    return float(xp.asnumpy(value))


def _gpu_trapezoid(values, coordinates):
    return xp.sum(0.5 * (values[:-1] + values[1:]) * (coordinates[1:] - coordinates[:-1]))


def _interface_positions(c_clamped, x_arr, y_arr, dx, perturbation_mode):
    """Find local interface positions for each transverse row on the GPU."""
    grad = xp.abs((c_clamped[1:, :] - c_clamped[:-1, :]) / dx)
    x_face = 0.5 * (x_arr[1:] + x_arr[:-1])  # grad lives between cells
    interface_pos = xp.full_like(y_arr, np.nan)
    ny_int = c_clamped.shape[1]

    for j in range(ny_int):
        grad_col = grad[:, j]
        if _scalar(xp.max(grad_col)) <= 1e-6:
            continue
        ix_peak = int(xp.argmax(grad_col))
        hw = min(10, ix_peak, grad.shape[0] - ix_peak - 1)
        if hw <= 0:
            continue
        sl = slice(ix_peak - hw, ix_peak + hw + 1)
        weights = grad_col[sl]
        ws = xp.sum(weights)
        if _scalar(ws) > 1e-12:
            interface_pos[j] = xp.sum(x_face[sl] * weights) / ws

    valid = ~xp.isnan(interface_pos)
    n_valid = int(xp.sum(valid))
    if n_valid <= ny_int // 2:
        return interface_pos, np.zeros(c_clamped.shape[1] // 2 + 1), 0.0

    pos_valid = interface_pos[valid]
    y_valid = y_arr[valid]
    pos_interp = np.interp(
        xp.asnumpy(y_arr), xp.asnumpy(y_valid), xp.asnumpy(pos_valid)
    )
    pos_fluct = pos_interp - np.mean(pos_interp)
    mode_amps = 2.0 * np.abs(np.fft.rfft(pos_fluct)) / len(pos_fluct)  # FFT over y
    pert_amp = float(mode_amps[perturbation_mode]) if perturbation_mode < len(mode_amps) else 0.0
    return interface_pos, mode_amps, pert_amp


def diagnostics_for_solver(solver):
    """Compute solver diagnostics, with the expensive spatial reductions on GPU."""
    ng = solver.ng
    cfg = solver.cfg
    W = cons_to_prim(solver.U, cfg.gamma)
    rho = W[iRHO, ng:-ng, ng:-ng]
    p = W[iPR, ng:-ng, ng:-ng]
    vx = W[iVX, ng:-ng, ng:-ng]
    vy = W[iVY, ng:-ng, ng:-ng]
    bx = W[iBX, ng:-ng, ng:-ng]
    by = W[iBY, ng:-ng, ng:-ng]
    bz = W[iBZ, ng:-ng, ng:-ng]
    c = W[iCLR, ng:-ng, ng:-ng]
    x_arr = solver.x[ng:-ng]
    y_arr = solver.y[ng:-ng]

    c_clamped = xp.clip(c, 0.0, 1.0)
    c_bar = xp.mean(c_clamped, axis=1)
    mw_integral = _scalar(_gpu_trapezoid(c_bar * (1.0 - c_bar), x_arr))
    mixed = xp.where((c_bar > 0.01) & (c_bar < 0.99))[0]
    mw_thresh = _scalar(x_arr[mixed[-1]] - x_arr[mixed[0]]) if mixed.size > 1 else 0.0

    c_bar_2d = c_bar[:, None]
    c_prime_sq = xp.mean((c_clamped - c_bar_2d) ** 2, axis=1)
    denom_mix = xp.maximum(c_bar * (1.0 - c_bar), 1e-14)
    if mixed.size > 1:
        mixedness_num = _scalar(_gpu_trapezoid(c_prime_sq[mixed], x_arr[mixed]))
        mixedness_den = _scalar(_gpu_trapezoid(denom_mix[mixed], x_arr[mixed]))
        mixedness = float(np.clip(1.0 - mixedness_num / max(mixedness_den, 1e-14), 0.0, 1.0))
    else:
        mixedness = 0.0

    interface_pos, mode_amps, pert_amp = _interface_positions(
        c_clamped, x_arr, y_arr, cfg.dx, cfg.perturbation_mode
    )
    valid = ~xp.isnan(interface_pos)
    n_valid = int(xp.sum(valid))

    dvydx = xp.gradient(vy, cfg.dx, axis=0)
    dvxdy = xp.gradient(vx, cfg.dy, axis=1)
    omega = dvydx - dvxdy
    enstrophy_global = _scalar(xp.sum(rho * omega**2) * cfg.dx * cfg.dy)
    # Interface-only measures. In MHD the vorticity deposited on the interface is
    # carried off along field lines by Alfven waves; that vorticity no longer drives
    # the RMI but still sits inside a fixed window, so lock on the mixing layer.
    mix_mask = (c_clamped > 0.05) & (c_clamped < 0.95)
    enstrophy_mix = _scalar(xp.sum(xp.where(mix_mask, rho * omega**2, 0.0)) * cfg.dx * cfg.dy)
    om_col = xp.sum(xp.where(mix_mask, omega, 0.0), axis=0) * cfg.dx
    n_half = 2 * cfg.perturbation_mode   # half wavelengths of the interface mode across Ly
    circulation = _scalar(xp.sum(xp.abs(om_col)) * cfg.dy) / n_half

    if n_valid > c_clamped.shape[1] // 2:
        x_contact = float(np.mean(xp.asnumpy(interface_pos[valid])))
        ix_lo = max(int(xp.searchsorted(x_arr, xp.asarray(x_contact - 0.5))), 0)
        ix_hi = min(int(xp.searchsorted(x_arr, xp.asarray(x_contact + 0.5))), rho.shape[0])
        if ix_hi - ix_lo > 3:
            enstrophy_local = _scalar(
                xp.sum(rho[ix_lo:ix_hi, :] * omega[ix_lo:ix_hi, :] ** 2)
                * cfg.dx * cfg.dy
            )
        else:
            enstrophy_local = enstrophy_global
    else:
        enstrophy_local = enstrophy_global

    v2 = vx**2 + vy**2
    b2 = bx**2 + by**2 + bz**2
    stag = _scalar(xp.max(p + 0.5 * rho * v2 + 0.5 * b2))

    div_b = xp.zeros_like(bx)
    if bx.shape[0] > 2 and bx.shape[1] > 2:
        div_b[1:-1, 1:-1] = (
            (0.5 * (bx[1:-1, 1:-1] + bx[2:, 1:-1]) -
             0.5 * (bx[:-2, 1:-1] + bx[1:-1, 1:-1])) / cfg.dx +
            (0.5 * (by[1:-1, 1:-1] + by[1:-1, 2:]) -
             0.5 * (by[1:-1, :-2] + by[1:-1, 1:-1])) / cfg.dy
        )
    div_b_max = _scalar(xp.max(xp.abs(div_b)))
    div_b_l2 = _scalar(xp.sqrt(xp.mean(div_b**2)))
    # Dimensionless divergence error h max|div B| / max|B|. A global |B| scale
    # is used because local normalisation blows up at the nulls of striped fields.
    b_max = _scalar(xp.max(xp.sqrt(b2)))
    div_b_rel_max = div_b_max * min(cfg.dx, cfg.dy) / b_max if b_max > 0 else 0.0
    energy_total = _scalar(xp.sum(solver.U[EN, ng:-ng, ng:-ng]) * cfg.dx * cfg.dy)

    x_downstream = cfg.interface_x + 0.3
    ix_ds = max(int(xp.searchsorted(x_arr, xp.asarray(x_downstream))), 0)
    vx_ds = vx[ix_ds:, :]
    rho_ds = rho[ix_ds:, :]
    jet_max_vx = _scalar(xp.max(vx_ds)) if vx_ds.size > 0 else 0.0
    mom_flux = rho_ds * vx_ds**2

    if solver._bar_mask_y is not None and solver._gap_mask_y is not None:
        bar_mask = xp.asarray(solver._bar_mask_y)
        gap_mask = xp.asarray(solver._gap_mask_y)
        n_bar = max(int(xp.sum(bar_mask)), 1)
        n_gap = max(int(xp.sum(gap_mask)), 1)
        mom_bar = _scalar(xp.mean(mom_flux[:, bar_mask])) if xp.any(bar_mask) else 0.0
        mom_gap = _scalar(xp.mean(mom_flux[:, gap_mask])) if xp.any(gap_mask) else 0.0
        total_mom = mom_bar * n_bar + mom_gap * n_gap
        collimation = mom_gap * n_gap / max(total_mom, 1e-14)
    else:
        mom_bar = _scalar(xp.mean(mom_flux)) if mom_flux.size > 0 else 0.0
        mom_gap = mom_bar
        collimation = 0.5

    return {
        "mixing_width_integral": mw_integral,
        "mixing_width_thresh": mw_thresh,
        "mixedness": mixedness,
        "enstrophy": enstrophy_global,
        "enstrophy_local": enstrophy_local,
        "enstrophy_mix": enstrophy_mix,
        "interface_circulation": circulation,
        "perturbation_amp": pert_amp,
        "mode_amps": mode_amps,
        "stag_pressure": stag,
        "divB_max": div_b_max,
        "divB_L2": div_b_l2,
        "divB_rel_max": div_b_rel_max,
        "energy_total": energy_total,
        "boundary_flux_cumulative": solver._cumulative_boundary_energy,
        "powell_cumulative": solver._cumulative_powell_energy,
        "jet_max_vx": jet_max_vx,
        "jet_collimation": collimation,
        "jet_momentum_flux_gap": mom_gap,
        "jet_momentum_flux_bar": mom_bar,
    }
