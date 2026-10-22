# MagSieve-3D: striped magnetic fields and the Richtmyer-Meshkov instability

A 2D and 3D compressible MHD solver (CuPy + a custom CUDA HLLD kernel) and the parameter study built on it. The study asks whether a spatially striped magnetic field can suppress ("lock") the Richtmyer-Meshkov instability of a shocked density interface while channelling the flow through the weak-field gaps.

**Result in one paragraph.** A uniform shock-parallel field locks the interface, in line with earlier work on magnetised RMI. Striped fields roughen the interface in proportion to their bar/gap contrast: the stripe-scale mode grows as a8 ~ 0.027 dP_B, converged over three resolutions. (An earlier "10% contrast" lock threshold did not survive the convergence study: it was measured against hydrodynamic enstrophy, which keeps growing with resolution.) A shock-parallel field is not compressed by the shock while the gas is, so a striped field that is in pressure balance upstream produces a steady field-aligned shear downstream. Magnetic tension does not resist this shear, and it deforms the interface at the stripe scale. Suppression and collimation therefore work against each other in this regime. Aligning the stripe phase with the interface ripple produces a predictable zero crossing of the RMI mode, but not a lock. In 3D, at the Lundquist number this grid resolves (S ~ 75), the field-aligned shear layers neither tear nor break down. See `paper/` for the write-up, including the grid-convergence study and its limitations.

## Layout

```
sieve/            solver package
  config.py       typed configuration (unknown keys raise ValueError)
  physics.py      CUDA HLLD wrapper, MUSCL, jump conditions, resistive terms, stripe equilibrium
  eos.py          two-material EOS: ideal gas, stiffened gas, Mie-Gruneisen (linear Us-Up)
  units.py        SI helpers: conductivity to code-unit diffusivity, Spitzer coefficient, liner EOS,
                  bremsstrahlung and Braginskii conduction coefficients
  hlld_kernel.cu  HLLD Riemann solver (one thread per face)
  solver.py       MHDSolver (SSP-RK3, GLM + Powell, BCs, energy budget) and PostProcessor
  solver3d.py     MHDSolver3D: unsplit HLLD in x, y, z (same kernel, rotated), z-periodic, 3D seeding,
                  current-density and generated-field diagnostics
  diagnostics.py  mixing, enstrophy, interface spectrum, div B, energy
main.py           CLI: verification tests, single cases, legacy campaign
examples/         MTIF liner flux compression; synthetic magnetic probes (SI units)
experiments/      sweeps A-H and analysis scripts; data/ holds the saved results
paper/            figure and .docx generators, figures/, magnetic_sieve.docx
rmi_output_gpu/   figures and GIFs written by the solver
```

## Installation

An NVIDIA GPU with CUDA 12 is required.

```
pip install -e .                  # package + dependencies (pyproject.toml)
pip install -e .[paper]          # also python-docx, for paper/make_paper.py
pip install -r requirements.txt  # or the exact tested versions
```

See `CONTRIBUTING.md` for the development workflow.

## Usage

```
python main.py --test all                     # verification suite, ~20 min on a Quadro P1000
python main.py --case striped --B0 3.5 --gap-fraction 0.9 --compare-hydro
python main.py --help
```

`--case` runs a 12 x 2 domain to t = 0.8 and writes figures and 61-frame GIFs to `rmi_output_gpu/`.

### Equation of state

Two materials, selected by the tracer: `eos_fuel` (where it is 0: `ideal` with `gamma`, or `stiffened` with
`p_inf`) and `eos_liner` (where it is 1: `ideal`, `stiffened` with `gamma_liner` and `p_inf_liner`, or
`mie_gruneisen` with `mg_rho0`, `mg_c0`, `mg_s`, `mg_gamma0`). Each law is written as eps = phi1(rho) p + phi2(rho);
mixed cells blend phi1 and phi2 linearly in the tracer, which is then a volume fraction advected
non-conservatively with the face values the energy flux uses (Shyue 1998), so a material interface at uniform
p and u stays in equilibrium. The HLLD kernel takes each face state's internal energy and sound speed from the
EOS; with one ideal gas the solver runs the original code path. Both the 2D and 3D solvers support it. The examples
take an aluminium liner in SI:

```
python examples/nearstar_mtif3_validation.py --eos-liner stiffened                # Al, gamma/p_inf fitted to its Hugoniot
python examples/nearstar_mtif3_validation.py --eos-liner mie_gruneisen            # Al: rho0 2700, c0 5328, s 1.338, Gamma0 2
python examples/mtif_flux_compression.py --eos-liner stiffened --p-inf-gpa 20 --gamma-liner 3.8
```

### Thermal conduction

Magnetized electron conduction, q = -[k_perp I + (k_par - k_perp) b b] . grad T with k_par = `cond_kappa0` T^5/2
and k_perp = k_par/(1 + x^2), x = omega_ce tau_e = `cond_x0` |B| T^3/2 / rho (T = p/rho), in the fuel only
(liner fraction below `cond_tracer_max`; liner faces and non-periodic domain faces are insulating, so total
energy is conserved exactly). Fluxes live on cell faces; the gradient normal to a face is a two-point
difference and the transverse gradients are van Leer limited (Sharma & Hammett 2007), which keeps strongly
anisotropic conduction from moving heat from cold to hot. It is a split step integrated with RKL2
super-time-stepping (Meyer, Balsara & Aslam 2014), about sqrt(dt/dt_explicit) stages, so it never limits the
hydrodynamic time step. `units.braginskii_kwargs` gives the Spitzer-Braginskii coefficients in code units (k_par
at 1 keV within 1.1% of the Spitzer value). The k_perp interpolation is the simple 1/(1 + x^2) form, not the
full Braginskii fit, and ion conduction is left out.

### Radiation

Optically thin bremsstrahlung, P = 1.69e-38 Z n_e^2 sqrt(T_eV) W/m^3 (NRL formulary, n in m^-3), acts on the
fuel only: where the liner fraction is below `rad_tracer_max` (0.1) and T is above `rad_T_min`. It is a
split step after each time step using the exact solution at fixed density, sqrt(T) falling linearly in time, so
it is stable for any time step. The energy removed accumulates in `cumulative_radiated_energy` and enters the
energy budget. `units.bremsstrahlung_kwargs` converts to code units; the examples take `--enable-rad`,
`--z-eff`, `--t-rad-min-ev`. At the examples' default parameters the fuel stays below 1 eV and nothing
radiates; a hot dense core does:

```
python examples/mtif_flux_compression.py --enable-rad --v-ref-kms 100 --rho-ref 100
```

### Resistivity

`eta_model` selects `constant` (`eta_resistivity` everywhere, the default), `material` (`eta_fuel` where the
tracer is 0, `eta_liner` where it is 1, log blend between) or `spitzer` (`eta_spitzer` T^-3/2 with T = p/rho
floored at `eta_T_min`, capped at `eta_max`). Face values use the harmonic mean, so a conductivity jump carries
the series resistance of the two cells; the resistive terms stay in face-flux form and conserve energy exactly,
and `cumulative_ohmic_heating` integrates eta J^2. The explicit time step uses the largest eta on the grid. The
examples take SI conductivities:

```
python examples/mtif_flux_compression.py --sigma-liner 3.5e7          # aluminium liner, S/m
python examples/nearstar_mtif3_validation.py --eta-model spitzer --sigma-liner 3.5e7
```

## Verification

`python main.py --test all` runs the full suite and exits non-zero on failure:

| test | checks |
|---|---|
| brio_wu | Brio-Wu shock tube structure |
| alfven | Alfvén wave convergence order (measured 1.96) |
| reflection | residual of fast and Alfvén pulses after leaving through the outflow boundary (3.5e-5, 7e-10) |
| conservation | energy budget closes to round-off (~1e-14); div B = 0 for Bz stripes, < 3% (h max\|div B\|/max\|B\|) otherwise |
| equilibrium | upstream gas stays at rest for pressure-balanced x stripes; misspelled config keys are rejected |
| z_invariance | z-invariant 3D runs reproduce the 2D solver: ideal, constant eta, and eta varying 8x across the interface |
| alfven_z | Alfven wave along z converges (orders 1.3, 1.9 at nz = 16-64) |
| resistive_z | weak B_x(z) decays as exp(-eta k^2 t) (error 3e-4); Ohmic heating conserves energy |
| resistive_layers | two-layer steady diffusion, eta jumps 4x: piecewise-linear B_z recovered to 3e-4 with harmonic face averaging (arithmetic: 1.7e-2) |
| ohmic | periodic box, eta varying 10x: thermal energy gained matches the integral of eta J^2 to 0.14%, total energy to 2e-14 |
| eos_impact | 10 km/s Al impact: stiffened Al on stiffened Cu vs the exact impedance match, and Mie-Gruneisen Al on Al vs the Us-Up Hugoniot; shock speeds, pressure, densities and velocity within 0.5% |
| eos_interface | Al slab in gas at uniform p and u: volume-fraction tracer keeps p uniform to 1e-9 (round-off times p_inf/p); mass-fraction blending, for comparison, is off by 3e5 p within 0.002 us |
| conduction | hot spot in a field with k_par/k_perp = 1e4: grid-aligned, the spread along and across B equals 2 D t to 0.04% and 1%; at 45 deg the numerical cross-field leakage is 2.7% (64^2) and 0.7% (128^2) of the parallel spread, second order; no new extrema, energy conserved to round-off |
| parity3d | 3D slabs reproduce the 2D solver to round-off for a stiffened Al/Cu impact (5e-14), bremsstrahlung decay (2e-15) and conduction in an oblique field (7e-15) |
| radiation | uniform hydrogen plasma, T0 = 500 eV, n = 1e20 cm^-3: T(t) matches (sqrt(T0) - K t/2)^2 to 1e-14 down to the 10 eV cutoff; E + E_rad conserved to 1e-15; no cooling in liner material or below the cutoff |
| eos_positivity | stiffened Al pulled apart at 5 and 8 km/s (cavitation) and a gas/Al interface: p + p_inf > 0, c^2 > 0, finite state |

## Reproducing the study

```
cd experiments
python sweeps.py C            # one sweep, results in data/exp_C.npz
python sweeps.py all          # sweeps A-H, about 2.7 h
python interface_spectrum.py  # RMI mode vs stripe mode
python enstrophy_location.py  # mixing-layer vs window enstrophy
python initial_conditions.py  # t = 0 fields (Fig. 1, 5)
python equilibrium_convergence.py  # upstream residual vs resolution (Fig. 2a)
python 3d_tearing_sweep.py   # 3D proof of concept, ~20 min (--quick: 2 min)
python 3d_tearing_sweep2.py  # resistive 3D, seed scan + resolution check, ~50 min
python ../paper/make_figures.py
python ../paper/make_paper.py
```

| sweep | question |
|---|---|
| A | small ripple, long domain: field-free-gap stripes vs hydro |
| B | uniform-temperature closure and a uniform-field control |
| C | gap-fraction sweep at B0 = 4 and 6 |
| D | weakest locking uniform field; weak-modulation stripes |
| E | stripe/ripple wavenumber ratio and phase at B0 = 6 |
| F | phase-aligned stripes at 20% and 30% modulation |
| G | resolution check (ny = 100 vs 200) |
| H | figure set: spectrum, enstrophy location, energy budget |

## Multi-physics MTIF capabilities

MagSieve-3D includes physics modules for magnetized target inertial fusion (MTIF) and magnetically driven
shock compression, each verified against an analytic solution in `python main.py --test all`:

- **Shock capturing and div B control:** HLLD Riemann solver, MUSCL reconstruction, GLM cleaning with Powell
  source terms. The divergence error at shocks is a few percent (h max|div B|/max|B| of 1.3-2.5% in the
  conservation test); in the MTIF examples the field is axial and div B = 0 exactly.
- **Variable resistivity:** constant, material-blended (liner/fuel) or Spitzer (T^-3/2 with a temperature floor
  and cap). The resistive terms are face fluxes, so energy is conserved exactly; harmonic face averaging gives
  the correct series resistance across a conductivity jump (two-layer steady diffusion recovered to 3e-4).
- **Two-material EOS:** ideal gas, stiffened gas and Mie-Gruneisen (linear Us-Up) for metallic liners. The
  interface uses a volume fraction advected non-conservatively (Shyue 1998), which keeps a gas/aluminium
  interface at uniform pressure to 1e-9. A 10 km/s aluminium impact matches the exact jump conditions to
  within 0.5%.
- **Optically thin bremsstrahlung:** an operator-split sink using the exact solution at fixed density, so it
  is stable at any time step. In the decay test T(t) matches the analytic solution to 1e-14 and E + E_rad is
  conserved to 7e-16.
- **Magnetized thermal conduction:** anisotropic, k_par ~ T^5/2 along B and k_perp = k_par/(1 + (omega_ce
  tau_e)^2) across it, with limited transverse gradients and RKL2 super-time-stepping. A hot spot with
  k_par/k_perp = 1e4 spreads along B as 2 D t and stays confined across it; for an oblique field the numerical
  cross-field leakage falls at second order with resolution.

All four modules run in 2D and 3D; z-invariant 3D slabs reproduce the 2D solver to round-off. Liners have no
strength or spall, and the example parameters are illustrative, not taken from any experiment.

### Examples

```bash
# liner-driven flux compression: stiffened-gas aluminium liner with finite conductivity
python examples/mtif_flux_compression.py --eos-liner stiffened --sigma-liner 3.5e7

# the same at hot, dense core scales (~100 eV), where bremsstrahlung matters
python examples/mtif_flux_compression.py --enable-rad --v-ref-kms 100 --rho-ref 100

# synthetic b-dot probe signals for an MTIF-3-style implosion, Mie-Gruneisen aluminium liner
python examples/nearstar_mtif3_validation.py --eos-liner mie_gruneisen --sigma-liner 3.5e7
```

## Benchmark: high-Mach MHD Richtmyer-Meshkov instability

A Mach-5 shock hits a single-mode interface between a light gas and one ten times denser. Baroclinic
vorticity deposited at the interface drives Richtmyer-Meshkov growth, spike roll-up and mixing. A uniform
field along the shock normal (B_x) suppresses it: the vorticity is carried away from the interface along
field lines by Alfven waves (Wheatley, Pullin & Samtaney 2005), so the interface amplitude stays bounded.

| | Hydro (B = 0) | MHD (B_x = 2.5) |
| :--- | :--- | :--- |
| Shock and interface | Mach 5, ρ₂/ρ₁ = 10, A = 0.82 (0.73 after compression) | same |
| Amplitude a(t) at t = 1.5 | 0.317, still growing | 0.064 (peak 0.092 at t ≈ 0.6), 80% below hydro |
| Interface | spike roll-up (mushroom) | bounded single mode |
| max \|ω\| at t = 1.5 (peak after impact) | 17.5 (57.8) | 11.6 (55.3) |
| Energy conservation error | 2.9 × 10⁻¹² | 1.6 × 10⁻¹² |
| h max\|∇·B\|/max\|B\| | n/a | 2.4% at shock passage, ~0.2% after |

2D, 1024 × 128 cells, single realization; no resolution study yet for this benchmark. Reproduce with
`python scripts/run_rmi_mhd.py && python scripts/plot_rmi_mhd.py`.

![RMI density and field lines](results/rmi_mhd/rmi_density_fieldlines.gif)
*Log density at matching times, hydrodynamic (left) and B_x = 2.5 (right); white lines are magnetic field
lines, the cyan line is the interface (heavy fraction 0.5).*

### Running it

```
python scripts/run_rmi_mhd.py            # configs/rmi_mhd.json: hydro and B_x = 2.5, 1024 x 128, ~20 min
python scripts/run_rmi_mhd.py --quick    # 256 x 32 check, ~1 min (results/rmi_mhd_quick)
python scripts/plot_rmi_mhd.py           # PNG/GIF into results/rmi_mhd
python main.py --test rmi_mhd            # smoke test
```

The set-up is in `sieve/problems/rmi_mhd.py`; the shock travels in +x and the interface is
x(y) = x0 + eta0 cos(k y). Jumps come from the Rankine-Hugoniot conditions, and the shock-interface
interaction is solved exactly in 1D for the Richtmyer and Meyer-Blewett reference growth rates. With B_x,
the incident, reflected and transmitted shocks must all be fast MHD shocks, so B0 above ~3.4 is rejected at
these parameters. Each case writes `diagnostics.csv`, `summary.json`, `frames.npz` and VTK frames
(`vtk/frame_NNNN.vti`, `rmi.pvd` for ParaView) to `results/rmi_mhd/<case>/`. The VTK frames and
`frames.npz` (about 140 MB per case) are not tracked in git.

## Model and numerical limitations

- Ideal gas (γ = 5/3) by default; stiffened-gas and Mie-Gruneisen liners have no strength, spall or phase change (tension is limited only by cavitation at c^2 = 0). The stiffened-gas fit to a Hugoniot is exact at one particle velocity only. Mixed cells use the isobaric mixture rule, which is exact for stiffened gases and approximate for Mie-Gruneisen. No physical viscosity; thermal conduction is electron-only with the simple k_perp = k_par/(1 + x^2) interpolation and insulating liner and domain faces, and at grid-oblique field angles the limited scheme adds numerical cross-field diffusion that falls at second order with resolution. Radiation is optically thin bremsstrahlung only (no line or recombination radiation, no opacity or reabsorption, T_e = T_i), coupled by first-order operator splitting; in mixed cells the temperature is taken from the mixture pressure and density.
- Resistivity is scalar (isotropic), with no Hall term. The `material` model takes eta from the passive tracer, so it is only as sharp as the numerically smeared liner/fuel interface; the `spitzer` model needs its temperature floor and cap, and in the cold, dense default of the examples it sits at the cap.
- Cell-centred magnetic field with GLM cleaning; the divergence error at shocks is a few percent (no constrained transport).
- Pressure balance of the striped field holds exactly in the initial data but only to second order in the discrete scheme.
- Shock-parallel stripes are restricted to the fast-shock regime, B0 < 6.55 sqrt(p_bar) at Mach 10; switch-on shocks are not modelled.
- Most sweeps use one resolution and one realisation. The lock boundary was rechecked at 2x and 4x resolution (`experiments/convergence_lock.py`): interface roughness converges, mixing-layer enstrophy of the hydrodynamic reference does not (no physical viscosity), so enstrophy ratios against hydro are not grid-converged.
- The legacy campaign in `main.py --campaign` uses locking criteria (window enstrophy, raw gap/bar momentum) that this study found unreliable for MHD. It is kept for reference and should not be run as is.

## Citation

If you use this code or its results, please cite it; GitHub shows a "Cite this repository" button generated
from `CITATION.cff`.

## License

No license has been chosen yet. Until one is added, the code is all rights reserved: you may read it, but
not reuse, modify or redistribute it without the author's permission.
