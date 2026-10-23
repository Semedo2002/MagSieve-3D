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
  solver.py       MHDSolver (SSP-RK3, constrained transport or GLM + Powell, Cartesian or cylindrical r-z,
                  BCs, energy budget) and PostProcessor
  ct.py           constrained transport: face-centred B, upwinded edge EMFs, vector-potential initialisation
  transport.py    Braginskii Z = 1 coefficients, Nernst field advection, Ettingshausen heat flux
  solver3d.py     MHDSolver3D: unsplit HLLD in x, y, z (same kernel, rotated), z-periodic, 3D seeding,
                  current-density and generated-field diagnostics
  diagnostics.py  mixing, enstrophy, interface spectrum, div B, energy
main.py           CLI: verification tests, single cases, legacy campaign
examples/         MTIF liner flux compression; synthetic magnetic probes (SI units); 3D target asymmetries;
                  MagLIF-style r-z implosion
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

### Constrained transport and geometry

The magnetic field is face centred and evolved by constrained transport (Gardiner & Stone 2005): edge EMFs
from the HLLD fluxes with upwinded corrections, so the face divergence stays at round-off (about 1e-15) in 2D
and 3D. `set_vector_potential` initialises any field divergence-free; otherwise faces are built from the cell
values. `div_b="glm"` keeps the cell-centred GLM/Powell scheme the published sieve study used (the experiment
scripts and the RMI benchmark pin it, so their results reproduce). `geometry="cylindrical"` runs the 2D solver
in axisymmetric (r, z): x is the radius with a reflecting axis at x = 0, y the axial coordinate, the
out-of-plane components are -v_phi and -B_phi; fluxes are area weighted, the geometric sources
(rho v_phi^2 - B_phi^2 + p_T)/r, -(rho v_r v_phi - B_r B_phi)/r and F_r[B_phi]/r are added, and CT updates
B_z with r-weighted edge EMFs (E_phi = 0 on the axis).

### Thermal conduction

Magnetized electron conduction, q = -[k_perp I + (k_par - k_perp) b b] . grad T with k_par = `cond_kappa0` T^5/2
and k_perp from Braginskii's Z = 1 fit, k_perp/k_par = (4.664 x^2 + 11.92)/(3.1616 (x^4 + 14.79 x^2 + 3.7703)),
x = omega_ce tau_e = `cond_x0` |B| T^3/2 / rho (T = p/rho), in the fuel (liner fraction below
`cond_tracer_max`). At a fuel-liner face the fuel conducts to the liner as an isothermal wall at `cond_wall_T`
(a Dirichlet condition half a cell away, with the fuel's anisotropic k, so a field parallel to the wall
insulates it); the heat goes into the liner cell, so total energy is conserved exactly. `cond_wall="insulating"`
and `cond_perp="simple"` (1/(1 + x^2)) restore the earlier choices; non-periodic domain faces are insulating. Fluxes live on cell faces; the gradient normal to a face is a two-point
difference and the transverse gradients are van Leer limited (Sharma & Hammett 2007), which keeps strongly
anisotropic conduction from moving heat from cold to hot. It is a split step integrated with RKL2
super-time-stepping (Meyer, Balsara & Aslam 2014), about sqrt(dt/dt_explicit) stages, so it never limits the
hydrodynamic time step. `units.braginskii_kwargs` gives the Spitzer-Braginskii coefficients in code units (k_par
at 1 keV within 1.1% of the Spitzer value). Ion conduction is left out.

### Nernst and Ettingshausen effects

`nernst_coeff` (from `units.braginskii_kwargs(..., nernst=True)`, or `--nernst` in the examples) adds the
Nernst term to Ohm's law, E_N = -v_N x B with v_N = -(beta_wedge/(e|B|)) grad_perp(kT_e) (Braginskii Z = 1
beta_wedge): the field is carried down the temperature gradient, toward cold walls. It enters CT as an
upwinded edge EMF, so div B stays at round-off, and the time step respects |v_N|. Its Onsager partner, the
Ettingshausen heat flux (beta_wedge kT_e/e) b x J, and the Poynting flux of E_N enter the energy equation as
face fluxes, so total energy is conserved exactly. 2D solver (Cartesian and r-z).

### Two-temperature plasma

`two_temp=True` carries the electron entropy s_e = p_e/rho^gamma as an advected scalar (electrons are
compressed adiabatically; shock heating goes to the ions), with theta_e = (1 + Z)/Z p_e/rho and theta_i = (1 + Z)
p_i/rho (`z_ion`). Electron-ion equilibration relaxes theta_e - theta_i exactly at rate (1 + Z) nu, nu =
`te_eq_coeff` rho/theta_e^3/2, at fixed total energy. Conduction, bremsstrahlung, Ohmic and Ettingshausen heating
act on the electrons. `units.twotemp_kwargs` gives the Spitzer equilibration coefficient in code units.

### Viscosity

`visc_coeff` adds Braginskii parallel viscosity, Pi = -3 eta_0 (b b - I/3)(b b : grad v - div v/3), which
reduces to isotropic Navier-Stokes where B = 0. Stresses are face fluxes of momentum and energy (energy
conserved exactly), integrated as an RKL2 split step like conduction (`units.viscosity_kwargs`).

### Liner strength and spall

`strength=True` gives the liner (volume fraction above 0.5) a deviatoric stress (S_xx, S_yy, S_xy) evolved by a
hypoelastic Jaumann rate with shear modulus `shear_modulus`, returned radially to the von Mises surface sqrt(3
J2) = `yield_strength` after each step. The stress enters the momentum and energy fluxes (and the hoop term in
r-z), and the time step uses the longitudinal speed c_L = sqrt(c^2 + 4G/(3 rho)). Where the tension exceeds
`spall_strength` the cell is marked damaged: no tension (p >= 0) and no strength from then on.
`units.strength_kwargs` converts SI moduli.

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

## Performance

The hot paths are fused CUDA kernels (conversions, MUSCL, the CT corner EMF; HLLD reads any axis without
transposing), the right-hand side has no host synchronisation, and in 2D with CT each SSP-RK3 step is captured
once as a CUDA graph and replayed (`use_cuda_graphs`), bit-identical to the uncaptured step. On a Quadro P1000
under Windows a 128 x 128 ideal MHD step went from about 140 ms to 22 ms, a 16 x 4 step from 140 ms to 9 ms.

## Verification

`python main.py --test all` runs the full suite and exits non-zero on failure:

| test | checks |
|---|---|
| brio_wu | Brio-Wu shock tube structure |
| alfven | Alfvén wave convergence order (measured 1.96) |
| reflection | residual of fast and Alfvén pulses after leaving through the outflow boundary (3.5e-5, 7e-10) |
| conservation | energy budget closes to round-off (~1e-14); with CT div B stays at 1e-15 for every field (GLM: 1.3e-2) |
| ct | field-loop advection one period: CT div B 7.5e-15 (GLM 4e-3), 78% of the magnetic energy kept at 128x64; a 3D field from a vector potential in a 3D flow keeps div B at 2.5e-15 |
| cylindrical | r-z: a static magnetized state stays static to 2e-15; cylindrical Noh problem: post-shock density 15.8 (exact 16), shock at r = 0.1975 (0.2), pre-shock density exact; energy budget 5e-14 |
| equilibrium | upstream gas stays at rest for pressure-balanced x stripes to truncation error, second order (1.1e-1, 2.6e-2 c_s at nx = 100, 200); misspelled config keys are rejected |
| z_invariance | z-invariant 3D runs reproduce the 2D solver: ideal, constant eta, and eta varying 8x across the interface |
| alfven_z | Alfven wave along z converges (orders 1.3, 1.9 at nz = 16-64) |
| resistive_z | weak B_x(z) decays as exp(-eta k^2 t) (error 3e-4); Ohmic heating conserves energy |
| resistive_layers | two-layer steady diffusion, eta jumps 4x: piecewise-linear B_z recovered to 3e-4 with harmonic face averaging (arithmetic: 1.7e-2) |
| ohmic | periodic box, eta varying 10x: thermal energy gained matches the integral of eta J^2 to 0.14%, total energy to 2e-14 |
| eos_impact | 10 km/s Al impact: stiffened Al on stiffened Cu vs the exact impedance match, and Mie-Gruneisen Al on Al vs the Us-Up Hugoniot; shock speeds, pressure, densities and velocity within 0.5% |
| eos_interface | Al slab in gas at uniform p and u: volume-fraction tracer keeps p uniform to 1e-9 (round-off times p_inf/p); mass-fraction blending, for comparison, is off by 3e5 p within 0.002 us |
| wall_conduction | fuel slab cooling into a cold liner wall: mean temperature follows the exact series to 0.3%, 10x slower with the field parallel to the wall (k_par/k_perp = 10) |
| transport | Braginskii fit limits; Nernst: field bumps in B_y (faces) and B_w (cells) travel down a temperature gradient along the exact path to 0.06 cells; Ettingshausen conserves energy to 2e-14 |
| twotemp | theta_e - theta_i relaxes as the exact ODE to 0.3% at fixed energy (1.5e-16); Sod: electron entropy behind the shock equals the upstream value (electrons compressed adiabatically), T_i/T_e >= 1.13 there; bremsstrahlung cools only the electrons |
| viscosity | sound-wave damping from Braginskii parallel viscosity: 4/3 eta k^2/(2 rho) for B along k, 1/3 for B across k, 4/3 (Navier-Stokes) for B = 0, all within 3%; energy 6e-14 |
| strength | stiffened Al, G = 26 GPa, Y = 0.29 GPa: elastic pulse at c_L to 0.2% (not bulk c0); 0.5 km/s impact: elastic precursor at the HEL to 0.9% moving at c_L, plastic wave on the yield surface sqrt(3 J2) = Y and plateau S_xx = -2Y/3 to 0.01%; a bar pulled apart spalls at the centre only; energy budget 9e-15 |
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

- **Shock capturing and exact div B:** HLLD Riemann solver, MUSCL reconstruction, constrained transport: the
  face divergence stays at round-off (1e-15) in 2D, 3D and r-z.
- **Axisymmetric (r, z) geometry:** for MagLIF-style liners without Cartesian grid imprint; verified on the
  cylindrical Noh problem.
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
- **Two-temperature plasma, Braginskii viscosity, Nernst and Ettingshausen transport.**
- **Liner strength:** von Mises plasticity with a Hugoniot elastic limit, and spall.
- **Magnetized thermal conduction:** anisotropic, k_par ~ T^5/2 along B and Braginskii k_perp across it, with limited transverse gradients and RKL2 super-time-stepping. A hot spot with
  k_par/k_perp = 1e4 spreads along B as 2 D t and stays confined across it; for an oblique field the numerical
  cross-field leakage falls at second order with resolution.

Resistivity, EOS, radiation, conduction, two-temperature and viscosity run in 2D and 3D; z-invariant 3D slabs
reproduce the 2D solver to round-off. Nernst, strength and r-z geometry are 2D. The example parameters are
illustrative, not taken from any experiment.

### Examples

```bash
# liner-driven flux compression: stiffened-gas aluminium liner with finite conductivity
python examples/mtif_flux_compression.py --eos-liner stiffened --sigma-liner 3.5e7

# the same at hot, dense core scales (~100 eV), where bremsstrahlung matters
python examples/mtif_flux_compression.py --enable-rad --v-ref-kms 100 --rho-ref 100

# synthetic b-dot probe signals for an MTIF-3-style implosion, Mie-Gruneisen aluminium liner
python examples/nearstar_mtif3_validation.py --eos-liner mie_gruneisen --sigma-liner 3.5e7

# magnetized conduction on a 0.5 mm target at ~100 eV (--braginskii: physical omega_ce tau_e, the
# default; --kappa-ratio R: fix k_par/k_perp = R in the initial fuel instead; --kappa-ratio 1: unmagnetized)
python examples/mtif_flux_compression.py --enable-conduction --v-ref-kms 100 --r-ref-mm 0.5

# 3D target asymmetries: axis tilt and azimuthal liner modes against 2D and 3D references
python examples/mtif_3d_asymmetry.py --tilts 0.5 1 2 --modes 2 4

# MagLIF-style implosion in axisymmetric (r, z) with an axial liner ripple (magneto-Rayleigh-Taylor seed);
# --enable-conduction --nernst --v-ref-kms 100 --r-ref-mm 0.5 adds magnetized transport on a hot small target
python examples/maglif_rz.py
```

With conduction on, the summaries report the heat conducted out of the hot spot (core gas above half the
peak core temperature) as a percentage of its peak thermal energy, and the field's suppression of conduction
there: the sum of grad T . K . grad T over the hot spot with the field off divided by the sum with it on, for
the state at turnaround. In 2D the field is axial, so all in-plane conduction is across B; the liner wall is
insulating. On the 0.5 mm, 100 km/s target the hot spot loses 1.7% of its peak thermal energy with
Braginskii magnetization and 2.8% unmagnetized (`--kappa-ratio 1`); at the examples' default scales (fuel near
0.01 eV) conduction is negligible.

### 3D multi-physics parity and magnetized conduction

The two-material EOS with volume-fraction interfaces, bremsstrahlung and anisotropic conduction all run in
the 3D solver, sharing their kernels with 2D; a z-invariant 3D slab reproduces the 2D solver to round-off
(`--test parity3d`). Conduction follows k_par ~ T^5/2 along B and k_perp = k_par/(1 + (omega_ce tau_e)^2)
across it, with Sharma & Hammett (2007) limited transverse gradients so that strongly anisotropic heat flow
never runs from cold to hot, and RKL2 super-time-stepping so that it never limits the hydrodynamic time step.

### 3D target asymmetries

`examples/mtif_3d_asymmetry.py` implodes the liner in 3D (64^2 x 32, z periodic) with a uniform axial seed
field, so div B = 0 for any target shape. A tilt is an axis displacement dx(z) = A sin(2 pi z/Lz) whose
largest slope equals the tilt angle (a uniform tilt cannot be periodic); the modes are liner radii
R (1 + 0.02 cos(m phi)). Conduction on, Braginskii magnetization, 0.5 mm / 100 km/s scales:

| case | convergence ratio | peak mean core B / B0 | peak-B spread in z | field tilt (rms B_perp/B_z) | hot-spot conduction loss | field suppression at turnaround |
|---|---|---|---|---|---|---|
| 2D baseline | 5.01 | 20.73 | - | - | 0.95% | 9.2e4 |
| 3D reference | 5.01 | 20.73 | 0 | 0 | 0.95% | 9.2e4 |
| tilt 0.5 deg | 5.01 | 20.73 | 6.3e-4 | 3.6e-3 | 0.95% | 7.1e4 |
| tilt 1 deg | 5.01 | 20.73 | 1.3e-3 | 7.1e-3 | 0.97% | 4.8e4 |
| tilt 2 deg | 4.96 | 20.64 | 2.1e-3 | 1.4e-2 | 1.08% | 2.1e4 |
| m = 2 | 4.73 | 20.24 | 0 | 0 | 1.29% | 1.8e5 |
| m = 4 | 4.48 | 19.76 | 0 | 0 | 1.08% | 1.0e5 |

The unperturbed 3D run matches the 2D baseline exactly. Up to 2 deg, tilt barely changes the compression
(peak field -0.4% at 2 deg) but bends the core field lines in proportion to the tilt, which opens a parallel
conduction path: the field's suppression of conduction in the hot spot falls 4.5x between 0 and 2 deg. The
2% azimuthal modes cost more peak field (-2.4% for m = 2, -4.7% for m = 4) and raise the conductive loss
through the larger hot-spot surface, while the field lines stay axial. The axis displacement is at most 0.18
cells at this resolution, so it is invisible in density slices and shows only in these integrated metrics;
energy closes to 1e-14 in every run.

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

- Ideal gas (γ = 5/3) by default; liner strength is elastic-perfectly plastic (constant G and Y, no work or rate hardening, no thermal softening) with a threshold spall model, and there is no phase change. The stiffened-gas fit to a Hugoniot is exact at one particle velocity only. Mixed cells use the isobaric mixture rule, which is exact for stiffened gases and approximate for Mie-Gruneisen. Viscosity is the Braginskii parallel part only (no gyroviscosity or perpendicular terms); thermal conduction is electron-only (Braginskii k_perp fit, no Righi-Leduc term), the liner is treated as an isothermal heat sink, and at grid-oblique field angles the limited scheme adds numerical cross-field diffusion that falls at second order with resolution. Nernst and Ettingshausen are 2D only. Radiation is optically thin bremsstrahlung only (no line or recombination radiation, no opacity or reabsorption), coupled by first-order operator splitting; in mixed cells the temperature is taken from the mixture pressure and density.
- Resistivity is scalar (isotropic), with no Hall term. The `material` model takes eta from the passive tracer, so it is only as sharp as the numerically smeared liner/fuel interface; the `spitzer` model needs its temperature floor and cap, and in the cold, dense default of the examples it sits at the cap.
- Constrained transport keeps div B at round-off; the cylindrical geometry is 2D (r, z) only, and the Cartesian 2D MTIF examples are still cross-sections with a four-fold grid imprint (use `examples/maglif_rz.py` for an axisymmetric implosion).
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
