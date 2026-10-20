# Magnetic sieve: striped magnetic fields and the Richtmyer–Meshkov instability

A 2D compressible MHD solver (CuPy + a custom CUDA HLLD kernel) and the parameter study built on it. The study asks whether a spatially striped magnetic field can suppress ("lock") the Richtmyer–Meshkov instability of a shocked density interface while channelling the flow through the weak-field gaps.

**Result in one paragraph.** A uniform shock-parallel field locks the interface, in line with earlier work on magnetised RMI. Striped fields lock only when the bar/gap field contrast is small (about 10%). A shock-parallel field is not compressed by the shock while the gas is, so a striped field that is in pressure balance upstream produces a steady field-aligned shear downstream. Magnetic tension does not resist this shear, and it deforms the interface at the stripe scale. Suppression and collimation therefore work against each other in this regime. Aligning the stripe phase with the interface ripple produces a predictable zero crossing of the RMI mode, but not a lock. See `paper/` for the write-up and its limitations.

## Layout

```
sieve/            solver package
  config.py       typed configuration (unknown keys raise ValueError)
  physics.py      CUDA HLLD wrapper, MUSCL, jump conditions, resistive terms, stripe equilibrium
  hlld_kernel.cu  HLLD Riemann solver (one thread per face)
  solver.py       MHDSolver (SSP-RK3, GLM + Powell, BCs, energy budget) and PostProcessor
  solver3d.py     MHDSolver3D: unsplit HLLD in x, y, z (same kernel, rotated), z-periodic, 3D seeding,
                  current-density and generated-field diagnostics
  diagnostics.py  mixing, enstrophy, interface spectrum, div B, energy
main.py           CLI: verification tests, single cases, legacy campaign
experiments/      sweeps A-H and analysis scripts; data/ holds the saved results
paper/            figure and .docx generators, figures/, magnetic_sieve.docx
rmi_output_gpu/   figures and GIFs written by the solver
```

## Requirements

An NVIDIA GPU with CUDA 12 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Usage

```
python main.py --test all                     # verification suite, ~3 min
python main.py --case striped --B0 3.5 --gap-fraction 0.9 --compare-hydro
python main.py --help
```

`--case` runs a 12 x 2 domain to t = 0.8 and writes figures and 61-frame GIFs to `rmi_output_gpu/`.

## Verification

`python main.py --test all` runs the full suite and exits non-zero on failure:

| test | checks |
|---|---|
| brio_wu | Brio–Wu shock tube structure |
| alfven | Alfvén wave convergence order (measured 1.96) |
| reflection | residual of fast and Alfvén pulses after leaving through the outflow boundary (3.5e-5, 7e-10) |
| conservation | energy budget closes to round-off (~1e-14); div B = 0 for Bz stripes, < 3% (h max\|div B\|/max\|B\|) otherwise |
| equilibrium | upstream gas stays at rest for pressure-balanced x stripes; misspelled config keys are rejected |
| z_invariance | z-invariant 3D runs reproduce the 2D solver, ideal and resistive (difference 1e-14) |
| alfven_z | Alfven wave along z converges (orders 1.3, 1.9 at nz = 16-64) |
| resistive_z | weak B_x(z) decays as exp(-eta k^2 t) (error 3e-4); Ohmic heating conserves energy |

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

## Benchmark: high-Mach RMI with magnetic suppression

A standalone problem in `sieve/problems/rmi_mhd.py`: a Mach-5 shock in a light gas hits a single-mode
interface x(y) = x0 + eta0 cos(k y) with a gas 10 times denser (A ~ 0.82), with no field, a uniform field
along the shock normal (B_x) or across it (B_y). Repo convention: the shock travels in +x.

```
python scripts/run_rmi_mhd.py            # configs/rmi_mhd.json: hydro and B_x = 2.5, 1024 x 128, ~30 min
python scripts/run_rmi_mhd.py --quick    # 256 x 32 check, ~1 min (results/rmi_mhd_quick)
python scripts/plot_rmi_mhd.py           # PNG/GIF into results/rmi_mhd
python main.py --test rmi_mhd            # smoke test
```

- The jumps come from Rankine-Hugoniot conditions. The shock-interface interaction is solved exactly in 1D
  (contact velocity, post-shock Atwood number) for the Richtmyer and Meyer-Blewett reference growth rates.
- With B_x, the incident, reflected and transmitted shocks must all be fast MHD shocks. Inadmissible B0 is
  rejected; at Mach 5 and rho2/rho1 = 10 the incident shock is the binding one (B0 < ~3.4).
- Per case, `results/rmi_mhd/<case>/` holds `diagnostics.csv` (a(t), bubble/spike positions, max vorticity,
  energy error, div B norms, minimum pressure and density), `summary.json`, `frames.npz`, and
  `vtk/frame_NNNN.vti` with `rmi.pvd` for ParaView (density, pressure, heavy fraction, velocity, B).
- `.gitignore` excludes the VTK frames and `frames.npz` (about 150 MB per case at full resolution).

## Model and numerical limitations

- 2D, ideal gas (γ = 5/3), constant resistivity, no physical viscosity or conduction.
- Cell-centred magnetic field with GLM cleaning; the divergence error at shocks is a few percent (no constrained transport).
- Pressure balance of the striped field holds exactly in the initial data but only to second order in the discrete scheme.
- Shock-parallel stripes are restricted to the fast-shock regime, B0 < 6.55 sqrt(p_bar) at Mach 10; switch-on shocks are not modelled.
- Most sweeps use one resolution and one realisation. Sweep G shows deep-stripe metrics are resolution sensitive; the hydrodynamic reference is converged.
- The legacy campaign in `main.py --campaign` uses locking criteria (window enstrophy, raw gap/bar momentum) that this study found unreliable for MHD. It is kept for reference and should not be run as is.
