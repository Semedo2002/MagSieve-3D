
# MagSieve-3D
<img width="3600" height="540" alt="anim_density" src="https://github.com/user-attachments/assets/d08222f1-6143-43a1-8ed0-2e1b94dce9d4" />

3D Ideal Magnetohydrodynamics solver engineered in Python/CUDA. Designed specifically to simulate hypersonic shock interface interactions and verify the Magnetic Sieve scaling law for directional plasma transport.
# MagSieve-3D: GPU-Accelerated MHD Solver

MagSieve-3D is a computational fluid dynamics engine designed to investigate the suppression and collimation of the Richtmyer-Meshkov Instability (RMI) under extreme Mach 10+ conditions. 

Unlike standard commercial solvers, MagSieve-3D was built from the ground up to utilize spatially modulated magnetic topologies, creating a Magnetic Sieve effect that transitions chaotic turbulent mixing into directional plasma micro jets.

#Core Numerical Architecture

The solver is architected for stability in high gradient, high Mach environments using a CUDA  backend:

* Accelerator: CuPy/NVRTC for parallelization of flux kernels.
* Flux Scheme: HLLD Approximate Riemann Solver (resolves all 5 characteristic MHD waves).
* Reconstruction: 2nd-order MUSCL with Monotonized Central limiting.
* Time Integration: 3rd-order Strong Stability Preserving Runge-Kutta.
* Divergence Cleaning: Hybrid implementation of Generalized Lagrange Multiplier and Powell source terms.

# Physics Conclusion -> The Magnetic Sieve

This solver provides the computational proof for the Magnetic Sieve Scaling Law. By applying a modulated transverse field, the solver demonstrates a phase transition where magnetic tension locks the interface against baroclinic torque.

# The Sieve Equation
The critical locking threshold is defined by the balance of local Alfvén velocity and the impulsive hydrodynamic growth rate:

$$B_{critical} = \Psi \cdot \sqrt{\rho_{avg}} \cdot (k_{int} \cdot \Delta U \cdot A^+ \cdot a_0^+)$$

# V&V

Mathematical rigor was established through standard benchmarks:
* Alfvén Convergence: 1.96 order accuracy.
* Brio-Wu Shock Tube: Resolution of rotational discontinuities and slow shocks.
* Parametric Sweeps: Verified the linear scaling law across Mach 5, 10, and 15 regimes.

# Requirements

To run the solver, an NVIDIA GPU with CUDA support is required.

* Python 3.8+
* CuPy: For GPU accelerated array operations.
* NumPy: For CPU side post-processing.
* Matplotlib / H5py: For visualization and data serialization.
* NVIDIA CUDA Toolki: Compatible with your CuPy version.

# Project Structure

* `/src`: Core solver kernels (Fluxes, Reconstruction, Divergence Cleaning).
* `/tests`: V&V benchmark scripts (Brio-Wu, Alfvén waves).
* `/experiments`: Configuration files for the Mach 10 RMI Sieve runs.
* `/data`: Raw .csv output from parametric sweeps.
* `/docs`: The 4-page technical whitepaper.

# Future Work

I am currently expanding the architecture to support:
1.  Multi-node MPI parallelization.
2.  Equation of State (EoS) modules for non-ideal plasma regimes.
3.  Resistive MHD terms for Low Lundquist number simulations.

---
Author: Abdelrahman Shaltout  
Affiliation: Independent Research / Swansea University Alumnus  
Contact: abdelrahmanshalt@gmail.com
---
# Legal & Intellectual Property
© 2026 Abdelrahman Shaltout. All Rights Reserved.

The source code and theoretical framework (MagSieve-3D) contained in this repository 
are proprietary. Unauthorized reproduction, distribution, or use of this code, 
in whole or in part, is strictly prohibited. 

This repository is for review and demonstration purposes only. If you are 
interested in utilizing this solver for research or commercial applications, 
please reach out via the contact information provided above.
