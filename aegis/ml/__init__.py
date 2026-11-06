"""Additive machine-learning extensions of AEGIS-MHD.

Rule for everything in this package: a model never replaces, displaces or bypasses the HLLD fluxes, the constrained
transport update or the conservation machinery. Models are predictors with an exact guard (fno_subgrid: radiation solve,
sub-grid stress), extra refinement flags (amr_trigger) or diagnostics that read solver or experiment output
(pinn_diagnostics). Removing a model leaves the numerical solver unchanged.
"""
