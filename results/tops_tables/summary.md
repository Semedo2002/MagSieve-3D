# Solver opacities against Los Alamos TOPS

Source tables: TOPS grey means (Magee et al. 1995; aphysics2.lanl.gov/apps) at TOPS's own temperatures on an
adaptive density grid. Errors are |lookup / TOPS - 1| through the solver's table lookup.

## Al

423 densities (1e-06 - 100 g/cc) x 74 TOPS temperatures (0.0005 - 100 keV); 0 points beyond TOPS's tables.

| check | quantity | points | median | 99th pct | max | within 5% |
|---|---|---|---|---|---|---|
| table nodes | kappa_R | 31302 | 2.2e-16 | 1.0e-14 | 1.24e-14 | 100.00% |
| table nodes | kappa_P | 31302 | 4.4e-16 | 1.8e-14 | 5.20e-14 | 100.00% |
| table nodes | Zbar | 31302 | 1.1e-16 | 2.8e-15 | 9.10e-15 | 100.00% |
| held-out midpoints | kappa_R | 31228 | 2.3e-05 | 2.5e-03 | 9.91e-03 | 100.00% |
| held-out midpoints | kappa_P | 31228 | 2.2e-05 | 2.8e-03 | 9.93e-03 | 100.00% |
| held-out midpoints | Zbar | 31228 | 3.4e-07 | 9.1e-04 | 9.84e-03 | 100.00% |
| separate request, off-grid densities | kappa_R | 2738 | 2.5e-05 | 1.7e-03 | 6.59e-03 | 100.00% |
| separate request, off-grid densities | kappa_P | 2738 | 2.9e-05 | 2.3e-03 | 1.64e-02 | 100.00% |
| separate request, off-grid densities | Zbar | 2738 | 9.3e-06 | 7.2e-04 | 7.40e-03 | 100.00% |
| separate request, table densities | kappa_R | 96 | 2.2e-16 | 2.5e-05 | 6.51e-05 | 100.00% |
| separate request, table densities | kappa_P | 96 | 2.2e-16 | 1.9e-05 | 4.15e-05 | 100.00% |
| separate request, table densities | Zbar | 96 | 1.1e-16 | 1.1e-15 | 1.33e-15 | 100.00% |

Between TOPS temperatures (no TOPS value exists there): error when every other temperature is dropped, i.e. at twice the real spacing, so larger than the real interpolation error:

| T band | kappa_R median / 95th | kappa_P median / 95th | Z median / 95th |
|---|---|---|---|
| 0-0.003 keV | 0.068 / 0.395 | 0.062 / 0.262 | 0.038 / 0.238 |
| 0.003-0.03 keV | 0.031 / 0.171 | 0.051 / 0.429 | 0.006 / 0.043 |
| 0.03-0.3 keV | 0.030 / 0.156 | 0.044 / 0.346 | 0.003 / 0.022 |
| 0.3-3 keV | 0.004 / 0.036 | 0.014 / 0.093 | 0.000 / 0.003 |
| 3-101 keV | 0.011 / 0.031 | 0.007 / 0.049 | 0.000 / 0.000 |

## Cu

409 densities (1e-06 - 100 g/cc) x 74 TOPS temperatures (0.0005 - 100 keV); 0 points beyond TOPS's tables.

| check | quantity | points | median | 99th pct | max | within 5% |
|---|---|---|---|---|---|---|
| table nodes | kappa_R | 30266 | 2.2e-16 | 1.2e-14 | 1.95e-14 | 100.00% |
| table nodes | kappa_P | 30266 | 4.4e-16 | 1.7e-14 | 8.46e-14 | 100.00% |
| table nodes | Zbar | 30266 | 1.1e-16 | 2.8e-15 | 1.02e-14 | 100.00% |
| held-out midpoints | kappa_R | 30192 | 2.2e-05 | 2.5e-03 | 1.25e-02 | 100.00% |
| held-out midpoints | kappa_P | 30192 | 2.4e-05 | 3.0e-03 | 1.03e-02 | 100.00% |
| held-out midpoints | Zbar | 30192 | 1.1e-05 | 9.8e-04 | 9.97e-03 | 100.00% |
| separate request, off-grid densities | kappa_R | 2738 | 2.4e-05 | 1.8e-03 | 9.91e-03 | 100.00% |
| separate request, off-grid densities | kappa_P | 2738 | 3.0e-05 | 2.2e-03 | 1.34e-02 | 100.00% |
| separate request, off-grid densities | Zbar | 2738 | 1.2e-05 | 7.5e-04 | 3.01e-03 | 100.00% |
| separate request, table densities | kappa_R | 96 | 2.2e-16 | 3.6e-05 | 4.72e-05 | 100.00% |
| separate request, table densities | kappa_P | 96 | 3.3e-16 | 2.8e-05 | 2.93e-05 | 100.00% |
| separate request, table densities | Zbar | 96 | 2.2e-16 | 1.8e-15 | 1.78e-15 | 100.00% |

Between TOPS temperatures (no TOPS value exists there): error when every other temperature is dropped, i.e. at twice the real spacing, so larger than the real interpolation error:

| T band | kappa_R median / 95th | kappa_P median / 95th | Z median / 95th |
|---|---|---|---|
| 0-0.003 keV | 0.077 / 0.557 | 0.046 / 0.301 | 0.042 / 0.335 |
| 0.003-0.03 keV | 0.014 / 0.210 | 0.012 / 0.275 | 0.003 / 0.020 |
| 0.03-0.3 keV | 0.035 / 0.259 | 0.037 / 0.623 | 0.004 / 0.021 |
| 0.3-3 keV | 0.005 / 0.059 | 0.016 / 0.184 | 0.000 / 0.006 |
| 3-101 keV | 0.015 / 0.098 | 0.018 / 0.359 | 0.000 / 0.001 |

