# Literature validation traces

This folder is the trace library read by `aegis.validation.scan_library` and
`examples/validate_experimental_shots.py`.

**It currently holds no digitized experimental traces.** Only `synthetic/` is populated, with self-test traces from
a simulation at a hidden parameter value. They test the tool chain and say nothing about any experiment. The real
experimental data bundled with the code are the Marsh (1980) Hugoniots in `aegis/data/experimental/`.

## Adding a digitized trace

Digitize the figure (for example with WebPlotDigitizer), put the points in a CSV here, and start the file with a
provenance header:

```
# name: z_liner_shot1234_bdot
# kind: bdot
# units: V
# provenance: digitized
# citation: A. Author et al., Phys. Plasmas 00, 000000 (2020), doi:10.xxxx/xxxxx
# figure: Fig. 5(b), upper trace
# facility: Z
# shot: 1234
# sigma: 5%
# t_scale: 1e-9
t,y
100.0,0.12
...
```

- Required for every file: `name`, `kind` (`bdot`, `visar`, `framing`, ...), `units`, `provenance`.
- `provenance` is `digitized`, `measured` or `synthetic`. `digitized` and `measured` files must also give `citation`
  and `figure`; otherwise the file is refused.
- Uncertainty is a third column, or a `sigma` header (absolute, or a percentage of the peak |y|). The real
  uncertainty belongs here, including the digitization error. chi^2_nu means little with a guessed sigma.
- `t_scale` and `y_scale` convert to the units of the simulated signal.
- HDF5 works too: one group per trace with datasets `t`, `y`, optional `sigma`, and the header fields as attributes.

Check copyright before committing digitized data from a paper, and keep the citation with the data.

## Comparing it with a simulation

A trace counts as validation only when the simulation represents that experiment: same geometry, drive, materials and
probe location. Set the shot up as a case, place probes (`BdotProbe`, `VisarProbe`, `FramingCamera`) at the physical
locations, and give the signals the trace's name. The demonstration sweep does not compare real shots with its
dimensionless test liner, and it lists them as not compared.
