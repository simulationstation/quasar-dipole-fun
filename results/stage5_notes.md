# Stage 5: Magnitude/redshift slicing stability tests

Run the slicing script to reproduce the magnitude- and redshift-binned dipoles from the Secrest+22 CatWISE AGN catalog.

## Quick start

```bash
python slice_secrest_dipole.py \
  --catalog ./data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --outdir ./results/slicing \
  --n-bootstrap 200
```

Use quantile bins by default (six bins). To specify explicit W1 edges:

```bash
python slice_secrest_dipole.py --w1-bins "13.5,14.0,14.5,15.0,15.5,16.0,16.4"
```

Enable fast sanity checks on a subset:

```bash
python slice_secrest_dipole.py --quick --quick-count 200000
```

Outputs include JSON summaries (`w1_bins.json`, `z_bins.json`) and a markdown report (`stage5_slicing_report.md`) under `results/slicing/`.
