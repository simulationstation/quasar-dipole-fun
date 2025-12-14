# Catalog-level dipole estimation pipeline

This repository connects local sky catalogs to the existing Bayesian dipole model comparison framework (`metropolis_hastings_sampler.py`). The focus is correctness and reproducibility: no data are downloaded, no external sky pixelization is required, and the pipeline fails fast when inputs are incomplete.

## What this provides

1. **Input schema** (`catalog_schema.md`): documents the required catalog columns and masking conventions.
2. **Dipole estimators** (`dipole_estimators.py`): vector and weighted dipoles, bootstrap uncertainties, jackknife splits, and null tests without HEALPix.
3. **Sky masks** (`sky_masks.py`): Galactic latitude and rectangular masks with a built-in equatorial→Galactic transform.
4. **CLI estimator** (`estimate_catalog_dipole.py`): turns a local CSV/Parquet catalog into a dipole amplitude, direction, and covariance, with optional null tests and jackknife diagnostics.
5. **Model interface** (`to_model_constraints.py`): converts catalog dipole outputs into the amplitude and angular inputs expected by the Bayesian sampler.

## Dependencies

- Python ≥ 3.9 (tested with 3.12)
- `numpy`, `pandas`
- `pyarrow` (only if reading Parquet)
- `astropy` (coordinate transforms for Galactic masking and mock generation)

If these packages are absent, the CLI will raise an import error rather than attempting to install anything automatically.

## Usage overview

1. Prepare a local catalog that matches `catalog_schema.md`.
2. Run the estimator (example with CatWISE-like columns):
   ```bash
   python estimate_catalog_dipole.py \
     --catalog catwise.csv \
     --ra-col ra --dec-col dec \
     --mag-col w1mag --mag-min 14 --mag-max 17 \
     --b-cut 20 --bootstrap 1000 --jackknife \
     --output dipole_result.json
   ```
3. Convert to sampler inputs (optionally comparing to a reference direction):
   ```bash
   python to_model_constraints.py --dipole-json dipole_result.json \
     --ref-lon 264 --ref-lat 48 \
     --output model_constraints.json
   ```

The resulting JSON can be ingested by `metropolis_hastings_sampler.py` by mapping `amplitude_mean`, `amplitude_sigma`, and (optionally) `angular_separation_deg` into the existing likelihood blocks.

## Validation and sanity checks

- **Bootstrap**: use `--bootstrap N` to obtain empirical amplitude/direction scatter.
- **Jackknife**: enable `--jackknife` to compare hemispheres and octants; differences highlight mask sensitivity.
- **Null tests**: `--null-tests N` randomizes longitudes to quantify how often noise mimics the measured amplitude.
- **Coverage warnings**: the estimator reports if hemispheres or octants are empty after cuts, signaling unreliable dipoles.

## What this does *not* claim

- No cosmological interpretation is performed here; outputs are catalog-level summary statistics only.
- No published dipole values are hardcoded; every result comes from the provided local catalog.
- No attempt is made to optimize speed; clarity and reproducibility take precedence.

## Extending to other catalogs

The structure is deliberately generic:

- Swap in radio catalogs by pointing `--catalog` to a radio source list and adjusting magnitude/weight flags accordingly.
- Replace the Gaussian angular likelihood in the sampler with a von Mises–Fisher kernel if desired; `to_model_constraints.py` keeps the interface minimal for such swaps.
- Additional masking strategies (e.g., complex survey footprints) can be layered by precomputing a boolean mask column and passing it via `--mask-col`.

If any required input is missing, the tools will halt with a descriptive error rather than proceed with fabricated or simulated data.

## CatWISE end-to-end reproduction

`reproduce_catwise_dipole.py` adds a streaming, reproducible pipeline for the CatWISE quasar sample. It accepts FITS/CSV/Parquet files stored locally (no downloads).

Example (CatWISE FITS from disk):

```bash
python reproduce_catwise_dipole.py \
  --input /path/to/catwise.fits \
  --outdir results/catwise \
  --ra-col ra --dec-col dec \
  --w1-col w1mpro --w1-min 14 --w1-max 17 \
  --qso-prob-col qso_prob --qso-prob-min 0.9 \
  --mask-gal-b-min 20 \
  --bootstrap 500 --chunk-size 200000 --seed 42
```

Outputs in `--outdir`:

- `catwise_dipole.json`: counts, applied cuts, dipole estimate, bootstrap summaries, CMB separation.
- `catwise_diagnostics.json`: hemisphere split dipoles and longitude-randomization p-value.
- `constraints_catwise.json`: sampler-ready payload.

If the QSO probability column is absent, add `--no-qso-prob-cut` to skip that filter.

## Mock validation

`run_mock_test.py` performs a full synthetic round-trip:

```bash
python run_mock_test.py
```

It generates a mock catalog with a known injected dipole, runs the reproduction pipeline, compares the recovered amplitude/direction to the truth, and exits non-zero if the discrepancy is large. The intermediate mock catalog and results live in `mock_run/`.

You can also generate a catalog directly:

```bash
python generate_mock_catalog.py --output mock.csv --n 80000 --dipole 0.02 --l-dipole 210 --b-dipole 30
```

## Feeding constraints into the sampler

`constraints_catwise.json` can be supplied to the Bayesian sampler to override the default CatWISE likelihood terms:

```bash
python metropolis_hastings_sampler.py --constraints results/catwise/constraints_catwise.json --compare-models --n-steps 5000 --burn-in 1000 --thin 10
```

When provided, the sampler prints the values it is using for `D_QSO_OBS`, `L_QSO_OBS_DEG`, `B_QSO_OBS_DEG`, `SIGMA_QSO`, and `SIGMA_QSO_DIR_DEG`, and embeds the constraint metadata in `posterior_summary.json`.
