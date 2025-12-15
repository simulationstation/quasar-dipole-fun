# ℓ-space dipole diagnostic

This tool builds HEALPix maps from the CatWISE AGN catalog (or any catalog with the same columns) and compares the observed low-multipole power to structured null hypotheses. It complements the catalog-level dipole estimator by asking whether the $ℓ=1$ signal is specific to the true sky or can be reproduced by selection effects.

## What the test measures

- Constructs count and overdensity maps per W1 slice (full catalog plus magnitude bins).
- Computes pseudo-$C_\ell$ for $\ell=1..\ell_\max$ on the cut sky, reporting the dipole fraction and low-$\ell$ ratios.
- Measures the map-based dipole direction and compares it to the catalog vector sum.
- Generates null distributions via RA scrambling (preserves declination structure), pixel shuffling (samples the parent exposure map), and optional harmonic phase randomization.

## Why RA-scramble is the right null

Permuting right ascension while holding declination fixed keeps declination-dependent coverage and selection artifacts intact (e.g., scan stripes or zero-point drifts). If the observed $C_1$ is reproduced by RA scrambles, the dipole can be explained by those declination systematics rather than true sky structure.

## Interpreting the outputs

- **Dipole dominance**: a high $C_1$ fraction with low $C_2$/$C_3$ supports a genuine large-scale dipole; significant $C_2$/$C_3$ hints at more complex anisotropy.
- **Null p-values**: if RA-scramble $p(C_1)$ is large, the dipole is likely driven by declination-dependent selection; if pixel-shuffle yields low $p(C_1)$ but $C_2$/$C_3$ are null, the dipole is difficult to mimic without the real sky.
- **Direction drift**: stable dipole direction across magnitude bins with declining amplitude is consistent with local-structure dilution. Large directional RMS with rising $C_2$/$C_3$ points to multiple components or residual systematics.
- **Cut-sky caution**: pseudo-$C_\ell$ values are divided by $f_\text{sky}$ but still suffer from mode mixing; rely on ratios and null comparisons rather than absolute amplitudes.

## Running examples

Full sample only (baseline cuts, default bins disabled):

```bash
python scripts/ell_space_test.py --catalog ./data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --tag ell_full_only --w1-bins "" --n-bins 1
```

Magnitude slicing with six quantile bins and default nulls:

```bash
python scripts/ell_space_test.py --catalog ./data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits \
  --tag ell_six_bins --nside 64 --ellmax 20 --n-bins 6
```

Quick smoke test (three bins, 50 null draws, $\ell_\max=10$):

```bash
python scripts/ell_space_test.py --mode quick --tag ell_quick \
  --catalog ./data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits
```

All outputs land in `results/ell_space/<tag>/` with HEALPix FITS maps, $C_\ell$ plots, null histograms, and a Markdown report summarizing the interpretations.
