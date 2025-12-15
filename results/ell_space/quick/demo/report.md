# ℓ-space dipole diagnostic

Catalog: `data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits`\n
NSIDE=64, ellmax=10, null draws=50

## Slices

- **full**: N=1401166, f_sky=0.210, C1=5.923e-05, dipole fraction=0.042, quad/dip=5.903, oct/dip=2.933\n- **bin_1**: N=467135, f_sky=0.210, C1=2.468e-04, dipole fraction=0.074, quad/dip=0.449, oct/dip=2.394\n- **bin_2**: N=468567, f_sky=0.210, C1=4.620e-05, dipole fraction=0.031, quad/dip=10.314, oct/dip=2.795\n- **bin_3**: N=467385, f_sky=0.210, C1=7.319e-05, dipole fraction=0.029, quad/dip=13.782, oct/dip=1.087\n
## Interpretation

- Pseudo-$C_\ell$ values divide by $f_{sky}$ but still retain cut-sky mode mixing; interpret ratios rather than absolute amplitudes.
- If RA-scramble p-values approach unity for $C_1$ while $C_2$/$C_3$ remain null, the dipole likely arises from declination-dependent selection.
- A high dipole fraction across bins with declining amplitude but stable direction supports a genuine large-scale structure dipole diluted at faint magnitudes.
- Growing $C_2$/$C_3$ or large direction drift with depth suggests multiple anisotropic components or residual systematics.

Null-test p-values are summarized in `ell_space_summary.json`.
