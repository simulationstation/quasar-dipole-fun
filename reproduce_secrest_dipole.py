#!/usr/bin/env python3
"""Reproduce the Secrest et al. CatWISE quasar dipole from their published catalog.

This script loads the actual Secrest+22 CatWISE AGN catalog and reproduces
the dipole measurement with documented cuts.

Reference: Secrest et al. 2022, ApJL 937 L31
Published dipole: D = 0.0154 ± 0.0015, (l, b) = (238.2°, 28.8°)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError as exc:
    raise SystemExit("astropy required") from exc


# Published Secrest+22 values for comparison
SECREST_PUBLISHED = {
    "amplitude": 0.0154,
    "amplitude_sigma": 0.0015,
    "l_deg": 238.2,
    "b_deg": 28.8,
    "l_sigma_deg": 8.0,  # approximate from paper
    "b_sigma_deg": 8.0,
    "N_sources": 1355352,  # after all cuts
    "reference": "Secrest et al. 2022, ApJL 937 L31"
}

# CMB dipole direction for comparison
CMB_L_DEG = 264.021
CMB_B_DEG = 48.253


def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    return np.array([
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    ])


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def compute_dipole(l_deg: np.ndarray, b_deg: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Compute dipole from Galactic coordinates.

    Returns: (amplitude, l_dip, b_dip, sum_vector)
    """
    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)
    cos_b = np.cos(b_rad)

    x = cos_b * np.cos(l_rad)
    y = cos_b * np.sin(l_rad)
    z = np.sin(b_rad)

    sum_vec = np.array([x.sum(), y.sum(), z.sum()])
    N = len(l_deg)

    amplitude = 3.0 * np.linalg.norm(sum_vec) / N if N > 0 else np.nan

    if np.linalg.norm(sum_vec) > 0:
        d_unit = sum_vec / np.linalg.norm(sum_vec)
        l_dip = math.degrees(math.atan2(d_unit[1], d_unit[0])) % 360.0
        b_dip = math.degrees(math.asin(np.clip(d_unit[2], -1, 1)))
    else:
        l_dip, b_dip = np.nan, np.nan

    return amplitude, l_dip, b_dip, sum_vec


def bootstrap_dipole(l_deg: np.ndarray, b_deg: np.ndarray, n_boot: int = 200,
                     seed: int = 42) -> Dict:
    """Bootstrap dipole uncertainties."""
    rng = np.random.default_rng(seed)
    N = len(l_deg)

    D_boot = []
    l_boot = []
    b_boot = []

    for _ in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        D, l, b, _ = compute_dipole(l_deg[idx], b_deg[idx])
        D_boot.append(D)
        l_boot.append(l)
        b_boot.append(b)

    D_boot = np.array(D_boot)
    l_boot = np.array(l_boot)
    b_boot = np.array(b_boot)

    # Direction uncertainty via angular separation from median
    median_l = np.median(l_boot)
    median_b = np.median(b_boot)
    median_vec = lb_to_unitvec(median_l, median_b)

    seps = []
    for l, b in zip(l_boot, b_boot):
        v = lb_to_unitvec(l, b)
        seps.append(angle_deg(v, median_vec))

    return {
        "n_bootstrap": n_boot,
        "amplitude_q16": float(np.percentile(D_boot, 16)),
        "amplitude_q50": float(np.percentile(D_boot, 50)),
        "amplitude_q84": float(np.percentile(D_boot, 84)),
        "amplitude_std": float(np.std(D_boot)),
        "l_median": float(median_l),
        "b_median": float(median_b),
        "direction_sigma_deg": float(np.percentile(seps, 68)),
    }


def compute_coverage(l_deg: np.ndarray, b_deg: np.ndarray) -> Dict:
    """Compute sky coverage diagnostics."""
    n_north = np.sum(b_deg > 0)
    n_south = np.sum(b_deg < 0)

    # RA from Galactic
    # (approximate - just for diagnostics)

    return {
        "N_total": len(l_deg),
        "N_north_gal": int(n_north),
        "N_south_gal": int(n_south),
        "hemisphere_ratio": float(n_north / n_south) if n_south > 0 else np.nan,
        "l_min": float(l_deg.min()),
        "l_max": float(l_deg.max()),
        "b_min": float(b_deg.min()),
        "b_max": float(b_deg.max()),
        "b_abs_max": float(np.abs(b_deg).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="./data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits",
                        help="Path to Secrest CatWISE AGN catalog")
    parser.add_argument("--outdir", default="./results/secrest_reproduction",
                        help="Output directory")
    parser.add_argument("--b-cut", type=float, default=30.0,
                        help="Galactic latitude cut |b| > b_cut (default: 30, Secrest uses 30)")
    parser.add_argument("--w1cov-min", type=float, default=80.0,
                        help="Minimum W1 coverage (default: 80, Secrest uses 80)")
    parser.add_argument("--mask-file", type=str, default=None,
                        help="Path to exclusion mask file (Secrest exclude_master_revised.fits)")
    parser.add_argument("--w1-min", type=float, default=None,
                        help="Minimum W1 magnitude")
    parser.add_argument("--w1-max", type=float, default=16.4,
                        help="Maximum W1 magnitude (Secrest uses 16.4)")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load catalog
    print(f"Loading catalog: {args.catalog}")
    tbl = Table.read(args.catalog)
    print(f"  Raw rows: {len(tbl)}")

    # Extract columns
    l_all = tbl['l'].data.astype(float)
    b_all = tbl['b'].data.astype(float)
    w1_all = tbl['w1'].data.astype(float)
    w1cov_all = tbl['w1cov'].data.astype(float) if 'w1cov' in tbl.colnames else None

    # Document cuts
    cuts_applied = []
    mask = np.ones(len(tbl), dtype=bool)

    # Apply exclusion mask (circular regions around bright sources)
    if args.mask_file is not None:
        print(f"  Applying exclusion mask: {args.mask_file}")
        mask_tbl = Table.read(args.mask_file)
        ra_all = tbl['ra'].data.astype(float)
        dec_all = tbl['dec'].data.astype(float)
        cat_coords = SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg, frame='icrs')

        exclude_mask = np.zeros(len(tbl), dtype=bool)
        for row in mask_tbl:
            if not row['use']:  # use=False means exclude
                center = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
                sep = cat_coords.separation(center).deg
                exclude_mask |= (sep < row['radius'])

        mask &= ~exclude_mask
        cuts_applied.append({
            "name": "Exclusion mask",
            "condition": f"Outside {len(mask_tbl)} exclusion regions",
            "N_excluded": int(exclude_mask.sum()),
            "N_after": int(mask.sum())
        })
        print(f"  After exclusion mask: {mask.sum()} (excluded {exclude_mask.sum()})")

    # W1 coverage cut (Secrest uses >= 80)
    if w1cov_all is not None and args.w1cov_min is not None:
        cov_mask = w1cov_all >= args.w1cov_min
        mask &= cov_mask
        cuts_applied.append({
            "name": "W1 coverage",
            "condition": f"w1cov >= {args.w1cov_min}",
            "N_after": int(mask.sum())
        })
        print(f"  After w1cov >= {args.w1cov_min}: {mask.sum()}")

    # Galactic latitude cut
    gal_mask = np.abs(b_all) > args.b_cut
    mask &= gal_mask
    cuts_applied.append({
        "name": "Galactic latitude",
        "condition": f"|b| > {args.b_cut} deg",
        "N_before": int((~gal_mask).sum() + gal_mask.sum()),
        "N_after": int(mask.sum())
    })
    print(f"  After |b| > {args.b_cut}°: {mask.sum()}")

    # W1 magnitude cuts
    if args.w1_min is not None:
        w1_min_mask = w1_all >= args.w1_min
        mask &= w1_min_mask
        cuts_applied.append({
            "name": "W1 minimum",
            "condition": f"W1 >= {args.w1_min}",
            "N_after": int(mask.sum())
        })
        print(f"  After W1 >= {args.w1_min}: {mask.sum()}")

    if args.w1_max is not None:
        w1_max_mask = w1_all <= args.w1_max
        mask &= w1_max_mask
        cuts_applied.append({
            "name": "W1 maximum",
            "condition": f"W1 <= {args.w1_max}",
            "N_after": int(mask.sum())
        })
        print(f"  After W1 <= {args.w1_max}: {mask.sum()}")

    # Apply mask
    l_cut = l_all[mask]
    b_cut = b_all[mask]
    N_final = len(l_cut)
    print(f"  Final sample: {N_final}")

    # Compute dipole
    print("\nComputing dipole...")
    D, l_dip, b_dip, _ = compute_dipole(l_cut, b_cut)
    print(f"  Amplitude: D = {D:.5f}")
    print(f"  Direction: (l, b) = ({l_dip:.2f}, {b_dip:.2f}) deg")

    # Bootstrap
    print(f"\nBootstrapping ({args.bootstrap} resamples)...")
    boot = bootstrap_dipole(l_cut, b_cut, n_boot=args.bootstrap, seed=args.seed)
    print(f"  Amplitude [16,50,84]: [{boot['amplitude_q16']:.5f}, {boot['amplitude_q50']:.5f}, {boot['amplitude_q84']:.5f}]")
    print(f"  Direction sigma: {boot['direction_sigma_deg']:.2f} deg")

    # Coverage
    coverage = compute_coverage(l_cut, b_cut)

    # Comparison to published values
    pub_vec = lb_to_unitvec(SECREST_PUBLISHED["l_deg"], SECREST_PUBLISHED["b_deg"])
    rec_vec = lb_to_unitvec(l_dip, b_dip)
    cmb_vec = lb_to_unitvec(CMB_L_DEG, CMB_B_DEG)

    sep_from_published = angle_deg(rec_vec, pub_vec)
    sep_from_cmb = angle_deg(rec_vec, cmb_vec)

    # Sigma-level comparison (rough)
    amp_sigma = boot['amplitude_std']
    amp_diff_sigma = abs(D - SECREST_PUBLISHED["amplitude"]) / amp_sigma if amp_sigma > 0 else np.nan
    dir_diff_sigma = sep_from_published / boot['direction_sigma_deg'] if boot['direction_sigma_deg'] > 0 else np.nan

    comparison = {
        "recovered": {
            "amplitude": D,
            "l_deg": l_dip,
            "b_deg": b_dip,
            "N_sources": N_final
        },
        "published_secrest": SECREST_PUBLISHED,
        "angular_separation_from_published_deg": sep_from_published,
        "angular_separation_from_cmb_deg": sep_from_cmb,
        "amplitude_diff_sigma": amp_diff_sigma,
        "direction_diff_sigma": dir_diff_sigma,
    }

    # Save results
    dipole_result = {
        "catalog": args.catalog,
        "N_raw": len(tbl),
        "N_final": N_final,
        "dipole": {
            "amplitude": D,
            "l_deg": l_dip,
            "b_deg": b_dip,
        },
        "bootstrap": boot,
        "comparison_to_published": comparison,
        "cmb_separation_deg": sep_from_cmb,
    }

    with open(outdir / "dipole.json", "w") as f:
        json.dump(dipole_result, f, indent=2)

    with open(outdir / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    with open(outdir / "cuts_used.json", "w") as f:
        json.dump({
            "b_cut_deg": args.b_cut,
            "w1_min": args.w1_min,
            "w1_max": args.w1_max,
            "cuts_applied": cuts_applied,
        }, f, indent=2)

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON TO PUBLISHED SECREST+22 RESULTS")
    print("="*60)
    print(f"{'Metric':<30} {'Recovered':>15} {'Published':>15}")
    print("-"*60)
    print(f"{'N sources':<30} {N_final:>15,} {SECREST_PUBLISHED['N_sources']:>15,}")
    print(f"{'Amplitude D':<30} {D:>15.5f} {SECREST_PUBLISHED['amplitude']:>15.5f}")
    print(f"{'l (deg)':<30} {l_dip:>15.2f} {SECREST_PUBLISHED['l_deg']:>15.2f}")
    print(f"{'b (deg)':<30} {b_dip:>15.2f} {SECREST_PUBLISHED['b_deg']:>15.2f}")
    print("-"*60)
    print(f"{'Angular separation':<30} {sep_from_published:>15.2f}°")
    print(f"{'Sep from CMB dipole':<30} {sep_from_cmb:>15.2f}°")
    print(f"{'Amplitude diff (sigma)':<30} {amp_diff_sigma:>15.2f}")
    print(f"{'Direction diff (sigma)':<30} {dir_diff_sigma:>15.2f}")
    print("="*60)

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
