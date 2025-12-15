#!/usr/bin/env python3
"""
Stage 3 Pipeline: CatWISE dipole with NVSS removal and ecliptic correction.

This script implements the full Stage 3 pipeline:
1. Load Secrest+22 CatWISE AGN catalog
2. Apply baseline cuts (|b|>30, w1cov>=80, W1<=16.4)
3. Remove NVSS counterparts via source_id join (Secrest method)
4. Apply ecliptic latitude bias correction (weight-based)
5. Compute dipole with bootstrap uncertainties
6. Generate sampler constraints JSON

Usage:
    python3 run_stage3_pipeline.py --outdir results/stage3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u


# Published Secrest+22 values for comparison
SECREST_PUBLISHED = {
    "amplitude": 0.0154,
    "amplitude_sigma": 0.0015,
    "l_deg": 238.2,
    "b_deg": 28.8,
    "N_sources": 1355352,
    "reference": "Secrest et al. 2022, ApJL 937 L31"
}

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


def compute_dipole(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, float, float, np.ndarray]:
    """Compute dipole from Galactic coordinates with optional weights."""
    if weights is None:
        weights = np.ones(len(l_deg))

    l_rad = np.radians(l_deg)
    b_rad = np.radians(b_deg)
    cos_b = np.cos(b_rad)

    x = cos_b * np.cos(l_rad) * weights
    y = cos_b * np.sin(l_rad) * weights
    z = np.sin(b_rad) * weights

    sum_vec = np.array([x.sum(), y.sum(), z.sum()])
    W = weights.sum()

    amplitude = 3.0 * np.linalg.norm(sum_vec) / W if W > 0 else np.nan

    if np.linalg.norm(sum_vec) > 0:
        d_unit = sum_vec / np.linalg.norm(sum_vec)
        l_dip = math.degrees(math.atan2(d_unit[1], d_unit[0])) % 360.0
        b_dip = math.degrees(math.asin(np.clip(d_unit[2], -1, 1)))
    else:
        l_dip, b_dip = np.nan, np.nan

    return amplitude, l_dip, b_dip, sum_vec


def bootstrap_dipole(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    weights: np.ndarray,
    n_boot: int = 200,
    seed: int = 42
) -> Dict[str, Any]:
    """Bootstrap dipole uncertainties with weights."""
    rng = np.random.default_rng(seed)
    N = len(l_deg)

    # Normalize weights for probability sampling
    prob = weights / weights.sum()

    D_boot = []
    l_boot = []
    b_boot = []

    for _ in range(n_boot):
        idx = rng.choice(N, size=N, replace=True, p=prob)
        D, l, b, _ = compute_dipole(l_deg[idx], b_deg[idx], weights[idx])
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


def compute_ecliptic_lat(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Compute ecliptic latitude for each source."""
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ecl = coords.geocentrictrueecliptic
    return ecl.lat.deg


def build_ecliptic_weights(
    beta_deg: np.ndarray,
    n_bins: int = 36,
    floor_fraction: float = 0.25
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build weights that flatten the ecliptic latitude distribution.

    Each source gets weight w = 1/n(beta_bin) where n(beta_bin) is the
    count in that source's ecliptic latitude bin.

    Args:
        beta_deg: Ecliptic latitude in degrees
        n_bins: Number of bins (36 = 5 degree bins)
        floor_fraction: Minimum count as fraction of median (to avoid exploding weights)

    Returns:
        weights: Array of weights (normalized so mean=1)
        info: Dictionary with bin details
    """
    # Create bins from -90 to +90
    bin_edges = np.linspace(-90, 90, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count sources per bin
    counts, _ = np.histogram(beta_deg, bins=bin_edges)

    # Apply floor to avoid exploding weights
    median_count = np.median(counts[counts > 0])
    floor_count = max(1, floor_fraction * median_count)
    counts_floored = np.maximum(counts, floor_count)

    # Compute inverse-count weights per bin
    bin_weights = median_count / counts_floored

    # Assign weights to each source
    bin_idx = np.digitize(beta_deg, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    weights = bin_weights[bin_idx]

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    info = {
        "n_bins": n_bins,
        "bin_edges_deg": bin_edges.tolist(),
        "bin_centers_deg": bin_centers.tolist(),
        "counts_raw": counts.tolist(),
        "counts_floored": counts_floored.tolist(),
        "floor_count": float(floor_count),
        "median_count": float(median_count),
        "bin_weights": bin_weights.tolist(),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "method": "inverse_count_per_bin",
        "notes": "Weights flatten beta distribution; floor prevents exploding weights"
    }

    return weights, info


def apply_nvss_removal_secrest(
    catwise_tbl: Table,
    nvss_crossmatch_path: Path,
    match_radius_arcsec: float = 40.0
) -> Tuple[Table, Dict[str, Any]]:
    """
    Remove NVSS counterparts using Secrest's pre-matched catalog.

    The NVSS_CatWISE2020_40arcsec_best_symmetric.fits file contains
    pre-matched NVSS-CatWISE sources. We join on source_id to identify
    and remove radio-loud AGN.

    Args:
        catwise_tbl: CatWISE AGN table with source_id column
        nvss_crossmatch_path: Path to NVSS-CatWISE crossmatch FITS
        match_radius_arcsec: Match radius used in crossmatch (for documentation)

    Returns:
        filtered_tbl: CatWISE table with NVSS matches removed
        stats: Dictionary with removal statistics
    """
    N_before = len(catwise_tbl)

    # Load pre-matched NVSS-CatWISE catalog
    nvss_cw = Table.read(nvss_crossmatch_path)
    N_nvss_matches = len(nvss_cw)

    # Get unique CatWISE source_ids that have NVSS matches
    nvss_source_ids = set(nvss_cw['source_id'].data.astype(str))

    # Filter out sources with NVSS matches
    catwise_source_ids = catwise_tbl['source_id'].data.astype(str)
    keep_mask = ~np.isin(catwise_source_ids, list(nvss_source_ids))

    filtered_tbl = catwise_tbl[keep_mask]
    N_after = len(filtered_tbl)
    N_removed = N_before - N_after

    stats = {
        "nvss_crossmatch_file": str(nvss_crossmatch_path),
        "match_radius_arcsec": match_radius_arcsec,
        "N_nvss_catwise_matches_total": N_nvss_matches,
        "N_before": N_before,
        "N_removed": N_removed,
        "N_after": N_after,
        "fraction_removed": float(N_removed / N_before) if N_before > 0 else 0.0,
        "method": "source_id_join",
        "notes": "Used Secrest pre-matched NVSS_CatWISE2020_40arcsec_best_symmetric catalog"
    }

    return filtered_tbl, stats


def run_pipeline(
    catalog_path: Path,
    nvss_crossmatch_path: Path,
    outdir: Path,
    b_cut: float = 30.0,
    w1cov_min: float = 80.0,
    w1_max: float = 16.4,
    n_bootstrap: int = 200,
    seed: int = 42,
    nvss_match_radius: float = 40.0,
    ecl_n_bins: int = 36,
) -> Dict[str, Any]:
    """Run full Stage 3 pipeline."""

    print(f"Loading catalog: {catalog_path}")
    tbl = Table.read(catalog_path)
    N_raw = len(tbl)
    print(f"  Raw rows: {N_raw}")

    # Extract columns
    l_all = tbl['l'].data.astype(float)
    b_all = tbl['b'].data.astype(float)
    ra_all = tbl['ra'].data.astype(float)
    dec_all = tbl['dec'].data.astype(float)
    w1_all = tbl['w1'].data.astype(float)
    w1cov_all = tbl['w1cov'].data.astype(float)

    # =========================================================
    # BASELINE: Apply standard cuts only
    # =========================================================
    print("\n=== BASELINE (no corrections) ===")
    baseline_mask = (
        (np.abs(b_all) > b_cut) &
        (w1cov_all >= w1cov_min) &
        (w1_all <= w1_max)
    )
    N_baseline = baseline_mask.sum()
    print(f"  After cuts: {N_baseline}")

    l_base = l_all[baseline_mask]
    b_base = b_all[baseline_mask]

    D_base, l_dip_base, b_dip_base, _ = compute_dipole(l_base, b_base)
    boot_base = bootstrap_dipole(l_base, b_base, np.ones(N_baseline), n_bootstrap, seed)

    print(f"  Dipole: D={D_base:.5f}, (l,b)=({l_dip_base:.2f}, {b_dip_base:.2f})")

    # Check direction vs published
    pub_vec = lb_to_unitvec(SECREST_PUBLISHED["l_deg"], SECREST_PUBLISHED["b_deg"])
    base_vec = lb_to_unitvec(l_dip_base, b_dip_base)
    sep_from_pub = angle_deg(base_vec, pub_vec)
    print(f"  Separation from published: {sep_from_pub:.2f} deg")

    if sep_from_pub > 3.0:
        raise RuntimeError(f"Baseline direction {sep_from_pub:.2f}° from published exceeds 3° threshold!")

    baseline_result = {
        "N_sources": int(N_baseline),
        "amplitude": float(D_base),
        "l_deg": float(l_dip_base),
        "b_deg": float(b_dip_base),
        "bootstrap": boot_base,
        "separation_from_published_deg": float(sep_from_pub)
    }

    # Save baseline
    (outdir / "baseline").mkdir(parents=True, exist_ok=True)
    with open(outdir / "baseline" / "dipole_baseline.json", "w") as f:
        json.dump(baseline_result, f, indent=2)
    with open(outdir / "baseline" / "cuts_baseline.json", "w") as f:
        json.dump({
            "b_cut_deg": b_cut,
            "w1cov_min": w1cov_min,
            "w1_max": w1_max,
            "N_raw": N_raw,
            "N_after": int(N_baseline)
        }, f, indent=2)

    # =========================================================
    # NVSS REMOVAL: Remove radio-loud AGN
    # =========================================================
    print("\n=== NVSS REMOVAL ===")
    tbl_cut = tbl[baseline_mask]
    tbl_nvss, nvss_stats = apply_nvss_removal_secrest(
        tbl_cut, nvss_crossmatch_path, nvss_match_radius
    )
    N_nvss = len(tbl_nvss)
    print(f"  Removed {nvss_stats['N_removed']} NVSS matches ({nvss_stats['fraction_removed']*100:.2f}%)")
    print(f"  After NVSS removal: {N_nvss}")

    l_nvss = tbl_nvss['l'].data.astype(float)
    b_nvss = tbl_nvss['b'].data.astype(float)
    ra_nvss = tbl_nvss['ra'].data.astype(float)
    dec_nvss = tbl_nvss['dec'].data.astype(float)

    D_nvss, l_dip_nvss, b_dip_nvss, _ = compute_dipole(l_nvss, b_nvss)
    boot_nvss = bootstrap_dipole(l_nvss, b_nvss, np.ones(N_nvss), n_bootstrap, seed)

    nvss_vec = lb_to_unitvec(l_dip_nvss, b_dip_nvss)
    sep_nvss_pub = angle_deg(nvss_vec, pub_vec)

    print(f"  Dipole: D={D_nvss:.5f}, (l,b)=({l_dip_nvss:.2f}, {b_dip_nvss:.2f})")
    print(f"  Separation from published: {sep_nvss_pub:.2f} deg")

    nvss_result = {
        "N_sources": int(N_nvss),
        "amplitude": float(D_nvss),
        "l_deg": float(l_dip_nvss),
        "b_deg": float(b_dip_nvss),
        "bootstrap": boot_nvss,
        "separation_from_published_deg": float(sep_nvss_pub)
    }

    # Also run with 30 arcsec and 20 arcsec for sensitivity
    sensitivity_results = {}
    for radius in [30, 20]:
        # Note: The pre-matched catalog uses 40 arcsec, so we can't truly vary this
        # Just document that 40 arcsec was used
        sensitivity_results[f"match_radius_{radius}_arcsec"] = {
            "note": "Pre-matched catalog uses 40 arcsec; cannot vary radius without re-crossmatching"
        }

    # Save NVSS results
    (outdir / "nvss_removal").mkdir(parents=True, exist_ok=True)
    with open(outdir / "nvss_removal" / "matched_fraction.json", "w") as f:
        json.dump(nvss_stats, f, indent=2)
    with open(outdir / "nvss_removal" / "dipole_nvss_removed.json", "w") as f:
        json.dump(nvss_result, f, indent=2)
    with open(outdir / "nvss_removal" / "cuts_nvss_removed.json", "w") as f:
        json.dump({
            "b_cut_deg": b_cut,
            "w1cov_min": w1cov_min,
            "w1_max": w1_max,
            "nvss_removal": True,
            "nvss_match_radius_arcsec": nvss_match_radius,
            "N_baseline": int(N_baseline),
            "N_after_nvss": int(N_nvss)
        }, f, indent=2)

    # =========================================================
    # ECLIPTIC CORRECTION: Weight-based flattening
    # =========================================================
    print("\n=== ECLIPTIC CORRECTION (baseline + ecliptic, no NVSS) ===")

    l_ecl = l_base.copy()
    b_ecl = b_base.copy()
    ra_ecl = ra_all[baseline_mask]
    dec_ecl = dec_all[baseline_mask]

    beta_ecl = compute_ecliptic_lat(ra_ecl, dec_ecl)
    weights_ecl, ecl_info = build_ecliptic_weights(beta_ecl, n_bins=ecl_n_bins)

    D_ecl, l_dip_ecl, b_dip_ecl, _ = compute_dipole(l_ecl, b_ecl, weights_ecl)
    boot_ecl = bootstrap_dipole(l_ecl, b_ecl, weights_ecl, n_bootstrap, seed)

    ecl_vec = lb_to_unitvec(l_dip_ecl, b_dip_ecl)
    sep_ecl_pub = angle_deg(ecl_vec, pub_vec)

    print(f"  Dipole: D={D_ecl:.5f}, (l,b)=({l_dip_ecl:.2f}, {b_dip_ecl:.2f})")
    print(f"  Separation from published: {sep_ecl_pub:.2f} deg")

    ecl_result = {
        "N_sources": int(N_baseline),
        "amplitude": float(D_ecl),
        "l_deg": float(l_dip_ecl),
        "b_deg": float(b_dip_ecl),
        "bootstrap": boot_ecl,
        "separation_from_published_deg": float(sep_ecl_pub)
    }

    # Save ecliptic results
    (outdir / "ecliptic_correction").mkdir(parents=True, exist_ok=True)
    with open(outdir / "ecliptic_correction" / "beta_hist.json", "w") as f:
        json.dump(ecl_info, f, indent=2)
    with open(outdir / "ecliptic_correction" / "dipole_ecl_corrected.json", "w") as f:
        json.dump(ecl_result, f, indent=2)

    # =========================================================
    # FINAL: Both corrections (NVSS removal + ecliptic correction)
    # =========================================================
    print("\n=== FINAL (NVSS removal + ecliptic correction) ===")

    beta_final = compute_ecliptic_lat(ra_nvss, dec_nvss)
    weights_final, ecl_info_final = build_ecliptic_weights(beta_final, n_bins=ecl_n_bins)

    D_final, l_dip_final, b_dip_final, _ = compute_dipole(l_nvss, b_nvss, weights_final)
    boot_final = bootstrap_dipole(l_nvss, b_nvss, weights_final, n_bootstrap, seed)

    final_vec = lb_to_unitvec(l_dip_final, b_dip_final)
    sep_final_pub = angle_deg(final_vec, pub_vec)
    cmb_vec = lb_to_unitvec(CMB_L_DEG, CMB_B_DEG)
    sep_final_cmb = angle_deg(final_vec, cmb_vec)

    print(f"  Dipole: D={D_final:.5f}, (l,b)=({l_dip_final:.2f}, {b_dip_final:.2f})")
    print(f"  Separation from published: {sep_final_pub:.2f} deg")
    print(f"  Separation from CMB: {sep_final_cmb:.2f} deg")

    final_result = {
        "N_sources": int(N_nvss),
        "amplitude": float(D_final),
        "l_deg": float(l_dip_final),
        "b_deg": float(b_dip_final),
        "bootstrap": boot_final,
        "separation_from_published_deg": float(sep_final_pub),
        "separation_from_cmb_deg": float(sep_final_cmb)
    }

    # Uncertainty estimates
    sigma_amp = 0.5 * (boot_final["amplitude_q84"] - boot_final["amplitude_q16"])
    sigma_dir = boot_final["direction_sigma_deg"]

    uncertainty = {
        "amplitude": float(D_final),
        "amplitude_q16": boot_final["amplitude_q16"],
        "amplitude_q50": boot_final["amplitude_q50"],
        "amplitude_q84": boot_final["amplitude_q84"],
        "amplitude_sigma": float(sigma_amp),
        "l_deg": float(l_dip_final),
        "b_deg": float(b_dip_final),
        "l_median": boot_final["l_median"],
        "b_median": boot_final["b_median"],
        "direction_sigma_deg": float(sigma_dir),
        "n_bootstrap": n_bootstrap
    }

    # Save final results
    (outdir / "final").mkdir(parents=True, exist_ok=True)
    with open(outdir / "final" / "dipole_final.json", "w") as f:
        json.dump(final_result, f, indent=2)
    with open(outdir / "final" / "cuts_final.json", "w") as f:
        json.dump({
            "b_cut_deg": b_cut,
            "w1cov_min": w1cov_min,
            "w1_max": w1_max,
            "nvss_removal": True,
            "nvss_match_radius_arcsec": nvss_match_radius,
            "ecliptic_correction": True,
            "ecliptic_n_bins": ecl_n_bins,
            "N_raw": N_raw,
            "N_baseline": int(N_baseline),
            "N_after_nvss": int(N_nvss)
        }, f, indent=2)
    with open(outdir / "final" / "uncertainty_final.json", "w") as f:
        json.dump(uncertainty, f, indent=2)
    with open(outdir / "final" / "ecliptic_weights_final.json", "w") as f:
        json.dump(ecl_info_final, f, indent=2)

    # =========================================================
    # COMPARISON TABLE
    # =========================================================
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Mode':<25} {'N':>10} {'D':>10} {'l':>8} {'b':>8} {'Sep(pub)':>10}")
    print("-" * 80)
    print(f"{'Published (Secrest+22)':<25} {SECREST_PUBLISHED['N_sources']:>10,} {SECREST_PUBLISHED['amplitude']:>10.5f} {SECREST_PUBLISHED['l_deg']:>8.2f} {SECREST_PUBLISHED['b_deg']:>8.2f} {0.0:>10.2f}")
    print(f"{'Baseline':<25} {N_baseline:>10,} {D_base:>10.5f} {l_dip_base:>8.2f} {b_dip_base:>8.2f} {sep_from_pub:>10.2f}")
    print(f"{'NVSS removed':<25} {N_nvss:>10,} {D_nvss:>10.5f} {l_dip_nvss:>8.2f} {b_dip_nvss:>8.2f} {sep_nvss_pub:>10.2f}")
    print(f"{'Ecliptic corrected':<25} {N_baseline:>10,} {D_ecl:>10.5f} {l_dip_ecl:>8.2f} {b_dip_ecl:>8.2f} {sep_ecl_pub:>10.2f}")
    print(f"{'Both corrections':<25} {N_nvss:>10,} {D_final:>10.5f} {l_dip_final:>8.2f} {b_dip_final:>8.2f} {sep_final_pub:>10.2f}")
    print("=" * 80)

    # =========================================================
    # GENERATE SAMPLER CONSTRAINTS
    # =========================================================
    constraints = {
        "catwise_dipole": {
            "amplitude": float(D_final),
            "amplitude_sigma": float(sigma_amp),
            "l_deg": float(l_dip_final),
            "b_deg": float(b_dip_final),
            "direction_sigma_deg": float(sigma_dir),
            "N_sources": int(N_nvss)
        },
        "derived_sigma_qso": float(sigma_amp),
        "derived_sigma_qso_dir_deg": float(sigma_dir),
        "source": {
            "catalog": str(catalog_path),
            "nvss_crossmatch": str(nvss_crossmatch_path),
            "cuts": {
                "b_cut_deg": b_cut,
                "w1cov_min": w1cov_min,
                "w1_max": w1_max
            },
            "corrections": {
                "nvss_removal": True,
                "ecliptic_correction": True
            },
            "reference": "Secrest et al. 2022, ApJL 937 L31"
        },
        "wagenveld_wall": {
            "note": "Using existing sampler priors for Wagenveld wall parameters"
        },
        "radio_ratio": {
            "note": "Using existing sampler encoding for radio ratio term"
        }
    }

    with open(outdir / "final" / "constraints.json", "w") as f:
        json.dump(constraints, f, indent=2)

    print(f"\nConstraints saved to {outdir / 'final' / 'constraints.json'}")
    print(f"  sigma_qso = {sigma_amp:.5f}")
    print(f"  sigma_qso_dir = {sigma_dir:.2f} deg")

    return {
        "baseline": baseline_result,
        "nvss_removed": nvss_result,
        "ecliptic_corrected": ecl_result,
        "final": final_result,
        "constraints": constraints
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/secrest/catwise_agns.fits",
                        help="Path to CatWISE AGN catalog")
    parser.add_argument("--nvss-crossmatch",
                        default="data/secrest/secrest+22_accepted/nvss/reference/NVSS_CatWISE2020_40arcsec_best_symmetric.fits",
                        help="Path to NVSS-CatWISE crossmatch")
    parser.add_argument("--outdir", default="results/stage3",
                        help="Output directory")
    parser.add_argument("--b-cut", type=float, default=30.0,
                        help="Galactic latitude cut |b| > b_cut")
    parser.add_argument("--w1cov-min", type=float, default=80.0,
                        help="Minimum W1 coverage")
    parser.add_argument("--w1-max", type=float, default=16.4,
                        help="Maximum W1 magnitude")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ecl-bins", type=int, default=36,
                        help="Number of ecliptic latitude bins for correction")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        catalog_path=Path(args.catalog),
        nvss_crossmatch_path=Path(args.nvss_crossmatch),
        outdir=outdir,
        b_cut=args.b_cut,
        w1cov_min=args.w1cov_min,
        w1_max=args.w1_max,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        ecl_n_bins=args.ecl_bins
    )


if __name__ == "__main__":
    main()
