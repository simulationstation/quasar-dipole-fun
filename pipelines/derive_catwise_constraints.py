#!/usr/bin/env python3
"""
Derive CatWISE/Secrest dipole constraints with optional systematics corrections.

This script provides a reproducible pipeline that:
  - Loads the CatWISE AGN catalog (as distributed via Secrest+22 Zenodo)
  - Applies the documented magnitude, coverage, and latitude cuts
  - Optionally removes radio-loud sources via NVSS/SUMSS crossmatch
  - Optionally applies an ecliptic-latitude systematic correction
  - Estimates the dipole and bootstrap uncertainties
  - Emits constraint JSON compatible with metropolis_hastings_sampler

Outputs are written under ``results/secrest_reproduction/<run-tag>/`` and include
constraints, coverage diagnostics, and (optionally) bootstrap samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u
except ImportError as exc:  # pragma: no cover - exercised in tests via message
    raise SystemExit(
        "astropy is required for coordinate handling. Install with `pip install astropy`."
    ) from exc

# Optional healpy diagnostics
try:  # pragma: no cover - optional dependency
    import healpy as hp

    HEALPY_AVAILABLE = True
except Exception:  # pragma: no cover - healpy often absent
    HEALPY_AVAILABLE = False

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M")


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l_rad = np.deg2rad(l_deg % 360.0)
    b_rad = np.deg2rad(np.clip(b_deg, -90.0, 90.0))
    cos_b = np.cos(b_rad)
    return np.column_stack(
        [cos_b * np.cos(l_rad), cos_b * np.sin(l_rad), np.sin(b_rad)]
    )


def unitvec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return float("nan"), float("nan")
    v = vec / norm
    l = math.degrees(math.atan2(v[1], v[0])) % 360.0
    b = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    return l, b


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        return float("nan")
    cosang = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def compute_dipole(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, np.ndarray]:
    if weights is None:
        weights = np.ones_like(l_deg, dtype=float)
    weights = np.asarray(weights, dtype=float)
    unit = lb_to_unitvec(l_deg, b_deg)
    sum_vec = np.sum(unit * weights[:, None], axis=0)
    wtot = float(np.sum(weights))
    amp = 3.0 * float(np.linalg.norm(sum_vec)) / wtot if wtot > 0 else float("nan")
    l_dip, b_dip = unitvec_to_lb(sum_vec)
    return amp, l_dip, b_dip, sum_vec


def bootstrap_dipole(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    beta_deg: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    compute_corrected: callable,
) -> Dict[str, Any]:
    n = len(l_deg)
    D_list: List[float] = []
    l_list: List[float] = []
    b_list: List[float] = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True, p=weights / weights.sum())
        D, l, b, _ = compute_corrected(l_deg[idx], b_deg[idx], beta_deg[idx], weights[idx])
        D_list.append(D)
        l_list.append(l)
        b_list.append(b)

    D_arr = np.asarray(D_list)
    l_arr = np.asarray(l_list)
    b_arr = np.asarray(b_list)

    med_l = float(np.nanmedian(l_arr))
    med_b = float(np.nanmedian(b_arr))
    med_vec = lb_to_unitvec(np.array([med_l]), np.array([med_b]))[0]

    seps = []
    for lv, bv in zip(l_arr, b_arr):
        vec = lb_to_unitvec(np.array([lv]), np.array([bv]))[0]
        seps.append(angle_between(vec, med_vec))

    return {
        "D_p16": float(np.nanpercentile(D_arr, 16)),
        "D_p50": float(np.nanpercentile(D_arr, 50)),
        "D_p84": float(np.nanpercentile(D_arr, 84)),
        "dir_sigma_deg": float(np.nanstd(seps)),
        "l_median": med_l,
        "b_median": med_b,
        "samples": pd.DataFrame(
            {"D": D_arr, "l_deg": l_arr, "b_deg": b_arr, "dir_sep_deg": seps}
        ),
    }


@dataclass
class EclipticCorrection:
    applied: bool
    model: str
    details: Dict[str, Any]
    weights: np.ndarray


# -----------------------------------------------------------
# Core computations
# -----------------------------------------------------------


def compute_ecliptic_lat(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ecl = coords.geocentrictrueecliptic
    return ecl.lat.deg


def build_weight_correction(beta_deg: np.ndarray, base_weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    abs_beta = np.abs(beta_deg)
    bins = np.linspace(0, 90, 19)
    counts, edges = np.histogram(abs_beta, bins=bins, weights=base_weights)
    med = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 1.0
    med = med if med > 0 else 1.0
    factors = med / (counts + 1e-6)
    factors = np.clip(factors, 0.2, 5.0)
    bin_idx = np.digitize(abs_beta, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(factors) - 1)
    weights = base_weights * factors[bin_idx]
    info = {
        "bins_deg": bins.tolist(),
        "counts": counts.tolist(),
        "factors": factors.tolist(),
        "median_count": med,
        "notes": "Weights flatten |beta| distribution via median-normalized bin factors",
    }
    return weights, info


def apply_ecliptic_correction(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    beta_deg: np.ndarray,
    base_weights: np.ndarray,
    model: str,
) -> Tuple[Tuple[float, float, float, np.ndarray], EclipticCorrection]:
    if not model or model == "none":
        amp, l_d, b_d, vec = compute_dipole(l_deg, b_deg, base_weights)
        return (amp, l_d, b_d, vec), EclipticCorrection(False, "none", {}, base_weights)

    model = model.lower()
    if model == "weight":
        weights, details = build_weight_correction(beta_deg, base_weights)
        amp, l_d, b_d, vec = compute_dipole(l_deg, b_deg, weights)
        details.update({"applied_model": "weight"})
        return (amp, l_d, b_d, vec), EclipticCorrection(True, "weight", details, weights)

    # Default to regression-style template subtraction
    unit = lb_to_unitvec(l_deg, b_deg)
    weights = base_weights
    S = np.sum(unit * weights[:, None], axis=0)
    template_coeff = np.abs(np.sin(np.deg2rad(beta_deg)))
    T = np.sum(unit * (weights * template_coeff)[:, None], axis=0)
    denom = float(np.dot(T, T)) + 1e-12
    a = float(np.dot(S, T) / denom) if denom > 0 else 0.0
    corrected = S - a * T
    amp = 3.0 * float(np.linalg.norm(corrected)) / float(np.sum(weights))
    l_d, b_d = unitvec_to_lb(corrected)
    details = {
        "template_norm": float(np.linalg.norm(T)),
        "template_coeff": "|sin(beta)|",
        "regression_coeff": a,
        "applied_model": "regress",
    }
    return (amp, l_d, b_d, corrected), EclipticCorrection(True, "regress", details, weights)


def sky_coverage(l_deg: np.ndarray, b_deg: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "N_total": int(len(l_deg)),
        "N_weighted": float(weights.sum()),
        "N_gal_north": int(np.sum(b_deg > 0)),
        "N_gal_south": int(np.sum(b_deg < 0)),
    }
    if summary["N_gal_south"] > 0:
        summary["hemisphere_ratio"] = float(summary["N_gal_north"] / summary["N_gal_south"])
    else:
        summary["hemisphere_ratio"] = float("inf")

    notes: List[str] = []
    summary["notes"] = notes

    if HEALPY_AVAILABLE:
        for nside in (16, 32):
            npix = hp.nside2npix(nside)
            theta = np.deg2rad(90.0 - b_deg)
            phi = np.deg2rad(l_deg)
            pix = hp.ang2pix(nside, theta, phi)
            counts = np.bincount(pix, weights=weights, minlength=npix)
            occupancy = float(np.sum(counts > 0) / npix)
            key = f"healpix_nside_{nside}"
            summary[key] = {
                "occupancy_fraction": occupancy,
                "mean_counts": float(np.mean(counts)),
                "std_counts": float(np.std(counts)),
            }
    else:
        notes.append("healpy not available; using hemisphere stats only")

    if summary["hemisphere_ratio"] != float("inf") and summary["hemisphere_ratio"] > 2.0:
        notes.append("Highly asymmetric Galactic hemisphere coverage")
    return summary


# -----------------------------------------------------------
# NVSS removal
# -----------------------------------------------------------


def nvss_stub() -> None:
    msg = (
        "NVSS removal requested but no catalog provided. "
        "Download the NVSS catalog (e.g., from https://www.cv.nrao.edu/nvss/NVSSlist.html) "
        "and supply it via --nvss-catalog. Aborting without modifying the catalog."
    )
    raise SystemExit(msg)


def load_nvss_catalog(path: Path) -> SkyCoord:
    tbl = Table.read(path)
    ra_col = None
    dec_col = None
    for cand in ("ra", "RA", "RAJ2000"):
        if cand in tbl.colnames:
            ra_col = cand
            break
    for cand in ("dec", "DEC", "DEJ2000"):
        if cand in tbl.colnames:
            dec_col = cand
            break
    if ra_col is None or dec_col is None:
        raise SystemExit("NVSS catalog must contain RA/Dec columns")
    ra = np.asarray(tbl[ra_col]).astype(float)
    dec = np.asarray(tbl[dec_col]).astype(float)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def apply_nvss_removal(
    cat_coords: SkyCoord,
    nvss_catalog: Optional[Path],
    radius_arcsec: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if nvss_catalog is None:
        nvss_stub()

    if not nvss_catalog.exists():
        raise SystemExit(f"NVSS catalog not found: {nvss_catalog}")

    nvss_coords = load_nvss_catalog(nvss_catalog)
    idx, sep2d, _ = cat_coords.match_to_catalog_sky(nvss_coords)
    mask_match = sep2d <= radius_arcsec * u.arcsec
    sep_arcsec = sep2d[mask_match].arcsec

    stats = {
        "match_radius_arcsec": radius_arcsec,
        "N_matched": int(np.sum(mask_match)),
        "N_catalog": len(cat_coords),
        "N_removed": int(np.sum(mask_match)),
        "separation_hist": {
            "bins_arcsec": [0, 5, 10, 20, 30, 45, 60],
            "counts": np.histogram(sep_arcsec, bins=[0, 5, 10, 20, 30, 45, 60])[0].tolist(),
        },
    }
    keep_mask = ~mask_match
    return keep_mask, stats


# -----------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=str,
                        default="data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits",
                        help="Path to CatWISE AGN catalog (FITS)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Run tag for outputs (default: UTC timestamp)")
    parser.add_argument("--w1-max", type=float, default=16.4, help="Maximum W1 magnitude")
    parser.add_argument("--bmin", type=float, default=30.0, help="Minimum |b| cut")
    parser.add_argument("--w1cov-min", type=float, default=80.0, help="Minimum W1 coverage")
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--apply-nvss-removal", action="store_true", help="Remove NVSS/SUMSS matches")
    parser.add_argument("--nvss-catalog", type=str, default=None, help="Path to NVSS catalog (FITS/CSV)")
    parser.add_argument("--nvss-tap", type=str, default=None,
                        help="(Optional) TAP query placeholder; not implemented for large downloads")
    parser.add_argument("--match-radius-arcsec", type=float, default=45.0, help="Match radius for NVSS")
    parser.add_argument("--apply-ecliptic-correction", action="store_true",
                        help="Apply ecliptic latitude correction")
    parser.add_argument("--ecl-model", type=str, default="regress", choices=["none", "regress", "weight"],
                        help="Ecliptic correction model")
    parser.add_argument("--save-bootstrap", action="store_true", help="Write bootstrap samples CSV.gz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    run_tag = args.run_tag or _now_tag()
    base_out = Path("results/secrest_reproduction") / run_tag
    base_out.mkdir(parents=True, exist_ok=True)

    cuts = {
        "w1_max": args.w1_max,
        "bmin": args.bmin,
        "w1cov_min": args.w1cov_min,
    }

    print(f"Loading catalog: {args.catalog}")
    tbl = Table.read(args.catalog)
    N_raw = len(tbl)

    # Extract coordinates
    if "l" in tbl.colnames and "b" in tbl.colnames:
        l_all = np.asarray(tbl["l"], dtype=float)
        b_all = np.asarray(tbl["b"], dtype=float)
        ra_all = np.asarray(tbl["ra"] if "ra" in tbl.colnames else tbl["RA"])
        dec_all = np.asarray(tbl["dec"] if "dec" in tbl.colnames else tbl.get("DEC", np.zeros_like(l_all)))
    elif "glon" in tbl.colnames and "glat" in tbl.colnames:
        l_all = np.asarray(tbl["glon"], dtype=float)
        b_all = np.asarray(tbl["glat"], dtype=float)
        ra_all = np.asarray(tbl["ra"])
        dec_all = np.asarray(tbl["dec"])
    elif "ra" in tbl.colnames and "dec" in tbl.colnames:
        ra_all = np.asarray(tbl["ra"], dtype=float)
        dec_all = np.asarray(tbl["dec"], dtype=float)
        coords = SkyCoord(ra=ra_all * u.deg, dec=dec_all * u.deg, frame="icrs")
        gal = coords.galactic
        l_all = gal.l.deg
        b_all = gal.b.deg
    else:
        raise SystemExit("Catalog must provide l/b or ra/dec columns")

    w1 = np.asarray(tbl["w1"], dtype=float) if "w1" in tbl.colnames else None
    w1cov = np.asarray(tbl["w1cov"], dtype=float) if "w1cov" in tbl.colnames else None

    mask = np.ones(N_raw, dtype=bool)
    if w1 is not None:
        mask &= w1 <= args.w1_max
    mask &= np.abs(b_all) >= args.bmin
    if w1cov is not None:
        mask &= w1cov >= args.w1cov_min

    if args.apply_nvss_removal:
        nvss_path = Path(args.nvss_catalog) if args.nvss_catalog else None
        cat_coords = SkyCoord(ra=ra_all * u.deg, dec=dec_all * u.deg, frame="icrs")
        keep, nvss_stats = apply_nvss_removal(cat_coords, nvss_path, args.match_radius_arcsec)
        mask &= keep
        with open(base_out / "nvss_match_stats.json", "w", encoding="utf-8") as f:
            json.dump(nvss_stats, f, indent=2)
        print(f"NVSS removal: removed {nvss_stats['N_removed']} matches")
        nvss_removed = True
    else:
        nvss_removed = False
        nvss_stats = None

    l = l_all[mask]
    b = b_all[mask]
    ra = ra_all[mask]
    dec = dec_all[mask]
    beta = compute_ecliptic_lat(ra, dec)

    weights = np.ones_like(l, dtype=float)

    (D, l_dip, b_dip, sum_vec), ecl = apply_ecliptic_correction(
        l, b, beta, weights, args.ecl_model if args.apply_ecliptic_correction else "none"
    )

    def corrected_dipole(lv: np.ndarray, bv: np.ndarray, betav: np.ndarray, wv: np.ndarray):
        return apply_ecliptic_correction(lv, bv, betav, wv, ecl.model if ecl.applied else "none")[0]

    boot = bootstrap_dipole(l, b, beta, ecl.weights, args.bootstrap, rng, corrected_dipole)

    coverage = sky_coverage(l, b, ecl.weights)

    N_after = int(len(l))
    sigma_amp_used = float(0.5 * (boot["D_p84"] - boot["D_p16"]))
    sigma_dir_used = float(boot["dir_sigma_deg"])

    notes: List[str] = []
    if not HEALPY_AVAILABLE:
        notes.append("healpy not installed; coverage limited to hemispheric diagnostics")
    if coverage.get("hemisphere_ratio", 1.0) != float("inf") and coverage.get("hemisphere_ratio", 0) > 2.0:
        notes.append("Coverage asymmetric across Galactic hemispheres")

    systematics = {
        "nvss_removed": nvss_removed,
        "ecliptic_corrected": ecl.applied,
        "ecliptic_model": ecl.model,
        "notes": notes,
    }
    if nvss_stats:
        systematics["nvss_stats_file"] = str(base_out / "nvss_match_stats.json")
    if ecl.applied:
        ecl_meta = ecl.details
        ecl_meta["weights_mean"] = float(np.mean(ecl.weights))
        ecl_meta["weights_std"] = float(np.std(ecl.weights))
        with open(base_out / "ecliptic_regression.json", "w", encoding="utf-8") as f:
            json.dump(ecl_meta, f, indent=2)

    constraints = {
        "catalog": os.path.abspath(args.catalog),
        "cuts": cuts,
        "N_raw": int(N_raw),
        "N_after": N_after,
        "dipole": {"D": float(D), "l_deg": float(l_dip), "b_deg": float(b_dip)},
        "bootstrap": {
            "D_p16": boot["D_p16"],
            "D_p50": boot["D_p50"],
            "D_p84": boot["D_p84"],
            "dir_sigma_deg": boot["dir_sigma_deg"],
        },
        "systematics": systematics,
        "sigma_dir_deg_used": sigma_dir_used,
        "sigma_amp_used": sigma_amp_used,
    }

    constraints_path = base_out / "dipole_constraints.json"
    with open(constraints_path, "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2)
    print(f"Saved constraints to {constraints_path}")

    with open(base_out / "sky_coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    if args.save_bootstrap:
        boot["samples"].to_csv(base_out / "dipole_bootstrap_samples.csv.gz", index=False, compression="gzip")

    print(
        f"Dipole D={D:.5f}, l={l_dip:.2f} deg, b={b_dip:.2f} deg | "
        f"sigma_amp≈{sigma_amp_used:.5f}, sigma_dir≈{sigma_dir_used:.2f} deg"
    )


if __name__ == "__main__":
    main()
