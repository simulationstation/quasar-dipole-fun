"""CLI to estimate a catalog dipole with mask and uncertainty controls.

This script is deliberately local-first and will refuse to run without
an on-disk catalog. No remote downloads or HEALPix dependencies are
used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd

from dipole_estimators import (
    DipoleEstimate,
    DipoleUncertainty,
    bootstrap_dipole,
    jackknife_by_hemisphere,
    jackknife_by_octant,
    randomized_null_test,
    simple_vector_dipole,
    weighted_dipole,
)
from sky_masks import (
    apply_boolean_mask,
    describe_mask,
    equatorial_to_galactic,
    galactic_latitude_mask,
    rectangular_mask,
)


def _load_catalog(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Unsupported catalog format. Use CSV or Parquet.")


def _parse_rect_masks(rects: List[str]):
    parsed = []
    for item in rects:
        try:
            lon_min, lon_max, lat_min, lat_max = map(float, item.split(","))
            parsed.append((lon_min, lon_max, lat_min, lat_max))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not parse rectangular mask '{item}'. Use lon_min,lon_max,lat_min,lat_max") from exc
    return parsed


def _prepare_coordinates(df: pd.DataFrame, args: argparse.Namespace):
    if args.lon_col and args.lat_col:
        lon = df[args.lon_col].to_numpy()
        lat = df[args.lat_col].to_numpy()
        frame = "provided lon/lat"
    elif args.ra_col and args.dec_col:
        if args.frame == "galactic":
            lon, lat = equatorial_to_galactic(df[args.ra_col], df[args.dec_col])
            frame = "galactic (derived from ra/dec)"
        else:
            lon = df[args.ra_col].to_numpy()
            lat = df[args.dec_col].to_numpy()
            frame = "equatorial ra/dec"
    else:
        raise ValueError("Either lon/lat or ra/dec columns must be provided.")
    return lon, lat, frame


def _apply_masks(df: pd.DataFrame, lon, lat, args: argparse.Namespace):
    total = len(df)
    mask = np.ones(total, dtype=bool)
    summaries = []

    if args.mask_col:
        col_mask = apply_boolean_mask(df[args.mask_col])
        mask &= col_mask
        summaries.append(describe_mask(mask.sum(), total, f"user mask column {args.mask_col}"))

    if args.mag_col:
        if args.mag_col not in df.columns:
            raise ValueError(f"Magnitude column '{args.mag_col}' missing")
        mag = df[args.mag_col].to_numpy()
        finite = np.isfinite(mag)
        mag_mask = finite
        if args.mag_min is not None:
            mag_mask &= mag >= args.mag_min
        if args.mag_max is not None:
            mag_mask &= mag <= args.mag_max
        mask &= mag_mask
        summaries.append(describe_mask(mask.sum(), total, "magnitude limits"))

    if args.b_cut is not None:
        gal_mask = galactic_latitude_mask(lon, lat, args.b_cut)
        mask &= gal_mask
        summaries.append(describe_mask(mask.sum(), total, f"|b| >= {args.b_cut} deg"))

    for lon_min, lon_max, lat_min, lat_max in _parse_rect_masks(args.mask_rect):
        rect_mask = rectangular_mask(lon, lat, lon_min, lon_max, lat_min, lat_max)
        mask &= ~rect_mask  # drop sources inside the rectangle
        summaries.append(
            describe_mask(mask.sum(), total, f"exclude rect lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}]")
        )

    return mask, summaries


def _check_coverage(lon, lat):
    msgs = []
    if len(lon) < 100:
        msgs.append("<100 sources after cuts; dipole likely noise dominated")
    # Hemisphere coverage
    if np.sum(lat >= 0) == 0 or np.sum(lat < 0) == 0:
        msgs.append("One hemisphere empty after cuts; dipole unreliable")
    # Simple octant check
    lon_wrapped = np.mod(lon, 360.0)
    edges = np.linspace(0, 360, 9)
    counts, _ = np.histogram(lon_wrapped, bins=edges)
    if (counts == 0).any():
        msgs.append("At least one longitude octant empty; consider relaxing masks")
    return msgs


def _save_output(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def estimate(args: argparse.Namespace):
    df = _load_catalog(args.catalog)
    lon, lat, frame = _prepare_coordinates(df, args)
    mask, mask_summaries = _apply_masks(df, lon, lat, args)

    if mask.sum() == 0:
        raise RuntimeError("All rows removed by masking; nothing to do")

    lon_sel = lon[mask]
    lat_sel = lat[mask]
    weights = None
    if args.weight_col:
        if args.weight_col not in df.columns:
            raise ValueError(f"Weight column '{args.weight_col}' missing")
        weights = df.loc[mask, args.weight_col].to_numpy()

    coverage_msgs = _check_coverage(lon_sel, lat_sel)

    if args.bootstrap and args.bootstrap > 0:
        dipole, unc = bootstrap_dipole(lon_sel, lat_sel, weights=weights, n_bootstrap=args.bootstrap)
    else:
        dipole = weighted_dipole(lon_sel, lat_sel, weights=weights)
        unc = DipoleUncertainty(np.nan, np.nan, np.nan, np.full((3, 3), np.nan))

    jackknife = None
    if args.jackknife:
        jackknife = {
            "hemisphere": [
                {"subset": res.subset_label, "amplitude": res.estimate.amplitude, "lon_deg": res.estimate.lon_deg, "lat_deg": res.estimate.lat_deg}
                for res in jackknife_by_hemisphere(lon_sel, lat_sel, weights)
            ],
            "octant": [
                {"subset": res.subset_label, "amplitude": res.estimate.amplitude, "lon_deg": res.estimate.lon_deg, "lat_deg": res.estimate.lat_deg}
                for res in jackknife_by_octant(lon_sel, lat_sel, weights)
            ],
        }

    null_amplitudes = None
    if args.null_tests and args.null_tests > 0:
        null_amplitudes = randomized_null_test(lon_sel, lat_sel, weights, n_realizations=args.null_tests, random_state=args.seed)
        null_summary = {
            "mean": float(np.mean(null_amplitudes)),
            "std": float(np.std(null_amplitudes, ddof=1)),
            "p_gt_data": float(np.mean(null_amplitudes >= dipole.amplitude)),
        }
    else:
        null_summary = None

    result = {
        "catalog": args.catalog,
        "frame": frame,
        "n_original": int(len(df)),
        "n_after_cuts": int(mask.sum()),
        "mask_summaries": [ms.__dict__ for ms in mask_summaries],
        "coverage_warnings": coverage_msgs,
        "dipole": {
            "amplitude": dipole.amplitude,
            "lon_deg": dipole.lon_deg,
            "lat_deg": dipole.lat_deg,
        },
        "uncertainty": {
            "amplitude_sigma": unc.amplitude_sigma,
            "lon_sigma_deg": unc.lon_sigma_deg,
            "lat_sigma_deg": unc.lat_sigma_deg,
            "covariance": unc.covariance.tolist(),
        },
        "jackknife": jackknife,
        "null_test": null_summary,
    }

    _save_output(args.output, result)

    print(json.dumps(result, indent=2))
    if coverage_msgs:
        print("Warnings: " + "; ".join(coverage_msgs), file=sys.stderr)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Path to CSV or Parquet catalog")
    parser.add_argument("--ra-col", default="ra", help="RA column (deg). Required if lon/lat not provided")
    parser.add_argument("--dec-col", default="dec", help="Dec column (deg). Required if lon/lat not provided")
    parser.add_argument("--lon-col", help="Longitude column (deg) to use directly")
    parser.add_argument("--lat-col", help="Latitude column (deg) to use directly")
    parser.add_argument("--frame", choices=["galactic", "equatorial"], default="galactic", help="Coordinate frame for dipole output")
    parser.add_argument("--mag-col", help="Magnitude column for cuts")
    parser.add_argument("--mag-min", type=float, help="Minimum magnitude (inclusive)")
    parser.add_argument("--mag-max", type=float, help="Maximum magnitude (inclusive)")
    parser.add_argument("--b-cut", type=float, help="Absolute Galactic latitude cut (deg)")
    parser.add_argument("--mask-col", help="Boolean column indicating sources to keep")
    parser.add_argument("--mask-rect", nargs="*", default=[], help="Rectangular mask(s) lon_min,lon_max,lat_min,lat_max in degrees")
    parser.add_argument("--weight-col", help="Weight column")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap realizations for uncertainties")
    parser.add_argument("--jackknife", action="store_true", help="Compute hemisphere and octant jackknife estimates")
    parser.add_argument("--null-tests", type=int, default=0, help="Number of randomized longitude null realizations")
    parser.add_argument("--seed", type=int, help="Random seed for bootstrap/null tests")
    parser.add_argument("--output", default="dipole_result.json", help="Output JSON file")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    estimate(args)
