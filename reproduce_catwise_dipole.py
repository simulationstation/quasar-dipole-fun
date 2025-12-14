#!/usr/bin/env python3
"""Reproduce the CatWISE quasar dipole from a local catalog file.

This script implements a streaming-friendly, end-to-end pipeline that:
  * loads a local CatWISE catalog (FITS recommended; CSV/Parquet supported)
  * applies configurable photometric and quality cuts
  * masks the Galactic plane using |b| >= bmin
  * estimates the number-count dipole with the classic estimator
        D = 3 |sum_i w_i n_i| / sum_i w_i
    where n_i is the unit vector toward source *i* and w_i is the Poisson
    bootstrap weight (unity for the point estimate)
  * derives uncertainties via a Poisson bootstrap that works chunk-by-chunk
  * performs basic diagnostics/null tests
  * writes constraints consumable by ``metropolis_hastings_sampler.py``

Example usage::

    python reproduce_catwise_dipole.py \
      --input /path/to/catwise.fits \
      --outdir results/catwise \
      --ra-col ra --dec-col dec \
      --w1-col w1mpro --w1-min 14.0 --w1-max 17.0 \
      --qso-prob-col qso_prob --qso-prob-min 0.9 \
      --mask-gal-b-min 20 \
      --bootstrap 500 \
      --seed 42 \
      --chunk-size 200000

The output directory will contain:
  * ``catwise_dipole.json``      : point estimate, cuts, bootstrap summaries
  * ``catwise_diagnostics.json`` : hemisphere split + longitude randomization
  * ``constraints_catwise.json`` : sampler-ready constraints payload
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    print(
        "ERROR: astropy is required for coordinate transforms. "
        "Install it with `pip install astropy` and retry.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

# CMB dipole direction (galactic) reused for separation diagnostics
from metropolis_hastings_sampler import B_CMB_DEG, L_CMB_DEG  # noqa: E402


def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    return np.array([
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    ])


def unitvec_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = vec
    lon = math.degrees(math.atan2(y, x)) % 360.0
    lat = math.degrees(math.atan2(z, math.hypot(x, y)))
    return lon, lat


def angle_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = float(np.dot(vec1, vec2))
    dot = max(min(dot, 1.0), -1.0)
    return math.degrees(math.acos(dot))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce CatWISE dipole from a local catalog")

    parser.add_argument("--input", required=True, type=Path, help="Path to local CatWISE catalog (FITS/CSV/Parquet)")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for dipole products")

    parser.add_argument("--ra-col", default="ra", help="Right ascension column name (degrees)")
    parser.add_argument("--dec-col", default="dec", help="Declination column name (degrees)")
    parser.add_argument("--w1-col", default="w1mpro", help="W1 magnitude column name")
    parser.add_argument("--w1-min", type=float, default=None, help="Minimum W1 magnitude (inclusive)")
    parser.add_argument("--w1-max", type=float, default=None, help="Maximum W1 magnitude (inclusive)")

    parser.add_argument("--qso-prob-col", default="qso_prob", help="QSO probability column name")
    parser.add_argument("--qso-prob-min", type=float, default=0.9, help="Minimum QSO probability")
    parser.add_argument("--no-qso-prob-cut", action="store_true", help="Skip QSO probability cut")

    parser.add_argument("--quality-flag-col", default=None, help="Optional quality flag column name")
    parser.add_argument(
        "--quality-flag-allowed",
        type=lambda s: [int(x) for x in s.split(",") if x.strip()],
        default=None,
        help="Comma-separated list of allowed quality flag values",
    )

    parser.add_argument("--mask-gal-b-min", type=float, default=20.0, help="Mask |b| < bmin (degrees)")

    parser.add_argument("--bootstrap", type=int, default=500, help="Number of Poisson bootstrap replicates")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Rows to process per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap and null tests")

    parser.add_argument("--randomization-trials", type=int, default=100, help="Longitude randomization trials")

    return parser.parse_args()


def load_fits_chunks(path: Path, columns: List[str], chunk_size: int) -> Iterator[Dict[str, np.ndarray]]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data  # type: ignore[index]
        n_rows = len(data)
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            subset = {col: np.asarray(data[col][start:stop]) for col in columns}
            yield subset


def load_csv_chunks(path: Path, columns: List[str], chunk_size: int) -> Iterator[Dict[str, np.ndarray]]:
    import pandas as pd

    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
        yield {col: chunk[col].to_numpy() for col in columns}


def load_parquet_chunks(path: Path, columns: List[str], chunk_size: int) -> Iterator[Dict[str, np.ndarray]]:
    import pandas as pd

    df = pd.read_parquet(path, columns=columns)
    for start in range(0, len(df), chunk_size):
        stop = min(start + chunk_size, len(df))
        slice_df = df.iloc[start:stop]
        yield {col: slice_df[col].to_numpy() for col in columns}


def iter_catalog(path: Path, columns: List[str], chunk_size: int) -> Iterator[Dict[str, np.ndarray]]:
    lower = path.suffix.lower()
    if lower in {".fits", ".fit", ".fits.gz"}:
        yield from load_fits_chunks(path, columns, chunk_size)
    elif lower == ".csv":
        yield from load_csv_chunks(path, columns, chunk_size)
    elif lower in {".parquet", ".pq"}:
        yield from load_parquet_chunks(path, columns, chunk_size)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def apply_cuts(
    data: Dict[str, np.ndarray],
    args: argparse.Namespace,
    warnings: List[str],
) -> np.ndarray:
    mask = np.ones(len(next(iter(data.values()))), dtype=bool)

    if args.w1_min is not None:
        mask &= data[args.w1_col] >= args.w1_min
    if args.w1_max is not None:
        mask &= data[args.w1_col] <= args.w1_max

    if not args.no_qso_prob_cut:
        if args.qso_prob_col not in data:
            warnings.append(
                f"Requested QSO probability cut but column '{args.qso_prob_col}' missing; skipping cut."
            )
        else:
            mask &= data[args.qso_prob_col] >= args.qso_prob_min

    if args.quality_flag_col:
        if args.quality_flag_col not in data:
            warnings.append(
                f"Quality flag column '{args.quality_flag_col}' missing; skipping quality cut."
            )
        elif args.quality_flag_allowed:
            allowed = np.array(args.quality_flag_allowed)
            mask &= np.isin(data[args.quality_flag_col], allowed)

    return mask


def compute_dipole(sum_vec: np.ndarray, weight_sum: float) -> Tuple[float, float, float]:
    norm = float(np.linalg.norm(sum_vec))
    amplitude = float(3.0 * norm / weight_sum) if weight_sum > 0 else float("nan")
    lon, lat = unitvec_to_lb(sum_vec)
    return amplitude, lon, lat


def bootstrap_summary(D_boot: np.ndarray, v_boot: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(D_boot)
    if not np.any(finite):
        return {k: float("nan") for k in ["median", "q16", "q84", "sigma_dir_deg"]}

    D_sel = D_boot[finite]
    v_sel = v_boot[finite]

    median = float(np.median(D_sel))
    q16, q84 = np.percentile(D_sel, [16, 84])

    vec_norms = np.linalg.norm(v_sel, axis=1)
    nonzero = vec_norms > 0
    normed_vecs = v_sel[nonzero] / vec_norms[nonzero, None]

    median_dir = normed_vecs.mean(axis=0)
    if np.linalg.norm(median_dir) > 0:
        median_dir /= np.linalg.norm(median_dir)
    offsets = np.array([angle_deg(median_dir, v) for v in normed_vecs])
    sigma_dir = float(np.percentile(offsets, 68)) if offsets.size else float("nan")

    return {
        "median": median,
        "q16": float(q16),
        "q84": float(q84),
        "sigma_dir_deg": sigma_dir,
    }


def randomize_longitudes(lon_deg: np.ndarray, lat_deg: np.ndarray, rng: np.random.Generator, trials: int) -> np.ndarray:
    amps = []
    for _ in range(trials):
        rand_lon = rng.uniform(0.0, 360.0, size=len(lon_deg))
        vecs = sph_to_cart(rand_lon, lat_deg)
        v_sum = vecs.sum(axis=0)
        amps.append(3.0 * np.linalg.norm(v_sum) / len(vecs))
    return np.asarray(amps)


def sph_to_cart(lon_deg: Iterable[float], lat_deg: Iterable[float]) -> np.ndarray:
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    columns = [args.ra_col, args.dec_col, args.w1_col]
    if not args.no_qso_prob_cut:
        columns.append(args.qso_prob_col)
    if args.quality_flag_col:
        columns.append(args.quality_flag_col)

    warnings: List[str] = []

    args.outdir.mkdir(parents=True, exist_ok=True)

    N_raw = 0
    N_after_cuts = 0
    N_after_mask = 0

    sum_vec = np.zeros(3, dtype=float)
    weight_sum = 0.0

    north_vec = np.zeros(3, dtype=float)
    south_vec = np.zeros(3, dtype=float)
    north_count = 0
    south_count = 0

    lat_list: List[float] = []
    lon_list: List[float] = []

    n_boot = args.bootstrap
    v_boot = np.zeros((n_boot, 3), dtype=float)
    w_boot = np.zeros(n_boot, dtype=float)

    for chunk in iter_catalog(args.input, columns, args.chunk_size):
        N_raw += len(next(iter(chunk.values())))

        mask = apply_cuts(chunk, args, warnings)
        if not mask.any():
            continue

        N_after_cuts += int(mask.sum())

        ra = np.asarray(chunk[args.ra_col][mask], dtype=float)
        dec = np.asarray(chunk[args.dec_col][mask], dtype=float)

        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        lon = coords.l.deg
        lat = coords.b.deg

        gal_mask = np.abs(lat) >= args.mask_gal_b_min
        if not gal_mask.any():
            continue

        lon = lon[gal_mask]
        lat = lat[gal_mask]

        N_after_mask += len(lon)

        lon_list.extend(lon.tolist())
        lat_list.extend(lat.tolist())

        vecs = sph_to_cart(lon, lat)
        sum_vec += vecs.sum(axis=0)
        weight_sum += len(vecs)

        north_mask = lat >= 0
        south_mask = ~north_mask
        if north_mask.any():
            north_vec += vecs[north_mask].sum(axis=0)
            north_count += int(north_mask.sum())
        if south_mask.any():
            south_vec += vecs[south_mask].sum(axis=0)
            south_count += int(south_mask.sum())

        if n_boot > 0:
            for i in range(n_boot):
                weights = rng.poisson(1.0, size=len(vecs))
                if weights.sum() == 0:
                    continue
                v_boot[i] += (weights[:, None] * vecs).sum(axis=0)
                w_boot[i] += float(weights.sum())

    lon_arr = np.asarray(lon_list, dtype=float)
    lat_arr = np.asarray(lat_list, dtype=float)

    D_amp, D_lon, D_lat = compute_dipole(sum_vec, weight_sum)
    D_vec_unit = sum_vec / np.linalg.norm(sum_vec) if np.linalg.norm(sum_vec) > 0 else sum_vec
    cmb_vec = lb_to_unitvec(L_CMB_DEG, B_CMB_DEG)
    cmb_sep = angle_deg(D_vec_unit, cmb_vec) if np.linalg.norm(D_vec_unit) > 0 else float("nan")

    D_boot = np.where(w_boot > 0, 3.0 * np.linalg.norm(v_boot, axis=1) / w_boot, np.nan)
    boot_summary = bootstrap_summary(D_boot, v_boot)

    north_amp, north_lon, north_lat = compute_dipole(north_vec, north_count)
    south_amp, south_lon, south_lat = compute_dipole(south_vec, south_count)
    ns_sep = angle_deg(
        north_vec / np.linalg.norm(north_vec),
        south_vec / np.linalg.norm(south_vec),
    ) if north_count > 0 and south_count > 0 else float("nan")

    rand_amps = randomize_longitudes(lon_arr, lat_arr, rng, args.randomization_trials) if len(lon_arr) else np.array([])
    p_value = float(np.mean(rand_amps >= D_amp)) if rand_amps.size else float("nan")

    dipole_payload = {
        "input": str(args.input),
        "counts": {
            "raw": N_raw,
            "after_cuts": N_after_cuts,
            "after_mask": N_after_mask,
        },
        "cuts": {
            "w1_min": args.w1_min,
            "w1_max": args.w1_max,
            "qso_prob_min": None if args.no_qso_prob_cut else args.qso_prob_min,
            "mask_gal_b_min": args.mask_gal_b_min,
            "quality_flag_col": args.quality_flag_col,
            "quality_flag_allowed": args.quality_flag_allowed,
        },
        "dipole": {
            "amplitude": D_amp,
            "lon_deg": D_lon,
            "lat_deg": D_lat,
            "cmb_separation_deg": cmb_sep,
            "estimator": "D = 3 |sum_i w_i n_i| / sum_i w_i",
        },
        "bootstrap": {
            "n": n_boot,
            "median": boot_summary.get("median"),
            "q16": boot_summary.get("q16"),
            "q84": boot_summary.get("q84"),
            "sigma_dir_deg": boot_summary.get("sigma_dir_deg"),
        },
        "randomization_trials": args.randomization_trials,
        "seed": args.seed,
    }

    diagnostics = {
        "hemisphere_split": {
            "north": {"amplitude": north_amp, "lon_deg": north_lon, "lat_deg": north_lat, "N": north_count},
            "south": {"amplitude": south_amp, "lon_deg": south_lon, "lat_deg": south_lat, "N": south_count},
            "separation_deg": ns_sep,
        },
        "longitude_randomization": {
            "trials": args.randomization_trials,
            "p_value": p_value,
            "mean_null_amplitude": float(rand_amps.mean()) if rand_amps.size else float("nan"),
        },
    }

    sigma_amp = None
    if all(np.isfinite([boot_summary.get("q16", np.nan), boot_summary.get("q84", np.nan)])):
        sigma_amp = 0.5 * (boot_summary["q84"] - boot_summary["q16"])

    constraints = {
        "catwise": {
            "D_QSO_OBS": D_amp,
            "L_QSO_OBS_DEG": D_lon,
            "B_QSO_OBS_DEG": D_lat,
            "SIGMA_QSO": sigma_amp,
            "SIGMA_QSO_DIR_DEG": boot_summary.get("sigma_dir_deg"),
        }
    }

    with open(args.outdir / "catwise_dipole.json", "w", encoding="utf-8") as f:
        json.dump(dipole_payload, f, indent=2)

    with open(args.outdir / "catwise_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    with open(args.outdir / "constraints_catwise.json", "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2)

    print("\n=== CatWISE dipole summary ===")
    print(f"Input catalog   : {args.input}")
    print(f"Output directory: {args.outdir}")
    print(f"Counts          : raw={N_raw}, after cuts={N_after_cuts}, after mask={N_after_mask}")
    print(f"Dipole          : D={D_amp:.5f}, l={D_lon:.2f} deg, b={D_lat:.2f} deg")
    print(
        "Bootstrap       : median={:.5f} (+{:.5f}/-{:.5f}), sigma_dir={:.2f} deg".format(
            boot_summary.get("median", float("nan")),
            boot_summary.get("q84", float("nan")) - boot_summary.get("median", 0.0),
            boot_summary.get("median", 0.0) - boot_summary.get("q16", float("nan")),
            boot_summary.get("sigma_dir_deg", float("nan")),
        )
    )
    print(f"CMB separation  : {cmb_sep:.2f} deg")
    print(f"Longitude null p: {p_value:.3f}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
