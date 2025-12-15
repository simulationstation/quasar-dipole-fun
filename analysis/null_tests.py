"""Null simulations for Stage 6 multipole-aware diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import healpy as hp
import numpy as np
from astropy.table import Table

from analysis.low_ell_axes import compute_low_ell_axes_from_arrays, load_maps
from quasar_dipole.ell_space import build_counts_map, compute_overdensity


def _load_catalog(catalog_path: Path) -> Table:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    return Table.read(catalog_path, memmap=True)


def _filter_unmasked(ra: np.ndarray, dec: np.ndarray, mask: np.ndarray, nside: int) -> np.ndarray:
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra % 360.0)
    pix = hp.ang2pix(nside, theta, phi)
    return ~mask[pix]


def _compute_axes_from_counts(counts: np.ndarray, mask: np.ndarray) -> List[Dict[str, object]]:
    delta, _ = compute_overdensity(counts, mask)
    return compute_low_ell_axes_from_arrays(delta, mask)


def _ra_scramble(
    ra: np.ndarray, dec: np.ndarray, base_mask: np.ndarray, nside: int, rng: np.random.Generator
) -> np.ndarray:
    shuffled_ra = rng.permutation(ra)
    return build_counts_map(shuffled_ra[base_mask], dec[base_mask], nside)


def _isotropic_catalog(
    n: int, mask: np.ndarray, nside: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    # Vectorized rejection sampling - generate in batches
    f_sky = float(np.sum(~mask)) / len(mask)
    # Overgenerate by 1/f_sky + 20% buffer
    batch_size = int(n / f_sky * 1.2) + 1000

    ra_list = []
    dec_list = []

    while len(ra_list) < n:
        # Generate batch of random points on sphere
        theta = np.arccos(1 - 2 * rng.random(batch_size))
        phi = rng.uniform(0.0, 2 * np.pi, batch_size)
        pix = hp.ang2pix(nside, theta, phi)

        # Keep only unmasked
        keep = ~mask[pix]
        ra_list.extend(np.rad2deg(phi[keep]).tolist())
        dec_list.extend((90.0 - np.rad2deg(theta[keep])).tolist())

    return np.asarray(ra_list[:n]), np.asarray(dec_list[:n])


def run_null_tests(
    delta_path: Path,
    mask_path: Path,
    catalog_path: Path,
    output_path: Path,
    n_draws: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    delta_map, mask = load_maps(delta_path, mask_path)
    nside = hp.get_nside(delta_map)

    catalog = _load_catalog(catalog_path)
    if "ra" not in catalog.colnames or "dec" not in catalog.colnames:
        raise ValueError("Catalog must provide 'ra' and 'dec' columns for null tests")

    ra = np.asarray(catalog["ra"])
    dec = np.asarray(catalog["dec"])
    base_mask = _filter_unmasked(ra, dec, mask, nside)
    n_sources = base_mask.sum()

    rng = np.random.default_rng(seed)
    observed_axes = compute_low_ell_axes_from_arrays(delta_map, mask)

    def summarize(samples: List[float]) -> Dict[str, float]:
        arr = np.asarray(samples)
        return {
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }

    distributions: Dict[str, Dict[str, List[float]]] = {
        "ra_scramble": {"C1": [], "C2": [], "C3": [], "coherence": []},
        "isotropic": {"C1": [], "C2": [], "C3": [], "coherence": []},
    }

    import sys
    for i in range(n_draws):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Null test {i + 1}/{n_draws}", file=sys.stderr, flush=True)

        # RA scramble null
        ra_counts = _ra_scramble(ra, dec, base_mask, nside, rng)
        ra_axes = _compute_axes_from_counts(ra_counts, mask)
        ra_coherence = _coherence_from_axes(ra_axes)
        _append_distributions(distributions["ra_scramble"], ra_axes, ra_coherence)

        # Isotropic Poisson null
        iso_ra, iso_dec = _isotropic_catalog(n_sources, mask, nside, rng)
        iso_counts = build_counts_map(iso_ra, iso_dec, nside)
        iso_axes = _compute_axes_from_counts(iso_counts, mask)
        iso_coherence = _coherence_from_axes(iso_axes)
        _append_distributions(distributions["isotropic"], iso_axes, iso_coherence)

    observed_coherence = _coherence_from_axes(observed_axes)

    payload = {
        "observed": {"axes": observed_axes, "coherence": observed_coherence},
        "distributions": {k: {key: val for key, val in v.items()} for k, v in distributions.items()},
    }

    for null_name, data in distributions.items():
        payload[null_name + "_summary"] = {
            "C1": summarize(data["C1"]),
            "C2": summarize(data["C2"]),
            "C3": summarize(data["C3"]),
            "coherence": summarize(data["coherence"]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def _coherence_from_axes(axes: List[Dict[str, object]]) -> float:
    from analysis.axis_coherence import compute_axis_coherence

    tmp_path = Path("/tmp/stage6_coherence.json")
    result = compute_axis_coherence(axes, tmp_path)
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass
    return float(result["coherence_deg"])


def _append_distributions(store: Dict[str, List[float]], axes: List[Dict[str, object]], coherence: float) -> None:
    axes_by_ell = {int(entry["ell"]): entry for entry in axes}
    for ell_key, label in [(1, "C1"), (2, "C2"), (3, "C3")]:
        store[label].append(float(axes_by_ell[ell_key]["C_ell"]))
    store["coherence"].append(float(coherence))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=Path("results/stage5/delta.fits"))
    parser.add_argument("--mask", type=Path, default=Path("results/stage5/mask.fits"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage6/null_distributions.json"),
    )
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_null_tests(args.delta, args.mask, args.catalog, args.output, args.n_draws, args.seed)


if __name__ == "__main__":
    main()
