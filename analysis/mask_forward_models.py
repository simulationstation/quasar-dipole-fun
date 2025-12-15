"""Mask-forward physical simulations contrasting kinematic and geometric models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import healpy as hp
import numpy as np

from analysis.low_ell_axes import compute_low_ell_axes_from_arrays, load_maps
from quasar_dipole.ell_space import build_counts_map, compute_overdensity


CMB_AXIS = {"l": 264.021, "b": 48.253}


def _unit_vector_from_lb(l_deg: float, b_deg: float) -> np.ndarray:
    theta = np.deg2rad(90.0 - b_deg)
    phi = np.deg2rad(l_deg)
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _sample_direction(mask: np.ndarray, nside: int, rng: np.random.Generator) -> tuple[float, float]:
    while True:
        theta = np.arccos(1 - 2 * rng.random())
        phi = rng.uniform(0.0, 2 * np.pi)
        pix = hp.ang2pix(nside, theta, phi)
        if not mask[pix]:
            return theta, phi


def _modulated_catalog(
    n: int,
    mask: np.ndarray,
    nside: int,
    rng: np.random.Generator,
    axis: Dict[str, float],
    dipole_amp: float = 0.0,
    quad_amp: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    axis_vec = _unit_vector_from_lb(axis["l"], axis["b"])
    ra_list: List[float] = []
    dec_list: List[float] = []

    # Ensure acceptance probabilities are positive
    norm = 1.0 + abs(dipole_amp) + abs(quad_amp)

    while len(ra_list) < n:
        theta, phi = _sample_direction(mask, nside, rng)
        direction = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        cos_gamma = float(np.dot(direction, axis_vec))
        p = 1.0 + dipole_amp * cos_gamma + quad_amp * 0.5 * (3 * cos_gamma**2 - 1)
        p = p / norm
        if p < 0:
            p = 0.0
        if rng.random() <= p:
            ra_list.append(np.rad2deg(phi))
            dec_list.append(90.0 - np.rad2deg(theta))

    return np.asarray(ra_list), np.asarray(dec_list)


def _power_and_axes_from_catalog(
    ra: np.ndarray, dec: np.ndarray, mask: np.ndarray, nside: int
) -> Dict[str, object]:
    counts = build_counts_map(ra, dec, nside)
    delta, _ = compute_overdensity(counts, mask)
    axes = compute_low_ell_axes_from_arrays(delta, mask)
    return {"axes": axes}


def simulate_models(
    delta_path: Path,
    mask_path: Path,
    catalog_path: Path,
    output_path: Path,
    seed: int = 42,
    dipole_amp: float = 0.00336,
    quad_fraction: float = 0.5,
) -> Dict[str, object]:
    delta_map, mask = load_maps(delta_path, mask_path)
    nside = hp.get_nside(delta_map)

    from analysis.null_tests import _load_catalog, _filter_unmasked

    catalog = _load_catalog(catalog_path)
    ra = np.asarray(catalog["ra"])
    dec = np.asarray(catalog["dec"])
    base_mask = _filter_unmasked(ra, dec, mask, nside)
    n_sources = int(base_mask.sum())

    rng = np.random.default_rng(seed)

    cmb_axis = {"l": CMB_AXIS["l"], "b": CMB_AXIS["b"]}
    model_results: Dict[str, object] = {}

    # Model 1: pure kinematic dipole
    kin_ra, kin_dec = _modulated_catalog(
        n_sources, mask, nside, rng, cmb_axis, dipole_amp=dipole_amp, quad_amp=0.0
    )
    kin_payload = _power_and_axes_from_catalog(kin_ra, kin_dec, mask, nside)
    kin_payload.update({"axis": cmb_axis, "dipole_amp": dipole_amp, "quad_amp": 0.0})
    model_results["kinematic_dipole"] = kin_payload

    # Model 2: geometric wall (dipole + quadrupole aligned)
    wall_quad = dipole_amp * quad_fraction
    wall_ra, wall_dec = _modulated_catalog(
        n_sources,
        mask,
        nside,
        rng,
        cmb_axis,
        dipole_amp=dipole_amp,
        quad_amp=wall_quad,
    )
    wall_payload = _power_and_axes_from_catalog(wall_ra, wall_dec, mask, nside)
    wall_payload.update(
        {
            "axis": cmb_axis,
            "dipole_amp": dipole_amp,
            "quad_amp": wall_quad,
            "quad_to_dipole_ratio": quad_fraction,
        }
    )
    model_results["geometric_wall"] = wall_payload

    output = {
        "n_sources": n_sources,
        "cmb_axis": cmb_axis,
        "model_results": model_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=Path("results/stage5/delta.fits"))
    parser.add_argument("--mask", type=Path, default=Path("results/stage5/mask.fits"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/stage6/model_forward.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dipole-amp", type=float, default=0.00336)
    parser.add_argument("--quad-fraction", type=float, default=0.5)
    args = parser.parse_args()

    simulate_models(
        args.delta,
        args.mask,
        args.catalog,
        args.output,
        seed=args.seed,
        dipole_amp=args.dipole_amp,
        quad_fraction=args.quad_fraction,
    )


if __name__ == "__main__":
    main()
