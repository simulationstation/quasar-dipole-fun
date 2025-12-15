"""Low-ℓ axis extraction for Stage 6 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import healpy as hp
import numpy as np


def load_maps(delta_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not delta_path.exists():
        raise FileNotFoundError(f"Delta map not found: {delta_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask map not found: {mask_path}")

    delta_map = hp.read_map(delta_path, dtype=float)
    mask_map = hp.read_map(mask_path, dtype=float)
    if delta_map.shape != mask_map.shape:
        raise ValueError("Delta and mask maps must have the same size")
    return delta_map, mask_map != 0


def _dominant_axis_for_ell(
    alm: np.ndarray, ell: int, lmax: int, mask: np.ndarray, nside: int
) -> Dict[str, float]:
    # Get l values for each alm index
    l_arr, m_arr = hp.Alm.getlm(lmax)
    ell_indices = l_arr == ell
    ell_only = np.zeros_like(alm)
    ell_only[ell_indices] = alm[ell_indices]

    map_ell = hp.alm2map(ell_only, nside=nside, lmax=lmax, verbose=False)
    map_ell = np.where(mask, 0.0, map_ell)
    if np.all(map_ell == 0):
        return {"l": float("nan"), "b": float("nan")}

    max_pix = int(np.argmax(np.abs(map_ell)))
    theta, phi = hp.pix2ang(nside, max_pix)
    l_deg = np.rad2deg(phi) % 360.0
    b_deg = 90.0 - np.rad2deg(theta)

    # Orient the axis toward the positive fluctuation
    if map_ell[max_pix] < 0:
        l_deg = (l_deg + 180.0) % 360.0
        b_deg = -b_deg

    return {"l": float(l_deg), "b": float(b_deg)}


def compute_low_ell_axes(
    delta_path: Path, mask_path: Path, output_path: Path, lmax: int = 3
) -> List[Dict[str, object]]:
    delta_map, mask = load_maps(delta_path, mask_path)
    results = compute_low_ell_axes_from_arrays(delta_map, mask, lmax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def compute_low_ell_axes_from_arrays(
    delta_map: np.ndarray, mask: np.ndarray, lmax: int = 3
) -> List[Dict[str, object]]:
    if delta_map.shape != mask.shape:
        raise ValueError("Delta map and mask must have the same shape")

    nside = hp.get_nside(delta_map)
    masked_delta = np.where(mask, 0.0, delta_map)
    alm = hp.map2alm(masked_delta, lmax=lmax, iter=3)
    cl = hp.alm2cl(alm)
    total_power = float(np.sum(cl[1 : lmax + 1]))

    results: List[Dict[str, object]] = []
    for ell in range(1, lmax + 1):
        axis = _dominant_axis_for_ell(alm, ell, lmax, mask, nside)
        c_ell = float(cl[ell]) if ell < len(cl) else float("nan")
        fraction = c_ell / total_power if total_power > 0 else float("nan")
        results.append(
            {
                "ell": ell,
                "axis": axis,
                "C_ell": c_ell,
                "fraction_of_total_power": fraction,
            }
        )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=Path("results/stage5/delta.fits"))
    parser.add_argument("--mask", type=Path, default=Path("results/stage5/mask.fits"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/stage6/ell_axes.json")
    )
    parser.add_argument("--lmax", type=int, default=3)
    args = parser.parse_args()

    compute_low_ell_axes(args.delta, args.mask, args.output, lmax=args.lmax)


if __name__ == "__main__":
    main()
