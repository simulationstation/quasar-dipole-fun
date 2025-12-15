"""Axis coherence test for low-ℓ multipoles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


Pairwise = Dict[str, float]


def load_axes(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Axis file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_coord(axis: Dict[str, float]) -> SkyCoord:
    return SkyCoord(axis["l"] * u.deg, axis["b"] * u.deg, frame="galactic")


def compute_axis_coherence(axes: List[Dict[str, object]], output_path: Path) -> Dict[str, object]:
    axes_by_ell = {int(entry["ell"]): entry["axis"] for entry in axes}
    required = [1, 2, 3]
    for ell in required:
        if ell not in axes_by_ell:
            raise ValueError(f"Missing ℓ={ell} axis in axes input")

    pairs = [(1, 2), (1, 3), (2, 3)]
    separations: Dict[str, float] = {}
    coord_cache = {ell: _to_coord(axes_by_ell[ell]) for ell in required}
    for a, b in pairs:
        key = f"ell{a}_ell{b}"
        sep = coord_cache[a].separation(coord_cache[b]).deg
        separations[key] = float(sep)

    coherence = float(np.mean(list(separations.values())))
    payload = {"separations_deg": separations, "coherence_deg": coherence}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axes", type=Path, default=Path("results/stage6/ell_axes.json"), help="Input axes JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage6/axis_coherence.json"),
        help="Output coherence JSON",
    )
    args = parser.parse_args()

    axes = load_axes(args.axes)
    compute_axis_coherence(axes, args.output)


if __name__ == "__main__":
    main()
