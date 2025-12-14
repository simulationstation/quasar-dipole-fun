#!/usr/bin/env python3
"""Generate a synthetic quasar catalog with an injected dipole.

The output is a CSV with columns: ra, dec, w1mpro, qso_prob.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from astropy.coordinates import CartesianRepresentation, SkyCoord
    import astropy.units as u
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "astropy is required for generate_mock_catalog.py; install with `pip install astropy`"
    ) from exc


def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    return np.array([
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    ])


def unitvec_to_radec(vec: np.ndarray) -> Tuple[float, float]:
    """Convert a 3D Cartesian vector to ICRS (RA, Dec) in degrees.

    The input can be any finite non-zero vector; it will be normalized before
    conversion. RA is wrapped to [0, 360) degrees.
    """

    v = np.asarray(vec, dtype=float).reshape(3,)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("vec must be a finite non-zero 3-vector")

    v = v / norm
    rep = CartesianRepresentation(*(v * u.one))
    coord = SkyCoord(rep, frame="icrs")
    ra = coord.spherical.lon.to_value(u.deg) % 360.0
    dec = coord.spherical.lat.to_value(u.deg)
    return float(ra), float(dec)


def sample_dipolar_sky(n: int, D_true: float, l_dipole: float, b_dipole: float, rng: np.random.Generator) -> np.ndarray:
    if D_true < 0 or D_true >= 1:
        raise ValueError("Dipole amplitude must satisfy 0 <= D_true < 1 for rejection sampling")

    target = lb_to_unitvec(l_dipole, b_dipole)
    accepted: list[np.ndarray] = []

    while len(accepted) < n:
        batch = max(10_000, int(1.2 * (n - len(accepted))))
        # Isotropic directions on the sphere
        lon = rng.uniform(0.0, 2 * math.pi, size=batch)
        cos_lat = rng.uniform(-1.0, 1.0, size=batch)
        lat = np.arcsin(cos_lat)

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        vecs = np.stack([x, y, z], axis=-1)

        cos_theta = vecs @ target
        # Rejection sampling with probability proportional to 1 + D_true * cos(theta)
        accept_prob = 1 + D_true * cos_theta
        accept_prob /= (1 + D_true)
        uniform = rng.random(size=batch)
        keep = uniform < accept_prob

        for v in vecs[keep]:
            accepted.append(v)
            if len(accepted) >= n:
                break

    return np.vstack(accepted[:n])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate isotropic sky points with an injected dipole")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--n", type=int, default=100_000, help="Number of sources to generate")
    parser.add_argument("--dipole", type=float, default=0.02, help="Dipole amplitude D_true")
    parser.add_argument("--l-dipole", type=float, default=210.0, help="Dipole longitude in Galactic coords (deg)")
    parser.add_argument("--b-dipole", type=float, default=30.0, help="Dipole latitude in Galactic coords (deg)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    vecs = sample_dipolar_sky(args.n, args.dipole, args.l_dipole, args.b_dipole, rng)
    ra_list, dec_list = zip(*(unitvec_to_radec(v) for v in vecs))

    w1 = np.full(args.n, 15.0)
    qso_prob = np.full(args.n, 0.99)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("ra,dec,w1mpro,qso_prob\n")
        for ra, dec, w1_val, qso in zip(ra_list, dec_list, w1, qso_prob):
            f.write(f"{ra:.8f},{dec:.8f},{w1_val:.4f},{qso:.4f}\n")

    print(f"Wrote mock catalog with N={args.n} to {args.output}")


if __name__ == "__main__":
    main()
