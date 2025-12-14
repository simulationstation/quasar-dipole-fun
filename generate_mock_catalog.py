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
    from astropy.coordinates import SkyCoord
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


def galactic_lb_to_icrs_radec(l_deg: np.ndarray, b_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform Galactic (l, b) coordinates to ICRS RA/Dec in degrees."""

    coord_gal = SkyCoord(l=np.asarray(l_deg) * u.deg, b=np.asarray(b_deg) * u.deg, frame="galactic")
    coord_icrs = coord_gal.icrs
    return coord_icrs.ra.deg, coord_icrs.dec.deg


def sample_dipolar_sky(
    n: int, D_true: float, l_dipole: float, b_dipole: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    if D_true < 0 or D_true >= 1:
        raise ValueError("Dipole amplitude must satisfy 0 <= D_true < 1 for rejection sampling")

    target = lb_to_unitvec(l_dipole, b_dipole)
    l_acc: list[float] = []
    b_acc: list[float] = []

    while len(l_acc) < n:
        batch = max(10_000, int(1.2 * (n - len(l_acc))))
        # Sample isotropic Galactic longitudes/latitudes
        l_batch = rng.uniform(0.0, 360.0, size=batch)
        sin_b = rng.uniform(-1.0, 1.0, size=batch)
        b_batch = np.degrees(np.arcsin(sin_b))

        l_rad = np.radians(l_batch)
        b_rad = np.radians(b_batch)
        cos_b = np.cos(b_rad)

        vecs = np.stack(
            [
                cos_b * np.cos(l_rad),
                cos_b * np.sin(l_rad),
                np.sin(b_rad),
            ],
            axis=-1,
        )

        cos_theta = vecs @ target
        accept_prob = (1.0 + D_true * cos_theta) / (1.0 + D_true)
        uniform = rng.random(size=batch)
        keep = uniform < accept_prob

        for l_val, b_val in zip(l_batch[keep], b_batch[keep]):
            l_acc.append(float(l_val))
            b_acc.append(float(b_val))
            if len(l_acc) >= n:
                break

    return np.array(l_acc[:n]), np.array(b_acc[:n])


def estimate_dipole_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[float, float, float]:
    coord_icrs = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    cart = coord_icrs.cartesian
    sum_vec = np.array([cart.x.sum().value, cart.y.sum().value, cart.z.sum().value])
    n = len(ra_deg)
    amplitude = float(3.0 * np.linalg.norm(sum_vec) / n) if n > 0 else float("nan")

    recovered_icrs = SkyCoord(x=sum_vec[0] * u.one, y=sum_vec[1] * u.one, z=sum_vec[2] * u.one, frame="icrs", representation_type="cartesian")
    recovered_gal = recovered_icrs.galactic
    return amplitude, float(recovered_gal.l.deg), float(recovered_gal.b.deg)


def run_self_check() -> None:
    print("Running self-check with a synthetic catalog...")

    n = 200_000
    D_true = 0.02
    l_true = 264.0
    b_true = 48.0
    seed = 2024

    rng = np.random.default_rng(seed)

    l_samples, b_samples = sample_dipolar_sky(n, D_true, l_true, b_true, rng)
    ra_samples, dec_samples = galactic_lb_to_icrs_radec(l_samples, b_samples)

    D_rec, l_rec, b_rec = estimate_dipole_from_radec(ra_samples, dec_samples)

    target_coord = SkyCoord(l=l_true * u.deg, b=b_true * u.deg, frame="galactic")
    recovered_coord = SkyCoord(l=l_rec * u.deg, b=b_rec * u.deg, frame="galactic")
    sep = target_coord.separation(recovered_coord).deg
    rel_err = abs(D_rec - D_true) / D_true if D_true > 0 else float("inf")

    print(f"True dipole:   D={D_true:.4f}, (l, b)=({l_true:.1f}, {b_true:.1f})")
    print(f"Recovered dipole: D={D_rec:.4f}, (l, b)=({l_rec:.1f}, {b_rec:.1f})")
    print(f"Angular separation: {sep:.3f} deg")
    print(f"Relative amplitude error: {rel_err * 100:.2f}%")

    failed = False
    if rel_err > 0.20:
        print("Self-check failed: amplitude error exceeds 20% tolerance")
        failed = True
    if sep > 10.0:
        print("Self-check failed: dipole direction differs by more than 10 degrees")
        failed = True

    if failed:
        raise SystemExit(1)
    print("Self-check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate isotropic sky points with an injected dipole")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--n", type=int, default=100_000, help="Number of sources to generate")
    parser.add_argument("--dipole", type=float, default=0.02, help="Dipole amplitude D_true")
    parser.add_argument("--l-dipole", type=float, default=210.0, help="Dipole longitude in Galactic coords (deg)")
    parser.add_argument("--b-dipole", type=float, default=30.0, help="Dipole latitude in Galactic coords (deg)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--self-check", action="store_true", help="Run a quick dipole recovery self-check and exit")
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    if args.output is None:
        parser.error("--output is required unless --self-check is used")

    rng = np.random.default_rng(args.seed)

    l_samples, b_samples = sample_dipolar_sky(args.n, args.dipole, args.l_dipole, args.b_dipole, rng)
    ra_list, dec_list = galactic_lb_to_icrs_radec(l_samples, b_samples)

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
