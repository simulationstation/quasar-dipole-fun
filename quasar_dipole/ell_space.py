"""HEALPix-based ℓ-space diagnostics for dipole robustness.

This module provides reusable helpers for the ℓ-space test that compares
low-multipole power in the observed catalog against structured null
hypotheses (RA scrambling, pixel shuffling, and optional phase
randomization). The functions are intentionally stateless so that CLI
front-ends can stay thin while keeping the logic testable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from secrest_utils import compute_dipole


@dataclass
class SliceResult:
    name: str
    w1_range: Optional[Tuple[float, float]]
    n_sources: int
    f_sky: float
    mean_per_pixel: float
    cl: List[float]
    dipole_fraction: float
    quadrupole_to_dipole: float
    octupole_to_dipole: float
    catalog_dipole_amp: float
    catalog_dipole_l_deg: float
    catalog_dipole_b_deg: float
    map_dipole_amp: float
    map_dipole_l_deg: float
    map_dipole_b_deg: float

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        if self.w1_range is not None:
            data["w1_range"] = list(self.w1_range)
        return data


@dataclass
class NullResult:
    name: str
    p_values: Dict[str, float]
    c1_samples: np.ndarray
    c2_samples: np.ndarray
    c3_samples: np.ndarray
    dipole_fraction_samples: np.ndarray
    direction_angle_deg: np.ndarray

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "p_values": self.p_values,
            "direction_angle_deg_stats": {
                "median": float(np.median(self.direction_angle_deg)),
                "p95": float(np.percentile(self.direction_angle_deg, 95)),
            },
        }


class EllSpaceMaps:
    def __init__(self, nside: int, mask: np.ndarray, parent_counts: np.ndarray):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.mask = mask
        self.parent_counts = parent_counts

    @property
    def f_sky(self) -> float:
        return 1.0 - float(np.mean(self.mask))


# ---------------------------------------------------------------------------
# Catalog loading and slicing
# ---------------------------------------------------------------------------


def load_catalog(path: Path) -> Table:
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {path}. Provide --catalog pointing to a local FITS file."
        )
    return Table.read(path, memmap=True)


def ensure_galactic_lat(tbl: Table) -> None:
    if "b" in tbl.colnames:
        return
    if "ra" not in tbl.colnames or "dec" not in tbl.colnames:
        raise ValueError("Catalog must provide RA and DEC columns to compute Galactic latitude.")
    coords = SkyCoord(tbl["ra"], tbl["dec"], unit=(u.deg, u.deg), frame="icrs")
    tbl["b"] = coords.galactic.b.to(u.deg)
    tbl["l"] = coords.galactic.l.to(u.deg)


# ---------------------------------------------------------------------------
# HEALPix utilities
# ---------------------------------------------------------------------------


def build_counts_map(ra_deg: np.ndarray, dec_deg: np.ndarray, nside: int) -> np.ndarray:
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg % 360.0)
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix).astype(float)
    return counts


def galactic_mask(nside: int, b_cut: float) -> np.ndarray:
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    b_deg = 90.0 - np.rad2deg(theta)
    return np.abs(b_deg) <= b_cut


def combine_masks(primary: np.ndarray, secondary: Optional[np.ndarray]) -> np.ndarray:
    if secondary is None:
        return primary
    if secondary.shape != primary.shape:
        raise ValueError("Mask shapes differ between primary and user-provided mask.")
    return primary | secondary.astype(bool)


def mask_from_fits(path: Optional[Path], nside: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Mask FITS not found: {path}")
    mask_map = hp.read_map(path, dtype=float)
    if mask_map.size != hp.nside2npix(nside):
        raise ValueError("Provided mask NSIDE does not match requested NSIDE.")
    return mask_map != 0


def compute_overdensity(counts: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    unmasked = counts[~mask]
    mean_n = float(np.mean(unmasked)) if unmasked.size > 0 else float("nan")
    delta = np.zeros_like(counts, dtype=float)
    if mean_n > 0.0:
        delta[~mask] = (counts[~mask] - mean_n) / mean_n
    return delta, mean_n


def pseudo_cl(delta_map: np.ndarray, mask: np.ndarray, ellmax: int) -> Tuple[np.ndarray, float]:
    sky_fraction = 1.0 - float(np.mean(mask))
    masked_delta = delta_map.copy()
    masked_delta[mask] = 0.0
    cl_raw = hp.anafast(masked_delta, lmax=ellmax)
    if sky_fraction > 0:
        cl_corr = cl_raw / sky_fraction
    else:
        cl_corr = np.full_like(cl_raw, float("nan"))
    return cl_corr, sky_fraction


def map_dipole_from_healpix(counts: np.ndarray, mask: np.ndarray, nside: int) -> Tuple[float, float, float]:
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg = np.rad2deg(phi)
    dec_deg = 90.0 - np.rad2deg(theta)
    coords = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs").galactic
    l_deg = coords.l.deg
    b_deg = coords.b.deg
    weights = counts.copy()
    weights[mask] = 0.0
    amp, l_d, b_d, _ = compute_dipole(l_deg, b_deg, weights=weights)
    return amp, l_d, b_d


# ---------------------------------------------------------------------------
# Null draws
# ---------------------------------------------------------------------------


def ra_scramble(
    ra: np.ndarray,
    dec: np.ndarray,
    base_mask: np.ndarray,
    nside: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled_ra = rng.permutation(ra)
    counts = build_counts_map(shuffled_ra[base_mask], dec[base_mask], nside)
    return counts


def pixel_shuffle(
    parent_counts: np.ndarray,
    n_sources: int,
    mask: np.ndarray,
    nside: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = parent_counts.copy()
    weights[mask] = 0.0
    total = weights.sum()
    if total == 0.0:
        raise ValueError("Parent exposure map is empty; cannot perform pixel shuffle null.")
    prob = weights / total
    pix_choices = rng.choice(np.arange(weights.size), size=n_sources, p=prob)
    counts = np.bincount(pix_choices, minlength=weights.size).astype(float)
    return counts


def phase_randomize(
    delta_map: np.ndarray, mask: np.ndarray, ellmax: int, rng: np.random.Generator
) -> np.ndarray:
    masked = delta_map.copy()
    masked[mask] = 0.0
    alm = hp.map2alm(masked, lmax=ellmax)
    l_arr, m_arr = hp.Alm.getlm(ellmax)
    for idx, (l, m) in enumerate(zip(l_arr, m_arr)):
        if m == 0 or l == 0:
            continue
        phase = rng.uniform(0, 2 * math.pi)
        amplitude = np.abs(alm[idx])
        alm[idx] = amplitude * np.exp(1j * phase)
    randomized = hp.alm2map(alm, nside=hp.npix2nside(mask.size), lmax=ellmax, verbose=False)
    randomized[mask] = 0.0
    return randomized


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_healpix_map(path: Path, m: np.ndarray, overwrite: bool = True) -> None:
    hp.write_map(path, m, overwrite=overwrite)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def plot_cl_curve(cl: np.ndarray, ellmax: int, title: str, outpath: Path) -> None:
    ell = np.arange(cl.size)
    plt.figure(figsize=(6, 4))
    plt.plot(ell, cl, marker="o")
    plt.xlim(0.5, ellmax + 0.5)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_null_hist(observed: float, samples: np.ndarray, title: str, outpath: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(samples, bins=30, alpha=0.7, color="steelblue")
    plt.axvline(observed, color="red", linestyle="--", label="Observed")
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_direction_scatter(
    bin_centers: List[float], angles: List[float], title: str, outpath: Path
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, angles, marker="o")
    plt.xlabel("W1 bin center")
    plt.ylabel("Dipole direction scatter (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

