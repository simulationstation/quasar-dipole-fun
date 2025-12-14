"""Dipole estimators that avoid external sky pixelization dependencies.

The estimators operate on arrays of sky longitudes and latitudes (degrees)
and return amplitudes and directions in the same frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass
class DipoleEstimate:
    amplitude: float
    lon_deg: float
    lat_deg: float
    vector: np.ndarray


@dataclass
class DipoleUncertainty:
    amplitude_sigma: float
    lon_sigma_deg: float
    lat_sigma_deg: float
    covariance: np.ndarray


@dataclass
class JackknifeResult:
    subset_label: str
    estimate: DipoleEstimate


# ------------------------- spherical helpers ------------------------- #

def sph_to_cart(lon_deg: Iterable[float], lat_deg: Iterable[float]) -> np.ndarray:
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def cart_to_sph(vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = vec
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.hypot(x, y))
    lon_deg = math.degrees(lon) % 360.0
    lat_deg = math.degrees(lat)
    return lon_deg, lat_deg


# ------------------------- dipole estimators ------------------------- #

def simple_vector_dipole(lon_deg: Iterable[float], lat_deg: Iterable[float]) -> DipoleEstimate:
    """Unweighted dipole using a normalized vector sum."""

    vecs = sph_to_cart(lon_deg, lat_deg)
    sum_vec = vecs.sum(axis=0)
    norm = np.linalg.norm(sum_vec)
    total = len(vecs)
    amplitude = float(norm / total) if total > 0 else float("nan")
    lon, lat = cart_to_sph(sum_vec)
    return DipoleEstimate(amplitude=amplitude, lon_deg=lon, lat_deg=lat, vector=sum_vec)


def weighted_dipole(
    lon_deg: Iterable[float], lat_deg: Iterable[float], weights: Optional[Iterable[float]] = None
) -> DipoleEstimate:
    """Weighted dipole estimator.

    Weights typically encode completeness or flux. Negative weights are
    not allowed; they are clipped to zero to avoid sign inversions.
    """

    vecs = sph_to_cart(lon_deg, lat_deg)
    if weights is None:
        w = np.ones(len(vecs))
    else:
        w = np.clip(np.asarray(weights, dtype=float), a_min=0.0, a_max=None)
    if len(w) != len(vecs):
        raise ValueError("weights must match the length of the coordinate arrays")

    weighted_vec = (vecs.T * w).T
    sum_vec = weighted_vec.sum(axis=0)
    norm = np.linalg.norm(sum_vec)
    weight_sum = w.sum()
    amplitude = float(norm / weight_sum) if weight_sum > 0 else float("nan")
    lon, lat = cart_to_sph(sum_vec)
    return DipoleEstimate(amplitude=amplitude, lon_deg=lon, lat_deg=lat, vector=sum_vec)


def bootstrap_dipole(
    lon_deg: Iterable[float],
    lat_deg: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[DipoleEstimate, DipoleUncertainty]:
    """Bootstrap dipole estimate and uncertainties."""

    rng = np.random.default_rng(random_state)
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    w = np.ones_like(lon) if weights is None else np.asarray(weights, dtype=float)
    if len(lon) != len(lat) or len(lon) != len(w):
        raise ValueError("lon/lat/weights lengths must match")

    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(lon), size=len(lon))
        boot = weighted_dipole(lon[idx], lat[idx], w[idx])
        estimates.append(boot)

    amplitudes = np.array([e.amplitude for e in estimates])
    lons = np.array([e.lon_deg for e in estimates])
    lats = np.array([e.lat_deg for e in estimates])

    # Directional statistics: recenter longitudes near the median to avoid wrap issues
    lon_center = np.median(lons)
    lons_centered = lon_center + np.mod(lons - lon_center + 180.0, 360.0) - 180.0

    amplitude_sigma = float(np.std(amplitudes, ddof=1)) if len(amplitudes) > 1 else float("nan")
    lon_sigma = float(np.std(lons_centered, ddof=1)) if len(lons_centered) > 1 else float("nan")
    lat_sigma = float(np.std(lats, ddof=1)) if len(lats) > 1 else float("nan")

    cov = np.cov(np.vstack([amplitudes, lons_centered, lats])) if len(amplitudes) > 1 else np.full((3, 3), np.nan)

    # Point estimate from the full sample (not the bootstrap mean)
    point = weighted_dipole(lon, lat, w)
    return point, DipoleUncertainty(
        amplitude_sigma=amplitude_sigma,
        lon_sigma_deg=lon_sigma,
        lat_sigma_deg=lat_sigma,
        covariance=cov,
    )


def jackknife_by_hemisphere(lon_deg: Iterable[float], lat_deg: Iterable[float], weights: Optional[Iterable[float]] = None):
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    w = np.ones_like(lon) if weights is None else np.asarray(weights, dtype=float)

    results = []
    north = lat >= 0
    south = lat < 0
    for label, mask in [("north", north), ("south", south)]:
        if mask.sum() == 0:
            continue
        results.append(JackknifeResult(label, weighted_dipole(lon[mask], lat[mask], w[mask])))
    return results


def jackknife_by_octant(lon_deg: Iterable[float], lat_deg: Iterable[float], weights: Optional[Iterable[float]] = None):
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    w = np.ones_like(lon) if weights is None else np.asarray(weights, dtype=float)

    results = []
    lon_wrapped = np.mod(lon, 360.0)
    octant_edges = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    for i in range(8):
        mask = (lon_wrapped >= octant_edges[i]) & (lon_wrapped < octant_edges[i + 1])
        if mask.sum() == 0:
            continue
        label = f"octant_{i + 1}"
        results.append(JackknifeResult(label, weighted_dipole(lon[mask], lat[mask], w[mask])))
    return results


def randomized_null_test(
    lon_deg: Iterable[float],
    lat_deg: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    n_realizations: int = 500,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Return amplitudes from dipoles with randomized longitudes (null test)."""

    rng = np.random.default_rng(random_state)
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    w = np.ones_like(lon) if weights is None else np.asarray(weights, dtype=float)

    amps = []
    for _ in range(n_realizations):
        shuffled_lon = rng.permutation(lon)
        est = weighted_dipole(shuffled_lon, lat, w)
        amps.append(est.amplitude)
    return np.asarray(amps)
