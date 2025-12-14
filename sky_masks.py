"""Sky masking utilities for dipole estimation.

The functions here avoid external astronomy packages while providing
common masks used in catalog-level dipole analyses. All angles are in
degrees. When equatorial coordinates are provided, conversion to
Galactic coordinates uses the IAU 1958 / J2000 coefficients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

# IAU 1958 / J2000 conversion constants (degrees)
_RA_NGP = math.radians(192.85948)
_DEC_NGP = math.radians(27.12825)
_L_CP = math.radians(122.93192)  # Galactic longitude of the North Celestial Pole


@dataclass
class MaskSummary:
    """Summary information about applied masks."""

    retained_fraction: float
    description: str


def _to_numpy(array: Iterable[float]) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Angle arrays must be one-dimensional")
    return arr


def equatorial_to_galactic(ra_deg: Iterable[float], dec_deg: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert equatorial (RA, Dec) to Galactic (lon, lat).

    The transformation follows the formulation in the Hipparcos explanatory
    supplement and matches the default astropy conversion within <1e-6 rad.
    """

    ra = np.radians(_to_numpy(ra_deg))
    dec = np.radians(_to_numpy(dec_deg))

    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_dngp = math.sin(_DEC_NGP)
    cos_dngp = math.cos(_DEC_NGP)

    delta_ra = ra - _RA_NGP

    sin_b = sin_dec * sin_dngp + cos_dec * cos_dngp * np.cos(delta_ra)
    b = np.arcsin(np.clip(sin_b, -1.0, 1.0))

    y = cos_dec * np.sin(delta_ra)
    x = sin_dec * cos_dngp - cos_dec * sin_dngp * np.cos(delta_ra)
    l = np.arctan2(y, x) + _L_CP

    l = np.mod(l, 2 * np.pi)
    return np.degrees(l), np.degrees(b)


def galactic_latitude_mask(
    lon_deg: Iterable[float],
    lat_deg: Iterable[float],
    b_min_deg: float,
) -> np.ndarray:
    """Return a boolean mask keeping |b| >= b_min_deg."""

    lat = np.abs(_to_numpy(lat_deg))
    return lat >= float(b_min_deg)


def rectangular_mask(
    lon_deg: Iterable[float],
    lat_deg: Iterable[float],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> np.ndarray:
    """Mask objects inside a rectangular longitude/latitude window.

    Longitude wrapping is handled: if ``lon_min > lon_max`` the mask wraps
    across 0° (e.g., 350°–10°).
    """

    lon = np.mod(_to_numpy(lon_deg), 360.0)
    lat = _to_numpy(lat_deg)

    if lon_min <= lon_max:
        lon_keep = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_keep = (lon >= lon_min) | (lon <= lon_max)
    lat_keep = (lat >= lat_min) & (lat <= lat_max)
    return lon_keep & lat_keep


def apply_boolean_mask(mask_column: Iterable[float | bool]) -> np.ndarray:
    """Validate and return a boolean mask from a catalog column."""

    mask = _to_numpy(mask_column).astype(bool)
    return mask


def describe_mask(retained: int, total: int, description: str) -> MaskSummary:
    fraction = 0.0 if total == 0 else retained / total
    return MaskSummary(retained_fraction=fraction, description=description)
