"""Convert catalog dipole estimates into inputs for the Bayesian model.

The output JSON is intentionally minimal and mirrors the inputs expected
by ``metropolis_hastings_sampler.py``: an amplitude with Gaussian
uncertainty and an angular separation (in degrees) with uncertainty.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict


def load_dipole(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def angular_separation(lon1_deg: float, lat1_deg: float, lon2_deg: float, lat2_deg: float) -> float:
    """Return great-circle separation in degrees."""

    lon1 = math.radians(lon1_deg)
    lat1 = math.radians(lat1_deg)
    lon2 = math.radians(lon2_deg)
    lat2 = math.radians(lat2_deg)
    dlon = lon1 - lon2
    cos_sep = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def convert(dipole_json: Dict, ref_lon: float | None, ref_lat: float | None) -> Dict:
    dipole = dipole_json["dipole"]
    unc = dipole_json.get("uncertainty", {})

    amp = float(dipole["amplitude"])
    amp_sigma = float(unc.get("amplitude_sigma", float("nan")))
    lon = float(dipole["lon_deg"])
    lat = float(dipole["lat_deg"])

    separation = None
    sep_sigma = None
    if ref_lon is not None and ref_lat is not None:
        separation = angular_separation(lon, lat, ref_lon, ref_lat)
        # A conservative angular uncertainty: use the larger of lon/lat sigmas if available.
        lon_sig = float(unc.get("lon_sigma_deg", float("nan")))
        lat_sig = float(unc.get("lat_sigma_deg", float("nan")))
        sep_sigma = max(abs(lon_sig), abs(lat_sig)) if not math.isnan(lon_sig) and not math.isnan(lat_sig) else float("nan")

    return {
        "catalog": dipole_json.get("catalog"),
        "frame": dipole_json.get("frame"),
        "n_after_cuts": dipole_json.get("n_after_cuts"),
        "amplitude_mean": amp,
        "amplitude_sigma": amp_sigma,
        "direction_lon_deg": lon,
        "direction_lat_deg": lat,
        "reference_lon_deg": ref_lon,
        "reference_lat_deg": ref_lat,
        "angular_separation_deg": separation,
        "angular_separation_sigma_deg": sep_sigma,
        "notes": "Assumes Gaussian amplitude likelihood; angular uncertainty approximated from lon/lat scatters.",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dipole-json", required=True, help="JSON produced by estimate_catalog_dipole.py")
    parser.add_argument("--ref-lon", type=float, help="Reference longitude (deg) for angular comparison")
    parser.add_argument("--ref-lat", type=float, help="Reference latitude (deg) for angular comparison")
    parser.add_argument("--output", default="model_constraints.json", help="Output JSON for the sampler")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dipole_json = load_dipole(args.dipole_json)
    payload = convert(dipole_json, args.ref_lon, args.ref_lat)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
