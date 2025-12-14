#!/usr/bin/env python3
"""Generate and validate a mock catalog end-to-end."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def angle_between(vec1, vec2):
    import numpy as np

    dot = float(np.dot(vec1, vec2))
    dot = max(min(dot, 1.0), -1.0)
    return math.degrees(math.acos(dot))


def lb_to_unitvec(l_deg, b_deg):
    import math
    import numpy as np

    l = math.radians(l_deg)
    b = math.radians(b_deg)
    return np.array([
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    ])


def main() -> None:
    outdir = Path("mock_run")
    outdir.mkdir(exist_ok=True)

    catalog = outdir / "mock_catalog.csv"
    results = outdir / "results"
    results.mkdir(exist_ok=True)

    true_D = 0.02
    true_l = 210.0
    true_b = 30.0

    print("Generating mock catalog...")
    subprocess.run(
        [
            sys.executable,
            "generate_mock_catalog.py",
            "--output",
            str(catalog),
            "--n",
            "80000",
            "--dipole",
            str(true_D),
            "--l-dipole",
            str(true_l),
            "--b-dipole",
            str(true_b),
            "--seed",
            "123",
        ],
        check=True,
    )

    print("Running dipole reproduction...")
    subprocess.run(
        [
            sys.executable,
            "reproduce_catwise_dipole.py",
            "--input",
            str(catalog),
            "--outdir",
            str(results),
            "--ra-col",
            "ra",
            "--dec-col",
            "dec",
            "--w1-col",
            "w1mpro",
            "--no-qso-prob-cut",
            "--mask-gal-b-min",
            "10",
            "--bootstrap",
            "200",
            "--seed",
            "999",
            "--chunk-size",
            "50000",
            "--randomization-trials",
            "50",
        ],
        check=True,
    )

    with open(results / "catwise_dipole.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    D_hat = float(payload["dipole"]["amplitude"])
    l_hat = float(payload["dipole"]["lon_deg"])
    b_hat = float(payload["dipole"]["lat_deg"])

    v_true = lb_to_unitvec(true_l, true_b)
    v_hat = lb_to_unitvec(l_hat, b_hat)
    ang = angle_between(v_true, v_hat)

    print(f"Recovered D={D_hat:.4f} vs true {true_D:.4f}")
    print(f"Direction separation={ang:.2f} deg")

    if abs(D_hat - true_D) > 0.01 or ang > 20.0:
        raise SystemExit("Mock test failed: recovered dipole too far from truth")

    print("Mock test passed.")


if __name__ == "__main__":
    main()
