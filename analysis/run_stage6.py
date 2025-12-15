"""Unified Stage-6 orchestrator for multipole-aware anisotropy diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis.axis_coherence import compute_axis_coherence
from analysis.low_ell_axes import compute_low_ell_axes
from analysis.mask_forward_models import simulate_models
from analysis.null_tests import run_null_tests


def _format_axes_table(axes: List[Dict[str, object]]) -> str:
    lines = ["| ℓ | l (deg) | b (deg) | C_ℓ | frac. power |", "|---|---------|---------|------|-------------|"]
    for entry in axes:
        axis = entry["axis"]
        lines.append(
            f"| {entry['ell']} | {axis['l']:.2f} | {axis['b']:.2f} | {entry['C_ell']:.3e} | {entry['fraction_of_total_power']:.3f} |"
        )
    return "\n".join(lines)


def _coherence_pvalues(observed: float, distributions: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    pvals: Dict[str, float] = {}
    for name, data in distributions.items():
        samples = np.asarray(data["coherence"])
        if samples.size == 0:
            pvals[name] = float("nan")
        else:
            pvals[name] = float(np.mean(samples <= observed))
    return pvals


def _verdict(axes: List[Dict[str, object]], pvals: Dict[str, float]) -> str:
    axes_by_ell = {int(entry["ell"]): entry for entry in axes}
    dipole = axes_by_ell.get(1, {})
    quadrupole = axes_by_ell.get(2, {})
    dip_power = float(dipole.get("C_ell", np.nan))
    quad_power = float(quadrupole.get("C_ell", np.nan))

    if not np.isfinite(dip_power) or not np.isfinite(quad_power):
        return "Mask/systematic dominated"

    if quad_power > dip_power:
        return "Geometric structure"

    if all(np.isfinite(list(pvals.values()))) and min(pvals.values()) < 0.05:
        return "Velocity-like"

    return "Mask/systematic dominated"


def run_stage6(
    delta_path: Path,
    mask_path: Path,
    catalog_path: Path,
    output_dir: Path,
    n_null: int = 500,
    seed: int = 42,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = compute_low_ell_axes(delta_path, mask_path, output_dir / "ell_axes.json")
    coherence = compute_axis_coherence(axes, output_dir / "axis_coherence.json")
    nulls = run_null_tests(
        delta_path,
        mask_path,
        catalog_path,
        output_dir / "null_distributions.json",
        n_draws=n_null,
        seed=seed,
    )
    models = simulate_models(
        delta_path,
        mask_path,
        catalog_path,
        output_dir / "model_forward.json",
        seed=seed,
    )

    coherence_pvals = _coherence_pvalues(coherence["coherence_deg"], nulls["distributions"])
    verdict = _verdict(axes, coherence_pvals)

    report_path = output_dir / "stage6_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Stage 6: Multipole-Aware Anisotropy Diagnostics\n\n")
        f.write("## Low-ℓ Axes and Power\n\n")
        f.write(_format_axes_table(axes) + "\n\n")
        f.write(
            f"Axis coherence: {coherence['coherence_deg']:.2f} deg (RA-scramble p={coherence_pvals.get('ra_scramble', float('nan')):.3f}, isotropic p={coherence_pvals.get('isotropic', float('nan')):.3f})\n\n"
        )
        f.write("## Null Tests\n\n")
        f.write("Coherence distributions compared against observed value.\n\n")
        f.write("## Mask-Forward Models\n\n")
        f.write("Kinematic and geometric forward models evaluated under the real mask.\n\n")
        f.write(f"## Verdict\n\n{verdict}\n")

    return report_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=Path("results/stage5/delta.fits"))
    parser.add_argument("--mask", type=Path, default=Path("results/stage5/mask.fits"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage6"))
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_stage6(args.delta, args.mask, args.catalog, args.output_dir, args.n_null, args.seed)


if __name__ == "__main__":
    main()
