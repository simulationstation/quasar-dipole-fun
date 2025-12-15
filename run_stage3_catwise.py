#!/usr/bin/env python3
"""
Run Stage-3 CatWISE analysis end-to-end.

This orchestrates constraint derivation under several systematic toggles and then
feeds the resulting constraints into the MCMC sampler for model comparison and
posterior estimation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

RUN_MODES = (
    "baseline",
    "nvss_only",
    "ecliptic_only",
    "nvss_plus_ecliptic",
)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M")


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)} (cwd={cwd or Path('.').resolve()})")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def determine_best_model(mode_dir: Path) -> str:
    comparison = json.loads((mode_dir / "model_comparison.json").read_text())
    ranked = sorted(comparison, key=lambda r: r["log_mean_exp"], reverse=True)
    return ranked[0]["model"]


def write_report(mode_dir: Path, constraints: Dict, best_summary: Dict) -> None:
    report = mode_dir / "REPORT.md"
    lines = [f"# Stage-3 {mode_dir.name} report", ""]
    lines.append("## Constraints")
    lines.append(
        f"* Catalog: `{constraints.get('catalog', 'unknown')}` with N_raw={constraints.get('N_raw')}"
    )
    lines.append(
        f"* Dipole: D={constraints['dipole']['D']:.5f}, l={constraints['dipole']['l_deg']:.2f}°, "
        f"b={constraints['dipole']['b_deg']:.2f}°"
    )
    lines.append(
        f"* bootstrap sigma_amp≈{constraints.get('sigma_amp_used')}, "
        f"sigma_dir≈{constraints.get('sigma_dir_deg_used')} deg"
    )
    lines.append("")

    lines.append("## Posterior (best model)")
    lines.append(f"* Model: {best_summary['model_mode']}")
    lines.append(
        f"* Log-likelihood: mean={best_summary['loglike_scores']['mean_loglike']:.3f}, "
        f"log_mean_exp={best_summary['loglike_scores']['log_mean_exp']:.3f}"
    )
    lines.append("* Key parameters:")
    for name, stats in best_summary.get("parameters", {}).items():
        lines.append(
            f"  - {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, R_hat={stats.get('R_hat')}"
        )

    report.write_text("\n".join(lines))


def aggregate_summary(stage_dir: Path, summaries: Dict[str, Dict]) -> None:
    out_lines = ["# Stage-3 mode comparison", "", "| Mode | Model | log_mean_exp | sigma_dir (deg) |", "| --- | --- | --- | --- |"]
    for mode, summary in summaries.items():
        out_lines.append(
            f"| {mode} | {summary['model_mode']} | {summary['loglike_scores']['log_mean_exp']:.3f} | "
            f"{summary.get('constraints', {}).get('values', {}).get('SIGMA_QSO_DIR_DEG', 'n/a')} |"
        )
    (stage_dir / "SUMMARY.md").write_text("\n".join(out_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/secrest/secrest+22_accepted/wise/reference/catwise_agns.fits")
    parser.add_argument("--nvss-catalog", default=None, help="Path to NVSS catalog (optional)")
    parser.add_argument("--run-tag", default=None, help="Run tag for outputs (default: UTC timestamp)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--ecl-model", default="regress", choices=["regress", "weight"])
    args = parser.parse_args()

    run_tag = args.run_tag or _timestamp()
    stage_dir = Path("results/stage3") / run_tag
    stage_dir.mkdir(parents=True, exist_ok=True)

    modes_to_run = []
    modes_to_run.append({"name": "baseline", "nvss": False, "ecl": False})
    if args.nvss_catalog:
        modes_to_run.append({"name": "nvss_only", "nvss": True, "ecl": False})
        modes_to_run.append({"name": "nvss_plus_ecliptic", "nvss": True, "ecl": True})
    else:
        print("NVSS catalog not provided; skipping NVSS-removal modes.")
    modes_to_run.append({"name": "ecliptic_only", "nvss": False, "ecl": True})

    best_summaries: Dict[str, Dict] = {}

    for mode in modes_to_run:
        mode_name = mode["name"]
        print(f"\n=== Running mode: {mode_name} ===")
        run_tag_mode = f"{run_tag}_{mode_name}"
        derive_args = [
            "python",
            "pipelines/derive_catwise_constraints.py",
            "--catalog",
            args.catalog,
            "--run-tag",
            run_tag_mode,
            "--bootstrap",
            str(args.bootstrap),
            "--seed",
            str(args.seed),
        ]
        if mode["nvss"]:
            derive_args.append("--apply-nvss-removal")
            derive_args += ["--nvss-catalog", args.nvss_catalog]
        if mode["ecl"]:
            derive_args.append("--apply-ecliptic-correction")
            derive_args += ["--ecl-model", args.ecl_model]

        run_cmd(derive_args)

        constraints_path = Path("results/secrest_reproduction") / run_tag_mode / "dipole_constraints.json"
        if not constraints_path.exists():
            raise SystemExit(f"Constraints file missing for mode {mode_name}: {constraints_path}")

        mode_dir = stage_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(constraints_path, mode_dir / "constraints.json")

        compare_cmd = [
            "python",
            "../../metropolis_hastings_sampler.py",
            "--compare-models",
            "--constraints",
            "constraints.json",
            "--seed",
            str(args.seed),
            "--quiet",
        ]
        run_cmd(compare_cmd, cwd=mode_dir)

        best_model = determine_best_model(mode_dir)
        print(f"Best model for {mode_name}: {best_model}")

        single_cmd = [
            "python",
            "../../metropolis_hastings_sampler.py",
            "--model",
            best_model,
            "--constraints",
            "constraints.json",
            "--seed",
            str(args.seed),
            "--quiet",
        ]
        run_cmd(single_cmd, cwd=mode_dir)

        constraints = json.loads((mode_dir / "constraints.json").read_text())
        best_summary = json.loads((mode_dir / "posterior_summary.json").read_text())
        best_summaries[mode_name] = best_summary
        write_report(mode_dir, constraints, best_summary)

    aggregate_summary(stage_dir, best_summaries)
    print(f"Stage-3 results written to {stage_dir}")


if __name__ == "__main__":
    main()
