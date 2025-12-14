#!/usr/bin/env python3
"""
Bootstrap WAIC uncertainty analysis for model comparison outputs.

Given one or more model_comparison.json files, this script loads the
per-model posterior_samples_*.csv files, recomputes WAIC totals using the
component-wise definition, and estimates uncertainty via bootstrap
resampling of posterior draws.
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap WAIC uncertainty analysis from model comparison outputs"
    )
    parser.add_argument(
        "comparison_files",
        nargs="*",
        default=["model_comparison.json"],
        help="model_comparison.json files to analyze (default: model_comparison.json)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for bootstrap resampling (default: 123)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output filename (only used when analyzing a single comparison file)",
    )
    return parser.parse_args()


def log_mean_exp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.mean(np.exp(x - m))))


def compute_waic_breakdown(
    ll_components: np.ndarray, component_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, float]]:
    breakdown: Dict[str, Dict[str, float]] = {}
    waic_components: List[float] = []

    for name, col in zip(component_names, ll_components.T):
        lppd_k = log_mean_exp(col)
        p_waic_k = float(np.var(col, ddof=0))
        waic_k = -2.0 * (lppd_k - p_waic_k)
        breakdown[name] = {"lppd": float(lppd_k), "p_waic": p_waic_k, "waic": waic_k}
        waic_components.append(waic_k)

    waic_components_arr = np.array(waic_components, dtype=float)
    waic_total = float(np.sum(waic_components_arr))
    leave_one_out = {
        name: float(np.sum(np.delete(waic_components_arr, i)))
        for i, name in enumerate(component_names)
    }
    return breakdown, waic_total, leave_one_out


def compute_waic_total(ll_components: np.ndarray) -> float:
    totals: List[float] = []
    for col in ll_components.T:
        lppd_k = log_mean_exp(col)
        p_waic_k = float(np.var(col, ddof=0))
        totals.append(-2.0 * (lppd_k - p_waic_k))
    return float(np.sum(totals))


def bootstrap_waic_totals(
    ll_components: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    n_samples = ll_components.shape[0]
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        resampled = ll_components[idx]
        boot[i] = compute_waic_total(resampled)
    return boot


def load_ll_components(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No headers found in {csv_path}")

        ll_cols = [c for c in reader.fieldnames if c.startswith("ll_")]
        if not ll_cols:
            raise ValueError(f"No ll_* columns found in {csv_path}")

        rows: List[List[float]] = []
        for row in reader:
            rows.append([float(row[c]) for c in ll_cols])

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    ll_components = np.array(rows, dtype=float)
    component_names = [c[3:] for c in ll_cols]
    return ll_components, component_names


def summarize_bootstrap(samples: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p5": float(np.percentile(samples, 5)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
    }


def analyze_comparison_file(
    comparison_path: str,
    n_bootstrap: int,
    seed: int,
    out_override: Optional[str],
    allow_override: bool,
) -> None:
    if not os.path.exists(comparison_path):
        print(f"Warning: comparison file not found: {comparison_path}")
        return

    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison_data = json.load(f)

    rng = np.random.default_rng(seed)
    comparison_dir = os.path.dirname(os.path.abspath(comparison_path)) or "."

    model_results = []
    model_names: List[str] = []
    bootstrap_totals: List[np.ndarray] = []

    for entry in comparison_data:
        mode = entry.get("model") or entry.get("model_mode")
        if not mode:
            print("Warning: skipping entry without model name")
            continue

        samples_rel = entry.get("samples_csv") or f"posterior_samples_{mode}.csv"
        samples_path = os.path.join(comparison_dir, samples_rel)
        if not os.path.exists(samples_path):
            print(f"Warning: samples CSV not found for {mode}: {samples_path}")
            continue

        try:
            ll_components, comp_names = load_ll_components(samples_path)
        except ValueError as exc:
            print(f"Warning: {exc}")
            continue

        breakdown, waic_total, leave_one_out = compute_waic_breakdown(
            ll_components, comp_names
        )
        boot = bootstrap_waic_totals(ll_components, n_bootstrap, rng)

        model_names.append(mode)
        bootstrap_totals.append(boot)
        model_results.append(
            {
                "model": mode,
                "components": comp_names,
                "waic_total": waic_total,
                "waic_components": breakdown,
                "waic_leave_one_out": leave_one_out,
                "bootstrap_summary": summarize_bootstrap(boot),
            }
        )

    if not model_results:
        print(f"Warning: no models analyzed for {comparison_path}")
        return

    boot_matrix = np.vstack(bootstrap_totals)
    best_counts = np.bincount(np.argmin(boot_matrix, axis=0), minlength=len(model_names))
    best_probabilities = {
        model_names[i]: float(best_counts[i] / np.sum(best_counts))
        for i in range(len(model_names))
    }

    delta_wall_full = None
    if "wall_only" in model_names and "full" in model_names:
        wall_idx = model_names.index("wall_only")
        full_idx = model_names.index("full")
        delta_samples = boot_matrix[wall_idx] - boot_matrix[full_idx]
        delta_wall_full = {
            "mean": float(np.mean(delta_samples)),
            "std": float(np.std(delta_samples)),
            "p5": float(np.percentile(delta_samples, 5)),
            "p50": float(np.percentile(delta_samples, 50)),
            "p95": float(np.percentile(delta_samples, 95)),
        }

    output = {
        "comparison_file": os.path.abspath(comparison_path),
        "bootstrap_samples": n_bootstrap,
        "seed": seed,
        "models": {r["model"]: r for r in model_results},
        "best_model_probability": best_probabilities,
        "delta_wall_only_vs_full": delta_wall_full,
    }

    base = os.path.splitext(os.path.basename(comparison_path))[0]
    if out_override and allow_override:
        out_path = out_override
    elif os.path.basename(comparison_path) == "model_comparison.json":
        out_path = os.path.join(comparison_dir, "waic_uncertainty_report.json")
    else:
        out_path = os.path.join(comparison_dir, f"{base}__waic_uncertainty.json")

    if out_override and not allow_override:
        print(
            "Warning: --out was provided but multiple comparison files are being analyzed;"
        )
        print("         using derived filenames for each file instead.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nComparison: {comparison_path}")
    for r in model_results:
        print(
            f"  {r['model']:<12} WAIC={r['waic_total']:.3f}  "
            f"bootstrap mean={r['bootstrap_summary']['mean']:.3f}"
        )
    print("  Best model probabilities:")
    for name, prob in best_probabilities.items():
        print(f"    {name:<12} {prob:.3f}")

    if delta_wall_full is not None:
        print("  ΔWAIC (wall_only - full):")
        print(
            "    mean={mean:.3f} std={std:.3f}  "
            "p5={p5:.3f} p50={p50:.3f} p95={p95:.3f}".format(**delta_wall_full)
        )

    print(f"  Saved uncertainty report to: {out_path}\n")


if __name__ == "__main__":
    args = parse_args()
    allow_override = len(args.comparison_files) == 1
    for comp_file in args.comparison_files:
        analyze_comparison_file(
            comp_file, args.bootstrap, args.seed, args.out, allow_override
        )
