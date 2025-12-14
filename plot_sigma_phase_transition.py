#!/usr/bin/env python3
"""
Plot helper for visualizing the sigma_dir sweep phase transition.

Reads sweep_results.json (produced by --sweep-sigma-dir) and plots the
probability proxy that the full model is best as a function of sigma_dir.
"""

import argparse
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_sweep(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("sweep_results.json must contain a list of entries")
    return data


def plot_phase_transition(sweep: List[dict], output: str, show: bool) -> None:
    if not sweep:
        raise ValueError("No sweep results found.")

    sigmas = np.array([float(entry["sigma_qso_dir"]) for entry in sweep], dtype=float)
    flags = np.array([1.0 if entry.get("best_model") == "full" else 0.0 for entry in sweep], dtype=float)

    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    flags = flags[order]

    plt.figure(figsize=(7, 4.5))
    plt.plot(sigmas, flags, "o-", label="full best (1=yes, 0=no)")

    if len(np.unique(flags)) > 1:
        # simple logistic fit using logit regression (no extra dependencies)
        eps = 1e-3
        y = np.clip(flags, eps, 1.0 - eps)
        logit = np.log(y / (1.0 - y))
        coef = np.polyfit(sigmas, logit, 1)
        xs = np.linspace(sigmas.min(), sigmas.max(), 200)
        ys = 1.0 / (1.0 + np.exp(-(coef[0] * xs + coef[1])))
        plt.plot(xs, ys, "--", label="logistic fit")

    plt.xlabel(r"$\sigma_{\mathrm{dir}}$ (deg)")
    plt.ylabel("P(full best)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved figure to {output}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sigma_dir sweep phase transition")
    parser.add_argument("--input", type=str, default="sweep_results.json", help="Path to sweep_results.json")
    parser.add_argument("--output", type=str, default="sigma_phase_transition.png", help="Output plot filename")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    sweep = load_sweep(args.input)
    plot_phase_transition(sweep, args.output, args.show)


if __name__ == "__main__":
    main()
