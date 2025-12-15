#!/usr/bin/env python3
"""Stage 5: magnitude/redshift slicing stability tests for Secrest+22 dipole."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.table import Table

from secrest_utils import (
    CMB_B_DEG,
    CMB_L_DEG,
    SECREST_CATALOG_DEFAULT,
    SECREST_PUBLISHED,
    angle_deg,
    angular_difference_deg,
    apply_baseline_cuts,
    bootstrap_dipole,
    compute_dipole,
    lb_to_unitvec,
    unitvec_to_lb,
)

RED_SHIFT_CANDIDATES = [
    "z",
    "Z",
    "redshift",
    "z_spec",
    "zphot",
    "z_phot",
    "photoz",
    "z_qso",
]


def parse_edges(edge_str: str) -> List[float]:
    try:
        edges = [float(x) for x in edge_str.split(",") if x.strip()]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError("Edges must be comma-separated floats") from exc
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("At least two edges are required")
    return edges


def quantile_edges(values: np.ndarray, n_bins: int) -> List[float]:
    if n_bins < 1:
        raise ValueError("Number of quantile bins must be >=1")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    unique_edges = np.unique(edges)
    if len(unique_edges) < 2:
        raise ValueError("Quantile edges collapsed; data may be constant")
    return unique_edges.tolist()


def bin_mask(values: np.ndarray, low: float, high: float, is_last: bool) -> np.ndarray:
    if is_last:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)


def find_best_redshift_column(tbl: Table) -> Tuple[Optional[str], Optional[np.ndarray], Dict[str, float]]:
    fractions: Dict[str, float] = {}
    best_col: Optional[str] = None
    best_vals: Optional[np.ndarray] = None
    best_frac = -np.inf

    for cand in RED_SHIFT_CANDIDATES:
        for col in tbl.colnames:
            if col.lower() == cand.lower():
                vals = np.asarray(tbl[col], dtype=float)
                valid = np.isfinite(vals)
                frac = float(valid.mean()) if len(vals) > 0 else 0.0
                fractions[col] = frac
                if frac > best_frac and frac >= 0.2:
                    best_col = col
                    best_vals = vals
                    best_frac = frac
    return best_col, best_vals, fractions


def compute_bin_results(
    l_bin: np.ndarray,
    b_bin: np.ndarray,
    n_bootstrap: int,
    seed: int,
    label: str,
) -> Dict[str, Any]:
    amp, l_dip, b_dip, _ = compute_dipole(l_bin, b_bin)
    boot = bootstrap_dipole(
        l_bin, b_bin, n_bootstrap=n_bootstrap, seed=seed
    ).as_dict()

    rec_vec = lb_to_unitvec([l_dip], [b_dip])[0]
    cmb_vec = lb_to_unitvec([CMB_L_DEG], [CMB_B_DEG])[0]
    pub_vec = lb_to_unitvec([SECREST_PUBLISHED["l_deg"]], [SECREST_PUBLISHED["b_deg"]])[0]

    return {
        "label": label,
        "N": int(len(l_bin)),
        "dipole": {"amplitude": amp, "l_deg": l_dip, "b_deg": b_dip},
        "bootstrap": boot,
        "separations_deg": {
            "to_cmb": angle_deg(rec_vec, cmb_vec),
            "to_secrest": angle_deg(rec_vec, pub_vec),
        },
    }


def direction_stability(bin_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vecs = []
    weights = []
    labels = []
    for res in bin_results:
        l = res["dipole"]["l_deg"]
        b = res["dipole"]["b_deg"]
        if not np.isfinite(l) or not np.isfinite(b):
            continue
        vecs.append(lb_to_unitvec([l], [b])[0])
        weights.append(res.get("N", 0))
        labels.append(res.get("label", "bin"))

    if not vecs:
        return {
            "mean_direction": {"l_deg": float("nan"), "b_deg": float("nan")},
            "rms": {"l_deg": float("nan"), "b_deg": float("nan"), "sep_deg": float("nan")},
            "max": {"l_deg": float("nan"), "b_deg": float("nan"), "sep_deg": float("nan")},
            "per_bin": [],
        }

    weights_arr = np.asarray(weights, dtype=float)
    vecs_arr = np.asarray(vecs)
    mean_vec = np.average(vecs_arr, axis=0, weights=weights_arr)
    mean_l, mean_b = unitvec_to_lb(mean_vec)

    per_bin = []
    l_dev = []
    b_dev = []
    sep_dev = []
    for label, vec, w, res in zip(labels, vecs_arr, weights_arr, bin_results):
        l_bin = res["dipole"]["l_deg"]
        b_bin = res["dipole"]["b_deg"]
        l_delta = angular_difference_deg(l_bin, mean_l)
        b_delta = b_bin - mean_b
        sep = angle_deg(vec, mean_vec)
        per_bin.append({"label": label, "l_dev_deg": l_delta, "b_dev_deg": b_delta, "sep_deg": sep, "weight": float(w)})
        l_dev.append((l_delta, w))
        b_dev.append((b_delta, w))
        sep_dev.append((sep, w))

    def rms_weighted(pairs: Iterable[Tuple[float, float]]) -> float:
        vals = np.array([p[0] for p in pairs], dtype=float)
        wts = np.array([p[1] for p in pairs], dtype=float)
        denom = wts.sum()
        return float(np.sqrt(np.sum(wts * vals ** 2) / denom)) if denom > 0 else float("nan")

    def max_abs(pairs: Iterable[Tuple[float, float]]) -> float:
        vals = np.array([p[0] for p in pairs], dtype=float)
        return float(np.max(np.abs(vals))) if len(vals) > 0 else float("nan")

    return {
        "mean_direction": {"l_deg": mean_l, "b_deg": mean_b},
        "rms": {
            "l_deg": rms_weighted(l_dev),
            "b_deg": rms_weighted(b_dev),
            "sep_deg": rms_weighted(sep_dev),
        },
        "max": {
            "l_deg": max_abs(l_dev),
            "b_deg": max_abs(b_dev),
            "sep_deg": max_abs(sep_dev),
        },
        "per_bin": per_bin,
    }


def format_bin_label(low: float, high: float, is_last: bool) -> str:
    right = "]" if is_last else ")"
    return f"[{low:.3f}, {high:.3f}{right}"


def run_slicing(args: argparse.Namespace) -> Dict[str, Any]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tbl = Table.read(args.catalog)
    mask, cuts = apply_baseline_cuts(
        tbl,
        b_cut=args.b_cut,
        w1cov_min=args.w1cov_min,
        w1_max=args.w1_max,
    )

    l_all = np.asarray(tbl["l"], dtype=float)[mask]
    b_all = np.asarray(tbl["b"], dtype=float)[mask]
    w1_all = np.asarray(tbl["w1"], dtype=float)[mask]
    subset_idx: Optional[np.ndarray] = None

    quick_note = None
    if args.quick and len(l_all) > args.quick_count:
        rng = np.random.default_rng(args.seed)
        subset_idx = rng.choice(len(l_all), size=args.quick_count, replace=False)
        l_all = l_all[subset_idx]
        b_all = b_all[subset_idx]
        w1_all = w1_all[subset_idx]
        quick_note = f"Used random subset of {args.quick_count} sources for quick mode."

    # Baseline dipole
    base_amp, base_l, base_b, _ = compute_dipole(l_all, b_all)
    base_boot = bootstrap_dipole(
        l_all, b_all, n_bootstrap=args.n_bootstrap, seed=args.seed
    ).as_dict()
    base_vec = lb_to_unitvec([base_l], [base_b])[0]
    cmb_vec = lb_to_unitvec([CMB_L_DEG], [CMB_B_DEG])[0]
    pub_vec = lb_to_unitvec([SECREST_PUBLISHED["l_deg"]], [SECREST_PUBLISHED["b_deg"]])[0]

    baseline = {
        "N": int(len(l_all)),
        "dipole": {"amplitude": base_amp, "l_deg": base_l, "b_deg": base_b},
        "bootstrap": base_boot,
        "separations_deg": {
            "to_cmb": angle_deg(base_vec, cmb_vec),
            "to_secrest": angle_deg(base_vec, pub_vec),
        },
    }

    def build_bins(values: np.ndarray, edges_arg: Optional[str], quantiles: Optional[int]) -> List[Tuple[float, float]]:
        if edges_arg and quantiles:
            raise ValueError("Specify either explicit edges or quantile count, not both")
        if edges_arg:
            edges = parse_edges(edges_arg)
        else:
            q = 6 if quantiles is None else quantiles
            edges = quantile_edges(values, q)
        bins = []
        for i in range(len(edges) - 1):
            bins.append((edges[i], edges[i + 1], i == len(edges) - 2))
        return bins

    def process_bins(values: np.ndarray, edges_arg: Optional[str], quantiles: Optional[int]) -> Dict[str, Any]:
        bin_defs = build_bins(values, edges_arg, quantiles)
        bin_results = []
        for i, (low, high, is_last) in enumerate(bin_defs):
            mask_bin = bin_mask(values, low, high, is_last)
            l_bin = l_all[mask_bin]
            b_bin = b_all[mask_bin]
            res = compute_bin_results(
                l_bin,
                b_bin,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + i,
                label=format_bin_label(low, high, is_last),
            )
            bin_results.append(res)
        stability = direction_stability(bin_results)
        edges = [
            {"low": float(low), "high": float(high), "inclusive_high": bool(is_last)}
            for (low, high, is_last) in bin_defs
        ]
        return {"bins": bin_results, "stability": stability, "edges": edges}

    # Magnitude slicing
    w1_data = process_bins(w1_all, args.w1_bins, args.w1_quantiles)
    w1_output = {
        "catalog": args.catalog,
        "cuts": cuts,
        "quick_note": quick_note,
        "baseline": baseline,
        "bins": w1_data["bins"],
        "stability": w1_data["stability"],
        "w1_edges": w1_data["edges"],
        "w1_quantiles": args.w1_quantiles,
    }

    with open(outdir / "w1_bins.json", "w") as f:
        json.dump(w1_output, f, indent=2)

    # Redshift slicing
    masked_tbl = tbl[mask]
    z_col, _, z_fractions = find_best_redshift_column(masked_tbl)
    z_output: Dict[str, Any] = {
        "catalog": args.catalog,
        "quick_note": quick_note,
        "detected_column": z_col,
        "fractions": z_fractions,
    }

    if z_col is not None:
        z_vals_all = np.asarray(masked_tbl[z_col], dtype=float)
        if subset_idx is not None:
            z_vals_all = z_vals_all[subset_idx]
        z_data = process_bins(z_vals_all, args.z_bins, args.z_quantiles)
        z_output.update(
            {
                "available": True,
                "baseline": baseline,
                "bins": z_data["bins"],
                "stability": z_data["stability"],
                "z_edges": z_data["edges"],
                "z_quantiles": args.z_quantiles,
            }
        )
        with open(outdir / "z_bins.json", "w") as f:
            json.dump(z_output, f, indent=2)
    else:
        z_output.update(
            {
                "available": False,
                "reason": "No usable redshift column with >=20% finite values",
            }
        )
        with open(outdir / "z_bins.json", "w") as f:
            json.dump(z_output, f, indent=2)

    # Report
    report_path = outdir / "stage5_slicing_report.md"
    write_report(
        report_path,
        baseline,
        w1_output,
        z_output,
    )

    return {"w1": w1_output, "z": z_output}


def write_report(
    report_path: Path,
    baseline: Dict[str, Any],
    w1_output: Dict[str, Any],
    z_output: Dict[str, Any],
) -> None:
    lines = []
    lines.append("# Stage 5: Dipole stability vs magnitude/redshift\n")
    lines.append("## Baseline dipole\n")
    lines.append(
        f"N = {baseline['N']:,}, D = {baseline['dipole']['amplitude']:.5f}, "
        f"(l,b) = ({baseline['dipole']['l_deg']:.2f}, {baseline['dipole']['b_deg']:.2f}) deg"
    )
    lines.append(
        f"Separation from CMB: {baseline['separations_deg']['to_cmb']:.2f}°, "
        f"from Secrest: {baseline['separations_deg']['to_secrest']:.2f}°\n"
    )
    if w1_output.get("quick_note"):
        lines.append(f"_Quick mode_: {w1_output['quick_note']}\n")

    def render_table(bin_results: List[Dict[str, Any]], header: str) -> List[str]:
        tbl_lines = [header, "| Bin | N | D | (l,b) deg | sep CMB | sep Secrest | dir scatter (med) |", "|---|---|---|---|---|---|---|"]
        for res in bin_results:
            dip = res["dipole"]
            seps = res["separations_deg"]
            boot = res["bootstrap"]
            tbl_lines.append(
                f"| {res['label']} | {res['N']:,} | {dip['amplitude']:.5f} | "
                f"({dip['l_deg']:.2f}, {dip['b_deg']:.2f}) | "
                f"{seps['to_cmb']:.2f}° | {seps['to_secrest']:.2f}° | "
                f"{boot['direction_scatter_q50']:.2f}° |"
            )
        tbl_lines.append("")
        return tbl_lines

    lines.extend(render_table(w1_output["bins"], "## W1 slicing"))

    w1_stab = w1_output["stability"]
    lines.append(
        f"Direction stability (W1): mean (l,b)=({w1_stab['mean_direction']['l_deg']:.2f}, {w1_stab['mean_direction']['b_deg']:.2f}), "
        f"RMS sep={w1_stab['rms']['sep_deg']:.2f}°, max sep={w1_stab['max']['sep_deg']:.2f}°\n"
    )

    if z_output.get("available"):
        lines.extend(render_table(z_output.get("bins", []), "## Redshift slicing"))
        z_stab = z_output.get("stability", {})
        lines.append(
            f"Direction stability (z): mean (l,b)=({z_stab.get('mean_direction', {}).get('l_deg', float('nan')):.2f}, "
            f"{z_stab.get('mean_direction', {}).get('b_deg', float('nan')):.2f}), "
            f"RMS sep={z_stab.get('rms', {}).get('sep_deg', float('nan')):.2f}°, "
            f"max sep={z_stab.get('max', {}).get('sep_deg', float('nan')):.2f}°\n"
        )
    else:
        lines.append("## Redshift slicing\n")
        lines.append("Redshift slicing skipped: " + z_output.get("reason", "not available") + "\n")

    interpretation = "Directions are stable across bins." if w1_stab["rms"]["sep_deg"] < 10 else "Direction varies noticeably across bins."
    lines.append("## Interpretation\n")
    lines.append(
        interpretation
        + " Stability metrics consider RMS and max angular separation relative to the weighted mean direction across bins.\n"
    )

    report_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=SECREST_CATALOG_DEFAULT, help="Path to Secrest CatWISE AGN catalog")
    parser.add_argument("--outdir", default="./results/slicing", help="Output directory for slicing results")
    parser.add_argument("--b-cut", type=float, default=30.0, help="Galactic latitude cut |b| > b_cut")
    parser.add_argument("--w1cov-min", type=float, default=80.0, help="Minimum W1 coverage")
    parser.add_argument("--w1-max", type=float, default=16.4, help="Maximum W1 magnitude")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w1-quantiles", type=int, default=6, help="Number of W1 quantile bins")
    parser.add_argument("--w1-bins", type=str, default=None, help="Explicit W1 bin edges (comma-separated)")
    parser.add_argument("--z-quantiles", type=int, default=6, help="Number of redshift quantile bins")
    parser.add_argument("--z-bins", type=str, default=None, help="Explicit redshift bin edges (comma-separated)")
    parser.add_argument("--quick", action="store_true", help="Run on a subset for fast sanity checks")
    parser.add_argument("--quick-count", type=int, default=200_000, help="Subset size when using --quick")
    args = parser.parse_args()

    run_slicing(args)


if __name__ == "__main__":
    main()
