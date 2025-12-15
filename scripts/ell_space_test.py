"""Referee-grade ℓ-space dipole diagnostic for CatWISE-like catalogs."""

from __future__ import annotations

"""CLI entrypoint for the ℓ-space dipole diagnostic."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from quasar_dipole.ell_space import (
    SliceResult,
    build_counts_map,
    combine_masks,
    compute_overdensity,
    ensure_galactic_lat,
    galactic_mask,
    load_catalog,
    map_dipole_from_healpix,
    mask_from_fits,
    phase_randomize,
    plot_cl_curve,
    plot_direction_scatter,
    plot_null_hist,
    pseudo_cl,
    ra_scramble,
    save_healpix_map,
    save_json,
    save_npz,
    pixel_shuffle,
)
from secrest_utils import (
    SECREST_CATALOG_DEFAULT,
    angle_deg,
    apply_baseline_cuts,
    compute_dipole,
    lb_to_unitvec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(SECREST_CATALOG_DEFAULT))
    parser.add_argument("--outdir", type=Path, default=Path("results/ell_space"))
    parser.add_argument("--tag", type=str, default="demo", help="Run tag for output folder")
    parser.add_argument("--nside", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--bcut", type=float, default=30.0)
    parser.add_argument("--w1max", type=float, default=16.4)
    parser.add_argument("--ellmax", type=int, default=20)
    parser.add_argument("--null-n", type=int, default=200, help="Number of null draws per test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=6, help="Number of W1 quantile bins")
    parser.add_argument(
        "--w1-bins",
        type=str,
        default=None,
        help="Comma-separated W1 edges (overrides --n-bins)",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional HEALPix FITS mask to AND with the Galactic mask",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "quick"],
        help="Quick mode limits bins and null draws for smoke tests",
    )
    parser.add_argument(
        "--enable-phase-null",
        action="store_true",
        help="Enable harmonic phase randomization null (off by default)",
    )
    return parser.parse_args()


def determine_bins(w1: np.ndarray, args: argparse.Namespace) -> List[Tuple[float, float]]:
    if args.w1_bins:
        edges = [float(x) for x in args.w1_bins.split(",")]
        if sorted(edges) != edges:
            raise ValueError("W1 bin edges must be sorted")
    else:
        n_bins = 3 if args.mode == "quick" else args.n_bins
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = list(np.quantile(w1, quantiles))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bins.append((float(lo), float(hi)))
    return bins


def build_masks(nside: int, bcut: float, parent_counts: np.ndarray, user_mask: Optional[np.ndarray]) -> np.ndarray:
    gal_mask = galactic_mask(nside, bcut)
    exposure_mask = parent_counts == 0.0
    combined = combine_masks(gal_mask | exposure_mask, user_mask)
    return combined


def slice_indices(w1: np.ndarray, base_mask: np.ndarray, w1_range: Optional[Tuple[float, float]]):
    if w1_range is None:
        return base_mask
    lo, hi = w1_range
    return base_mask & (w1 >= lo) & (w1 <= hi)


def run_nulls(
    slice_name: str,
    ra: np.ndarray,
    dec: np.ndarray,
    l: np.ndarray,
    b: np.ndarray,
    counts_map: np.ndarray,
    mask_pix: np.ndarray,
    ellmax: int,
    nside: int,
    n_draws: int,
    rng: np.random.Generator,
    parent_counts: np.ndarray,
    enable_phase: bool,
    observed_dir: Tuple[float, float],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    observed_vec = lb_to_unitvec([observed_dir[0]], [observed_dir[1]])[0]
    delta_obs, _ = compute_overdensity(counts_map, mask_pix)
    cl_obs, _ = pseudo_cl(delta_obs, mask_pix, ellmax)
    dipole_fraction_obs = float(cl_obs[1] / cl_obs[1 : ellmax + 1].sum()) if cl_obs[1:].sum() > 0 else float("nan")

    null_specs = [
        ("ra_scramble", lambda: ra_scramble(ra, dec, np.ones_like(ra, dtype=bool), nside, rng)),
        (
            "pixel_shuffle",
            lambda: pixel_shuffle(parent_counts, len(ra), mask_pix, nside, rng),
        ),
    ]
    if enable_phase:
        null_specs.append(
            (
                "phase_randomization",
                lambda: phase_randomize(delta_obs, mask_pix, ellmax, rng),
            )
        )

    for null_name, generator in null_specs:
        c1_list: List[float] = []
        c2_list: List[float] = []
        c3_list: List[float] = []
        dip_frac_list: List[float] = []
        angle_list: List[float] = []

        for _ in range(n_draws):
            if null_name == "phase_randomization":
                delta = generator()
                cl_null, _ = pseudo_cl(delta, mask_pix, ellmax)
                counts_for_dipole = counts_map  # phase randomization keeps counts fixed
            else:
                counts_for_dipole = generator()
                delta, _ = compute_overdensity(counts_for_dipole, mask_pix)
                cl_null, _ = pseudo_cl(delta, mask_pix, ellmax)

            total_power = cl_null[1 : ellmax + 1].sum()
            dip_frac = float(cl_null[1] / total_power) if total_power > 0 else float("nan")
            dip_amp, dip_l, dip_b = map_dipole_from_healpix(counts_for_dipole, mask_pix, nside)
            null_vec = lb_to_unitvec([dip_l], [dip_b])[0]
            angle_list.append(angle_deg(observed_vec, null_vec))

            c1_list.append(float(cl_null[1]))
            c2_list.append(float(cl_null[2]))
            c3_list.append(float(cl_null[3]))
            dip_frac_list.append(dip_frac)

        c1_arr = np.asarray(c1_list)
        c2_arr = np.asarray(c2_list)
        c3_arr = np.asarray(c3_list)
        dip_frac_arr = np.asarray(dip_frac_list)
        angles_arr = np.asarray(angle_list)

        pvals = {
            "C1": float(np.mean(c1_arr >= cl_obs[1])),
            "C2": float(np.mean(c2_arr >= cl_obs[2])),
            "C3": float(np.mean(c3_arr >= cl_obs[3])),
            "dipole_fraction": float(np.mean(dip_frac_arr >= dipole_fraction_obs)),
        }

        results.append(
            {
                "name": null_name,
                "p_values": pvals,
                "c1_samples": c1_arr,
                "c2_samples": c2_arr,
                "c3_samples": c3_arr,
                "dipole_fraction": dip_frac_arr,
                "direction_angle_deg": angles_arr,
            }
        )

    return results


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.mode == "quick":
        args.ellmax = 10
        args.null_n = min(args.null_n, 50)

    outbase = args.outdir / args.tag
    outbase.mkdir(parents=True, exist_ok=True)

    tbl = load_catalog(args.catalog)
    ensure_galactic_lat(tbl)

    if "w1" not in tbl.colnames:
        raise ValueError("Catalog must include a 'w1' column for magnitude slicing.")

    mask, cuts = apply_baseline_cuts(tbl, b_cut=args.bcut, w1cov_min=80.0, w1_max=args.w1max)
    if not np.any(mask):
        raise RuntimeError("No sources survived the baseline cuts; aborting.")

    ra_all = np.asarray(tbl["ra"], dtype=float)
    dec_all = np.asarray(tbl["dec"], dtype=float)
    l_all = np.asarray(tbl["l"], dtype=float)
    b_all = np.asarray(tbl["b"], dtype=float)
    w1_all = np.asarray(tbl["w1"], dtype=float)

    parent_counts = build_counts_map(ra_all[mask], dec_all[mask], args.nside)
    user_mask = mask_from_fits(args.mask, args.nside)
    mask_pix = build_masks(args.nside, args.bcut, parent_counts, user_mask)

    slices: List[Tuple[str, Optional[Tuple[float, float]]]] = [("full", None)]
    w1_bins = determine_bins(w1_all[mask], args)
    for i, (lo, hi) in enumerate(w1_bins, start=1):
        slices.append((f"bin_{i}", (lo, hi)))

    slice_results: List[SliceResult] = []
    null_results: Dict[str, List[Dict[str, object]]] = {}
    dir_scatter_centers: List[float] = []
    dir_scatter_angles: List[float] = []

    for name, w1_range in slices:
        slice_mask = slice_indices(w1_all, mask, w1_range)
        n_slice = int(slice_mask.sum())
        if n_slice == 0:
            print(f"Slice {name} is empty after cuts; skipping.")
            continue

        ra = ra_all[slice_mask]
        dec = dec_all[slice_mask]
        l = l_all[slice_mask]
        b = b_all[slice_mask]
        counts = build_counts_map(ra, dec, args.nside)
        delta, mean_n = compute_overdensity(counts, mask_pix)
        cl, f_sky = pseudo_cl(delta, mask_pix, args.ellmax)

        cl_slice = [float(x) for x in cl[: args.ellmax + 1]]
        total_power = cl[1 : args.ellmax + 1].sum()
        dip_frac = float(cl[1] / total_power) if total_power > 0 else float("nan")
        quad_ratio = float(cl[2] / cl[1]) if cl[1] != 0 else float("nan")
        oct_ratio = float(cl[3] / cl[1]) if cl[1] != 0 else float("nan")

        cat_amp, cat_l, cat_b, cat_vec = compute_dipole(l, b)
        map_amp, map_l, map_b = map_dipole_from_healpix(counts, mask_pix, args.nside)

        slice_result = SliceResult(
            name=name,
            w1_range=w1_range,
            n_sources=n_slice,
            f_sky=f_sky,
            mean_per_pixel=mean_n,
            cl=cl_slice,
            dipole_fraction=dip_frac,
            quadrupole_to_dipole=quad_ratio,
            octupole_to_dipole=oct_ratio,
            catalog_dipole_amp=cat_amp,
            catalog_dipole_l_deg=cat_l,
            catalog_dipole_b_deg=cat_b,
            map_dipole_amp=map_amp,
            map_dipole_l_deg=map_l,
            map_dipole_b_deg=map_b,
        )
        slice_results.append(slice_result)

        slice_dir = outbase / name
        slice_dir.mkdir(parents=True, exist_ok=True)
        save_healpix_map(slice_dir / "counts.fits", counts)
        save_healpix_map(slice_dir / "delta.fits", delta)
        save_healpix_map(slice_dir / "mask.fits", mask_pix.astype(float))
        plot_cl_curve(cl, args.ellmax, f"{name} C_ell", slice_dir / "cl.png")

        nulls = run_nulls(
            name,
            ra,
            dec,
            l,
            b,
            counts,
            mask_pix,
            args.ellmax,
            args.nside,
            args.null_n,
            rng,
            parent_counts,
            args.enable_phase_null,
            (map_l, map_b),
        )
        null_results[name] = nulls

        for null_entry in nulls:
            save_npz(
                slice_dir / f"{null_entry['name']}_nulls.npz",
                c1=null_entry["c1_samples"],
                c2=null_entry["c2_samples"],
                c3=null_entry["c3_samples"],
                dipole_fraction=null_entry["dipole_fraction"],
                direction_angle_deg=null_entry["direction_angle_deg"],
            )
            plot_null_hist(
                cl[1],
                null_entry["c1_samples"],
                f"{name} {null_entry['name']} C1",
                slice_dir / f"{null_entry['name']}_c1_hist.png",
            )

        if w1_range is not None:
            center = 0.5 * (w1_range[0] + w1_range[1])
            dir_scatter_centers.append(center)
            dir_scatter_angles.append(angle_deg(lb_to_unitvec([map_l], [map_b])[0], lb_to_unitvec([slice_results[0].map_dipole_l_deg], [slice_results[0].map_dipole_b_deg])[0]))

    summary = {
        "catalog": str(args.catalog),
        "cuts": cuts,
        "nside": args.nside,
        "ellmax": args.ellmax,
        "null_n": args.null_n,
        "slices": [s.to_dict() for s in slice_results],
        "nulls": {
            name: [
                {
                    "name": nres["name"],
                    "p_values": nres["p_values"],
                    "direction_angle_deg_stats": {
                        "median": float(np.median(nres["direction_angle_deg"])),
                        "p95": float(np.percentile(nres["direction_angle_deg"], 95)),
                    },
                }
                for nres in nulls
            ]
            for name, nulls in null_results.items()
        },
    }
    save_json(outbase / "ell_space_summary.json", summary)

    if dir_scatter_centers:
        plot_direction_scatter(
            dir_scatter_centers,
            dir_scatter_angles,
            "Dipole direction drift vs W1 bin",
            outbase / "direction_scatter.png",
        )

    report_path = outbase / "report.md"
    with open(report_path, "w") as f:
        f.write("# ℓ-space dipole diagnostic\n\n")
        f.write(f"Catalog: `{args.catalog}`\\n\n")
        f.write(f"NSIDE={args.nside}, ellmax={args.ellmax}, null draws={args.null_n}\n\n")
        f.write("## Slices\n\n")
        for s in slice_results:
            f.write(
                f"- **{s.name}**: N={s.n_sources}, f_sky={s.f_sky:.3f}, "
                f"C1={s.cl[1]:.3e}, dipole fraction={s.dipole_fraction:.3f}, "
                f"quad/dip={s.quadrupole_to_dipole:.3f}, oct/dip={s.octupole_to_dipole:.3f}\\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "- Pseudo-$C_\ell$ values divide by $f_{sky}$ but still retain cut-sky mode mixing; interpret ratios rather than absolute amplitudes.\n"
        )
        f.write(
            "- If RA-scramble p-values approach unity for $C_1$ while $C_2$/$C_3$ remain null, the dipole likely arises from declination-dependent selection.\n"
        )
        f.write(
            "- A high dipole fraction across bins with declining amplitude but stable direction supports a genuine large-scale structure dipole diluted at faint magnitudes.\n"
        )
        f.write(
            "- Growing $C_2$/$C_3$ or large direction drift with depth suggests multiple anisotropic components or residual systematics.\n"
        )
        f.write("\nNull-test p-values are summarized in `ell_space_summary.json`.\n")

    print(f"Wrote summary to {report_path}")


if __name__ == "__main__":
    main()
