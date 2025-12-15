#!/usr/bin/env python3
"""
Stage 4: Posterior Predictive Checks in Sky Space

Tests whether wall_only vs full models produce distinguishable sky patterns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import healpy as hp

# Constants
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
N_SOURCES = 1_400_000
N_DRAWS = 200
B_CUT = 30.0  # Galactic latitude mask
SEED = 42
C_KMS = 299792.458
V_CMB_KMS = 369.82

# CMB dipole direction (Galactic)
L_CMB, B_CMB = 264.021, 48.253

# CatWISE observed dipole
CATWISE_L, CATWISE_B = 238.2, 28.8
CATWISE_D = 0.0154


def lb_to_vec(l_deg: float, b_deg: float) -> np.ndarray:
    """Convert Galactic (l,b) to unit vector."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])


def vec_to_lb(v: np.ndarray) -> Tuple[float, float]:
    """Convert unit vector to Galactic (l,b)."""
    v = v / np.linalg.norm(v)
    l = np.degrees(np.arctan2(v[1], v[0])) % 360
    b = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    return l, b


def angular_sep(l1: float, b1: float, l2: float, b2: float) -> float:
    """Angular separation in degrees."""
    v1, v2 = lb_to_vec(l1, b1), lb_to_vec(l2, b2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


def build_cmb_basis(n_cmb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build orthonormal basis centered on CMB direction."""
    z_hat = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(z_hat, n_cmb)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_cmb, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def alpha_phi_to_unitvec(alpha_deg: float, phi_deg: float,
                         n_axis: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Convert axis-centric spherical coords to unit vector:
      n = cos(alpha)*n_axis + sin(alpha)*(cos(phi)*e1 + sin(phi)*e2)
    """
    alpha, phi = np.radians(alpha_deg), np.radians(phi_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cp, sp = np.cos(phi), np.sin(phi)
    return ca * n_axis + sa * (cp * e1 + sp * e2)


# Precompute CMB basis
n_CMB = lb_to_vec(L_CMB, B_CMB)
v_CMB_vec = V_CMB_KMS * n_CMB
e1_CMB, e2_CMB = build_cmb_basis(n_CMB)


def get_galactic_mask(nside: int, b_cut: float) -> np.ndarray:
    """Create mask for |b| > b_cut."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    b = 90 - np.degrees(theta)
    return np.abs(b) > b_cut


def compute_model_vectors(params: Dict, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute D_Q_vec and D_R_vec from posterior parameters.
    Replicates the sampler's compute_model_vectors function.
    """
    D_wall = params['D_wall']
    alpha_wall = params['alpha_wall_deg']
    phi_wall = params['phi_wall_deg']
    A_Q = params.get('A_Q', 1.0)
    A_R = params.get('A_R', 1.0)

    if model == 'wall_only':
        v_drift = 0.0
        l_drift, b_drift = 0.0, 0.0
    else:
        v_drift = params['v_drift_kms']
        l_drift = params['l_drift_deg']
        b_drift = params['b_drift_deg']

    # Drift velocity vector
    n_drift = lb_to_vec(l_drift, b_drift) if v_drift > 0 else np.zeros(3)
    v_drift_vec = v_drift * n_drift

    # Total velocity
    v_tot_vec = v_CMB_vec + v_drift_vec
    v_tot_mag = np.linalg.norm(v_tot_vec)
    beta_tot = v_tot_mag / C_KMS
    n_vtot = v_tot_vec / v_tot_mag if v_tot_mag > 1e-10 else n_CMB

    # Kinematic dipoles
    D_kin_Q = A_Q * beta_tot
    D_kin_R = A_R * beta_tot
    D_kin_Q_vec = D_kin_Q * n_vtot
    D_kin_R_vec = D_kin_R * n_vtot

    # Wall contribution
    if abs(D_wall) > 1e-10:
        n_wall = alpha_phi_to_unitvec(alpha_wall, phi_wall, n_CMB, e1_CMB, e2_CMB)
        D_wall_vec = D_wall * n_wall
    else:
        D_wall_vec = np.zeros(3)

    D_Q_vec = D_kin_Q_vec + D_wall_vec
    D_R_vec = D_kin_R_vec + D_wall_vec

    return D_Q_vec, D_R_vec


def generate_dipolar_sky(D: float, l_dip: float, b_dip: float,
                         n_sources: int, mask: np.ndarray,
                         rng: np.random.Generator, nside: int) -> np.ndarray:
    """
    Generate mock sky with dipolar modulation.
    P(n_hat) ∝ 1 + D * (n_hat · d_hat)
    """
    d_vec = lb_to_vec(l_dip, b_dip)
    npix = hp.nside2npix(nside)

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    cos_angle = x * d_vec[0] + y * d_vec[1] + z * d_vec[2]
    prob = 1 + D * cos_angle
    prob[~mask] = 0
    prob = np.maximum(prob, 0)
    prob /= prob.sum()

    pix_idx = rng.choice(npix, size=n_sources, p=prob)
    counts = np.bincount(pix_idx, minlength=npix).astype(float)
    return counts


def measure_dipole_from_map(counts: np.ndarray, mask: np.ndarray, nside: int) -> Tuple[float, float, float]:
    """Measure dipole from HEALPix count map."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    w = counts.copy()
    w[~mask] = 0
    W = w.sum()

    if W == 0:
        return np.nan, np.nan, np.nan

    sum_vec = np.array([(w * x).sum(), (w * y).sum(), (w * z).sum()])
    D = 3.0 * np.linalg.norm(sum_vec) / W
    l_dip, b_dip = vec_to_lb(sum_vec)
    return D, l_dip, b_dip


def compute_hemispheric_asymmetry(counts: np.ndarray, mask: np.ndarray, nside: int) -> Dict[str, float]:
    """Compute hemispheric asymmetry metrics."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    b = 90 - np.degrees(theta)

    north = (b > 0) & mask
    south = (b < 0) & mask

    N_north = counts[north].sum()
    N_south = counts[south].sum()
    ns_asymmetry = (N_north - N_south) / (N_north + N_south) if (N_north + N_south) > 0 else 0

    D, l_dip, b_dip = measure_dipole_from_map(counts, mask, nside)
    d_vec = lb_to_vec(l_dip, b_dip)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    cos_angle = x * d_vec[0] + y * d_vec[1] + z * d_vec[2]

    forward = (cos_angle > 0) & mask
    backward = (cos_angle < 0) & mask

    N_forward = counts[forward].sum()
    N_backward = counts[backward].sum()
    dipole_contrast = (N_forward - N_backward) / (N_forward + N_backward) if (N_forward + N_backward) > 0 else 0

    return {
        'ns_asymmetry': float(ns_asymmetry),
        'dipole_contrast': float(dipole_contrast),
        'N_north': float(N_north),
        'N_south': float(N_south)
    }


def run_posterior_predictive(model: str, samples_path: Path, outdir: Path,
                             rng: np.random.Generator) -> Dict[str, Any]:
    """Run posterior predictive checks for one model."""
    print(f"\n{'='*60}")
    print(f"Processing model: {model}")
    print(f"{'='*60}")

    df = pd.read_csv(samples_path)
    n_samples = len(df)

    if n_samples > N_DRAWS:
        idx = rng.choice(n_samples, size=N_DRAWS, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    mask = get_galactic_mask(NSIDE, B_CUT)

    input_dipoles = []
    measured_dipoles = []
    hemispheric_stats = []
    tracer_q_dipoles = []
    tracer_r_dipoles = []

    print(f"Generating {len(df)} posterior predictive realizations...")

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(df)}")

        params = row.to_dict()

        # Compute predicted dipole vectors
        D_Q_vec, D_R_vec = compute_model_vectors(params, model)

        D_Q = np.linalg.norm(D_Q_vec)
        D_R = np.linalg.norm(D_R_vec)

        if D_Q > 1e-10:
            l_Q, b_Q = vec_to_lb(D_Q_vec)
        else:
            l_Q, b_Q = CATWISE_L, CATWISE_B

        if D_R > 1e-10:
            l_R, b_R = vec_to_lb(D_R_vec)
        else:
            l_R, b_R = CATWISE_L, CATWISE_B

        input_dipoles.append({'D': D_Q, 'l': l_Q, 'b': b_Q})
        tracer_q_dipoles.append({'D': D_Q, 'l': l_Q, 'b': b_Q})
        tracer_r_dipoles.append({'D': D_R, 'l': l_R, 'b': b_R})

        # Generate mock sky
        counts = generate_dipolar_sky(D_Q, l_Q, b_Q, N_SOURCES, mask, rng, NSIDE)

        # Measure dipole from mock sky
        D_meas, l_meas, b_meas = measure_dipole_from_map(counts, mask, NSIDE)
        measured_dipoles.append({'D': D_meas, 'l': l_meas, 'b': b_meas})

        # Hemispheric asymmetry
        hem_stats = compute_hemispheric_asymmetry(counts, mask, NSIDE)
        hemispheric_stats.append(hem_stats)

    return {
        'model': model,
        'n_realizations': len(df),
        'input_dipoles': input_dipoles,
        'measured_dipoles': measured_dipoles,
        'hemispheric_stats': hemispheric_stats,
        'tracer_q_dipoles': tracer_q_dipoles,
        'tracer_r_dipoles': tracer_r_dipoles
    }


def compute_direction_stability(results: Dict) -> Dict[str, Any]:
    """Task 2: Directional stability analysis."""
    input_dip = results['input_dipoles']
    meas_dip = results['measured_dipoles']

    input_meas_seps = []
    for inp, meas in zip(input_dip, meas_dip):
        sep = angular_sep(inp['l'], inp['b'], meas['l'], meas['b'])
        input_meas_seps.append(sep)

    cmb_seps = [angular_sep(d['l'], d['b'], L_CMB, B_CMB) for d in meas_dip]
    catwise_seps = [angular_sep(d['l'], d['b'], CATWISE_L, CATWISE_B) for d in meas_dip]

    l_vals = [d['l'] for d in input_dip]
    b_vals = [d['b'] for d in input_dip]

    # Input dipole statistics
    input_cmb_seps = [angular_sep(d['l'], d['b'], L_CMB, B_CMB) for d in input_dip]
    input_catwise_seps = [angular_sep(d['l'], d['b'], CATWISE_L, CATWISE_B) for d in input_dip]

    return {
        'input_meas_separation': {
            'mean': float(np.mean(input_meas_seps)),
            'std': float(np.std(input_meas_seps)),
            'median': float(np.median(input_meas_seps)),
            'q16': float(np.percentile(input_meas_seps, 16)),
            'q84': float(np.percentile(input_meas_seps, 84))
        },
        'input_cmb_separation': {
            'mean': float(np.mean(input_cmb_seps)),
            'std': float(np.std(input_cmb_seps)),
            'median': float(np.median(input_cmb_seps))
        },
        'input_catwise_separation': {
            'mean': float(np.mean(input_catwise_seps)),
            'std': float(np.std(input_catwise_seps)),
            'median': float(np.median(input_catwise_seps))
        },
        'measured_cmb_separation': {
            'mean': float(np.mean(cmb_seps)),
            'std': float(np.std(cmb_seps)),
            'median': float(np.median(cmb_seps))
        },
        'measured_catwise_separation': {
            'mean': float(np.mean(catwise_seps)),
            'std': float(np.std(catwise_seps)),
            'median': float(np.median(catwise_seps))
        },
        'direction_scatter': {
            'l_std': float(np.std(l_vals)),
            'b_std': float(np.std(b_vals)),
            'l_mean': float(np.mean(l_vals)),
            'b_mean': float(np.mean(b_vals)),
            'l_range': [float(np.min(l_vals)), float(np.max(l_vals))],
            'b_range': [float(np.min(b_vals)), float(np.max(b_vals))]
        }
    }


def compute_hemispheric_test(results: Dict) -> Dict[str, Any]:
    """Task 3: Hemispheric asymmetry analysis."""
    hem_stats = results['hemispheric_stats']

    ns_asym = [h['ns_asymmetry'] for h in hem_stats]
    dip_contrast = [h['dipole_contrast'] for h in hem_stats]

    real_ns_asym = 0.0  # CatWISE is approximately balanced after |b|>30 cut

    return {
        'ns_asymmetry': {
            'mean': float(np.mean(ns_asym)),
            'std': float(np.std(ns_asym)),
            'median': float(np.median(ns_asym)),
            'q16': float(np.percentile(ns_asym, 16)),
            'q84': float(np.percentile(ns_asym, 84))
        },
        'dipole_contrast': {
            'mean': float(np.mean(dip_contrast)),
            'std': float(np.std(dip_contrast)),
            'median': float(np.median(dip_contrast)),
            'q16': float(np.percentile(dip_contrast, 16)),
            'q84': float(np.percentile(dip_contrast, 84))
        },
        'observed_ns_asymmetry': real_ns_asym
    }


def compute_tracer_consistency(results: Dict) -> Dict[str, Any]:
    """Task 4: Tracer-independence test."""
    q_dip = results['tracer_q_dipoles']
    r_dip = results['tracer_r_dipoles']

    alignments = []
    amp_ratios = []

    for q, r in zip(q_dip, r_dip):
        sep = angular_sep(q['l'], q['b'], r['l'], r['b'])
        alignments.append(sep)
        if r['D'] > 0:
            amp_ratios.append(q['D'] / r['D'])

    return {
        'qr_alignment': {
            'mean': float(np.mean(alignments)),
            'std': float(np.std(alignments)),
            'median': float(np.median(alignments)),
            'q16': float(np.percentile(alignments, 16)),
            'q84': float(np.percentile(alignments, 84)),
            'max': float(np.max(alignments))
        },
        'amplitude_ratio': {
            'mean': float(np.mean(amp_ratios)),
            'std': float(np.std(amp_ratios)),
            'median': float(np.median(amp_ratios))
        },
        'interpretation': (
            'wall_only: perfect alignment (0 deg), stable ratio. '
            'full: increased scatter if drift adds tracer-dependent kinematic component.'
        )
    }


def compute_ppc_pvalues(results: Dict) -> Dict[str, Any]:
    """Task 5: Posterior predictive p-values."""
    input_dip = results['input_dipoles']
    meas_dip = results['measured_dipoles']
    hem_stats = results['hemispheric_stats']

    obs_D = CATWISE_D
    obs_cmb_sep = angular_sep(CATWISE_L, CATWISE_B, L_CMB, B_CMB)

    sim_D = [d['D'] for d in input_dip]
    sim_cmb_sep = [angular_sep(d['l'], d['b'], L_CMB, B_CMB) for d in input_dip]
    sim_catwise_sep = [angular_sep(d['l'], d['b'], CATWISE_L, CATWISE_B) for d in input_dip]

    p_amplitude = float(np.mean(np.array(sim_D) > obs_D))
    p_cmb_sep = float(np.mean(np.array(sim_cmb_sep) > obs_cmb_sep))
    p_direction = float(np.mean(np.array(sim_catwise_sep) < 10))

    return {
        'amplitude': {
            'observed': obs_D,
            'sim_mean': float(np.mean(sim_D)),
            'sim_std': float(np.std(sim_D)),
            'p_value': p_amplitude,
            'interpretation': 'P(D_sim > D_obs)'
        },
        'cmb_separation': {
            'observed': obs_cmb_sep,
            'sim_mean': float(np.mean(sim_cmb_sep)),
            'sim_std': float(np.std(sim_cmb_sep)),
            'p_value': p_cmb_sep,
            'interpretation': 'P(sep_sim > sep_obs)'
        },
        'direction_recovery': {
            'fraction_within_10deg': p_direction,
            'sim_catwise_sep_mean': float(np.mean(sim_catwise_sep)),
            'sim_catwise_sep_std': float(np.std(sim_catwise_sep))
        }
    }


def main():
    outdir = Path('results/stage4')
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    models = {
        'wall_only': outdir / 'wall_only_samples.csv',
        'full': outdir / 'full_samples.csv'
    }

    all_results = {}

    for model, samples_path in models.items():
        if not samples_path.exists():
            print(f"WARNING: {samples_path} not found, skipping {model}")
            continue

        results = run_posterior_predictive(model, samples_path, outdir, rng)

        dir_stability = compute_direction_stability(results)
        with open(outdir / f'{model}_direction_stability.json', 'w') as f:
            json.dump(dir_stability, f, indent=2)
        print(f"\n{model} direction stability:")
        print(f"  Input-measured sep: {dir_stability['input_meas_separation']['mean']:.2f} ± {dir_stability['input_meas_separation']['std']:.2f} deg")
        print(f"  Input direction scatter: l_std={dir_stability['direction_scatter']['l_std']:.1f}, b_std={dir_stability['direction_scatter']['b_std']:.1f}")
        print(f"  Mean sep from CatWISE: {dir_stability['input_catwise_separation']['mean']:.2f} deg")

        hem_test = compute_hemispheric_test(results)
        with open(outdir / f'{model}_hemispheric_test.json', 'w') as f:
            json.dump(hem_test, f, indent=2)
        print(f"\n{model} hemispheric asymmetry:")
        print(f"  N-S asymmetry: {hem_test['ns_asymmetry']['mean']:.4f} ± {hem_test['ns_asymmetry']['std']:.4f}")

        tracer_test = compute_tracer_consistency(results)
        with open(outdir / f'{model}_tracer_consistency.json', 'w') as f:
            json.dump(tracer_test, f, indent=2)
        print(f"\n{model} tracer consistency:")
        print(f"  Q-R alignment: {tracer_test['qr_alignment']['mean']:.2f} ± {tracer_test['qr_alignment']['std']:.2f} deg")
        print(f"  Amplitude ratio: {tracer_test['amplitude_ratio']['mean']:.2f} ± {tracer_test['amplitude_ratio']['std']:.2f}")

        ppc = compute_ppc_pvalues(results)
        with open(outdir / f'{model}_ppc.json', 'w') as f:
            json.dump(ppc, f, indent=2)
        print(f"\n{model} PPC p-values:")
        print(f"  Amplitude: p = {ppc['amplitude']['p_value']:.3f} (sim_mean={ppc['amplitude']['sim_mean']:.4f}, obs={ppc['amplitude']['observed']:.4f})")
        print(f"  CMB separation: p = {ppc['cmb_separation']['p_value']:.3f}")

        all_results[model] = {
            'direction_stability': dir_stability,
            'hemispheric_test': hem_test,
            'tracer_consistency': tracer_test,
            'ppc': ppc
        }

    with open(outdir / 'stage4_combined.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    if 'wall_only' in all_results and 'full' in all_results:
        wo = all_results['wall_only']
        fu = all_results['full']

        print(f"\n{'Metric':<40} {'wall_only':>12} {'full':>12}")
        print("-"*70)
        print(f"{'Direction: l scatter (deg)':<40} {wo['direction_stability']['direction_scatter']['l_std']:>12.1f} {fu['direction_stability']['direction_scatter']['l_std']:>12.1f}")
        print(f"{'Direction: b scatter (deg)':<40} {wo['direction_stability']['direction_scatter']['b_std']:>12.1f} {fu['direction_stability']['direction_scatter']['b_std']:>12.1f}")
        print(f"{'Sep from CatWISE (deg)':<40} {wo['direction_stability']['input_catwise_separation']['mean']:>12.1f} {fu['direction_stability']['input_catwise_separation']['mean']:>12.1f}")
        print(f"{'Q-R tracer alignment (deg)':<40} {wo['tracer_consistency']['qr_alignment']['mean']:>12.2f} {fu['tracer_consistency']['qr_alignment']['mean']:>12.2f}")
        print(f"{'Q-R alignment scatter (deg)':<40} {wo['tracer_consistency']['qr_alignment']['std']:>12.2f} {fu['tracer_consistency']['qr_alignment']['std']:>12.2f}")
        print(f"{'Amplitude ratio Q/R':<40} {wo['tracer_consistency']['amplitude_ratio']['mean']:>12.2f} {fu['tracer_consistency']['amplitude_ratio']['mean']:>12.2f}")
        print(f"{'PPC amplitude p-value':<40} {wo['ppc']['amplitude']['p_value']:>12.3f} {fu['ppc']['amplitude']['p_value']:>12.3f}")
        print(f"{'PPC CMB sep p-value':<40} {wo['ppc']['cmb_separation']['p_value']:>12.3f} {fu['ppc']['cmb_separation']['p_value']:>12.3f}")
        print(f"{'Direction within 10° of CatWISE':<40} {wo['ppc']['direction_recovery']['fraction_within_10deg']:>12.1%} {fu['ppc']['direction_recovery']['fraction_within_10deg']:>12.1%}")

        print("\n" + "-"*70)
        print("KEY DISCRIMINATORS:")
        print("-"*70)

        # Tracer independence
        wo_align = wo['tracer_consistency']['qr_alignment']['mean']
        fu_align = fu['tracer_consistency']['qr_alignment']['mean']
        if wo_align < 0.1 and fu_align > 1.0:
            print("✓ TRACER INDEPENDENCE: wall_only shows perfect Q-R alignment (geometric)")
            print(f"  full shows {fu_align:.1f}° scatter (kinematic component varies)")
        elif wo_align < fu_align:
            print(f"  wall_only: {wo_align:.2f}° Q-R alignment")
            print(f"  full: {fu_align:.2f}° Q-R alignment")

        # Direction scatter
        wo_l_std = wo['direction_stability']['direction_scatter']['l_std']
        fu_l_std = fu['direction_stability']['direction_scatter']['l_std']
        if fu_l_std > wo_l_std * 1.5:
            print(f"✓ DIRECTION SCATTER: full shows {fu_l_std/wo_l_std:.1f}x more l scatter than wall_only")

    print("\nStage 4 complete. Results saved to results/stage4/")


if __name__ == '__main__':
    main()
