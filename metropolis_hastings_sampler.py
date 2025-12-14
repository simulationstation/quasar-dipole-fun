#!/usr/bin/env python3
"""
Metropolis-Hastings sampler for a "bubble mapping" dipole model with model comparison.

Physical interpretation:
  - Drift component: frame motion through parent trough, produces kinematic-like dipole
  - Wall component: off-center geometric term, catalog-independent residual

Model modes:
  - full: drift + wall (all parameters free)
  - wall_only: v_drift=0, only CMB velocity contributes to beta_tot
  - drift_only: D_wall=0, no Wagenveld wall constraints
  - cmb_only: v_drift=0 and D_wall=0, only CMB velocity + catalog scalars

Vector model:
  v_tot_vec = v_cmb_vec + v_drift_vec
  beta_tot  = |v_tot_vec| / c
  n_vtot    = v_tot_vec / |v_tot_vec|

  D_kin_Q_vec = (A_Q * beta_tot) * n_vtot
  D_kin_R_vec = (A_R * beta_tot) * n_vtot
  D_wall_vec  = D_wall * n_wall

  D_Q_vec = D_kin_Q_vec + D_wall_vec
  D_R_vec = D_kin_R_vec + D_wall_vec

Likelihood components (5 terms):
  1) wagenveld_wall_amp:    D_wall   ~ N(0.0081, 0.0014)
  2) wagenveld_wall_angle:  alpha_w  ~ N(39 deg, 8 deg)  (polar angle from CMB axis)
  3) catwise_amp:           |D_Q|    ~ N(0.01554, sigma_qso)
  4) catwise_dir:           angle(dhat_Q, n_obs) ~ N(0, sigma_dir)
  5) radio_ratio:           |D_R| / D_kin_R ~ N(3.67, 0.49)

Outputs:
  - posterior_samples.csv     (always writes full 8-parameter vector + derived + ll components)
  - posterior_summary.json    (per-model summary: posterior stats, WAIC, diagnostics)
  - model_comparison.json     (if --compare-models, ranked list with WAIC and deltas)

Notes on WAIC here:
  - The 5 likelihood "components" are treated as separable pseudo-data contributions.
  - WAIC is computed component-wise and summed (lppd and p_waic additive under independence).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -------------------------
# Physical constants
# -------------------------

C_KMS: float = 299_792.458   # speed of light [km/s]
V_CMB_KMS: float = 369.82    # CMB dipole velocity magnitude [km/s]

# -------------------------
# Fixed inputs (data)
# -------------------------

# CMB dipole direction in Galactic coords (l,b) degrees
L_CMB_DEG = 264.021
B_CMB_DEG = 48.253

# Wagenveld wall constraints (previously "residual")
D_WALL_MEAN = 0.0081
D_WALL_SIG  = 0.0014
ANG_WALL_CMB_MEAN_DEG = 39.0
ANG_WALL_CMB_SIG_DEG  = 8.0

# CatWISE quasar observed dipole amplitude and direction (galactic)
D_QSO_OBS = 0.01554
L_QSO_OBS_DEG = 238.2
B_QSO_OBS_DEG = 28.8

# Radio ratio constraint (Böhme)
R_RADIO_MEAN = 3.67
R_RADIO_SIG  = 0.49

# -------------------------
# Tunables (defaults; can override via CLI)
# -------------------------

SIGMA_QSO = 0.0010
SIGMA_QSO_DIR_DEG = 10.0

N_STEPS   = 250_000
BURN_IN   = 50_000
THIN      = 20
N_CHAINS  = 6

# Faster defaults for --compare-models when user hasn't overridden
N_STEPS_COMPARE   = 80_000
BURN_IN_COMPARE   = 20_000

# Output files
WRITE_DIAGNOSTICS = True
DIAGNOSTICS_CSV   = "angle_posterior.csv"
SAMPLES_CSV       = "posterior_samples.csv"
SUMMARY_JSON      = "posterior_summary.json"

# Proposal step sizes
STEP_V_DRIFT    = 50.0
STEP_L_DRIFT    = 5.0
STEP_B_DRIFT    = 3.0
STEP_D_WALL     = 0.00035
STEP_ALPHA_WALL = 2.5
STEP_PHI_WALL   = 8.0
STEP_A_Q        = 0.5
STEP_A_R        = 0.5

# Prior bounds
V_DRIFT_MAX = 2000.0
D_WALL_MAX  = 0.05
A_MAX       = 50.0

# Numerical stability
EPS_SIN  = 1e-12
EPS_NORM = 1e-15

# Model modes
MODEL_MODES = ["full", "wall_only", "drift_only", "cmb_only"]

# Full parameter order (always 8)
FULL_PARAM_NAMES = [
    "v_drift_kms", "l_drift_deg", "b_drift_deg",
    "D_wall", "alpha_wall_deg", "phi_wall_deg",
    "A_Q", "A_R",
]

# Likelihood component names (fixed across all modes)
LL_COMPONENT_NAMES = [
    "wagenveld_wall_amp",
    "wagenveld_wall_angle",
    "catwise_amp",
    "catwise_dir",
    "radio_ratio",
]
N_LL_COMPONENTS = len(LL_COMPONENT_NAMES)

# Active parameters per mode
ACTIVE_PARAMS: Dict[str, List[str]] = {
    "full":      ["v_drift_kms", "l_drift_deg", "b_drift_deg", "D_wall", "alpha_wall_deg", "phi_wall_deg", "A_Q", "A_R"],
    "wall_only": ["D_wall", "alpha_wall_deg", "phi_wall_deg", "A_Q", "A_R"],
    "drift_only":["v_drift_kms", "l_drift_deg", "b_drift_deg", "A_Q", "A_R"],
    "cmb_only":  ["A_Q", "A_R"],
}

# Fixed parameter values per mode
FIXED_PARAMS: Dict[str, Dict[str, float]] = {
    "full": {},
    "wall_only": {"v_drift_kms": 0.0, "l_drift_deg": 0.0, "b_drift_deg": 0.0},
    "drift_only": {"D_wall": 0.0, "alpha_wall_deg": 0.0, "phi_wall_deg": 0.0},
    "cmb_only": {"v_drift_kms": 0.0, "l_drift_deg": 0.0, "b_drift_deg": 0.0,
                 "D_wall": 0.0, "alpha_wall_deg": 0.0, "phi_wall_deg": 0.0},
}

# -------------------------
# Geometry helpers
# -------------------------

def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    """Convert Galactic lon/lat in degrees to 3D unit vector."""
    l = np.deg2rad(l_deg % 360.0)
    b = np.deg2rad(np.clip(b_deg, -90.0, 90.0))
    cb = math.cos(float(b))
    return np.array([cb * math.cos(float(l)), cb * math.sin(float(l)), math.sin(float(b))], dtype=float)

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees; returns nan if either has ~0 norm."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < EPS_NORM or nv < EPS_NORM:
        return float("nan")
    cuv = float(np.dot(u, v) / (nu * nv))
    cuv = max(-1.0, min(1.0, cuv))
    return float(np.rad2deg(math.acos(cuv)))

def log_gauss(x: float, mu: float, sig: float) -> float:
    """Log of unnormalized Gaussian N(mu, sig^2)."""
    z = (x - mu) / sig
    return -0.5 * z * z

def wrap_lon(l_deg: float) -> float:
    """Wrap longitude to [0, 360)."""
    return l_deg % 360.0

def reflect_lat(b_deg: float) -> Tuple[float, float]:
    """
    Reflect latitude into [-90, 90] and return (b_reflected, l_shift_deg).
    Reflection corresponds to flipping over the pole, adding 180° to longitude.
    """
    b = b_deg
    l_shift = 0.0
    while b > 90.0:
        b = 180.0 - b
        l_shift += 180.0
    while b < -90.0:
        b = -180.0 - b
        l_shift += 180.0
    return b, l_shift

def wrap_angle(x: float, period: float) -> float:
    """Wrap angle to [0, period)."""
    return x % period

def reflect_alpha(alpha: float) -> Tuple[float, float]:
    """
    Reflect polar angle alpha into [0, 180] and return (alpha_reflected, phi_shift_deg).
    """
    phi_shift = 0.0
    while alpha < 0.0 or alpha > 180.0:
        if alpha < 0.0:
            alpha = -alpha
            phi_shift += 180.0
        if alpha > 180.0:
            alpha = 360.0 - alpha
            phi_shift += 180.0
    return alpha, phi_shift

def build_cmb_basis(n_cmb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build orthonormal basis (e1, e2) perpendicular to n_cmb."""
    axes = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)
    dots = np.abs(axes @ n_cmb)
    ref = axes[int(np.argmin(dots))]

    e1 = np.cross(ref, n_cmb)
    e1n = float(np.linalg.norm(e1))
    if e1n < EPS_NORM:
        raise RuntimeError("Failed to build CMB basis (e1 norm too small).")
    e1 = e1 / e1n

    e2 = np.cross(n_cmb, e1)
    e2n = float(np.linalg.norm(e2))
    if e2n < EPS_NORM:
        raise RuntimeError("Failed to build CMB basis (e2 norm too small).")
    e2 = e2 / e2n
    return e1, e2

def alpha_phi_to_unitvec(alpha_deg: float, phi_deg: float,
                         n_axis: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Convert axis-centric spherical coords to unit vector:
      n = cos(alpha)*n_axis + sin(alpha)*(cos(phi)*e1 + sin(phi)*e2)
    """
    a = math.radians(alpha_deg)
    p = math.radians(phi_deg)
    ca, sa = math.cos(a), math.sin(a)
    cp, sp = math.cos(p), math.sin(p)
    return ca * n_axis + sa * (cp * e1 + sp * e2)

def log_sum_exp(arr: np.ndarray) -> float:
    """Stable log-sum-exp."""
    amax = float(np.max(arr))
    if not np.isfinite(amax):
        return amax
    return amax + float(np.log(np.sum(np.exp(arr - amax))))

def log_mean_exp(arr: np.ndarray) -> float:
    """Stable log-mean-exp: log(mean(exp(arr)))."""
    n = int(arr.shape[0])
    if n <= 0:
        return -float("inf")
    return log_sum_exp(arr) - math.log(n)

# -------------------------
# Precompute fixed quantities
# -------------------------

n_CMB = lb_to_unitvec(L_CMB_DEG, B_CMB_DEG)
v_CMB_vec = V_CMB_KMS * n_CMB
n_QSO_obs = lb_to_unitvec(L_QSO_OBS_DEG, B_QSO_OBS_DEG)
e1_CMB, e2_CMB = build_cmb_basis(n_CMB)

# -------------------------
# Model configuration
# -------------------------

@dataclass(frozen=True)
class ModelConfig:
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in MODEL_MODES:
            raise ValueError(f"Unknown model mode: {self.mode}. Must be one of {MODEL_MODES}")

    @property
    def active_params(self) -> List[str]:
        return ACTIVE_PARAMS[self.mode]

    @property
    def fixed_params(self) -> Dict[str, float]:
        return FIXED_PARAMS[self.mode]

    @property
    def n_params(self) -> int:
        return len(self.active_params)

    @property
    def step_sizes(self) -> Dict[str, float]:
        all_steps = {
            "v_drift_kms": STEP_V_DRIFT,
            "l_drift_deg": STEP_L_DRIFT,
            "b_drift_deg": STEP_B_DRIFT,
            "D_wall": STEP_D_WALL,
            "alpha_wall_deg": STEP_ALPHA_WALL,
            "phi_wall_deg": STEP_PHI_WALL,
            "A_Q": STEP_A_Q,
            "A_R": STEP_A_R,
        }
        return {p: all_steps[p] for p in self.active_params}

    def has_drift(self) -> bool:
        return "v_drift_kms" in self.active_params

    def has_wall(self) -> bool:
        return "D_wall" in self.active_params

    def active_to_full_theta(self, theta_active: np.ndarray) -> np.ndarray:
        """Convert active parameter vector to full 8-parameter vector."""
        theta_full = np.zeros(8, dtype=float)

        for pname, pval in self.fixed_params.items():
            theta_full[FULL_PARAM_NAMES.index(pname)] = float(pval)

        for i, pname in enumerate(self.active_params):
            theta_full[FULL_PARAM_NAMES.index(pname)] = float(theta_active[i])

        return theta_full

    def get_initial_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Randomized initial theta for active parameters."""
        defaults: Dict[str, float] = {
            "v_drift_kms": 100.0,
            "l_drift_deg": float(rng.uniform(0.0, 360.0)),
            "b_drift_deg": float(rng.uniform(-60.0, 60.0)),
            "D_wall": 0.0081,
            "alpha_wall_deg": 39.0,
            "phi_wall_deg": float(rng.uniform(0.0, 360.0)),
            "A_Q": 6.0,
            "A_R": 2.0,
        }
        return np.array([defaults[p] for p in self.active_params], dtype=float)

# -------------------------
# Model physics
# -------------------------

def compute_model_vectors(theta_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]:
    """
    Compute model dipole vectors from full 8-parameter theta.

    Returns:
      D_Q_vec, D_R_vec, beta_tot, n_vtot, D_kin_Q, D_kin_R
    """
    v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R = [float(x) for x in theta_full]

    # Drift velocity vector
    n_drift = lb_to_unitvec(l_drift, b_drift)
    v_drift_vec = v_drift * n_drift

    # Total velocity vector
    v_tot_vec = v_CMB_vec + v_drift_vec
    v_tot_mag = float(np.linalg.norm(v_tot_vec))

    beta_tot = v_tot_mag / C_KMS
    n_vtot = v_tot_vec / v_tot_mag if v_tot_mag > EPS_NORM else n_CMB

    # Kinematic dipoles
    D_kin_Q = A_Q * beta_tot
    D_kin_R = A_R * beta_tot
    D_kin_Q_vec = D_kin_Q * n_vtot
    D_kin_R_vec = D_kin_R * n_vtot

    # Wall contribution (skip if D_wall == 0)
    if abs(D_wall) > EPS_NORM:
        n_wall = alpha_phi_to_unitvec(alpha_wall, phi_wall, n_CMB, e1_CMB, e2_CMB)
        D_wall_vec = D_wall * n_wall
    else:
        D_wall_vec = np.zeros(3, dtype=float)

    D_Q_vec = D_kin_Q_vec + D_wall_vec
    D_R_vec = D_kin_R_vec + D_wall_vec
    return D_Q_vec, D_R_vec, beta_tot, n_vtot, D_kin_Q, D_kin_R

# -------------------------
# Prior + likelihood
# -------------------------

def log_prior(theta_active: np.ndarray, config: ModelConfig) -> float:
    """Log-prior over active parameters only (with hard bounds)."""
    theta_full = config.active_to_full_theta(theta_active)
    v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R = [float(x) for x in theta_full]

    # Bounds
    if config.has_drift():
        if not (0.0 <= v_drift <= V_DRIFT_MAX): return -np.inf
        if not (0.0 <= l_drift < 360.0): return -np.inf
        if not (-90.0 <= b_drift <= 90.0): return -np.inf

    if config.has_wall():
        if not (0.0 <= D_wall <= D_WALL_MAX): return -np.inf
        if not (0.0 <= alpha_wall <= 180.0): return -np.inf
        if not (0.0 <= phi_wall < 360.0): return -np.inf

    if not (0.0 <= A_Q <= A_MAX): return -np.inf
    if not (0.0 <= A_R <= A_MAX): return -np.inf

    lp = 0.0

    # Drift direction uniform on sphere => prior ∝ cos(b)
    if config.has_drift():
        cb = math.cos(math.radians(b_drift))
        if cb <= EPS_SIN:
            return -np.inf
        lp += math.log(cb)

    # Wall direction uniform on sphere => prior ∝ sin(alpha)
    if config.has_wall():
        sa = math.sin(math.radians(alpha_wall))
        if sa <= EPS_SIN:
            return -np.inf
        lp += math.log(sa)

    return float(lp)

def loglike_components(theta_active: np.ndarray, config: ModelConfig,
                       sigma_qso: float, sigma_qso_dir: float) -> Tuple[float, np.ndarray]:
    """
    Component-wise log-likelihood vector (length 5) + total.

    If wall inactive, components 0 and 1 are 0.0.
    On invalid evaluation returns (-inf, [-inf]*5).
    """
    theta_full = config.active_to_full_theta(theta_active)
    D_wall = float(theta_full[3])
    alpha_wall = float(theta_full[4])

    D_Q_vec, D_R_vec, _, _, _, D_kin_R = compute_model_vectors(theta_full)
    D_Q_amp = float(np.linalg.norm(D_Q_vec))
    D_R_amp = float(np.linalg.norm(D_R_vec))

    ll_vec = np.zeros(N_LL_COMPONENTS, dtype=float)

    # 0) wall amplitude
    if config.has_wall():
        ll_vec[0] = log_gauss(D_wall, D_WALL_MEAN, D_WALL_SIG)

    # 1) wall angle (alpha)
    if config.has_wall():
        ll_vec[1] = log_gauss(alpha_wall, ANG_WALL_CMB_MEAN_DEG, ANG_WALL_CMB_SIG_DEG)

    # 2) CatWISE amplitude
    ll_vec[2] = log_gauss(D_Q_amp, D_QSO_OBS, sigma_qso)

    # 3) CatWISE direction
    if D_Q_amp < EPS_NORM:
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf, dtype=float)
    dhat_Q = D_Q_vec / D_Q_amp
    ang = angle_deg(dhat_Q, n_QSO_obs)
    if np.isnan(ang):
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf, dtype=float)
    ll_vec[3] = log_gauss(float(ang), 0.0, sigma_qso_dir)

    # 4) Radio ratio
    if D_kin_R < EPS_NORM:
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf, dtype=float)
    r_model = D_R_amp / float(D_kin_R)
    ll_vec[4] = log_gauss(float(r_model), R_RADIO_MEAN, R_RADIO_SIG)

    ll_total = float(np.sum(ll_vec))
    if not np.isfinite(ll_total):
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf, dtype=float)
    return ll_total, ll_vec

def log_posterior(theta_active: np.ndarray, config: ModelConfig,
                  sigma_qso: float, sigma_qso_dir: float) -> float:
    lp = log_prior(theta_active, config)
    if not np.isfinite(lp):
        return -np.inf
    ll_total, _ = loglike_components(theta_active, config, sigma_qso, sigma_qso_dir)
    if not np.isfinite(ll_total):
        return -np.inf
    return float(lp + ll_total)

# -------------------------
# Proposals
# -------------------------

def propose(theta_active: np.ndarray, rng: np.random.Generator, config: ModelConfig,
            step_overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Random-walk Gaussian proposal for active params with coordinate wrapping/reflection."""
    theta_new = theta_active.copy()

    steps = config.step_sizes.copy()
    if step_overrides:
        # Only active params are allowed
        for k, v in step_overrides.items():
            if k in steps:
                steps[k] = float(v)

    for i, pname in enumerate(config.active_params):
        theta_new[i] += float(rng.normal(0.0, steps[pname]))

    # Convert to full for coordinate constraints
    theta_full = config.active_to_full_theta(theta_new)
    v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R = [float(x) for x in theta_full]

    if config.has_drift():
        b_drift, l_shift = reflect_lat(b_drift)
        l_drift = wrap_lon(l_drift + l_shift)
        if v_drift < 0.0:
            v_drift = -v_drift

    if config.has_wall():
        alpha_wall, phi_shift = reflect_alpha(alpha_wall)
        phi_wall = wrap_angle(phi_wall + phi_shift, 360.0)
        if D_wall < 0.0:
            D_wall = -D_wall

    if A_Q < 0.0: A_Q = -A_Q
    if A_R < 0.0: A_R = -A_R

    theta_full2 = np.array([v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R], dtype=float)
    theta_out = np.array([theta_full2[FULL_PARAM_NAMES.index(p)] for p in config.active_params], dtype=float)
    return theta_out

# -------------------------
# MCMC
# -------------------------

def run_chain(theta0: np.ndarray, n_steps: int, rng: np.random.Generator,
              burn_in: int, thin: int, config: ModelConfig,
              sigma_qso: float, sigma_qso_dir: float,
              step_overrides: Optional[Dict[str, float]] = None,
              chain_id: int = 0, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run a single Metropolis-Hastings chain.

    Returns:
      samples          shape (S, n_active)
      ll_totals        shape (S,)
      ll_components    shape (S, 5)
      acceptance_rate  float
    """
    theta = theta0.copy()
    ll_total, ll_vec = loglike_components(theta, config, sigma_qso, sigma_qso_dir)
    lp = log_prior(theta, config)
    logp = lp + ll_total

    if not np.isfinite(logp):
        raise RuntimeError(f"Chain {chain_id}: initial theta has -inf posterior; pick a better start.")

    accepted = 0
    samples: List[np.ndarray] = []
    ll_totals: List[float] = []
    ll_components: List[np.ndarray] = []

    for t in range(int(n_steps)):
        th_prop = propose(theta, rng, config, step_overrides)
        ll_total_p, ll_vec_p = loglike_components(th_prop, config, sigma_qso, sigma_qso_dir)
        lp_p = log_prior(th_prop, config)
        logp_p = lp_p + ll_total_p

        if np.isfinite(logp_p):
            u = float(rng.random())
            log_u = math.log(u) if u > 0.0 else -np.inf
            if log_u < (logp_p - logp):
                theta = th_prop
                logp = logp_p
                ll_total = ll_total_p
                ll_vec = ll_vec_p
                accepted += 1

        if t >= burn_in and ((t - burn_in) % thin == 0):
            samples.append(theta.copy())
            ll_totals.append(float(ll_total))
            ll_components.append(ll_vec.copy())

        if verbose and (t + 1) % 50_000 == 0:
            print(f"  Chain {chain_id}: step {t+1}/{n_steps}  accept_rate={accepted/(t+1):.3f}")

    return np.array(samples, dtype=float), np.array(ll_totals, dtype=float), np.array(ll_components, dtype=float), float(accepted / n_steps)

def run_multi_chain(config: ModelConfig, n_chains: int, n_steps: int, burn_in: int, thin: int,
                    sigma_qso: float, sigma_qso_dir: float,
                    step_overrides: Optional[Dict[str, float]] = None,
                    base_seed: int = 42, verbose: bool = True
                    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Returns:
      chains             list of arrays (S_c, n_active)
      ll_totals_list     list of arrays (S_c,)
      ll_components_list list of arrays (S_c, 5)
      acceptance_rates   list of floats
    """
    chains: List[np.ndarray] = []
    ll_totals_list: List[np.ndarray] = []
    ll_components_list: List[np.ndarray] = []
    acceptance_rates: List[float] = []

    for c in range(int(n_chains)):
        seed = int(base_seed + c * 1000)
        rng = np.random.default_rng(seed)
        theta0 = config.get_initial_theta(rng)

        if verbose:
            print(f"\nStarting chain {c} with seed={seed}")

        samples, ll_totals, ll_comps, acc = run_chain(
            theta0, n_steps, rng, burn_in, thin, config,
            sigma_qso, sigma_qso_dir, step_overrides,
            chain_id=c, verbose=verbose
        )

        chains.append(samples)
        ll_totals_list.append(ll_totals)
        ll_components_list.append(ll_comps)
        acceptance_rates.append(acc)

        if verbose:
            print(f"  Chain {c} done: {len(samples)} samples, acceptance={acc:.3f}")

    return chains, ll_totals_list, ll_components_list, acceptance_rates

# -------------------------
# WAIC (component-wise)
# -------------------------

def compute_waic_components(ll_components: np.ndarray) -> Tuple[Tuple[float, float, float], Dict[str, Dict[str, float]]]:
    """
    Compute WAIC from component-wise log-likelihoods.

    Args:
      ll_components: array (S, K) where K=5.

    Returns:
      (lppd, p_waic, waic), breakdown dict keyed by component name.
    """
    if ll_components.ndim != 2 or ll_components.shape[1] != N_LL_COMPONENTS:
        raise ValueError(f"ll_components must have shape (S, {N_LL_COMPONENTS}). Got {ll_components.shape}.")

    S = int(ll_components.shape[0])
    if S < 2:
        breakdown = {name: {"lppd": float("nan"), "p_waic": 0.0, "waic": float("nan")}
                     for name in LL_COMPONENT_NAMES}
        return (float("nan"), 0.0, float("nan")), breakdown

    lppd_k = np.zeros(N_LL_COMPONENTS, dtype=float)
    p_waic_k = np.zeros(N_LL_COMPONENTS, dtype=float)

    for k in range(N_LL_COMPONENTS):
        col = ll_components[:, k]
        lppd_k[k] = log_mean_exp(col)
        p_waic_k[k] = float(np.var(col, ddof=0))  # ddof=0 for population variance

    lppd = float(np.sum(lppd_k))
    p_waic = float(np.sum(p_waic_k))
    waic = float(-2.0 * (lppd - p_waic))

    breakdown: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(LL_COMPONENT_NAMES):
        waic_k = float(-2.0 * (lppd_k[k] - p_waic_k[k]))
        breakdown[name] = {"lppd": float(lppd_k[k]), "p_waic": float(p_waic_k[k]), "waic": waic_k}

    return (lppd, p_waic, waic), breakdown

def compute_ll_component_stats(ll_components: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(LL_COMPONENT_NAMES):
        col = ll_components[:, k]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return stats

# -------------------------
# Diagnostics (R-hat, ESS)
# -------------------------

def gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    """Gelman-Rubin R-hat for each parameter in active space."""
    m = len(chains)
    if m < 2:
        return np.full(chains[0].shape[1], np.nan, dtype=float)

    n = min(c.shape[0] for c in chains)
    if n < 2:
        return np.full(chains[0].shape[1], np.nan, dtype=float)

    chains2 = [c[:n] for c in chains]
    p = chains2[0].shape[1]
    rhat = np.ones(p, dtype=float)

    for j in range(p):
        chain_means = np.array([np.mean(c[:, j]) for c in chains2], dtype=float)
        chain_vars = np.array([np.var(c[:, j], ddof=1) for c in chains2], dtype=float)

        W = float(np.mean(chain_vars))
        B = float(n * np.var(chain_means, ddof=1))

        if W < EPS_NORM:
            rhat[j] = 1.0 if B < EPS_NORM else np.nan
            continue

        var_plus = ((n - 1) / n) * W + (1 / n) * B
        rhat[j] = math.sqrt(var_plus / W)

    return rhat

def effective_sample_size(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Simple initial-monotone ACF ESS estimate per parameter."""
    n, p = samples.shape
    if max_lag is None:
        max_lag = min(1000, n // 5)
    max_lag = max(1, min(int(max_lag), n // 2))

    ess = np.zeros(p, dtype=float)

    for j in range(p):
        x = samples[:, j]
        xc = x - np.mean(x)
        var = float(np.var(x, ddof=0))
        if var < EPS_NORM:
            ess[j] = float(n)
            continue

        acf_sum = 0.0
        for lag in range(1, max_lag + 1):
            n_pairs = n - lag
            if n_pairs < 1:
                break
            acf = float(np.sum(xc[:n_pairs] * xc[lag:lag + n_pairs]) / (n_pairs * var))
            if acf < 0.0:
                break
            acf_sum += acf

        tau = 1.0 + 2.0 * acf_sum
        ess[j] = float(n / max(tau, 1.0))

    return ess

def combined_ess(chains: List[np.ndarray]) -> np.ndarray:
    return effective_sample_size(np.vstack(chains), max_lag=None) if chains else np.array([])

# -------------------------
# Posterior stats
# -------------------------

def compute_param_stats(samples: np.ndarray, names: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        x = samples[:, i]
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        stats[name] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(q50),
            "q16": float(q16),
            "q84": float(q84),
        }
    return stats

def compute_derived_quantities(samples_active: np.ndarray, config: ModelConfig) -> Tuple[np.ndarray, ...]:
    """Derived arrays: beta_tot, v_tot_kms, D_kin_Q, D_kin_R, catwise_sep, radio_ratio."""
    S = int(samples_active.shape[0])
    beta = np.empty(S, dtype=float)
    vtot = np.empty(S, dtype=float)
    dkinq = np.empty(S, dtype=float)
    dkinr = np.empty(S, dtype=float)
    sep = np.empty(S, dtype=float)
    ratio = np.empty(S, dtype=float)

    for k in range(S):
        theta_full = config.active_to_full_theta(samples_active[k])
        D_Q_vec, D_R_vec, beta_k, _, D_kin_Q, D_kin_R = compute_model_vectors(theta_full)

        beta[k] = beta_k
        vtot[k] = beta_k * C_KMS
        dkinq[k] = D_kin_Q
        dkinr[k] = D_kin_R

        D_Q_amp = float(np.linalg.norm(D_Q_vec))
        D_R_amp = float(np.linalg.norm(D_R_vec))

        if D_Q_amp > EPS_NORM:
            sep[k] = angle_deg(D_Q_vec / D_Q_amp, n_QSO_obs)
        else:
            sep[k] = float("nan")

        if D_kin_R > EPS_NORM:
            ratio[k] = D_R_amp / D_kin_R
        else:
            ratio[k] = float("nan")

    return beta, vtot, dkinq, dkinr, sep, ratio

def compute_derived_stats(samples_active: np.ndarray, config: ModelConfig) -> Dict[str, Any]:
    beta, vtot, dkinq, dkinr, sep, ratio = compute_derived_quantities(samples_active, config)

    def safe_stats(arr: np.ndarray) -> Dict[str, float]:
        v = arr[~np.isnan(arr)]
        if v.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "q16": float("nan"), "q84": float("nan")}
        q16, q50, q84 = np.percentile(v, [16, 50, 84])
        return {"mean": float(np.mean(v)), "median": float(q50), "q16": float(q16), "q84": float(q84)}

    out: Dict[str, Any] = {
        "beta_tot": safe_stats(beta),
        "v_tot_kms": safe_stats(vtot),
        "D_kin_Q": safe_stats(dkinq),
        "D_kin_R": safe_stats(dkinr),
        "catwise_separation": safe_stats(sep),
        "radio_ratio": safe_stats(ratio),
    }

    if config.has_wall():
        idx = config.active_params.index("alpha_wall_deg")
        alpha = samples_active[:, idx]
        out["alpha_wall_fraction_in_31_47"] = float(np.mean((alpha >= 31.0) & (alpha <= 47.0)))

    return out

# -------------------------
# Export
# -------------------------

def write_samples_csv(
    chains: List[np.ndarray],
    ll_totals_list: List[np.ndarray],
    ll_components_list: List[np.ndarray],
    config: ModelConfig,
    filename: str,
) -> None:
    """
    Write samples to CSV with a fixed column schema across modes.

    Columns:
      chain_id, model_mode,
      v_drift_kms,l_drift_deg,b_drift_deg,D_wall,alpha_wall_deg,phi_wall_deg,A_Q,A_R,
      beta_tot,v_tot_kms,
      ll_<component0..4>,
      loglike
    """
    cols = (
        ["chain_id", "model_mode"]
        + FULL_PARAM_NAMES
        + ["beta_tot", "v_tot_kms"]
        + ["ll_" + name for name in LL_COMPONENT_NAMES]
        + ["loglike"]
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")

        for chain_id, (samples, ll_totals, ll_comps) in enumerate(zip(chains, ll_totals_list, ll_components_list)):
            for row_active, ll_total, ll_vec in zip(samples, ll_totals, ll_comps):
                theta_full = config.active_to_full_theta(row_active)
                _, _, beta_tot, _, _, _ = compute_model_vectors(theta_full)
                v_tot_kms = beta_tot * C_KMS

                parts: List[str] = [str(chain_id), config.mode]
                parts += [f"{float(v):.8g}" for v in theta_full]
                parts += [f"{float(beta_tot):.8g}", f"{float(v_tot_kms):.4f}"]
                parts += [f"{float(v):.6f}" for v in ll_vec]
                parts += [f"{float(ll_total):.6f}"]
                f.write(",".join(parts) + "\n")

def write_summary_json(
    chains: List[np.ndarray],
    ll_totals_list: List[np.ndarray],
    ll_components_list: List[np.ndarray],
    acceptance_rates: List[float],
    rhat: np.ndarray,
    ess: np.ndarray,
    config: ModelConfig,
    waic_total: Tuple[float, float, float],
    waic_breakdown: Dict[str, Dict[str, float]],
    ll_comp_stats: Dict[str, Dict[str, float]],
    filename: str,
) -> None:
    combined = np.vstack(chains)
    combined_ll = np.concatenate(ll_totals_list)
    lppd, p_waic, waic = waic_total

    param_stats = compute_param_stats(combined, config.active_params)
    derived_stats = compute_derived_stats(combined, config)

    for i, name in enumerate(config.active_params):
        param_stats[name]["R_hat"] = float(rhat[i]) if np.isfinite(rhat[i]) else None
        param_stats[name]["ESS"] = float(ess[i]) if np.isfinite(ess[i]) else None

    summary: Dict[str, Any] = {
        "model_mode": config.mode,
        "active_parameters": config.active_params,
        "fixed_parameters": config.fixed_params,
        "n_chains": len(chains),
        "samples_per_chain": int(chains[0].shape[0]) if chains else 0,
        "total_samples": int(combined.shape[0]),
        "acceptance_rates": [float(a) for a in acceptance_rates],
        "waic": {"lppd": float(lppd), "p_waic": float(p_waic), "waic": float(waic)},
        "waic_components": waic_breakdown,
        "ll_component_stats": ll_comp_stats,
        "ll_total_stats": {
            "mean": float(np.mean(combined_ll)),
            "std": float(np.std(combined_ll)),
            "min": float(np.min(combined_ll)),
            "max": float(np.max(combined_ll)),
        },
        "parameters": param_stats,
        "derived": derived_stats,
        "convergence": {
            "R_hat": {name: (float(rhat[i]) if np.isfinite(rhat[i]) else None)
                      for i, name in enumerate(config.active_params)},
            "ESS": {name: (float(ess[i]) if np.isfinite(ess[i]) else None)
                    for i, name in enumerate(config.active_params)},
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

# -------------------------
# Single-model runner
# -------------------------

def run_single_model(config: ModelConfig, args: argparse.Namespace, verbose: bool = True) -> Dict[str, Any]:
    all_steps = {
        "v_drift_kms": args.step_v_drift,
        "l_drift_deg": args.step_l_drift,
        "b_drift_deg": args.step_b_drift,
        "D_wall": args.step_d_wall,
        "alpha_wall_deg": args.step_alpha_wall,
        "phi_wall_deg": args.step_phi_wall,
        "A_Q": args.step_a_q,
        "A_R": args.step_a_r,
    }
    step_overrides = {k: float(v) for k, v in all_steps.items() if k in config.active_params}

    if verbose:
        print("=" * 60)
        print(f"Model: {config.mode}")
        print(f"Active parameters: {config.active_params}")
        print(f"Fixed: {config.fixed_params}")
        print("=" * 60)

    chains, ll_totals_list, ll_components_list, acceptance_rates = run_multi_chain(
        config=config,
        n_chains=args.n_chains,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        thin=args.thin,
        sigma_qso=args.sigma_qso,
        sigma_qso_dir=args.sigma_qso_dir,
        step_overrides=step_overrides,
        base_seed=args.seed,
        verbose=verbose,
    )

    combined = np.vstack(chains)
    combined_ll_total = np.concatenate(ll_totals_list)
    combined_ll_comp = np.vstack(ll_components_list)

    rhat = gelman_rubin(chains)
    ess = combined_ess(chains)

    waic_total, waic_breakdown = compute_waic_components(combined_ll_comp)
    ll_comp_stats = compute_ll_component_stats(combined_ll_comp)

    if verbose:
        lppd, p_waic, waic = waic_total
        print("\n" + "-" * 40)
        print("WAIC (component-wise):")
        print(f"  Total: lppd={lppd:.4f}, p_waic={p_waic:.4f}, WAIC={waic:.4f}")
        print("\n  Per-component breakdown:")
        print(f"  {'Component':<25} {'lppd':>10} {'p_waic':>10} {'waic':>10}")
        print("  " + "-" * 55)
        for name in LL_COMPONENT_NAMES:
            b = waic_breakdown[name]
            print(f"  {name:<25} {b['lppd']:>10.4f} {b['p_waic']:>10.4f} {b['waic']:>10.4f}")

        print("\nLog-likelihood total stats:")
        print(f"  mean = {np.mean(combined_ll_total):.4f}, std = {np.std(combined_ll_total):.4f}")
        print(f"  min  = {np.min(combined_ll_total):.4f}, max = {np.max(combined_ll_total):.4f}")

    return {
        "config": config,
        "chains": chains,
        "ll_totals_list": ll_totals_list,
        "ll_components_list": ll_components_list,
        "acceptance_rates": acceptance_rates,
        "R_hat": rhat,
        "ess": ess,
        "waic_total": waic_total,
        "waic_breakdown": waic_breakdown,
        "ll_comp_stats": ll_comp_stats,
        "combined": combined,
        "combined_ll_total": combined_ll_total,
        "combined_ll_comp": combined_ll_comp,
    }

# -------------------------
# Model comparison
# -------------------------

def run_model_comparison(args: argparse.Namespace, verbose: bool = True) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # Use faster defaults only if user kept the full-run defaults unchanged
    n_steps = args.n_steps if args.n_steps != N_STEPS else N_STEPS_COMPARE
    burn_in = args.burn_in if args.burn_in != BURN_IN else BURN_IN_COMPARE

    # Clone args with overridden step/burn for comparison
    class CompareArgs(argparse.Namespace):
        pass

    cargs = CompareArgs(**vars(args))
    cargs.n_steps = n_steps
    cargs.burn_in = burn_in

    print("=" * 60)
    print("MODEL COMPARISON")
    print(f"Running all {len(MODEL_MODES)} models with n_steps={n_steps}, burn_in={burn_in}")
    print("=" * 60)

    for mode in MODEL_MODES:
        print("\n" + "=" * 60)
        print(f"Running model: {mode}")
        print("=" * 60)
        config = ModelConfig(mode)
        results.append(run_single_model(config, cargs, verbose=verbose))

    # Ranked by WAIC
    ranked = sorted(results, key=lambda r: r["waic_total"][2])
    best = ranked[0]
    best_waic = best["waic_total"][2]
    best_breakdown = best["waic_breakdown"]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS (ranked by WAIC)")
    print("=" * 70)
    print(f"\n{'Model':<15} {'WAIC':>10} {'ΔWAIC':>10} {'p_waic':>10} {'lppd':>10} {'n_par':>6}")
    print("-" * 70)

    for r in ranked:
        mode = r["config"].mode
        lppd, p_waic, waic = r["waic_total"]
        n_par = r["config"].n_params
        delta = waic - best_waic
        delta_str = f"{delta:+.2f}" if delta > 0 else "(best)"
        print(f"{mode:<15} {waic:>10.2f} {delta_str:>10} {p_waic:>10.2f} {lppd:>10.2f} {n_par:>6}")

        if delta > 0:
            comp_deltas: List[Tuple[str, float]] = []
            for name in LL_COMPONENT_NAMES:
                d = r["waic_breakdown"][name]["waic"] - best_breakdown[name]["waic"]
                if abs(d) > 0.01:
                    comp_deltas.append((name, float(d)))
            comp_deltas.sort(key=lambda x: -abs(x[1]))
            top2 = comp_deltas[:2]
            if top2:
                contrib = ", ".join([f"{n}:{d:+.2f}" for n, d in top2])
                print(f"{'':15} top ΔWAIC contributors: {contrib}")

    print("\n" + "-" * 70)
    print("Log-likelihood stats per model:")
    print("-" * 70)
    for r in ranked:
        ll = r["combined_ll_total"]
        print(f"{r['config'].mode:<15} mean={np.mean(ll):>8.2f}, std={np.std(ll):>6.2f}, "
              f"range=[{np.min(ll):.2f}, {np.max(ll):.2f}]")

    return results

# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bubble-mapping dipole MCMC sampler with model comparison")

    parser.add_argument("--model", type=str, default="full", choices=MODEL_MODES,
                        help="Model mode (default: full)")
    parser.add_argument("--compare-models", action="store_true",
                        help="Run all models and compare WAIC")

    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--thin", type=int, default=THIN)
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigma-qso", type=float, default=SIGMA_QSO)
    parser.add_argument("--sigma-qso-dir", type=float, default=SIGMA_QSO_DIR_DEG)

    parser.add_argument("--step-v-drift", type=float, default=STEP_V_DRIFT)
    parser.add_argument("--step-l-drift", type=float, default=STEP_L_DRIFT)
    parser.add_argument("--step-b-drift", type=float, default=STEP_B_DRIFT)
    parser.add_argument("--step-d-wall", type=float, default=STEP_D_WALL)
    parser.add_argument("--step-alpha-wall", type=float, default=STEP_ALPHA_WALL)
    parser.add_argument("--step-phi-wall", type=float, default=STEP_PHI_WALL)
    parser.add_argument("--step-a-q", type=float, default=STEP_A_Q)
    parser.add_argument("--step-a-r", type=float, default=STEP_A_R)

    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()

# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    if args.compare_models:
        results = run_model_comparison(args, verbose=verbose)

        comparison_file = "model_comparison.json"
        ranked = sorted(results, key=lambda r: r["waic_total"][2])

        out: List[Dict[str, Any]] = []
        for r in ranked:
            lppd, p_waic, waic = r["waic_total"]
            out.append({
                "model": r["config"].mode,
                "waic": float(waic),
                "lppd": float(lppd),
                "p_waic": float(p_waic),
                "n_params": int(r["config"].n_params),
                "active_params": r["config"].active_params,
                "mean_acceptance": float(np.mean(r["acceptance_rates"])),
                "waic_components": r["waic_breakdown"],
            })

        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved comparison to: {comparison_file}")
        return

    # Single model run
    config = ModelConfig(args.model)

    if verbose:
        print("=" * 60)
        print("Bubble-Mapping Dipole MCMC Sampler")
        print("=" * 60)
        print(f"Model: {config.mode}")
        print(f"Active parameters: {config.active_params}")
        print(f"Fixed: {config.fixed_params}")
        print(f"N_STEPS={args.n_steps}, BURN_IN={args.burn_in}, THIN={args.thin}")
        print(f"N_CHAINS={args.n_chains}, SEED={args.seed}")
        print(f"V_CMB = {V_CMB_KMS:.2f} km/s toward (l={L_CMB_DEG:.3f}, b={B_CMB_DEG:.3f})")
        print("=" * 60)

    result = run_single_model(config, args, verbose=verbose)

    chains = result["chains"]
    ll_totals_list = result["ll_totals_list"]
    ll_components_list = result["ll_components_list"]
    acceptance_rates = result["acceptance_rates"]
    rhat = result["R_hat"]
    ess = result["ess"]
    waic_total = result["waic_total"]
    waic_breakdown = result["waic_breakdown"]
    ll_comp_stats = result["ll_comp_stats"]
    combined = result["combined"]

    if verbose:
        print("\n" + "-" * 40)
        print("Acceptance rates per chain:")
        for c, acc in enumerate(acceptance_rates):
            print(f"  Chain {c}: {acc:.3f}")

        print("\n" + "-" * 40)
        print("Convergence diagnostics:")
        print(f"{'Parameter':<16} {'R-hat':>8} {'ESS':>10}")
        for i, name in enumerate(config.active_params):
            r_str = f"{rhat[i]:.4f}" if np.isfinite(rhat[i]) else "N/A"
            e_str = f"{ess[i]:.1f}" if np.isfinite(ess[i]) else "N/A"
            warn = ""
            if np.isfinite(rhat[i]) and rhat[i] > 1.05:
                warn += " ***"
            if np.isfinite(ess[i]) and ess[i] < 100:
                warn += " (low)"
            print(f"{name:<16} {r_str:>8} {e_str:>10}{warn}")

        print("\n" + "-" * 40)
        print("Posterior summaries:")
        for i, nm in enumerate(config.active_params):
            x = combined[:, i]
            q16, q50, q84 = np.percentile(x, [16, 50, 84])
            print(f"{nm:16s}  mean={np.mean(x):10.4f} std={np.std(x):10.4f}  "
                  f"median={q50:10.4f}  [16,84]=[{q16:10.4f},{q84:10.4f}]")

        derived = compute_derived_stats(combined, config)
        print("\n" + "-" * 40)
        print("Derived quantities:")
        for key in ["beta_tot", "v_tot_kms", "D_kin_Q", "D_kin_R"]:
            d = derived[key]
            print(f"{key}: mean={d['mean']:.6g}  median={d['median']:.6g}  "
                  f"[16,84]=[{d['q16']:.6g},{d['q84']:.6g}]")

        cw = derived["catwise_separation"]
        print(f"\nCatWISE separation: mean={cw['mean']:.2f}  median={cw['median']:.2f}  "
              f"[16,84]=[{cw['q16']:.2f},{cw['q84']:.2f}] deg")

        rr = derived["radio_ratio"]
        print(f"Radio ratio: mean={rr['mean']:.3f}  median={rr['median']:.3f}  "
              f"[16,84]=[{rr['q16']:.3f},{rr['q84']:.3f}]  (target: {R_RADIO_MEAN:.2f}±{R_RADIO_SIG:.2f})")

        if "alpha_wall_fraction_in_31_47" in derived:
            print(f"\nFraction alpha_wall in [31,47]: {derived['alpha_wall_fraction_in_31_47']:.3f}")

    # Export
    if verbose:
        print("\n" + "-" * 40)
        print("Exporting results...")

    write_samples_csv(chains, ll_totals_list, ll_components_list, config, SAMPLES_CSV)
    if verbose:
        print(f"  Wrote samples to: {SAMPLES_CSV}")

    write_summary_json(
        chains=chains,
        ll_totals_list=ll_totals_list,
        ll_components_list=ll_components_list,
        acceptance_rates=acceptance_rates,
        rhat=rhat,
        ess=ess,
        config=config,
        waic_total=waic_total,
        waic_breakdown=waic_breakdown,
        ll_comp_stats=ll_comp_stats,
        filename=SUMMARY_JSON,
    )
    if verbose:
        print(f"  Wrote summary to: {SUMMARY_JSON}")

    if WRITE_DIAGNOSTICS and config.has_wall():
        alpha_idx = config.active_params.index("alpha_wall_deg")
        alpha_samples = combined[:, alpha_idx]
        with open(DIAGNOSTICS_CSV, "w", encoding="utf-8") as f:
            f.write("alpha_wall_deg\n")
            for a in alpha_samples:
                f.write(f"{float(a):.6f}\n")
        if verbose:
            print(f"  Wrote alpha_wall posterior to: {DIAGNOSTICS_CSV}")

    if verbose:
        print("\n" + "=" * 60)
        print("Done.")
        print("=" * 60)

if __name__ == "__main__":
    main()
