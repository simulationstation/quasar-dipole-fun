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

Model:
  v_tot_vec = v_cmb_vec + v_drift_vec
  beta_tot = |v_tot_vec| / c
  n_vtot = v_tot_vec / |v_tot_vec|

  D_kin_Q_vec = (A_Q * beta_tot) * n_vtot
  D_kin_R_vec = (A_R * beta_tot) * n_vtot
  D_wall_vec = D_wall * n_wall

  D_Q_vec = D_kin_Q_vec + D_wall_vec
  D_R_vec = D_kin_R_vec + D_wall_vec

WAIC is computed using 5 likelihood components:
  - wagenveld_wall_amp: D_wall amplitude constraint
  - wagenveld_wall_angle: alpha_wall angle constraint
  - catwise_amp: CatWISE QSO amplitude
  - catwise_dir: CatWISE QSO direction
  - radio_ratio: Radio dipole ratio constraint
"""

import math
import numpy as np
import argparse
import json
from typing import List, Tuple, Dict, Any, Optional

# -------------------------
# Physical constants
# -------------------------

C_KMS = 299792.458  # speed of light in km/s
V_CMB_KMS = 369.82  # CMB dipole velocity magnitude in km/s

# -------------------------
# Fixed inputs (data)
# -------------------------

L_CMB_DEG = 264.021
B_CMB_DEG = 48.253

D_WALL_MEAN = 0.0081
D_WALL_SIG  = 0.0014
ANG_WALL_CMB_MEAN_DEG = 39.0
ANG_WALL_CMB_SIG_DEG  = 8.0

D_QSO_OBS = 0.01554
L_QSO_OBS_DEG = 238.2
B_QSO_OBS_DEG = 28.8

R_RADIO_MEAN = 3.67
R_RADIO_SIG  = 0.49

# -------------------------
# Tunables
# -------------------------

SIGMA_QSO = 0.0010
SIGMA_QSO_DIR_DEG = 10.0
N_STEPS   = 250_000
BURN_IN   = 50_000
THIN      = 20
N_CHAINS  = 6

N_STEPS_COMPARE   = 80_000
BURN_IN_COMPARE   = 20_000

WRITE_DIAGNOSTICS = True
DIAGNOSTICS_CSV   = "angle_posterior.csv"
SAMPLES_CSV       = "posterior_samples.csv"
SUMMARY_JSON      = "posterior_summary.json"

STEP_V_DRIFT    = 50.0
STEP_L_DRIFT    = 5.0
STEP_B_DRIFT    = 3.0
STEP_D_WALL     = 0.00035
STEP_ALPHA_WALL = 2.5
STEP_PHI_WALL   = 8.0
STEP_A_Q        = 0.5
STEP_A_R        = 0.5

V_DRIFT_MAX = 2000.0
D_WALL_MAX  = 0.05
A_MAX       = 50.0

EPS_SIN = 1e-12
EPS_NORM = 1e-15

# Model modes
MODEL_MODES = ['full', 'wall_only', 'drift_only', 'cmb_only']

# Full parameter names (all 8)
FULL_PARAM_NAMES = [
    "v_drift_kms", "l_drift_deg", "b_drift_deg",
    "D_wall", "alpha_wall_deg", "phi_wall_deg",
    "A_Q", "A_R"
]

# Likelihood component names (5 components)
LL_COMPONENT_NAMES = [
    "wagenveld_wall_amp",
    "wagenveld_wall_angle",
    "catwise_amp",
    "catwise_dir",
    "radio_ratio"
]
N_LL_COMPONENTS = len(LL_COMPONENT_NAMES)

# Active parameters per mode
ACTIVE_PARAMS = {
    'full': ["v_drift_kms", "l_drift_deg", "b_drift_deg", "D_wall", "alpha_wall_deg", "phi_wall_deg", "A_Q", "A_R"],
    'wall_only': ["D_wall", "alpha_wall_deg", "phi_wall_deg", "A_Q", "A_R"],
    'drift_only': ["v_drift_kms", "l_drift_deg", "b_drift_deg", "A_Q", "A_R"],
    'cmb_only': ["A_Q", "A_R"],
}

# Fixed parameter values per mode
FIXED_PARAMS = {
    'full': {},
    'wall_only': {"v_drift_kms": 0.0, "l_drift_deg": 0.0, "b_drift_deg": 0.0},
    'drift_only': {"D_wall": 0.0, "alpha_wall_deg": 0.0, "phi_wall_deg": 0.0},
    'cmb_only': {"v_drift_kms": 0.0, "l_drift_deg": 0.0, "b_drift_deg": 0.0,
                 "D_wall": 0.0, "alpha_wall_deg": 0.0, "phi_wall_deg": 0.0},
}

# -------------------------
# Helper functions
# -------------------------

def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    l = np.deg2rad(l_deg % 360.0)
    b = np.deg2rad(np.clip(b_deg, -90.0, 90.0))
    cb = math.cos(b)
    return np.array([cb * math.cos(l), cb * math.sin(l), math.sin(b)], dtype=float)

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < EPS_NORM or norm_v < EPS_NORM:
        return np.nan
    cuv = float(np.dot(u, v) / (norm_u * norm_v))
    cuv = max(-1.0, min(1.0, cuv))
    return float(np.rad2deg(math.acos(cuv)))

def log_gauss(x: float, mu: float, sig: float) -> float:
    z = (x - mu) / sig
    return -0.5 * z * z

def wrap_lon(l_deg: float) -> float:
    return l_deg % 360.0

def reflect_lat(b_deg: float) -> Tuple[float, float]:
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
    return x % period

def reflect_alpha(alpha: float) -> Tuple[float, float]:
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
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dots = np.abs(axes @ n_cmb)
    ref_idx = np.argmin(dots)
    ref = axes[ref_idx]

    e1 = np.cross(ref, n_cmb)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_cmb, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2

def alpha_phi_to_unitvec(alpha_deg: float, phi_deg: float,
                          n_axis: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    alpha = math.radians(alpha_deg)
    phi = math.radians(phi_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cp, sp = math.cos(phi), math.sin(phi)
    return ca * n_axis + sa * (cp * e1 + sp * e2)

# -------------------------
# Precomputed fixed quantities
# -------------------------

n_CMB = lb_to_unitvec(L_CMB_DEG, B_CMB_DEG)
v_CMB_vec = V_CMB_KMS * n_CMB
n_QSO_obs = lb_to_unitvec(L_QSO_OBS_DEG, B_QSO_OBS_DEG)
e1_CMB, e2_CMB = build_cmb_basis(n_CMB)

# -------------------------
# Model configuration
# -------------------------

class ModelConfig:
    """Configuration for a specific model mode."""

    def __init__(self, mode: str):
        if mode not in MODEL_MODES:
            raise ValueError(f"Unknown model mode: {mode}. Must be one of {MODEL_MODES}")
        self.mode = mode
        self.active_params = ACTIVE_PARAMS[mode]
        self.fixed_params = FIXED_PARAMS[mode]
        self.n_params = len(self.active_params)

        # Build index mapping: active_idx -> full_idx
        self.active_to_full = [FULL_PARAM_NAMES.index(p) for p in self.active_params]

        # Step sizes for active params
        self.step_sizes = self._get_step_sizes()

    def _get_step_sizes(self) -> Dict[str, float]:
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

    def active_to_full_theta(self, theta_active: np.ndarray) -> np.ndarray:
        """Convert active parameter vector to full 8-parameter vector."""
        theta_full = np.zeros(8)

        # Set fixed values
        for pname, pval in self.fixed_params.items():
            idx = FULL_PARAM_NAMES.index(pname)
            theta_full[idx] = pval

        # Set active values
        for i, pname in enumerate(self.active_params):
            idx = FULL_PARAM_NAMES.index(pname)
            theta_full[idx] = theta_active[i]

        return theta_full

    def get_initial_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Get initial theta for active parameters."""
        # Default starting values for all params
        defaults = {
            "v_drift_kms": 100.0,
            "l_drift_deg": rng.uniform(0.0, 360.0),
            "b_drift_deg": rng.uniform(-60.0, 60.0),
            "D_wall": 0.0081,
            "alpha_wall_deg": 39.0,
            "phi_wall_deg": rng.uniform(0.0, 360.0),
            "A_Q": 6.0,
            "A_R": 2.0,
        }
        return np.array([defaults[p] for p in self.active_params], dtype=float)

    def has_drift(self) -> bool:
        return "v_drift_kms" in self.active_params

    def has_wall(self) -> bool:
        return "D_wall" in self.active_params

# -------------------------
# Model physics
# -------------------------

def compute_model_vectors(theta_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]:
    """Compute model dipole vectors from full 8-parameter theta."""
    v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R = theta_full

    n_drift = lb_to_unitvec(l_drift, b_drift)
    v_drift_vec = v_drift * n_drift
    v_tot_vec = v_CMB_vec + v_drift_vec
    v_tot_mag = np.linalg.norm(v_tot_vec)

    beta_tot = v_tot_mag / C_KMS
    n_vtot = v_tot_vec / v_tot_mag if v_tot_mag > EPS_NORM else n_CMB

    D_kin_Q = A_Q * beta_tot
    D_kin_R = A_R * beta_tot

    D_kin_Q_vec = D_kin_Q * n_vtot
    D_kin_R_vec = D_kin_R * n_vtot

    # Only compute wall term if D_wall is nonzero (avoids unnecessary work for drift_only/cmb_only)
    if abs(D_wall) > EPS_NORM:
        n_wall = alpha_phi_to_unitvec(alpha_wall, phi_wall, n_CMB, e1_CMB, e2_CMB)
        D_wall_vec = D_wall * n_wall
    else:
        D_wall_vec = np.zeros(3)

    D_Q_vec = D_kin_Q_vec + D_wall_vec
    D_R_vec = D_kin_R_vec + D_wall_vec

    return D_Q_vec, D_R_vec, beta_tot, n_vtot, D_kin_Q, D_kin_R

# -------------------------
# Prior and likelihood
# -------------------------

def log_prior(theta_active: np.ndarray, config: ModelConfig) -> float:
    """Compute log-prior for active parameters only."""
    theta_full = config.active_to_full_theta(theta_active)
    v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R = theta_full

    # Bounds for active params
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

    log_p = 0.0

    # Drift direction prior (uniform on sphere => prior ∝ cos(b))
    # Reject if cos(b) too small (poles are measure-zero)
    if config.has_drift():
        cb_drift = math.cos(math.radians(b_drift))
        if cb_drift <= EPS_SIN:
            return -np.inf
        log_p += math.log(cb_drift)

    # Wall direction prior (uniform on sphere => prior ∝ sin(alpha))
    # Reject if sin(alpha) too small (poles are measure-zero)
    if config.has_wall():
        sa_wall = math.sin(math.radians(alpha_wall))
        if sa_wall <= EPS_SIN:
            return -np.inf
        log_p += math.log(sa_wall)

    return log_p

def loglike_components(theta_active: np.ndarray, config: ModelConfig,
                       sigma_qso: float, sigma_qso_dir: float) -> Tuple[float, np.ndarray]:
    """
    Compute log-likelihood and return component-wise breakdown.

    Returns:
        (ll_total, ll_vec) where ll_vec has shape (5,) for the 5 components:
        [wagenveld_wall_amp, wagenveld_wall_angle, catwise_amp, catwise_dir, radio_ratio]

    Components that don't apply to a model (e.g., wall components for drift_only)
    are set to 0.0 (neutral contribution).
    """
    theta_full = config.active_to_full_theta(theta_active)
    D_wall = theta_full[3]
    alpha_wall = theta_full[4]

    D_Q_vec, D_R_vec, beta_tot, n_vtot, D_kin_Q, D_kin_R = compute_model_vectors(theta_full)

    D_Q_amp = float(np.linalg.norm(D_Q_vec))
    D_R_amp = float(np.linalg.norm(D_R_vec))

    # Initialize component vector
    ll_vec = np.zeros(N_LL_COMPONENTS)

    # Component 0: wagenveld_wall_amp
    # Component 1: wagenveld_wall_angle
    if config.has_wall():
        ll_vec[0] = log_gauss(D_wall, D_WALL_MEAN, D_WALL_SIG)
        ll_vec[1] = log_gauss(alpha_wall, ANG_WALL_CMB_MEAN_DEG, ANG_WALL_CMB_SIG_DEG)
    # else: already 0.0

    # Component 2: catwise_amp
    ll_vec[2] = log_gauss(D_Q_amp, D_QSO_OBS, sigma_qso)

    # Component 3: catwise_dir
    if D_Q_amp < EPS_NORM:
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf)
    dhat_Q = D_Q_vec / D_Q_amp
    ang_Q_dir = angle_deg(dhat_Q, n_QSO_obs)
    if np.isnan(ang_Q_dir):
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf)
    ll_vec[3] = log_gauss(ang_Q_dir, 0.0, sigma_qso_dir)

    # Component 4: radio_ratio
    if D_kin_R < EPS_NORM:
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf)
    R_model = D_R_amp / D_kin_R
    ll_vec[4] = log_gauss(R_model, R_RADIO_MEAN, R_RADIO_SIG)

    ll_total = float(np.sum(ll_vec))

    if not np.isfinite(ll_total):
        return -np.inf, np.full(N_LL_COMPONENTS, -np.inf)

    return ll_total, ll_vec

def log_likelihood(theta_active: np.ndarray, config: ModelConfig,
                   sigma_qso: float, sigma_qso_dir: float) -> float:
    """Compute total log-likelihood for active parameters."""
    ll_total, _ = loglike_components(theta_active, config, sigma_qso, sigma_qso_dir)
    return ll_total

def log_posterior(theta_active: np.ndarray, config: ModelConfig,
                  sigma_qso: float, sigma_qso_dir: float) -> float:
    lp = log_prior(theta_active, config)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta_active, config, sigma_qso, sigma_qso_dir)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# -------------------------
# Proposal
# -------------------------

def propose(theta: np.ndarray, rng: np.random.Generator, config: ModelConfig,
            step_overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Generate proposal for active parameters."""
    theta_new = theta.copy()
    steps = config.step_sizes.copy()
    if step_overrides:
        steps.update(step_overrides)

    for i, pname in enumerate(config.active_params):
        theta_new[i] += rng.normal(0.0, steps[pname])

    # Build full theta for coordinate wrapping
    theta_full = config.active_to_full_theta(theta_new)
    v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R = theta_full

    # Wrap/reflect coordinates
    if config.has_drift():
        b_drift, l_shift = reflect_lat(b_drift)
        l_drift = wrap_lon(l_drift + l_shift)
        if v_drift < 0.0: v_drift = -v_drift

    if config.has_wall():
        alpha_wall, phi_shift = reflect_alpha(alpha_wall)
        phi_wall = wrap_angle(phi_wall + phi_shift, 360.0)
        if D_wall < 0.0: D_wall = -D_wall

    if A_Q < 0.0: A_Q = -A_Q
    if A_R < 0.0: A_R = -A_R

    # Reconstruct active theta
    theta_full = np.array([v_drift, l_drift, b_drift, D_wall, alpha_wall, phi_wall, A_Q, A_R])
    theta_out = np.array([theta_full[FULL_PARAM_NAMES.index(p)] for p in config.active_params])

    return theta_out

# -------------------------
# MCMC chain
# -------------------------

def run_chain(theta0: np.ndarray, n_steps: int, rng: np.random.Generator,
              burn_in: int, thin: int, config: ModelConfig,
              sigma_qso: float, sigma_qso_dir: float,
              step_overrides: Optional[Dict[str, float]] = None,
              chain_id: int = 0, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run a single MCMC chain.

    Returns:
        (samples, log_likelihoods, ll_components, acceptance_rate)
        where ll_components has shape (n_samples, 5)
    """
    theta = theta0.copy()
    ll_current, ll_vec_current = loglike_components(theta, config, sigma_qso, sigma_qso_dir)
    logp = log_prior(theta, config) + ll_current

    if not np.isfinite(logp):
        raise RuntimeError(f"Chain {chain_id}: Initial theta has -inf posterior.")

    accepted = 0
    samples = []
    log_likes = []
    ll_components_list = []

    for t in range(n_steps):
        th_prop = propose(theta, rng, config, step_overrides)
        ll_prop, ll_vec_prop = loglike_components(th_prop, config, sigma_qso, sigma_qso_dir)
        lp_prop = log_prior(th_prop, config)
        logp_prop = lp_prop + ll_prop

        if np.isfinite(logp_prop):
            u = rng.random()
            log_u = math.log(u) if u > 0.0 else -np.inf
            if log_u < (logp_prop - logp):
                theta = th_prop
                logp = logp_prop
                ll_current = ll_prop
                ll_vec_current = ll_vec_prop
                accepted += 1

        if t >= burn_in and ((t - burn_in) % thin == 0):
            samples.append(theta.copy())
            log_likes.append(ll_current)
            ll_components_list.append(ll_vec_current.copy())

        if verbose and (t + 1) % 50_000 == 0:
            acc_rate = accepted / (t + 1)
            print(f"  Chain {chain_id}: Step {t+1}/{n_steps}  accept_rate={acc_rate:.3f}")

    return (np.array(samples), np.array(log_likes),
            np.array(ll_components_list), accepted / n_steps)

# -------------------------
# WAIC computation (component-wise)
# -------------------------

def compute_waic_components(ll_components: np.ndarray) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute WAIC from component-wise log-likelihoods.

    Args:
        ll_components: array of shape (S, K) where S = samples, K = 5 components

    Returns:
        (lppd_total, p_waic_total, waic_total, component_details)

    For each component k:
      lppd_k = log(mean(exp(ll[:,k])))  [using log-sum-exp trick]
      p_waic_k = var(ll[:,k], ddof=0)

    Total:
      lppd = sum(lppd_k)
      p_waic = sum(p_waic_k)
      waic = -2 * (lppd - p_waic)
    """
    n_samples, n_components = ll_components.shape

    if n_samples < 2:
        return float(np.nan), 0.0, float(np.nan), {}

    lppd_k = np.zeros(n_components)
    p_waic_k = np.zeros(n_components)

    for k in range(n_components):
        ll_k = ll_components[:, k]

        # Check if this component has any non-zero values
        if np.all(ll_k == 0.0):
            # Component not active for this model
            lppd_k[k] = 0.0
            p_waic_k[k] = 0.0
        else:
            # Log-sum-exp for numerical stability
            ll_max = np.max(ll_k)
            lppd_k[k] = ll_max + np.log(np.mean(np.exp(ll_k - ll_max)))
            p_waic_k[k] = np.var(ll_k, ddof=0)

    lppd_total = float(np.sum(lppd_k))
    p_waic_total = float(np.sum(p_waic_k))
    waic_total = -2.0 * (lppd_total - p_waic_total)

    component_details = {
        "lppd_components": {name: float(lppd_k[i]) for i, name in enumerate(LL_COMPONENT_NAMES)},
        "p_waic_components": {name: float(p_waic_k[i]) for i, name in enumerate(LL_COMPONENT_NAMES)},
        "waic_components": {name: float(-2.0 * (lppd_k[i] - p_waic_k[i]))
                           for i, name in enumerate(LL_COMPONENT_NAMES)}
    }

    return lppd_total, p_waic_total, waic_total, component_details

def compute_waic(log_likes: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute WAIC from total log-likelihood samples (legacy single-datum).
    Kept for backwards compatibility.
    """
    n = len(log_likes)
    if n < 2:
        return float(np.nan), 0.0, float(np.nan)

    ll_max = np.max(log_likes)
    lppd = ll_max + np.log(np.mean(np.exp(log_likes - ll_max)))
    p_waic = np.var(log_likes, ddof=0)
    waic = -2.0 * (lppd - p_waic)

    return float(lppd), float(p_waic), float(waic)

# -------------------------
# Convergence diagnostics
# -------------------------

def gelman_rubin(chains: List[np.ndarray]) -> np.ndarray:
    n_chains = len(chains)
    if n_chains < 2:
        return np.full(chains[0].shape[1], np.nan)

    min_samples = min(c.shape[0] for c in chains)
    if min_samples < 2:
        return np.full(chains[0].shape[1], np.nan)

    n_samples = min_samples
    chains = [c[:n_samples] for c in chains]
    n_params = chains[0].shape[1]
    R_hat = np.ones(n_params)

    for p in range(n_params):
        param_chains = [c[:, p] for c in chains]
        chain_means = np.array([np.mean(pc) for pc in param_chains])
        chain_vars = np.array([np.var(pc, ddof=1) if len(pc) > 1 else 0.0 for pc in param_chains])

        W = np.mean(chain_vars)
        B = n_samples * np.var(chain_means, ddof=1) if n_chains > 1 else 0.0

        if W < EPS_NORM:
            R_hat[p] = 1.0 if B < EPS_NORM else np.nan
            continue

        var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        R_hat[p] = math.sqrt(var_plus / W)

    return R_hat

def effective_sample_size(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    n_samples, n_params = samples.shape
    if max_lag is None:
        max_lag = min(1000, n_samples // 5)
    max_lag = max(1, min(max_lag, n_samples // 2))

    ess = np.zeros(n_params)
    for p in range(n_params):
        x = samples[:, p]
        x_centered = x - np.mean(x)
        var_x = np.var(x, ddof=0)

        if var_x < EPS_NORM:
            ess[p] = float(n_samples)
            continue

        acf_sum = 0.0
        for lag in range(1, max_lag + 1):
            n_pairs = n_samples - lag
            if n_pairs < 1:
                break
            acf = np.sum(x_centered[:n_pairs] * x_centered[lag:lag + n_pairs]) / (n_pairs * var_x)
            if acf < 0.0:
                break
            acf_sum += acf

        tau = 1.0 + 2.0 * acf_sum
        ess[p] = n_samples / max(tau, 1.0)

    return ess

def combined_ess(chains: List[np.ndarray]) -> np.ndarray:
    if len(chains) == 0:
        return np.array([])
    combined = np.vstack(chains)
    return effective_sample_size(combined)

# -------------------------
# Statistics
# -------------------------

def compute_param_stats(samples: np.ndarray, names: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for i, name in enumerate(names):
        x = samples[:, i]
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        stats[name] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(q50),
            "q16": float(q16),
            "q84": float(q84)
        }
    return stats

def compute_derived_quantities(samples: np.ndarray, config: ModelConfig) -> Tuple[np.ndarray, ...]:
    """Compute derived quantities. Returns arrays for each quantity."""
    n_samples = len(samples)
    beta_tot_arr = np.empty(n_samples)
    v_tot_kms_arr = np.empty(n_samples)
    D_kin_Q_arr = np.empty(n_samples)
    D_kin_R_arr = np.empty(n_samples)
    catwise_sep_arr = np.empty(n_samples)
    radio_ratio_arr = np.empty(n_samples)

    for k in range(n_samples):
        theta_full = config.active_to_full_theta(samples[k])
        D_Q_vec, D_R_vec, beta_tot, n_vtot, D_kin_Q, D_kin_R = compute_model_vectors(theta_full)

        beta_tot_arr[k] = beta_tot
        v_tot_kms_arr[k] = beta_tot * C_KMS
        D_kin_Q_arr[k] = D_kin_Q
        D_kin_R_arr[k] = D_kin_R

        D_Q_amp = np.linalg.norm(D_Q_vec)
        D_R_amp = np.linalg.norm(D_R_vec)

        if D_Q_amp > EPS_NORM:
            catwise_sep_arr[k] = angle_deg(D_Q_vec / D_Q_amp, n_QSO_obs)
        else:
            catwise_sep_arr[k] = np.nan

        if D_kin_R > EPS_NORM:
            radio_ratio_arr[k] = D_R_amp / D_kin_R
        else:
            radio_ratio_arr[k] = np.nan

    return beta_tot_arr, v_tot_kms_arr, D_kin_Q_arr, D_kin_R_arr, catwise_sep_arr, radio_ratio_arr

def compute_derived_stats(samples: np.ndarray, config: ModelConfig) -> Dict[str, Any]:
    derived = compute_derived_quantities(samples, config)
    beta_tot, v_tot_kms, D_kin_Q, D_kin_R, catwise_sep, radio_ratio = derived

    def safe_stats(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return {"mean": np.nan, "median": np.nan, "q16": np.nan, "q84": np.nan}
        q16, q50, q84 = np.percentile(valid, [16, 50, 84])
        return {"mean": float(np.mean(valid)), "median": float(q50),
                "q16": float(q16), "q84": float(q84)}

    result = {
        "beta_tot": safe_stats(beta_tot),
        "v_tot_kms": safe_stats(v_tot_kms),
        "D_kin_Q": safe_stats(D_kin_Q),
        "D_kin_R": safe_stats(D_kin_R),
        "catwise_separation": safe_stats(catwise_sep),
        "radio_ratio": safe_stats(radio_ratio)
    }

    if config.has_wall():
        alpha_wall_idx = config.active_params.index("alpha_wall_deg")
        alpha_wall = samples[:, alpha_wall_idx]
        in_1sigma = np.sum((alpha_wall >= 31.0) & (alpha_wall <= 47.0))
        result["alpha_wall_fraction_in_31_47"] = float(in_1sigma / len(samples))

    return result

def compute_ll_component_stats(ll_components: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for each log-likelihood component."""
    stats = {}
    for i, name in enumerate(LL_COMPONENT_NAMES):
        ll_k = ll_components[:, i]
        if np.all(ll_k == 0.0):
            stats[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        else:
            stats[name] = {
                "mean": float(np.mean(ll_k)),
                "std": float(np.std(ll_k)),
                "min": float(np.min(ll_k)),
                "max": float(np.max(ll_k))
            }
    return stats

# -------------------------
# Multi-chain runner
# -------------------------

def run_multi_chain(config: ModelConfig, n_chains: int, n_steps: int, burn_in: int, thin: int,
                    sigma_qso: float, sigma_qso_dir: float,
                    step_overrides: Optional[Dict[str, float]] = None,
                    base_seed: int = 42, verbose: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Run multiple chains.

    Returns:
        (chains, log_likes_per_chain, ll_components_per_chain, acceptance_rates)
    """
    chains = []
    log_likes_list = []
    ll_components_list = []
    acceptance_rates = []

    for c in range(n_chains):
        seed = base_seed + c * 1000
        rng = np.random.default_rng(seed)
        theta0 = config.get_initial_theta(rng)

        if verbose:
            print(f"\nStarting chain {c} with seed={seed}")

        samples, log_likes, ll_comps, acc = run_chain(
            theta0, n_steps, rng, burn_in, thin, config,
            sigma_qso, sigma_qso_dir, step_overrides,
            chain_id=c, verbose=verbose
        )
        chains.append(samples)
        log_likes_list.append(log_likes)
        ll_components_list.append(ll_comps)
        acceptance_rates.append(acc)

        if verbose:
            print(f"  Chain {c} done: {len(samples)} samples, acceptance={acc:.3f}")

    return chains, log_likes_list, ll_components_list, acceptance_rates

# -------------------------
# Export
# -------------------------

def write_samples_csv(chains: List[np.ndarray], log_likes_list: List[np.ndarray],
                      ll_components_list: List[np.ndarray],
                      config: ModelConfig, filename: str):
    """Write samples to CSV with all 8 parameters, derived columns, and 5 ll components."""
    with open(filename, 'w') as f:
        # Header: chain_id, mode, 8 params, derived, total ll, 5 ll components
        cols = ["chain_id", "model_mode"] + FULL_PARAM_NAMES + ["beta_tot", "v_tot_kms", "loglike"]
        cols += ["ll_" + name for name in LL_COMPONENT_NAMES]
        f.write(",".join(cols) + "\n")

        for chain_id, (samples, log_likes, ll_comps) in enumerate(zip(chains, log_likes_list, ll_components_list)):
            for row, ll, ll_vec in zip(samples, log_likes, ll_comps):
                theta_full = config.active_to_full_theta(row)
                D_Q_vec, D_R_vec, beta_tot, _, _, _ = compute_model_vectors(theta_full)
                v_tot_kms = beta_tot * C_KMS

                line = f"{chain_id},{config.mode}"
                # Write all 8 parameters in canonical order
                for val in theta_full:
                    line += f",{val:.8g}"
                line += f",{beta_tot:.8g},{v_tot_kms:.4f},{ll:.6f}"
                # Write 5 ll components
                for ll_k in ll_vec:
                    line += f",{ll_k:.6f}"
                line += "\n"
                f.write(line)

def write_summary_json(chains: List[np.ndarray], log_likes_list: List[np.ndarray],
                       ll_components_list: List[np.ndarray],
                       acceptance_rates: List[float], R_hat: np.ndarray, ess: np.ndarray,
                       config: ModelConfig, waic_result: Tuple[float, float, float, Dict],
                       filename: str):
    """Write posterior summary to JSON with model metadata and WAIC components."""
    combined = np.vstack(chains)
    combined_ll_comps = np.vstack(ll_components_list)
    lppd, p_waic, waic, waic_components = waic_result

    param_stats = compute_param_stats(combined, config.active_params)
    derived_stats = compute_derived_stats(combined, config)
    ll_comp_stats = compute_ll_component_stats(combined_ll_comps)

    for i, name in enumerate(config.active_params):
        param_stats[name]["R_hat"] = float(R_hat[i]) if np.isfinite(R_hat[i]) else None
        param_stats[name]["ESS"] = float(ess[i]) if np.isfinite(ess[i]) else None

    summary = {
        "model_mode": config.mode,
        "active_parameters": config.active_params,
        "fixed_parameters": config.fixed_params,
        "n_chains": len(chains),
        "samples_per_chain": int(chains[0].shape[0]) if chains else 0,
        "total_samples": int(combined.shape[0]),
        "acceptance_rates": [float(a) for a in acceptance_rates],
        "waic": {
            "lppd": lppd,
            "p_waic": p_waic,
            "waic": waic
        },
        "waic_components": waic_components,
        "ll_component_stats": ll_comp_stats,
        "parameters": param_stats,
        "derived": derived_stats,
        "convergence": {
            "R_hat": {name: (float(R_hat[i]) if np.isfinite(R_hat[i]) else None)
                      for i, name in enumerate(config.active_params)},
            "ESS": {name: (float(ess[i]) if np.isfinite(ess[i]) else None)
                    for i, name in enumerate(config.active_params)}
        }
    }

    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)

# -------------------------
# Single model run
# -------------------------

def run_single_model(config: ModelConfig, args, verbose: bool = True) -> Dict[str, Any]:
    """Run MCMC for a single model. Returns results dict."""
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
    # Filter to active params only
    step_overrides = {k: v for k, v in all_steps.items() if k in config.active_params}

    if verbose:
        print("="*60)
        print(f"Model: {config.mode}")
        print(f"Active parameters: {config.active_params}")
        print(f"Fixed: {config.fixed_params}")
        print("="*60)

    chains, log_likes_list, ll_components_list, acceptance_rates = run_multi_chain(
        config, args.n_chains, args.n_steps, args.burn_in, args.thin,
        args.sigma_qso, args.sigma_qso_dir, step_overrides,
        args.seed, verbose
    )

    combined = np.vstack(chains)
    combined_ll = np.concatenate(log_likes_list)
    combined_ll_comps = np.vstack(ll_components_list)

    R_hat = gelman_rubin(chains)
    ess = combined_ess(chains)
    waic_result = compute_waic_components(combined_ll_comps)

    if verbose:
        lppd, p_waic, waic, waic_details = waic_result
        print("\n" + "-"*40)
        print("WAIC (component-wise):")
        print(f"  lppd   = {lppd:.4f}")
        print(f"  p_waic = {p_waic:.4f}")
        print(f"  WAIC   = {waic:.4f}")
        print("\n  Component breakdown:")
        for name in LL_COMPONENT_NAMES:
            lppd_k = waic_details["lppd_components"][name]
            p_k = waic_details["p_waic_components"][name]
            waic_k = waic_details["waic_components"][name]
            print(f"    {name:22s}: lppd={lppd_k:8.3f}, p_waic={p_k:8.3f}, waic={waic_k:8.3f}")

        print(f"\nLog-likelihood stats (total):")
        print(f"  mean = {np.mean(combined_ll):.4f}, std = {np.std(combined_ll):.4f}")
        print(f"  min  = {np.min(combined_ll):.4f}, max = {np.max(combined_ll):.4f}")

    return {
        "config": config,
        "chains": chains,
        "log_likes_list": log_likes_list,
        "ll_components_list": ll_components_list,
        "acceptance_rates": acceptance_rates,
        "R_hat": R_hat,
        "ess": ess,
        "waic_result": waic_result,
        "combined": combined,
        "combined_ll": combined_ll,
        "combined_ll_comps": combined_ll_comps
    }

# -------------------------
# Model comparison
# -------------------------

def run_model_comparison(args, verbose: bool = True) -> List[Dict[str, Any]]:
    """Run all models and compare WAIC."""
    results = []

    # Use comparison defaults if not overridden
    n_steps = args.n_steps if args.n_steps != N_STEPS else N_STEPS_COMPARE
    burn_in = args.burn_in if args.burn_in != BURN_IN else BURN_IN_COMPARE

    # Create modified args
    class CompareArgs:
        pass
    cargs = CompareArgs()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(cargs, attr, getattr(args, attr))
    cargs.n_steps = n_steps
    cargs.burn_in = burn_in

    print("="*60)
    print("MODEL COMPARISON")
    print(f"Running all {len(MODEL_MODES)} models with n_steps={n_steps}, burn_in={burn_in}")
    print("="*60)

    for mode in MODEL_MODES:
        print(f"\n{'='*60}")
        print(f"Running model: {mode}")
        print("="*60)

        config = ModelConfig(mode)
        result = run_single_model(config, cargs, verbose=verbose)
        results.append(result)

    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS (ranked by WAIC)")
    print("="*60)

    # Sort by WAIC (lower is better)
    sorted_results = sorted(results, key=lambda r: r["waic_result"][2])

    print(f"\n{'Model':<15} {'WAIC':>12} {'lppd':>12} {'p_waic':>10} {'n_params':>10}")
    print("-"*60)

    best_waic = sorted_results[0]["waic_result"][2]
    for r in sorted_results:
        mode = r["config"].mode
        lppd, p_waic, waic, waic_details = r["waic_result"]
        n_params = r["config"].n_params
        delta = waic - best_waic
        delta_str = f"(+{delta:.1f})" if delta > 0 else "(best)"
        ll_mean = np.mean(r["combined_ll"])
        ll_std = np.std(r["combined_ll"])
        print(f"{mode:<15} {waic:>12.2f} {lppd:>12.2f} {p_waic:>10.2f} {n_params:>10} {delta_str}")
        print(f"{'':15} ll: mean={ll_mean:.2f}, std={ll_std:.2f}, "
              f"range=[{np.min(r['combined_ll']):.2f},{np.max(r['combined_ll']):.2f}]")

    # Show component-wise ΔWAIC contributions
    print("\n" + "="*60)
    print("COMPONENT-WISE WAIC CONTRIBUTIONS")
    print("="*60)

    best_result = sorted_results[0]
    best_waic_comps = best_result["waic_result"][3]["waic_components"]

    for r in sorted_results[1:]:  # Skip the best model
        mode = r["config"].mode
        waic_comps = r["waic_result"][3]["waic_components"]
        print(f"\n{mode} vs {best_result['config'].mode}:")

        delta_list = []
        for name in LL_COMPONENT_NAMES:
            delta_k = waic_comps[name] - best_waic_comps[name]
            delta_list.append((name, delta_k))

        # Sort by absolute contribution
        delta_list.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, delta_k in delta_list:
            if abs(delta_k) > 0.01:  # Only show non-trivial contributions
                sign = "+" if delta_k > 0 else ""
                print(f"  {name:22s}: {sign}{delta_k:.3f}")

    return results

# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Bubble-mapping dipole MCMC sampler with model comparison")

    parser.add_argument("--model", type=str, default="full", choices=MODEL_MODES,
                        help=f"Model mode (default: full)")
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

if __name__ == "__main__":
    args = parse_args()
    verbose = not args.quiet

    if args.compare_models:
        results = run_model_comparison(args, verbose)

        # Save comparison summary
        comparison_file = "model_comparison.json"
        comparison_data = []
        for r in sorted(results, key=lambda x: x["waic_result"][2]):
            lppd, p_waic, waic, waic_details = r["waic_result"]
            comparison_data.append({
                "model": r["config"].mode,
                "waic": waic,
                "lppd": lppd,
                "p_waic": p_waic,
                "waic_components": waic_details["waic_components"],
                "n_params": r["config"].n_params,
                "active_params": r["config"].active_params,
                "mean_acceptance": float(np.mean(r["acceptance_rates"]))
            })
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nSaved comparison to: {comparison_file}")

    else:
        config = ModelConfig(args.model)

        if verbose:
            print("="*60)
            print("Bubble-Mapping Dipole MCMC Sampler")
            print("="*60)
            print(f"Model: {config.mode}")
            print(f"Active parameters: {config.active_params}")
            print(f"Fixed: {config.fixed_params}")
            print(f"N_STEPS={args.n_steps}, BURN_IN={args.burn_in}, THIN={args.thin}")
            print(f"N_CHAINS={args.n_chains}, SEED={args.seed}")
            print(f"V_CMB = {V_CMB_KMS:.2f} km/s")
            print("="*60)

        result = run_single_model(config, args, verbose)

        chains = result["chains"]
        log_likes_list = result["log_likes_list"]
        ll_components_list = result["ll_components_list"]
        acceptance_rates = result["acceptance_rates"]
        R_hat = result["R_hat"]
        ess = result["ess"]
        waic_result = result["waic_result"]
        combined = result["combined"]

        if verbose:
            print("\n" + "-"*40)
            print("Acceptance rates per chain:")
            for c, acc in enumerate(acceptance_rates):
                print(f"  Chain {c}: {acc:.3f}")

            print("\n" + "-"*40)
            print("Convergence diagnostics:")
            print(f"{'Parameter':<16} {'R-hat':>8} {'ESS':>10}")
            for i, name in enumerate(config.active_params):
                r_str = f"{R_hat[i]:.4f}" if np.isfinite(R_hat[i]) else "N/A"
                e_str = f"{ess[i]:.1f}" if np.isfinite(ess[i]) else "N/A"
                warning = ""
                if np.isfinite(R_hat[i]) and R_hat[i] > 1.05: warning += " ***"
                if np.isfinite(ess[i]) and ess[i] < 100: warning += " (low)"
                print(f"{name:<16} {r_str:>8} {e_str:>10}{warning}")

            print("\n" + "-"*40)
            print("Posterior summaries:")
            for i, nm in enumerate(config.active_params):
                x = combined[:, i]
                q16, q50, q84 = np.percentile(x, [16, 50, 84])
                print(f"{nm:16s}  mean={np.mean(x):10.4f} std={np.std(x):10.4f}  "
                      f"median={q50:10.4f}  [16,84]=[{q16:10.4f},{q84:10.4f}]")

            derived = compute_derived_stats(combined, config)
            print("\n" + "-"*40)
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
            print("\n" + "-"*40)
            print("Exporting results...")

        write_samples_csv(chains, log_likes_list, ll_components_list, config, SAMPLES_CSV)
        if verbose:
            print(f"  Wrote samples to: {SAMPLES_CSV}")

        write_summary_json(chains, log_likes_list, ll_components_list,
                           acceptance_rates, R_hat, ess,
                           config, waic_result, SUMMARY_JSON)
        if verbose:
            print(f"  Wrote summary to: {SUMMARY_JSON}")

        if WRITE_DIAGNOSTICS and config.has_wall():
            alpha_idx = config.active_params.index("alpha_wall_deg")
            alpha_samples = combined[:, alpha_idx]
            with open(DIAGNOSTICS_CSV, 'w') as f:
                f.write("alpha_wall_deg\n")
                for a in alpha_samples:
                    f.write(f"{a:.6f}\n")
            if verbose:
                print(f"  Wrote alpha_wall posterior to: {DIAGNOSTICS_CSV}")

        if verbose:
            print("\n" + "="*60)
            print("Done.")
            print("="*60)
