#!/usr/bin/env python3
"""
Metropolis-Hastings sampler for a "kinematic + universal residual dipole" model.

Model:
  D_Q_model = D_kin_Q * n_CMB + D_res * n_res
  D_R_model = D_kin_R * n_CMB + D_res * n_res

Likelihood constraints (Gaussian):
  Wagenveld residual amplitude: D_res = 0.0081 ± 0.0014
  Wagenveld angle offset: angle(n_res, n_CMB) = 39° ± 8°
  CatWISE quasar dipole amplitude: |D_Q_model| = 0.01554 ± sigma_Q
  Radio ratio (Böhme): |D_R_model| / D_kin_R = 3.67 ± 0.49

Notes:
- This is intentionally phenomenological. It estimates the residual vector and catalog-dependent kinematic amplitudes.
- Replace sigma_Q with the published uncertainty you trust when you have it.
"""

import math
import numpy as np

# -------------------------
# Fixed inputs (data)
# -------------------------

# CMB dipole direction in Galactic coords (l,b) degrees
L_CMB_DEG = 264.021
B_CMB_DEG = 48.253

# Wagenveld residual constraints
D_RES_MEAN = 0.0081
D_RES_SIG  = 0.0014
ANG_RES_CMB_MEAN_DEG = 39.0
ANG_RES_CMB_SIG_DEG  = 8.0

# CatWISE quasar observed dipole amplitude and direction
D_QSO_OBS = 0.01554
L_QSO_OBS_DEG = 238.2
B_QSO_OBS_DEG = 28.8

# Radio ratio constraint (Böhme)
R_RADIO_MEAN = 3.67
R_RADIO_SIG  = 0.49

# -------------------------
# Tunables (you can change)
# -------------------------

SIGMA_QSO = 0.0010  # placeholder; set to your preferred/quoted uncertainty
N_STEPS   = 250_000
BURN_IN   = 50_000
THIN      = 20

# Diagnostics
WRITE_DIAGNOSTICS = True
DIAGNOSTICS_CSV   = "angle_posterior.csv"

# Proposal step sizes
STEP_DRES      = 0.00035
STEP_L_DEG     = 3.0
STEP_B_DEG     = 2.0
STEP_DKIN_Q    = 0.00040
STEP_DKIN_R    = 0.00040

# Prior bounds (simple, broad)
D_MAX = 0.05

rng = np.random.default_rng(7)

# -------------------------
# Helper functions
# -------------------------

def lb_to_unitvec(l_deg: float, b_deg: float) -> np.ndarray:
    """Convert Galactic lon/lat in degrees to 3D unit vector."""
    l = np.deg2rad(l_deg % 360.0)
    b = np.deg2rad(np.clip(b_deg, -90.0, 90.0))
    cb = math.cos(b)
    return np.array([cb * math.cos(l), cb * math.sin(l), math.sin(b)], dtype=float)

def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees."""
    cuv = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    cuv = max(-1.0, min(1.0, cuv))
    return float(np.rad2deg(math.acos(cuv)))

def log_gauss(x: float, mu: float, sig: float) -> float:
    """Log of unnormalized Gaussian N(mu,sig^2) evaluated at x."""
    z = (x - mu) / sig
    return -0.5 * z * z

def wrap_lon(l_deg: float) -> float:
    return l_deg % 360.0

def reflect_lat(b_deg: float) -> float:
    """
    Reflect latitude into [-90, 90] while keeping continuity.
    If you step beyond poles, reflect like spherical coordinates.
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

# -------------------------
# Model and posterior
# -------------------------

n_CMB = lb_to_unitvec(L_CMB_DEG, B_CMB_DEG)

def log_prior(theta: np.ndarray) -> float:
    """
    theta = [D_res, l_res_deg, b_res_deg, D_kin_Q, D_kin_R]
    Prior: uniform within bounds, plus uniform-on-sphere for direction via cos(b) Jacobian.
    """
    D_res, l_res, b_res, DkQ, DkR = theta

    if not (0.0 <= D_res <= D_MAX): return -np.inf
    if not (0.0 <= DkQ  <= D_MAX): return -np.inf
    if not (0.0 <= DkR  <= D_MAX): return -np.inf
    if not (0.0 <= l_res < 360.0): return -np.inf
    if not (-90.0 <= b_res <= 90.0): return -np.inf

    # Uniform on sphere for direction => prior ∝ cos(b)
    cb = math.cos(math.radians(b_res))
    if cb <= 0.0:  # at poles cos(b)=0, effectively measure-zero
        return -np.inf
    return math.log(cb)

def log_likelihood(theta: np.ndarray) -> float:
    D_res, l_res, b_res, DkQ, DkR = theta
    n_res = lb_to_unitvec(l_res, b_res)

    # Wagenveld residual amplitude constraint
    ll = log_gauss(D_res, D_RES_MEAN, D_RES_SIG)

    # Wagenveld angle constraint: angle between residual direction and CMB direction
    ang = angle_deg(n_res, n_CMB)
    ll += log_gauss(ang, ANG_RES_CMB_MEAN_DEG, ANG_RES_CMB_SIG_DEG)

    # Model vectors
    D_res_vec = D_res * n_res
    D_Q_vec   = DkQ * n_CMB + D_res_vec
    D_R_vec   = DkR * n_CMB + D_res_vec

    D_Q_amp = float(np.linalg.norm(D_Q_vec))
    D_R_amp = float(np.linalg.norm(D_R_vec))

    # CatWISE amplitude constraint
    ll += log_gauss(D_Q_amp, D_QSO_OBS, SIGMA_QSO)

    # Radio ratio constraint: |D_R| / DkR = 3.67 ± 0.49
    # Guard against division by ~0
    if DkR <= 1e-6:
        return -np.inf
    R_model = D_R_amp / DkR
    ll += log_gauss(R_model, R_RADIO_MEAN, R_RADIO_SIG)

    return ll

def log_posterior(theta: np.ndarray) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# -------------------------
# Metropolis-Hastings sampler
# -------------------------

def propose(theta: np.ndarray) -> np.ndarray:
    D_res, l_res, b_res, DkQ, DkR = theta.copy()

    D_res += rng.normal(0.0, STEP_DRES)
    l_res += rng.normal(0.0, STEP_L_DEG)
    b_res += rng.normal(0.0, STEP_B_DEG)
    DkQ   += rng.normal(0.0, STEP_DKIN_Q)
    DkR   += rng.normal(0.0, STEP_DKIN_R)

    # Wrap/reflect angular coordinates
    b_res, l_shift = reflect_lat(b_res)
    l_res = wrap_lon(l_res + l_shift)

    # Enforce non-negativity softly by reflecting (keeps proposals in support)
    if D_res < 0.0: D_res = -D_res
    if DkQ  < 0.0: DkQ  = -DkQ
    if DkR  < 0.0: DkR  = -DkR

    return np.array([D_res, l_res, b_res, DkQ, DkR], dtype=float)

def run_chain(theta0: np.ndarray, n_steps: int):
    theta = theta0.copy()
    logp  = log_posterior(theta)
    if not np.isfinite(logp):
        raise RuntimeError("Initial theta has -inf posterior. Choose a better start.")

    accepted = 0
    samples = []

    for t in range(n_steps):
        th_prop = propose(theta)
        logp_prop = log_posterior(th_prop)

        if np.isfinite(logp_prop):
            # MH acceptance
            if math.log(rng.random()) < (logp_prop - logp):
                theta = th_prop
                logp = logp_prop
                accepted += 1

        # Save thinned post-burn samples
        if t >= BURN_IN and ((t - BURN_IN) % THIN == 0):
            samples.append(theta.copy())

        if (t + 1) % 50_000 == 0:
            acc_rate = accepted / (t + 1)
            print(f"Step {t+1}/{n_steps}  accept_rate={acc_rate:.3f}  current={theta}")

    return np.array(samples), accepted / n_steps

# -------------------------
# Initialization
# -------------------------

# A reasonable starting point near Wagenveld residual mean, with direction offset ~39 deg
# We'll pick l,b near the CMB direction but shifted; the sampler will move it.
theta0 = np.array([
    0.0081,   # D_res
    240.0,    # l_res
    20.0,     # b_res
    0.0080,   # D_kin_Q
    0.0040,   # D_kin_R
], dtype=float)

# -------------------------
# Run and summarize
# -------------------------

if __name__ == "__main__":
    print("Running Metropolis-Hastings...")
    print(f"N_STEPS={N_STEPS}, BURN_IN={BURN_IN}, THIN={THIN}, SIGMA_QSO={SIGMA_QSO}")
    samples, acc = run_chain(theta0, N_STEPS)
    print(f"\nDone. Saved samples: {len(samples)}  Overall acceptance: {acc:.3f}")

    # Posterior summaries
    names = ["D_res", "l_res_deg", "b_res_deg", "D_kin_Q", "D_kin_R"]
    for i, nm in enumerate(names):
        x = samples[:, i]
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        mean = np.mean(x)
        std  = np.std(x)
        print(f"{nm:10s}  mean={mean:.6f} std={std:.6f}  "
              f"median={q50:.6f}  [16,84]=[{q16:.6f},{q84:.6f}]")

    # Derived quantities: predicted amplitudes and radio ratio
    n_res_m = lb_to_unitvec(np.median(samples[:,1]), np.median(samples[:,2]))
    D_res_m, DkQ_m, DkR_m = np.median(samples[:,0]), np.median(samples[:,3]), np.median(samples[:,4])
    D_res_vec_m = D_res_m * n_res_m
    D_Q_vec_m   = DkQ_m * n_CMB + D_res_vec_m
    D_R_vec_m   = DkR_m * n_CMB + D_res_vec_m
    D_Q_amp_m   = float(np.linalg.norm(D_Q_vec_m))
    D_R_amp_m   = float(np.linalg.norm(D_R_vec_m))
    R_m         = D_R_amp_m / DkR_m

    ang_m = angle_deg(n_res_m, n_CMB)

    print("\nDerived (using medians):")
    print(f"angle(res, CMB) = {ang_m:.2f} deg")
    print(f"|D_Q_model|     = {D_Q_amp_m:.5f}  (target {D_QSO_OBS:.5f})")
    print(f"|D_R_model|     = {D_R_amp_m:.5f}")
    print(f"R_radio_model   = {R_m:.3f}  (target {R_RADIO_MEAN:.2f}±{R_RADIO_SIG:.2f})")

    # -------------------------
    # Diagnostics: angle posterior computed per-sample
    # -------------------------
    print("\n" + "="*60)
    print("DIAGNOSTICS: Per-sample angle computations")
    print("="*60)

    n_samples = len(samples)
    alpha_samples = np.empty(n_samples)
    catwise_sep_samples = np.empty(n_samples)
    n_QSO_obs = lb_to_unitvec(L_QSO_OBS_DEG, B_QSO_OBS_DEG)

    for k in range(n_samples):
        D_res_k, l_res_k, b_res_k, DkQ_k, DkR_k = samples[k]
        n_res_k = lb_to_unitvec(l_res_k, b_res_k)

        # Angle between residual direction and CMB direction
        alpha_samples[k] = angle_deg(n_res_k, n_CMB)

        # Model quasar dipole vector and direction
        D_Q_vec_k = DkQ_k * n_CMB + D_res_k * n_res_k
        D_Q_amp_k = np.linalg.norm(D_Q_vec_k)
        if D_Q_amp_k > 1e-12:
            dhat_Q_k = D_Q_vec_k / D_Q_amp_k
        else:
            dhat_Q_k = n_CMB  # fallback, shouldn't happen
        catwise_sep_samples[k] = angle_deg(dhat_Q_k, n_QSO_obs)

    # Alpha (residual-CMB angle) statistics
    alpha_mean = np.mean(alpha_samples)
    alpha_std = np.std(alpha_samples)
    alpha_q16, alpha_median, alpha_q84 = np.percentile(alpha_samples, [16, 50, 84])

    print(f"\nalpha = angle(n_res, n_CMB) over {n_samples} samples:")
    print(f"  mean   = {alpha_mean:.2f} deg")
    print(f"  std    = {alpha_std:.2f} deg")
    print(f"  median = {alpha_median:.2f} deg")
    print(f"  [16,84] = [{alpha_q16:.2f}, {alpha_q84:.2f}] deg")

    # Fraction within 1-sigma of Wagenveld constraint (39 +/- 8 => [31, 47])
    in_1sigma = np.sum((alpha_samples >= 31.0) & (alpha_samples <= 47.0))
    frac_1sigma = in_1sigma / n_samples
    print(f"\n  Fraction with alpha in [31, 47] deg: {frac_1sigma:.3f} ({in_1sigma}/{n_samples})")

    # Warning if alpha posterior not centered near 39 deg
    if alpha_median < 31.0 or alpha_median > 47.0:
        print(f"\n  *** WARNING: alpha median ({alpha_median:.2f} deg) is OUTSIDE the 1-sigma")
        print(f"      Wagenveld constraint range [31, 47] deg. This may indicate a")
        print(f"      likelihood conflict or multi-modal posterior. ***")
    else:
        print(f"\n  (OK: alpha median is within [31, 47] deg)")

    # CatWISE model dipole direction separation from observed
    sep_mean = np.mean(catwise_sep_samples)
    sep_q16, sep_median, sep_q84 = np.percentile(catwise_sep_samples, [16, 50, 84])

    print(f"\nCatWISE model dipole direction vs observed (l={L_QSO_OBS_DEG}, b={B_QSO_OBS_DEG}):")
    print(f"  Angular separation:")
    print(f"    mean   = {sep_mean:.2f} deg")
    print(f"    median = {sep_median:.2f} deg")
    print(f"    [16,84] = [{sep_q16:.2f}, {sep_q84:.2f}] deg")

    # Optional: write alpha posterior to CSV
    if WRITE_DIAGNOSTICS:
        with open(DIAGNOSTICS_CSV, 'w') as f:
            f.write("alpha_deg\n")
            for a in alpha_samples:
                f.write(f"{a:.6f}\n")
        print(f"\n  Wrote alpha posterior to: {DIAGNOSTICS_CSV}")
