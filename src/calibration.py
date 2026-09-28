import json
import multiprocessing
import os
import pickle
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import boxcox as scipy_boxcox
from scipy.stats import yeojohnson as scipy_yeojohnson
from scipy.stats import norm, qmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from bin_weights import build_bin_weights
from base_plates import (
    ENERGY_EDGES,
    PHI_TARGET_IN_LETHARGY,
    TARGET_VALUES,
    extract_flux,
    generate_weight_windows,
    plot_mcmc_validation_comparison,
    postprocess_statepoint,
    run_with_weight_windows,
)
from config import WEIGHT_MODE
from lethargy_converter import convert_if_lethargy

# One OpenTURNS thread per process; bin fits run in parallel across processes.
ot.ResourceMap.SetAsUnsignedInteger("TBB-ThreadsNumber", 1)

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

CSV_PATH = "examples/demo_hcpb_vv.csv"

VAL_FRACTION = 0.20
RANDOM_SEED = 12

Y_TRANSFORM = "boxcox"       # "boxcox" | "yeojohnson" | "log_eps"
LOG_FLOOR = 1e-30
# Box-Cox only: force λ=0 (log) on a bin whose fitted λ is above LOG_OVERRIDE_LAMBDA
# and whose target lies below the LOG_OVERRIDE_TARGET_Q quantile of its training values.
LOG_OVERRIDE_LAMBDA = 0.5
LOG_OVERRIDE_TARGET_Q = 0.10

MATERN_NU = 0.5              # 0.5 | 1.5 | 2.5
INITIAL_SCALE = 0.5
SCALE_LB = 5e-2
SCALE_UB = 5.0
NOISE_FLOOR = 1e-8
USE_MC_NOISE = True

N_MLE_RESTARTS = 5
GPR_N_JOBS = os.cpu_count() // 2 - 1

SOBOL_SAMPLE_SIZE = 8192
GPR_CACHE_PATH = "trained_gps"   # None: always retrain
VAL_RESULTS_CSV = "gpr_validation_results.csv"

UNC_LIMIT = 0.50             # drop a train or validation point in a bin above this relative sigma
SURVIVAL_MIN = 0.70          # warn when fewer samples than this survive UNC_LIMIT
POOR_R2 = 0.80
PRINT_POOR_BINS = False
N_SPECTRUM_EXAMPLES = 6

LIKELIHOOD_SIGMA_PCT = 10.0
SIGMA_Z_FLOOR = 1e-3
SAMPLING_DRAWS = 700
SAMPLING_TUNE = 1500
SAMPLING_CHAINS = 8
SAMPLING_CORES = os.cpu_count()

PP_CRED = 94

VAL_WW_DIR = "val_ww"
VAL_CE_DIR = "val_ce"
VAL_CE_PARTICLES = 200_000
VAL_CE_BATCHES = 100

WW_BATCHES = 100
WW_INACTIVE_BATCHES = 50
WW_PARTICLES = 60_000
MGXS_PARTICLES = 35_000

SAVE_PLOTS = True

# ==================================================================

csv_header = pd.read_csv(CSV_PATH, nrows=1)

MAT_COLS = sorted(
    (c for c in csv_header.columns if c.startswith("mat_")),
    key=lambda name: int(name.split("_")[1]),
)
MATERIALS = csv_header[MAT_COLS].iloc[0].tolist()

RAW_INPUT_COLS = sorted(
    (c for c in csv_header.columns if c.startswith("x") and c[1:].isdigit()),
    key=lambda name: int(name[1:]),
)
N_THICKNESS_INPUTS = len(RAW_INPUT_COLS) - 1
INPUT_COLS = ["x1"] + [f"t{i}" for i in range(1, len(RAW_INPUT_COLS))] + ["L"]
if "L" not in csv_header.columns:
    raise ValueError("CSV is missing the plate size column 'L'.")

BIN_COLS_FULL = sorted(
    (c for c in csv_header.columns if c.startswith("y_") and c[2:].isdigit()),
    key=lambda name: int(name.split("_")[1]),
)
SIGMA_COLS_FULL = sorted(
    (c for c in csv_header.columns if c.startswith("sigma_") and c[6:].isdigit()),
    key=lambda name: int(name.split("_")[1]),
)
N_BINS_FULL = len(BIN_COLS_FULL)

if len(TARGET_VALUES) != N_BINS_FULL or (len(ENERGY_EDGES) - 1) != N_BINS_FULL:
    raise ValueError(
        f"Bin count mismatch: CSV={N_BINS_FULL}, "
        f"TARGET_VALUES={len(TARGET_VALUES)}, ENERGY_EDGES={len(ENERGY_EDGES) - 1}."
    )

ENERGY_EDGES_FULL = np.asarray(ENERGY_EDGES, dtype=float)
y_obs_full = np.asarray(
    convert_if_lethargy(TARGET_VALUES, ENERGY_EDGES, PHI_TARGET_IN_LETHARGY),
    dtype=float,
)
target_sum = y_obs_full.sum() or 1.0
Y_obs_norm_full = y_obs_full / target_sum

ACTIVE_MASK = y_obs_full != 0.0
ACTIVE_IDX = np.where(ACTIVE_MASK)[0]
BIN_COLS = [BIN_COLS_FULL[i] for i in ACTIVE_IDX]
SIGMA_COLS = [SIGMA_COLS_FULL[i] for i in ACTIVE_IDX]
N_BINS = len(BIN_COLS)
Y_obs_norm = Y_obs_norm_full[ACTIVE_IDX]
ENERGY_MID_EV = 0.5 * (ENERGY_EDGES_FULL[:-1] + ENERGY_EDGES_FULL[1:])[ACTIVE_IDX]

n_skipped = int((~ACTIVE_MASK).sum())
if n_skipped:
    skipped = [BIN_COLS_FULL[i] for i in range(N_BINS_FULL) if not ACTIVE_MASK[i]]
    print(f"\n[Active bins] Skipping {n_skipped} zero-target bin(s): {skipped}")
else:
    print("\n[Active bins] No zero-target bins.")

bin_weights_full = build_bin_weights(WEIGHT_MODE, mask=ACTIVE_MASK)
BIN_WEIGHTS = bin_weights_full[ACTIVE_IDX] * N_BINS
print(f"[Weights] WEIGHT_MODE={WEIGHT_MODE!r}")

df = pd.read_csv(CSV_PATH)

def bounds_to_thicknesses(bounds):
    """Cumulative radii (x1, x2, ..., xN) to (x1, t1, ..., t_{N-1})."""
    thicknesses = np.empty_like(bounds)
    thicknesses[:, 0] = bounds[:, 0]
    thicknesses[:, 1:] = np.diff(bounds, axis=1)
    return thicknesses

def thicknesses_to_bounds(thicknesses):
    """Inverse of bounds_to_thicknesses."""
    bounds = np.empty_like(thicknesses)
    bounds[:, 0] = thicknesses[:, 0]
    for i in range(1, thicknesses.shape[1]):
        bounds[:, i] = bounds[:, i - 1] + thicknesses[:, i]
    return bounds

X_bounds_all  = df[RAW_INPUT_COLS].values.astype(float)   # cumulative boundaries (N, n_in)
X_thick_all   = bounds_to_thicknesses(X_bounds_all)       # (x1, t1, ..., t_{N-1})
L_all         = df[["L"]].values.astype(float)             # plate transverse size (N, 1)
X_all         = np.hstack([X_thick_all, L_all])            # independent params incl. L
Y_all        = df[BIN_COLS].values.astype(float)         # shape (N, N_BINS)
Sigma_all   = df[SIGMA_COLS].values.astype(float)

# ONLY reject if a bin is < 0 (invalid for log scaling).
# High-uncertainty samples are masked per-bin in the GPR loop.
valid_mask  = (Y_all >= 0).all(axis=1)

n_rejected  = (~valid_mask).sum()
n_kept      = valid_mask.sum()

print(f"\n[Filter] Samples with any bin < 0 : {n_rejected} rejected")
print(f"         Samples remaining          : {n_kept}")

X_all = X_all[valid_mask]
Y_all = Y_all[valid_mask]
Sigma_all   = Sigma_all[valid_mask]   # keep in sync

rel_unc_surv = np.full_like(Y_all, np.nan, dtype=float)
np.divide(Sigma_all, Y_all, out=rel_unc_surv, where=Y_all > 0)
# NaN comparisons are False → y≤0 / missing unc counts as non-surviving
survive_frac = np.mean(rel_unc_surv <= UNC_LIMIT, axis=0)   # length N_BINS (active)

low_surv_mask = survive_frac < SURVIVAL_MIN
n_low_surv    = int(low_surv_mask.sum())

if n_low_surv:
    print(f"\n[WARNING] {n_low_surv} bin(s) have survival fraction < {SURVIVAL_MIN:.0%} "
          f"at UNC_LIMIT={UNC_LIMIT:.0%} — bins are kept; continuing:")
    for i in np.where(low_surv_mask)[0]:
        print(f"           {BIN_COLS[i]}  survival={survive_frac[i]:.1%}")
else:
    print(f"\n[Survival] All {N_BINS} active bins have survival ≥ {SURVIVAL_MIN:.0%} "
          f"at UNC_LIMIT={UNC_LIMIT:.0%}.")

X_train, X_val, Y_train, Y_val, Sigma_train, Sigma_val = train_test_split(
    X_all, Y_all, Sigma_all,
    test_size=VAL_FRACTION,
    random_state=RANDOM_SEED,
    shuffle=True,
)

N_train = len(X_train)
N_val   = len(X_val)

_ALLOWED_Y_TRANSFORMS = ("boxcox", "yeojohnson", "log_eps")
if Y_TRANSFORM not in _ALLOWED_Y_TRANSFORMS:
    raise ValueError(
        f"Y_TRANSFORM={Y_TRANSFORM!r} is not supported. "
        f"Choose one of {_ALLOWED_Y_TRANSFORMS}."
    )

def as_1d(arr):
    return np.asarray(arr, dtype=float).reshape(-1)

def safe_scale(std_val):
    s = float(std_val)
    return 1.0 if (not np.isfinite(s) or s < 1e-12) else s

class LogEpsStdTransform:
    """Per-bin log(Y + ε) + z-score. ε fitted on training positives."""

    name = "log_eps"

    def __init__(self, eps=None):
        self.eps_ = None if eps is None else float(eps)
        self.mean_ = None
        self.scale_ = None

    def _raw(self, y):
        return np.log(np.maximum(y, 0.0) + self.eps_)

    def fit_transform(self, y_2d):
        y = as_1d(y_2d)
        pos = y[y > 0]
        if pos.size == 0:
            self.eps_ = float(LOG_FLOOR)
        else:
            self.eps_ = max(float(pos.min()) / 10.0, 1e-12 * float(np.median(pos)))
        z = self._raw(y)
        self.mean_ = float(z.mean())
        self.scale_ = safe_scale(z.std())
        return ((z - self.mean_) / self.scale_).reshape(-1, 1)

    def transform(self, y_2d):
        z = self._raw(as_1d(y_2d))
        return ((z - self.mean_) / self.scale_).reshape(-1, 1)

    def inverse_transform(self, z_2d):
        z = as_1d(z_2d)
        return np.maximum(np.exp(z * self.scale_ + self.mean_) - self.eps_, 0.0).reshape(-1, 1)

    def sigma_to_z(self, y, sigma_y):
        y = as_1d(y)
        sigma_y = as_1d(sigma_y)
        return sigma_y / ((np.maximum(y, 0.0) + self.eps_) * self.scale_)

    def relative_to_z(self, y, rel_frac):
        """Map a relative error fraction (e.g. 0.10 = 10%) at level y → σ_Z."""
        y = float(y)
        return float(self.sigma_to_z([y], [abs(rel_frac) * abs(y)])[0])

    def to_spec(self):
        return {
            "name": self.name,
            "mean_": float(self.mean_),
            "scale_": float(self.scale_),
            "eps_": float(self.eps_),
        }

    @classmethod
    def from_spec(cls, spec):
        tf = cls(eps=spec.get("eps_", spec.get("log_floor", LOG_FLOOR)))
        tf.mean_ = float(spec["mean_"])
        tf.scale_ = float(spec["scale_"])
        tf.eps_ = float(spec.get("eps_", tf.eps_))
        return tf

class BoxCoxStdTransform:
    """Per-bin Box-Cox + z-score. λ fitted on training positives.

    Non-positive values are clipped to y_floor_ (0.1 × min positive train)
    before applying Box-Cox so occasional MC zeros do not crash the map.
    """

    name = "boxcox"

    def __init__(self, lam=None, y_floor=None):
        self.lam_ = None if lam is None else float(lam)
        self.y_floor_ = None if y_floor is None else float(y_floor)
        self.mean_ = None
        self.scale_ = None

    def _boxcox(self, y):
        y = np.maximum(as_1d(y), self.y_floor_)
        lam = self.lam_
        if abs(lam) < 1e-8:
            return np.log(y)
        return (np.power(y, lam) - 1.0) / lam

    def fit_transform(self, y_2d, lam=None):
        """Fit λ by maximum likelihood, or keep the given ``lam`` fixed."""
        y = as_1d(y_2d)
        pos = y[y > 0]
        if pos.size < 8:
            # Degenerate bin — fall back to λ=0 (log) on a tiny floor
            self.y_floor_ = float(LOG_FLOOR)
            self.lam_ = 0.0
            z = np.log(np.maximum(y, self.y_floor_))
        else:
            self.y_floor_ = max(float(pos.min()) / 10.0, float(LOG_FLOOR))
            if lam is None:
                # Fit λ on strictly positive training values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _zt, lam = scipy_boxcox(pos)
            self.lam_ = float(lam)
            z = self._boxcox(y)
        self.mean_ = float(z.mean())
        self.scale_ = safe_scale(z.std())
        return ((z - self.mean_) / self.scale_).reshape(-1, 1)

    def transform(self, y_2d):
        z = self._boxcox(y_2d)
        return ((z - self.mean_) / self.scale_).reshape(-1, 1)

    def inverse_transform(self, z_2d):
        u = as_1d(z_2d) * self.scale_ + self.mean_
        lam = self.lam_
        if abs(lam) < 1e-8:
            y = np.exp(u)
        elif lam > 0:
            # u ≤ -1/λ is the image of y ≤ 0: flux is floored at zero there
            y = np.power(np.maximum(lam * u + 1.0, 0.0), 1.0 / lam)
        else:
            # λ < 0: u ≥ -1/λ has no preimage (y → ∞); clip the base
            base = np.maximum(lam * u + 1.0, 1e-30)
            y = np.power(base, 1.0 / lam)
        return y.reshape(-1, 1)

    def sigma_to_z(self, y, sigma_y):
        # d(bc)/dy = y^(λ-1)  (and 1/y for λ=0)
        y = np.maximum(as_1d(y), self.y_floor_)
        sigma_y = as_1d(sigma_y)
        lam = self.lam_
        if abs(lam) < 1e-8:
            dbc_dy = 1.0 / y
        else:
            dbc_dy = np.power(y, lam - 1.0)
        return sigma_y * dbc_dy / self.scale_

    def relative_to_z(self, y, rel_frac):
        y = float(max(y, self.y_floor_))
        return float(self.sigma_to_z([y], [abs(rel_frac) * y])[0])

    def to_spec(self):
        return {
            "name": self.name,
            "mean_": float(self.mean_),
            "scale_": float(self.scale_),
            "lam_": float(self.lam_),
            "y_floor_": float(self.y_floor_),
        }

    @classmethod
    def from_spec(cls, spec):
        tf = cls(lam=spec.get("lam_", 0.0), y_floor=spec.get("y_floor_", LOG_FLOOR))
        tf.mean_ = float(spec["mean_"])
        tf.scale_ = float(spec["scale_"])
        tf.lam_ = float(spec.get("lam_", 0.0))
        tf.y_floor_ = float(spec.get("y_floor_", LOG_FLOOR))
        return tf

class YeoJohnsonStdTransform:
    """Per-bin Yeo-Johnson + z-score. λ fitted on the training sample.

    Defined at zero and for negative values, so MC zeros are not clipped.
    λ=1 is the identity on y≥0.
    """

    name = "yeojohnson"

    def __init__(self, lam=None):
        self.lam_ = None if lam is None else float(lam)
        self.mean_ = None
        self.scale_ = None

    def _yeojohnson(self, y):
        y = as_1d(y)
        lam = self.lam_
        out = np.empty_like(y)
        pos = y >= 0.0
        neg = ~pos
        if abs(lam) < 1e-8:
            out[pos] = np.log1p(y[pos])
        else:
            out[pos] = (np.power(y[pos] + 1.0, lam) - 1.0) / lam
        if abs(lam - 2.0) < 1e-8:
            out[neg] = -np.log1p(-y[neg])
        else:
            out[neg] = -(np.power(-y[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
        return out

    def fit_transform(self, y_2d):
        y = as_1d(y_2d)
        if y.size < 8 or np.unique(y).size < 2:
            self.lam_ = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, lam = scipy_yeojohnson(y)
            self.lam_ = float(lam)
        z = self._yeojohnson(y)
        self.mean_ = float(z.mean())
        self.scale_ = safe_scale(z.std())
        return ((z - self.mean_) / self.scale_).reshape(-1, 1)

    def transform(self, y_2d):
        z = self._yeojohnson(y_2d)
        return ((z - self.mean_) / self.scale_).reshape(-1, 1)

    def inverse_transform(self, z_2d):
        u = as_1d(z_2d) * self.scale_ + self.mean_
        lam = self.lam_
        y = np.empty_like(u)
        pos = u >= 0.0
        neg = ~pos
        if abs(lam) < 1e-8:
            y[pos] = np.expm1(u[pos])
        else:
            base = np.maximum(lam * u[pos] + 1.0, 1e-30)
            y[pos] = np.power(base, 1.0 / lam) - 1.0
        if abs(lam - 2.0) < 1e-8:
            y[neg] = 1.0 - np.exp(-u[neg])
        else:
            base = np.maximum(1.0 - (2.0 - lam) * u[neg], 1e-30)
            y[neg] = 1.0 - np.power(base, 1.0 / (2.0 - lam))
        return y.reshape(-1, 1)

    def sigma_to_z(self, y, sigma_y):
        # dψ/dy = (y+1)^(λ-1) for y≥0, (1-y)^(1-λ) for y<0
        y = as_1d(y)
        sigma_y = as_1d(sigma_y)
        lam = self.lam_
        deriv = np.empty_like(y)
        pos = y >= 0.0
        deriv[pos] = np.power(y[pos] + 1.0, lam - 1.0)
        deriv[~pos] = np.power(1.0 - y[~pos], 1.0 - lam)
        return sigma_y * deriv / self.scale_

    def relative_to_z(self, y, rel_frac):
        y = float(y)
        return float(self.sigma_to_z([y], [abs(rel_frac) * abs(y)])[0])

    def to_spec(self):
        return {
            "name": self.name,
            "mean_": float(self.mean_),
            "scale_": float(self.scale_),
            "lam_": float(self.lam_),
        }

    @classmethod
    def from_spec(cls, spec):
        tf = cls(lam=spec.get("lam_", 1.0))
        tf.mean_ = float(spec["mean_"])
        tf.scale_ = float(spec["scale_"])
        tf.lam_ = float(spec.get("lam_", 1.0))
        return tf


def make_y_transform():
    """Factory for the configured Y_TRANSFORM."""
    if Y_TRANSFORM == "boxcox":
        return BoxCoxStdTransform()
    if Y_TRANSFORM == "yeojohnson":
        return YeoJohnsonStdTransform()
    if Y_TRANSFORM == "log_eps":
        return LogEpsStdTransform()
    raise ValueError(f"Unknown Y_TRANSFORM={Y_TRANSFORM!r}")

def y_transform_from_spec(spec):
    """Restore a transform from a cache payload dict."""
    name = spec.get("name")
    # Backward compat: old caches only stored log_floor/mean_/scale_
    if name is None:
        if "lam_" in spec:
            name = "boxcox"
        else:
            name = "log_eps"
    if name == "boxcox":
        return BoxCoxStdTransform.from_spec(spec)
    if name == "yeojohnson":
        return YeoJohnsonStdTransform.from_spec(spec)
    if name == "log_eps":
        return LogEpsStdTransform.from_spec(spec)
    raise ValueError(f"Unknown transform name in cache: {name!r}")

y_transformers  = []
log_override_bins = []
Y_train_z       = np.zeros_like(Y_train)
Y_val_z         = np.zeros_like(Y_val)
masks_train     = []

print(f"\n[Y transform] Using Y_TRANSFORM={Y_TRANSFORM!r}")
for b in range(N_BINS):
    # Mask out samples for this specific bin if they exceed UNC_LIMIT
    mask_b = (Sigma_train[:, b] / np.maximum(Y_train[:, b], 1e-20)) <= UNC_LIMIT
    masks_train.append(mask_b)

    tf = make_y_transform()
    Y_train_z[:, b] = tf.fit_transform(Y_train[:, b].reshape(-1, 1)).flatten()
    if (
        Y_TRANSFORM == "boxcox"
        and tf.lam_ > LOG_OVERRIDE_LAMBDA
        and Y_obs_norm[b] < np.quantile(Y_train[:, b], LOG_OVERRIDE_TARGET_Q)
    ):
        fitted_lam = tf.lam_
        Y_train_z[:, b] = tf.fit_transform(Y_train[:, b].reshape(-1, 1), lam=0.0).flatten()
        log_override_bins.append((BIN_COLS[b], fitted_lam))
    Y_val_z[:, b]   = tf.transform(Y_val[:, b].reshape(-1, 1)).flatten()
    y_transformers.append(tf)

if Y_TRANSFORM == "boxcox":
    lams = np.array([tf.lam_ for tf in y_transformers], dtype=float)
    print(
        f"             Box-Cox λ: median={np.median(lams):.3f}  "
        f"[{lams.min():.3f}, {lams.max():.3f}]  "
        f"(λ=0 ≡ log; λ=0.5 ≈ sqrt)"
    )
    print(
        f"             Log override (fitted λ > {LOG_OVERRIDE_LAMBDA}, target below "
        f"train q{100 * LOG_OVERRIDE_TARGET_Q:.0f}): {len(log_override_bins)} bin(s)"
    )
    for name, fitted_lam in log_override_bins:
        print(f"               {name}: λ {fitted_lam:.3f} → 0")
elif Y_TRANSFORM == "yeojohnson":
    lams = np.array([tf.lam_ for tf in y_transformers], dtype=float)
    print(
        f"             Yeo-Johnson λ: median={np.median(lams):.3f}  "
        f"[{lams.min():.3f}, {lams.max():.3f}]  "
        f"(λ=1 ≡ identity on y≥0)"
    )
elif Y_TRANSFORM == "log_eps":
    eps = np.array([tf.eps_ for tf in y_transformers], dtype=float)
    print(
        f"             log_eps ε: median={np.median(eps):.3e}  "
        f"[{eps.min():.3e}, {eps.max():.3e}]"
    )

Sigma_train_z = np.zeros_like(Sigma_train)
for b in range(N_BINS):
    Sigma_train_z[:, b] = y_transformers[b].sigma_to_z(
        Y_train[:, b], Sigma_train[:, b]
    )

marginals = []
prior_bounds = []

x_scaler = MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)

for i, col in enumerate(INPUT_COLS):
    lo  = X_train[:, i].min()
    hi  = X_train[:, i].max()
    prior_bounds.append((lo, hi))
    marginals.append(ot.Uniform(0.0, 1.0))

input_distribution = ot.JointDistribution(marginals)

n_inputs = len(INPUT_COLS)
X_ot = ot.Sample(X_train_scaled.tolist())

print(f"\n[GPR]   n_inputs={n_inputs}, N_train={N_train}, kernel=Matern(nu={MATERN_NU}), trend=Constant")
print(f"        Heteroscedastic noise : {USE_MC_NOISE}")

gpr_models      = []          # callable ot.Function metamodels (log-std space)
gpr_results     = []          # GaussianProcessRegressionResult per bin
gpr_cov_models  = []          # fitted MaternModel per bin (for inspection)
train_r2_list   = []          # R² on training set (log-std space)
loo_r2_list     = []          # leave-one-out R² (log-std space) when computable
fit_degree_list = []          # "GPR" or "const" if the fit fell back
fit_sparse_list = []          # kernel label, e.g. "M2.5"

class ConstantModel:
    """Fallback constant predictor used when a bin's GPR fit fails."""
    def __init__(self, val, n_inputs):
        self.val = float(val)
        self.n_inputs = int(n_inputs)

    def __call__(self, x):
        n = len(x) if hasattr(x, "__len__") else 1
        return ot.Sample([[self.val]] * n)

    def gradient(self, x):
        return ot.Matrix(self.n_inputs, 1)

_GPR_CACHE_COMPAT_KEYS = (
    "n_bins", "n_inputs", "bin_cols", "input_cols", "active_idx",
    "matern_nu", "initial_scale", "scale_lb", "scale_ub", "noise_floor",
    "use_mc_noise", "n_train", "n_val",
    "random_seed", "val_fraction", "unc_limit", "y_transform",
    "log_override_lambda", "log_override_target_q",
    "csv_basename", "x_train_checksum", "y_train_checksum",
)

def _gpr_cache_paths(path):
    """Resolve cache directory + payload / OT-study file paths."""
    root = os.path.abspath(os.path.expanduser(str(path)))
    # Allow either a directory or a stem like "trained_gps.pkl"
    if root.endswith(".pkl") or root.endswith(".xml") or root.endswith(".json"):
        root = os.path.splitext(root)[0]
    return {
        "root": root,
        "meta": os.path.join(root, "meta.json"),
        "payload": os.path.join(root, "payload.pkl"),
        "ot_xml": os.path.join(root, "ot_results.xml"),
    }

def _gpr_cache_meta():
    """Fingerprint of everything that must match for a cache hit."""
    return {
        "n_bins": int(N_BINS),
        "n_inputs": int(n_inputs),
        "bin_cols": list(BIN_COLS),
        "input_cols": list(INPUT_COLS),
        "active_idx": [int(i) for i in ACTIVE_IDX],
        "matern_nu": float(MATERN_NU),
        "initial_scale": float(INITIAL_SCALE),
        "scale_lb": float(SCALE_LB),
        "scale_ub": float(SCALE_UB),
        "noise_floor": float(NOISE_FLOOR),
        "use_mc_noise": bool(USE_MC_NOISE),
        "n_train": int(N_train),
        "n_val": int(N_val),
        "random_seed": int(RANDOM_SEED),
        "val_fraction": float(VAL_FRACTION),
        "unc_limit": float(UNC_LIMIT),
        "y_transform": str(Y_TRANSFORM),
        "log_override_lambda": float(LOG_OVERRIDE_LAMBDA),
        "log_override_target_q": float(LOG_OVERRIDE_TARGET_Q),
        "csv_basename": os.path.basename(CSV_PATH),
        # Cheap fingerprints so a different CSV with the same shape is rejected
        "x_train_checksum": float(np.sum(X_train)),
        "y_train_checksum": float(np.sum(Y_train)),
    }

def _gpr_cache_mismatches(saved_meta, current_meta):
    mismatches = []
    for key in _GPR_CACHE_COMPAT_KEYS:
        sv, cv = saved_meta.get(key), current_meta.get(key)
        if key.endswith("_checksum"):
            if sv is None or cv is None or not np.isclose(sv, cv, rtol=0.0, atol=1e-6):
                mismatches.append(f"{key}: cache={sv!r} vs current={cv!r}")
        elif sv != cv:
            mismatches.append(f"{key}: cache={sv!r} vs current={cv!r}")
    return mismatches

def _save_gpr_cache(path):
    """Persist fitted GPs + transforms so the next run can skip training."""
    paths = _gpr_cache_paths(path)
    os.makedirs(paths["root"], exist_ok=True)

    meta = _gpr_cache_meta()
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    payload = {
        "y_transformers": [tf.to_spec() for tf in y_transformers],
        "x_scaler": {
            "scale_": np.asarray(x_scaler.scale_, dtype=float),
            "min_": np.asarray(x_scaler.min_, dtype=float),
            "data_min_": np.asarray(x_scaler.data_min_, dtype=float),
            "data_max_": np.asarray(x_scaler.data_max_, dtype=float),
            "data_range_": np.asarray(x_scaler.data_range_, dtype=float),
            "n_features_in_": int(x_scaler.n_features_in_),
        },
        "masks_train": [np.asarray(m, dtype=bool) for m in masks_train],
        "X_train_scaled": np.asarray(X_train_scaled, dtype=float),
        "train_r2_list": list(train_r2_list),
        "loo_r2_list": list(loo_r2_list),
        "fit_degree_list": list(fit_degree_list),
        "fit_sparse_list": list(fit_sparse_list),
        # Constant-fallback bins have result=None; store the mean they predict
        "const_means": {
            int(b): float(gpr_models[b].val)
            for b in range(N_BINS)
            if gpr_results[b] is None and hasattr(gpr_models[b], "val")
        },
    }
    with open(paths["payload"], "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # OpenTURNS Study for the per-bin GaussianProcessRegressionResult objects
    study = ot.Study()
    study.setStorageManager(ot.XMLStorageManager(paths["ot_xml"]))
    n_saved = 0
    for b, res in enumerate(gpr_results):
        if res is not None:
            study.add(f"result_{b}", res)
            n_saved += 1
    study.save()
    print(f"[GPR cache] Saved {n_saved}/{N_BINS} OT results (+ payload) → {paths['root']}")

def _load_gpr_cache(path):
    """Load cached GPs if present and compatible. Returns True on success."""
    paths = _gpr_cache_paths(path)
    if not (os.path.isfile(paths["meta"]) and os.path.isfile(paths["payload"])):
        print(f"[GPR cache] No cache found at {paths['root']} — will train.")
        return False

    with open(paths["meta"], "r", encoding="utf-8") as f:
        saved_meta = json.load(f)
    mismatches = _gpr_cache_mismatches(saved_meta, _gpr_cache_meta())
    if mismatches:
        print(f"[GPR cache] Incompatible cache at {paths['root']} — will retrain.")
        for msg in mismatches:
            print(f"             • {msg}")
        return False

    with open(paths["payload"], "rb") as f:
        payload = pickle.load(f)

    # Restore Y-transforms and input scaler so they stay locked to the GPs
    global y_transformers, x_scaler, masks_train, X_train_scaled, X_val_scaled
    global gpr_models, gpr_results, gpr_cov_models
    global train_r2_list, loo_r2_list, fit_degree_list, fit_sparse_list

    y_transformers = [y_transform_from_spec(spec) for spec in payload["y_transformers"]]

    xs = MinMaxScaler()
    xs.scale_ = np.asarray(payload["x_scaler"]["scale_"], dtype=float)
    xs.min_ = np.asarray(payload["x_scaler"]["min_"], dtype=float)
    xs.data_min_ = np.asarray(payload["x_scaler"]["data_min_"], dtype=float)
    xs.data_max_ = np.asarray(payload["x_scaler"]["data_max_"], dtype=float)
    xs.data_range_ = np.asarray(payload["x_scaler"]["data_range_"], dtype=float)
    xs.n_features_in_ = int(payload["x_scaler"]["n_features_in_"])
    xs.n_samples_seen_ = int(N_train)
    x_scaler = xs

    masks_train = [np.asarray(m, dtype=bool) for m in payload["masks_train"]]
    X_train_scaled = np.asarray(payload["X_train_scaled"], dtype=float)
    # Keep validation inputs on the *cached* scaler (identical when checksums match)
    X_val_scaled = X_val * x_scaler.scale_ + x_scaler.min_
    train_r2_list = list(payload["train_r2_list"])
    loo_r2_list = list(payload["loo_r2_list"])
    fit_degree_list = list(payload["fit_degree_list"])
    fit_sparse_list = list(payload["fit_sparse_list"])
    const_means = {int(k): float(v) for k, v in payload.get("const_means", {}).items()}

    gpr_models, gpr_results, gpr_cov_models = [], [], []
    study = None
    if os.path.isfile(paths["ot_xml"]):
        study = ot.Study()
        study.setStorageManager(ot.XMLStorageManager(paths["ot_xml"]))
        study.load()

    for b in range(N_BINS):
        result = None
        name = f"result_{b}"
        has_obj = False
        if study is not None:
            try:
                has_obj = bool(study.hasObject(name))
            except Exception:
                has_obj = False
        if has_obj:
            try:
                result = ot.GaussianProcessRegressionResult()
                study.fillObject(name, result)
                metamodel = result.getMetaModel()
                cov_model = result.getCovarianceModel()
            except Exception as e:
                print(f"        Bin {b:2d}  failed to restore OT result ({e}) — const fallback")
                result = None
                has_obj = False
        if not has_obj:
            mean_val = const_means.get(b, float(np.mean(Y_train_z[:, b])))
            metamodel = ConstantModel(mean_val, n_inputs)
            cov_model = None
            if b < len(fit_degree_list):
                fit_degree_list[b] = "const"

        gpr_models.append(metamodel)
        gpr_results.append(result)
        gpr_cov_models.append(cov_model)

    # Recompute Z targets with the restored transformers (keeps val metrics consistent)
    for b in range(N_BINS):
        Y_train_z[:, b] = y_transformers[b].transform(Y_train[:, b].reshape(-1, 1)).flatten()
        Y_val_z[:, b] = y_transformers[b].transform(Y_val[:, b].reshape(-1, 1)).flatten()

    print(f"[GPR cache] Loaded {sum(r is not None for r in gpr_results)}/{N_BINS} "
          f"GPs from {paths['root']} — skipping training.")
    return True

def _make_lhs_starts(n_inputs, n_restarts, scale_lb, scale_ub, seed):
    """Latin Hypercube design in log-uniform([scale_lb, scale_ub])^n_inputs.

    Returns an ot.Sample of starting points for MLE.  Including the
    initial-scale point as the first row guarantees that MultiStart at least
    matches the default OT single-start behavior.
    """
    rows = [[float(INITIAL_SCALE)] * n_inputs]
    if n_restarts > 1:
        sampler = qmc.LatinHypercube(d=n_inputs, seed=seed)
        u = sampler.random(n_restarts)
        log_lb, log_ub = np.log(scale_lb), np.log(scale_ub)
        rows.extend(np.exp(log_lb + u * (log_ub - log_lb)).tolist())
    return ot.Sample(rows)

def _build_gpr(X_ot, Y_b, noise_var_list, n_inputs, nu, init_scale,
                scale_lb, scale_ub, basis, starting_sample):
    """Fit GaussianProcessFitter + GaussianProcessRegression with anisotropic
    Matern, a constant trend, heteroscedastic noise and MultiStart MLE."""
    cov = ot.MaternModel([float(init_scale)] * n_inputs, [1.0], float(nu))

    fitter = ot.GaussianProcessFitter(X_ot, Y_b, cov, basis)
    if noise_var_list is not None:
        fitter.setNoise(noise_var_list)
    lb = ot.Point([scale_lb] * n_inputs)
    ub = ot.Point([scale_ub] * n_inputs)
    fitter.setOptimizationBounds(ot.Interval(lb, ub))

    # MultiStart: OT runs the inner solver from each starting point and
    # returns the kernel hyperparameters with the highest marginal likelihood.
    if starting_sample.getSize() > 1:
        base_solver = ot.TNC()
        multi       = ot.MultiStart(base_solver, starting_sample)
        fitter.setOptimizationAlgorithm(multi)
    fitter.run()

    gpr_algo = ot.GaussianProcessRegression(fitter.getResult())
    gpr_algo.run()
    return gpr_algo.getResult()

def _pin_ot_to_one_thread():
    """Force OpenTURNS/TBB to a single thread (one core per bin process)."""
    ot.ResourceMap.SetAsUnsignedInteger("TBB-ThreadsNumber", 1)

def _fit_one_bin(b, starting_rows, result_xml_path):
    """Fit one bin's GPR. Module-level for ProcessPoolExecutor (fork-inherited state).

    Returns a picklable dict. OT results are written to ``result_xml_path`` (when
    the fit succeeds) because GaussianProcessRegressionResult is not picklable.
    """
    _pin_ot_to_one_thread()

    mask_b = masks_train[b]
    X_ot   = ot.Sample(X_train_scaled[mask_b].tolist())
    Y_b    = ot.Sample(Y_train_z[mask_b, b].reshape(-1, 1).tolist())
    starting_sample = ot.Sample(starting_rows)

    if USE_MC_NOISE:
        noise_var = np.maximum(Sigma_train_z[mask_b, b] ** 2, NOISE_FLOOR).tolist()
    else:
        noise_var = [NOISE_FLOOR] * int(mask_b.sum())

    trend_basis = ot.ConstantBasisFactory(n_inputs).build()
    trend_label = "GPR"

    # GP fit with retry on nugget inflation if Cholesky fails
    result    = None
    metamodel = None
    last_err  = None
    for noise_scale in (1.0, 10.0, 100.0):
        try:
            scaled_noise = [v * noise_scale for v in noise_var]
            result    = _build_gpr(
                X_ot, Y_b, scaled_noise, n_inputs, MATERN_NU, INITIAL_SCALE,
                SCALE_LB, SCALE_UB, trend_basis, starting_sample,
            )
            metamodel = result.getMetaModel()
            break
        except Exception as e:
            last_err = e
            continue

    const_mean = None
    fit_warn = None
    if metamodel is None:
        const_mean = float(np.mean(Y_train_z[:, b]))
        metamodel   = ConstantModel(const_mean, n_inputs)
        result      = None
        trend_label = "const"
        fit_warn = f"Bin {b:2d}  GPR fit failed ({last_err}) — using constant mean predictor"

    # Training R² in Z-space
    Y_pred_train_z = np.array(metamodel(X_ot)).flatten()
    y_masked = Y_train_z[mask_b, b]
    ss_res   = float(np.sum((y_masked - Y_pred_train_z) ** 2))
    ss_tot   = float(np.sum((y_masked - y_masked.mean()) ** 2))
    r2_train = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Approximate LOO R² (Rasmussen & Williams eq. 5.12)
    loo_r2 = float("nan")
    scales = None
    if result is not None:
        try:
            cov  = result.getCovarianceModel()
            scales = [float(s) for s in cov.getScale()]
            K    = np.array(cov.discretize(X_ot))
            K   += np.diag(noise_var)
            Kinv = np.linalg.inv(K)
            resid_tr = y_masked - Y_pred_train_z
            alpha    = Kinv @ resid_tr
            kinv_diag = np.clip(np.diag(Kinv), 1e-30, None)
            y_loo     = Y_pred_train_z - alpha / kinv_diag
            ss_res_l  = float(np.sum((y_masked - y_loo) ** 2))
            loo_r2    = 1.0 - ss_res_l / ss_tot if ss_tot > 1e-12 else float("nan")
        except Exception:
            pass

        # Persist OT result for the parent process (not picklable)
        study = ot.Study()
        study.setStorageManager(ot.XMLStorageManager(result_xml_path))
        study.add("result", result)
        study.save()

    return {
        "b": b,
        "ok": result is not None,
        "const_mean": const_mean,
        "trend_label": trend_label,
        "r2_train": float(r2_train),
        "loo_r2": float(loo_r2) if loo_r2 == loo_r2 else float("nan"),
        "scales": scales,
        "fit_warn": fit_warn,
        "xml_path": result_xml_path if result is not None else None,
    }

def _load_bin_ot_result(xml_path):
    """Restore a GaussianProcessRegressionResult saved by `_fit_one_bin`."""
    study = ot.Study()
    study.setStorageManager(ot.XMLStorageManager(xml_path))
    study.load()
    result = ot.GaussianProcessRegressionResult()
    study.fillObject("result", result)
    return result

loaded_from_cache = False
if GPR_CACHE_PATH:
    loaded_from_cache = _load_gpr_cache(GPR_CACHE_PATH)

if not loaded_from_cache:
    _starting_sample = _make_lhs_starts(
        n_inputs, N_MLE_RESTARTS, SCALE_LB, SCALE_UB, RANDOM_SEED
    )
    _starting_rows = [list(row) for row in _starting_sample]

    _n_jobs = GPR_N_JOBS
    _n_jobs = max(1, min(int(_n_jobs), N_BINS))

    print(f"\n[GPR]   MultiStart MLE: {len(_starting_sample)} starting points "
          f"(1 default + {N_MLE_RESTARTS} LHS log-uniform)")
    print(f"        Trend                 : Constant")
    print(f"        Parallel fit          : {_n_jobs} process(es) "
          f"(1 TBB thread each, {N_BINS} bins)")

    # Parent also pinned: avoids TBB oversubscription before/after the pool.
    _pin_ot_to_one_thread()

    _tmp_dir = tempfile.mkdtemp(prefix="gpr_bin_fit_")
    _bin_payloads = [None] * N_BINS
    try:
        if _n_jobs == 1:
            for b in range(N_BINS):
                xml_path = os.path.join(_tmp_dir, f"result_{b}.xml")
                _bin_payloads[b] = _fit_one_bin(b, _starting_rows, xml_path)
                if _bin_payloads[b]["fit_warn"]:
                    print(f"        {_bin_payloads[b]['fit_warn']}")
                if (b + 1) % 5 == 0 or b == N_BINS - 1:
                    p = _bin_payloads[b]
                    cov_str = ""
                    if p["scales"] is not None:
                        cov_str = f"  scale={[f'{s:.2f}' for s in p['scales']]}"
                    print(f"        Bin {b+1:2d}/{N_BINS}  R²_train={p['r2_train']:.3f}  "
                          f"R²_LOO={p['loo_r2']:.3f}{cov_str}")
        else:
            # fork: workers inherit training arrays / masks / config without pickling.
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=_n_jobs, mp_context=ctx) as pool:
                futures = {
                    pool.submit(
                        _fit_one_bin, b, _starting_rows,
                        os.path.join(_tmp_dir, f"result_{b}.xml"),
                    ): b
                    for b in range(N_BINS)
                }
                n_done = 0
                for fut in as_completed(futures):
                    payload = fut.result()
                    b = payload["b"]
                    _bin_payloads[b] = payload
                    if payload["fit_warn"]:
                        print(f"        {payload['fit_warn']}")
                    n_done += 1
                    if n_done % 5 == 0 or n_done == N_BINS:
                        cov_str = ""
                        if payload["scales"] is not None:
                            cov_str = f"  scale={[f'{s:.2f}' for s in payload['scales']]}"
                        print(f"        Done {n_done:2d}/{N_BINS}  "
                              f"(last bin {b+1}: R²_train={payload['r2_train']:.3f}  "
                              f"R²_LOO={payload['loo_r2']:.3f}{cov_str})")

        # Assemble results in bin order
        for b in range(N_BINS):
            p = _bin_payloads[b]
            if p["ok"] and p["xml_path"]:
                result = _load_bin_ot_result(p["xml_path"])
                metamodel = result.getMetaModel()
                cov_model = result.getCovarianceModel()
            else:
                result = None
                metamodel = ConstantModel(p["const_mean"], n_inputs)
                cov_model = None

            gpr_models.append(metamodel)
            gpr_results.append(result)
            gpr_cov_models.append(cov_model)
            fit_degree_list.append(p["trend_label"])
            fit_sparse_list.append(f"M{MATERN_NU}")
            train_r2_list.append(p["r2_train"])
            loo_r2_list.append(p["loo_r2"])
    finally:
        shutil.rmtree(_tmp_dir, ignore_errors=True)

    if GPR_CACHE_PATH:
        try:
            _save_gpr_cache(GPR_CACHE_PATH)
        except Exception as e:
            print(f"[GPR cache] WARNING: failed to save cache ({e})")

print("\n[Validation] Running predictions on held-out set …")

X_val_ot      = ot.Sample(X_val_scaled.tolist())
Y_pred_val_z  = np.zeros((N_val, N_BINS))   # GPR output — log-std space
Y_pred_val    = np.zeros((N_val, N_BINS))   # back-transformed via exp(z·scale + mean)

for b, model in enumerate(gpr_models):
    pred_z              = np.array(model(X_val_ot)).flatten()
    Y_pred_val_z[:, b]  = pred_z
    Y_pred_val[:, b]    = y_transformers[b].inverse_transform(
        pred_z.reshape(-1, 1)
    ).flatten()
# Same per-bin cut as training: score only validation rows with σ/y ≤ UNC_LIMIT.
masks_val = [
    (Sigma_val[:, b] / np.maximum(Y_val[:, b], 1e-20)) <= UNC_LIMIT
    for b in range(N_BINS)
]
n_val_kept = np.array([int(m.sum()) for m in masks_val])
print(f"[Validation] UNC_LIMIT={UNC_LIMIT:.0%} also applied to the hold-out, per bin")
print(f"             surviving rows: min={n_val_kept.min()}  "
      f"median={np.median(n_val_kept):.0f}  max={n_val_kept.max()}  (of {N_val})")

# Per-bin validation metrics
val_r2_list      = []     # in original spectrum space
val_r2_z_list   = []     # in log-std space — apples-to-apples with train R²
val_rmse_list    = []
val_mae_list     = []

for b in range(N_BINS):
    keep = masks_val[b]
    y_true = Y_val[keep, b]
    y_pred = Y_pred_val[keep, b]
    if y_true.size < 2:
        val_r2_list.append(float("nan"))
        val_rmse_list.append(float("nan"))
        val_mae_list.append(float("nan"))
        val_r2_z_list.append(float("nan"))
        continue
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-20 * len(y_true) else float("nan")
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    val_r2_list.append(r2)
    val_rmse_list.append(rmse)
    val_mae_list.append(mae)

    # Same metric in transformed space, so the Box-Cox inverse is not mixed in.
    y_true_z = Y_val_z[keep, b]
    y_pred_z = Y_pred_val_z[keep, b]
    ss_res_z = np.sum((y_true_z - y_pred_z) ** 2)
    ss_tot_z = np.sum((y_true_z - y_true_z.mean()) ** 2)
    r2_z = 1 - ss_res_z / ss_tot_z if ss_tot_z > 1e-20 else float("nan")
    val_r2_z_list.append(r2_z)

# Summary table
print("\n" + "─" * 100)
print(f"{'Bin':<6} {'Model':>5} {'Kernel':>6} {'Train R²(Z)':>12} {'LOO R²(Z)':>10} "
      f"{'Val R²(Z)':>10} {'Val R²(orig)':>13} {'Val RMSE':>10}")
print("─" * 100)
for b in range(N_BINS):
    flag = " ← poor" if val_r2_list[b] < 0.80 else ""
    print(
        f"{BIN_COLS[b]:<6}  {str(fit_degree_list[b]):>4}  {str(fit_sparse_list[b]):>6}  "
        f"{train_r2_list[b]:>12.4f}  {loo_r2_list[b]:>10.4f}  "
        f"{val_r2_z_list[b]:>10.4f}  {val_r2_list[b]:>12.4f}  "
        f"{val_rmse_list[b]:>9.5f}{flag}"
    )

print("  Note: Train/LOO/Val R²(Z) are in transformed Z-space (apples-to-apples).")
print("        Val scores use only hold-out rows with σ/y ≤ UNC_LIMIT.")
print("        Val R²(orig) and RMSE are in the original spectrum space.")
print(f"        Y_TRANSFORM={Y_TRANSFORM!r}. A bin with high Val R²(Z) but low")
print("        Val R²(orig) signals inverse-transform amplifying tail error.")

sigma_gp_z_val = np.full((N_val, N_BINS), np.nan)
for b in range(N_BINS):
    if gpr_results[b] is not None:
        try:
            gpcc  = ot.GaussianProcessConditionalCovariance(gpr_results[b])
            var_b = np.array(gpcc.getConditionalMarginalVariance(X_val_ot)).flatten()
            sigma_gp_z_val[:, b] = np.sqrt(np.maximum(var_b, 0.0) + NOISE_FLOOR)
        except Exception as e:
            print(f"        Bin {b:2d}  cond. variance failed: {e}")

scaled_resid = (Y_pred_val_z - Y_val_z) / sigma_gp_z_val
for b in range(N_BINS):
    scaled_resid[~masks_val[b], b] = np.nan
scaled_resid = scaled_resid[np.isfinite(scaled_resid)]

if scaled_resid.size == 0:
    print("\n[GPR residuals] skipped — no finite predictive σ.")
else:
    sr_mean, sr_std = float(scaled_resid.mean()), float(scaled_resid.std())
    frac_2s = float(np.mean(np.abs(scaled_resid) <= 2.0))
    print(f"\n[GPR residuals] mean={sr_mean:+.3f} (ideal 0)  std={sr_std:.3f} (ideal 1)")
    print(f"                within ±2σ: {100*frac_2s:.1f}%  (ideal ≈ 95.4%)")
    print(f"                std<1 → over-conservative σ; std>1 → σ underestimated")

    fig_sr, ax_sr = plt.subplots(figsize=(8, 5))
    ax_sr.hist(scaled_resid, bins=200, density=True, color="#2c6e9c",
               edgecolor="none", alpha=0.85)
    xx = np.linspace(-7, 7, 400)
    ax_sr.plot(xx, norm.pdf(xx), color="#e67e22", lw=2.2, label=r"$\mathcal{N}(0,1)$")
    ax_sr.set_xlabel(r"Scaled residual $(p-y)/\sigma$", fontsize=12)
    ax_sr.set_ylabel("Density", fontsize=12)
    ax_sr.set_xlim(-7, 7)
    ax_sr.set_title(f"GPR standardized residuals  (mean={sr_mean:+.2f}, std={sr_std:.2f})",
                    fontweight="bold")
    ax_sr.legend(fontsize=11)
    ax_sr.grid(alpha=0.2, ls=":")
    plt.tight_layout()
    if SAVE_PLOTS:
        fig_sr.savefig("gpr_standardized_residuals.png", dpi=300)
    plt.close(fig_sr)

fig_bar, ax = plt.subplots(figsize=(14, 4))
bin_idx = np.asarray(ACTIVE_IDX, dtype=int)
bars = ax.bar(bin_idx, val_r2_list, width=0.8, color=[
    "#2ecc71" if r >= 0.95 else "#f39c12" if r >= POOR_R2 else "#e74c3c"
    for r in val_r2_list
], edgecolor="white", linewidth=0.3)
ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.4)
ax.axhline(0.95, color="#2ecc71", lw=1.0, ls="--", alpha=0.7, label="R²=0.95")
ax.axhline(POOR_R2, color="#f39c12", lw=1.0, ls="--", alpha=0.7, label=f"R²={POOR_R2}")
first = int(bin_idx[0]) if N_BINS else 0
last = int(bin_idx[-1]) if N_BINS else 0
span = max(1, last - first + 1)
raw_step = max(1, int(np.ceil(span / 15)))
step = next(
    (s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000) if s >= raw_step),
    raw_step,
)
tick0 = first - (first % step)
ax.set_xticks(np.arange(tick0, last + 1, step))
ax.set_xlim(first - 1, last + 1)
ax.tick_params(axis="x", labelsize=9)
ax.set_xlabel("Spectrum bin")
ax.set_ylabel("Validation R²")
ax.set_title("GPR Surrogate — Validation R² per Bin", fontweight="bold")
ax.set_ylim(max(0, min(val_r2_list) - 0.05), 1.02)
ax.legend(fontsize=8)
plt.tight_layout()
if SAVE_PLOTS:
    fig_bar.savefig("gpr_r2_per_bin.png", dpi=300)

gpr_rel_err_pct = np.zeros(N_BINS)
for b in range(N_BINS):
    keep = masks_val[b]
    residuals_pct = 100.0 * np.abs(Y_val[keep, b] - Y_pred_val[keep, b]) / (np.abs(Y_val[keep, b]) + 1e-14)
    gpr_rel_err_pct[b] = np.percentile(residuals_pct, 95) if residuals_pct.size else float("nan")

n_examples = min(N_SPECTRUM_EXAMPLES, N_val)
rng = np.random.default_rng(RANDOM_SEED)
example_idx = rng.choice(N_val, size=n_examples, replace=False)

fig_spec, axes_s = plt.subplots(2, 3, figsize=(14, 8))
axes_s_flat = axes_s.flatten()
e_x = ENERGY_MID_EV

for k, idx in enumerate(example_idx):
    ax = axes_s_flat[k]

    y_true  = np.clip(Y_val[idx],      1e-10, None)
    y_pred  = np.clip(Y_pred_val[idx], 1e-10, None)

    # ±2σ_MC band around MC truth (measurement uncertainty)
    sigma_mc = Sigma_val[idx]                          # shape (26,)
    mc_lo    = np.clip(y_true - 2 * sigma_mc, 1e-10, None)
    mc_hi    =         y_true + 2 * sigma_mc

    mc_rel_pct = 100.0 * sigma_mc / (np.abs(y_true) + 1e-14)
    sigma_total_rel_pct = np.sqrt(mc_rel_pct**2 + gpr_rel_err_pct**2)
    tot_rel_factor = 1.0 + 2.0 * sigma_total_rel_pct / 100.0
    tot_lo = y_pred / tot_rel_factor
    tot_hi = y_pred * tot_rel_factor

    ax.set_xscale("log")
    ax.set_yscale("log")

    # MC truth: scatter points in dark color
    ax.scatter(e_x, y_true, s=40, marker="o", color="#2c3e50",
              alpha=0.7, edgecolors="white", linewidth=0.5, label="MC truth", zorder=3)
    # MC error band around truth
    ax.fill_between(e_x, mc_lo,  mc_hi,
                    alpha=0.15, color="#2c3e50", label="±2σ MC", zorder=1)

    # GPR prediction
    ax.scatter(e_x, y_pred, s=40, marker="s", color="#e74c3c",
              alpha=0.7, edgecolors="white", linewidth=0.5, label="GPR pred", zorder=3)
    # Combined error band around the GPR prediction
    ax.fill_between(e_x, tot_lo, tot_hi,
                    alpha=0.15, color="#e74c3c", label="±2σ total", zorder=1)

    # Cap y-range to the data points so huge GP unc bands don't crush the view;
    # still leave ~decade of margin so some of an oversized band remains visible.
    y_vis = np.concatenate([y_true, y_pred])
    y_vis = y_vis[np.isfinite(y_vis) & (y_vis > 1e-10)]
    if y_vis.size:
        ax.set_ylim(np.min(y_vis) * 0.3, np.max(y_vis) * 10.0)

    ax.set_title(f"Val sample #{idx}", fontsize=14)
    ax.set_xlabel("Energy [eV]", fontsize=13)
    ax.set_ylabel("Normalized flux (log)", fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)

fig_spec.suptitle(
    "GPR vs MC Truth — relative ±2σ bands (discrete points, no interpolation)",
    fontsize=16, fontweight="bold",
)
plt.tight_layout()
if SAVE_PLOTS:
    fig_spec.savefig("gpr_spectrum_examples.png", dpi=300)
plt.close(fig_spec)

Sigma_pred_val = np.abs(Y_pred_val) * (gpr_rel_err_pct[np.newaxis, :] / 100.0)
# shape: (N_val, N_BINS)  — symmetric absolute uncertainty on the GPR prediction

n_rows = int(np.ceil(N_BINS / 6))
fig_scatter, axes = plt.subplots(n_rows, 6, figsize=(6 * 2.8, n_rows * 2.8))
axes_flat = axes.flatten()

for b in range(N_BINS):
    ax = axes_flat[b]
    keep = masks_val[b]
    y_true  = Y_val[keep, b]
    y_pred  = Y_pred_val[keep, b]
    y_err   = 0.0 * Sigma_pred_val[keep, b]   # ±2σ half-width for each point
    if y_true.size == 0:
        ax.set_visible(False)
        continue

    lo_lim = min(y_true.min(), (y_pred - y_err).min())
    hi_lim = max(y_true.max(), (y_pred + y_err).max())

    ax.errorbar(
        y_true, y_pred,
        yerr=y_err,
        fmt="o",
        ms=3,
        alpha=0.55,
        color="#3498db",
        ecolor="#3498db",
        elinewidth=0.6,
        capsize=1.5,
        capthick=0.6,
        markeredgewidth=0,
        label="GPR ±0σ",
        zorder=2,
    )
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "r--", lw=1.0, zorder=3)
    ax.set_title(f"B{b}  R²={val_r2_list[b]:.3f}", fontsize=7, fontweight="bold")
    ax.tick_params(labelsize=5)
    ax.set_xlabel("True", fontsize=5)
    ax.set_ylabel("GPR", fontsize=5)

# hide unused subplots
for idx in range(N_BINS, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig_scatter.suptitle(
    f"GPR Predicted vs True — Validation Set (N={N_val})  [error bars = ±2σ GPR]",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
if SAVE_PLOTS:
    fig_scatter.savefig("gpr_scatter_per_bin.png", dpi=300, bbox_inches="tight")

poor_bins = [b for b, r in enumerate(val_r2_list) if r < POOR_R2]

if poor_bins and PRINT_POOR_BINS:
    print(f"\n[Diagnostics] Poor bins (R² < {POOR_R2}): "
          f"{[BIN_COLS[b] for b in poor_bins]}")

    _N_DIAG_COLS  = 3
    _N_DIAG_PANELS = len(INPUT_COLS) + 2
    _N_DIAG_ROWS  = int(np.ceil(_N_DIAG_PANELS / _N_DIAG_COLS))

    for b in poor_bins:
        keep = masks_val[b]
        y_true    = Y_val[keep, b]
        y_pred    = Y_pred_val[keep, b]
        x_kept    = X_val[keep]
        residuals = y_true - y_pred
        if y_true.size == 0:
            continue

        fig_diag, axes_d = plt.subplots(
            _N_DIAG_ROWS, _N_DIAG_COLS,
            figsize=(4.3 * _N_DIAG_COLS, 3.5 * _N_DIAG_ROWS),
        )
        axes_d_flat = np.atleast_1d(axes_d).flatten()
        fig_diag.suptitle(
            f"Diagnostics — {BIN_COLS[b]}  (val R²={val_r2_list[b]:.3f})",
            fontsize=11, fontweight="bold",
        )

        # Panels 0..len(INPUT_COLS)-1: residuals vs each input (incl. L)
        for j, col in enumerate(INPUT_COLS):
            ax = axes_d_flat[j]
            ax.scatter(x_kept[:, j], residuals, s=14, alpha=0.6,
                       color="#3498db", edgecolors="none")
            ax.axhline(0, color="red", lw=1.0, ls="--")
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel("Residual (true − GPR)", fontsize=8)
            ax.set_title(f"Residuals vs {col}", fontsize=9)

        # Next panel: output distribution — original space, train vs val
        ax_dist = axes_d_flat[len(INPUT_COLS)]
        ax_dist.hist(Y_train[:, b], bins=25, color="#2ecc71", alpha=0.7,
                     edgecolor="white", label="Train (orig)")
        ax_dist.hist(y_true,   bins=25, color="#e74c3c", alpha=0.5,
                     edgecolor="white", label="Val (orig)")
        ax_dist.set_xlabel("Output value (original space)", fontsize=8)
        ax_dist.set_ylabel("Count", fontsize=8)
        ax_dist.set_title(
            f"Output dist  Z({Y_TRANSFORM}): μ={y_transformers[b].mean_:.2f} "
            f"σ={y_transformers[b].scale_:.2f}",
            fontsize=9,
        )
        ax_dist.legend(fontsize=7)

        # Last panel: predicted vs true scatter for this bin
        ax_sc = axes_d_flat[len(INPUT_COLS) + 1]
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax_sc.scatter(y_true, y_pred, s=14, alpha=0.6,
                      color="#9b59b6", edgecolors="none")
        ax_sc.plot([lo, hi], [lo, hi], "r--", lw=1.0)
        ax_sc.set_xlabel("True", fontsize=8)
        ax_sc.set_ylabel("GPR predicted", fontsize=8)
        ax_sc.set_title("Predicted vs True", fontsize=9)

        # Hide any leftover unused panels (grid may overshoot by <3 cells).
        for idx in range(len(INPUT_COLS) + 2, len(axes_d_flat)):
            axes_d_flat[idx].set_visible(False)

        plt.tight_layout()
        fname = f"gpr_diagnostics_{BIN_COLS[b]}.png"
        if SAVE_PLOTS:
            fig_diag.savefig(fname, dpi=300)
            print(f"[Plot]  Saved {fname}")
        plt.close(fig_diag)
elif poor_bins and not PRINT_POOR_BINS:
    print(f"\n[Diagnostics] There are bins with val R² < {POOR_R2} but PRINT_POOR_BINS=False, so skipping per-bin diagnostics.")
else:
    print(f"\n[Diagnostics] All bins have val R² ≥ {POOR_R2} — no poor bins to diagnose.")

print(f"\n[Sobol] Sampling-based indices on GPR (N={SOBOL_SAMPLE_SIZE} per bin)…")

sobol_experiment = ot.SobolIndicesExperiment(
    input_distribution, SOBOL_SAMPLE_SIZE, True   # True = compute total indices too
)
sobol_design = sobol_experiment.generate()

S1 = np.zeros((N_BINS, len(INPUT_COLS)))   # first-order indices
ST = np.zeros((N_BINS, len(INPUT_COLS)))   # total-order indices

for b, model in enumerate(gpr_models):
    try:
        Y_design = model(sobol_design)
        sobol_alg = ot.SaltelliSensitivityAlgorithm(
            sobol_design, Y_design, SOBOL_SAMPLE_SIZE
        )
        S1[b, :] = np.array(sobol_alg.getFirstOrderIndices())
        ST[b, :] = np.array(sobol_alg.getTotalOrderIndices())
    except Exception as e:
        print(f"        Bin {b:2d}  Sobol failed: {e}")

fig_sobol, axes_sob = plt.subplots(1, 3, figsize=(18, 5))
_palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"]
colors = [_palette[j % len(_palette)] for j in range(len(INPUT_COLS))]
ST_minus_S1 = ST - S1

for j, (col, c) in enumerate(zip(INPUT_COLS, colors)):
    axes_sob[0].plot(ENERGY_MID_EV, S1[:, j], "o-", color=c, lw=1.5, ms=4, label=col)
    axes_sob[1].plot(ENERGY_MID_EV, ST[:, j], "s--", color=c, lw=1.5, ms=4, label=col)
    axes_sob[2].plot(ENERGY_MID_EV, ST_minus_S1[:, j], "^-.", color=c, lw=1.5, ms=4, label=col)

for ax, title in zip(
    axes_sob,
    ["First-order Sobol S1", "Total-order Sobol ST", "Interaction ST − S1"],
):
    ax.set_xscale("log")
    ax.set_xlabel("Energy [eV]", fontsize=14)
    ax.set_ylabel("Sobol index", fontsize=14)
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="gray", lw=0.5)

plt.tight_layout()
if SAVE_PLOTS:
    fig_sobol.savefig("gpr_sobol_indices.png", dpi=300)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  Bins with val R² ≥ 0.95 : {sum(r >= 0.95 for r in val_r2_list if not np.isnan(r)):>3d} / {N_BINS}")
print(f"  Bins with val R² ≥ {POOR_R2:.2f} : {sum(r >= POOR_R2 for r in val_r2_list if not np.isnan(r)):>3d} / {N_BINS}")
print(f"  Bins with val R²  < {POOR_R2:.2f} : {sum(r <  POOR_R2 for r in val_r2_list if not np.isnan(r)):>3d} / {N_BINS}")
print(f"  Overall mean val R²      : {np.nanmean(val_r2_list):.4f}")
print(f"  Bins that fell back to const : {sum(label == 'const' for label in fit_degree_list):>3d} / {N_BINS}")

# Per-bin MC uncertainty stats on the *full* filtered dataset (train+val
# combined; both went through the same Y > 0 / UNC_LIMIT filter).
rel_unc_full = np.zeros((N_train + N_val, N_BINS))
Y_full       = np.vstack([Y_train, Y_val])
S_full       = np.vstack([Sigma_train, Sigma_val])
np.divide(S_full, Y_full, out=rel_unc_full, where=Y_full > 0)

rows = []
for b in range(N_BINS):
    if gpr_cov_models[b] is not None:
        scales_b = list(gpr_cov_models[b].getScale())
    else:
        scales_b = [float("nan")] * n_inputs

    rel = rel_unc_full[:, b]
    row = {
        "bin":              BIN_COLS[b],
        "bin_idx":          b,
        "trend":            fit_degree_list[b],
        "kernel":           fit_sparse_list[b],
        "train_r2_z":       train_r2_list[b],
        "loo_r2_z":         loo_r2_list[b],
        "val_r2_z":         val_r2_z_list[b],
        "val_r2_orig":      val_r2_list[b],
        "val_rmse":         val_rmse_list[b],
        "val_mae":          val_mae_list[b],
        "y_mean":           float(np.mean(Y_full[:, b])),
        "y_median":         float(np.median(Y_full[:, b])),
        "rel_unc_mean":     float(np.mean(rel)),
        "rel_unc_median":   float(np.median(rel)),
        "rel_unc_p75":      float(np.percentile(rel, 75)),
        "rel_unc_p95":      float(np.percentile(rel, 95)),
        "rel_unc_max":      float(np.max(rel)),
    }
    for j, col in enumerate(INPUT_COLS):
        row[f"scale_{col}"] = scales_b[j] if j < len(scales_b) else float("nan")
    rows.append(row)

val_results_df = pd.DataFrame(rows)
val_results_df.to_csv(VAL_RESULTS_CSV, index=False)
print(f"[Save] Wrote per-bin validation+noise stats → {VAL_RESULTS_CSV}")

plt.close("all")

# Transform observed spectrum to Z-space (same transform fitted on training data)
Y_obs_norm_z = np.array([
    float(y_transformers[b].transform([[Y_obs_norm[b]]])[0, 0])
    for b in range(N_BINS)
])

sigma_pct = np.broadcast_to(
    np.asarray(LIKELIHOOD_SIGMA_PCT, dtype=float),
    (N_BINS,),
).copy()
if sigma_pct.shape != (N_BINS,):
    raise ValueError(
        f"LIKELIHOOD_SIGMA_PCT must be a scalar or a sequence of length {N_BINS}; "
        f"got shape {sigma_pct.shape}."
    )

Sigma_obs = np.array([
    y_transformers[b].relative_to_z(Y_obs_norm[b], sigma_pct[b] / 100.0)
    for b in range(N_BINS)
], dtype=float)
Sigma_obs = np.maximum(Sigma_obs, SIGMA_Z_FLOOR)
print(f"[Likelihood] σ range: [{Sigma_obs.min():.4f}, {Sigma_obs.max():.4f}]  "
      f"(from LIKELIHOOD_SIGMA_PCT={LIKELIHOOD_SIGMA_PCT}, Y_TRANSFORM={Y_TRANSFORM})")

input_scale   = x_scaler.scale_.copy()    # shape (n_inputs,)
input_shift   = x_scaler.min_.copy()      # shape (n_inputs,)  (= -X_min * scale_)

assert MATERN_NU in (0.5, 1.5, 2.5), "kernel formula below assumes half-integer nu"

gp_train_x, gp_lengthscale, gp_amplitude2, gp_weights, gp_trend = [], [], [], [], []

for b in range(N_BINS):
    res    = gpr_results[b]
    mask_b = masks_train[b]
    Xb     = X_train_scaled[mask_b]

    if res is None:                                  # const-fallback bin
        gp_train_x.append(Xb); gp_lengthscale.append(np.ones(n_inputs)); gp_amplitude2.append(0.0)
        gp_weights.append(np.zeros(Xb.shape[0]))
        gp_trend.append(float(np.mean(Y_train_z[:, b])))
        continue

    cov   = res.getCovarianceModel()
    theta = np.asarray(cov.getScale(), dtype=float)
    amp2  = float(np.asarray(cov.getAmplitude(), dtype=float)[0]) ** 2

    # OT's own dual weights and constant-trend coefficient (no re-derivation!)
    gamma = np.asarray(res.getCovarianceCoefficients(), dtype=float).flatten()
    beta  = float(np.asarray(res.getTrendCoefficients(), dtype=float).flatten()[0])

    gp_train_x.append(Xb); gp_lengthscale.append(theta); gp_amplitude2.append(amp2)
    gp_weights.append(gamma); gp_trend.append(beta)

def matern_kernel(r, amp2):
    """Matern kernel value vector for half-integer nu."""
    if MATERN_NU == 0.5:
        return amp2 * np.exp(-r)
    if MATERN_NU == 1.5:
        s = np.sqrt(3.0)
        return amp2 * (1.0 + s * r) * np.exp(-s * r)
    s = np.sqrt(5.0)  # 2.5
    return amp2 * (1.0 + s * r + (5.0 / 3.0) * r**2) * np.exp(-s * r)

def matern_kernel_derivative(r, amp2):
    """Returns c(r) such that dk/dx_j = c(r) * u_j / theta_j  (r cancels inside)."""
    if MATERN_NU == 0.5:
        # dk/dx_j = -amp2 e^{-r} u_j/(r theta_j); keep 1/r (guard r=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 1e-12, -amp2 * np.exp(-r) / r, 0.0)
    if MATERN_NU == 1.5:
        s = np.sqrt(3.0)
        return -3.0 * amp2 * np.exp(-s * r)          # r cancels analytically
    s = np.sqrt(5.0)  # 2.5
    return -(5.0 / 3.0) * amp2 * (1.0 + s * r) * np.exp(-s * r)

def gp_predict(theta_scaled):
    """theta_scaled (n_inputs,) → (mu (N_BINS,), dmu/dx_scaled (N_BINS, n_inputs))."""
    mu  = np.empty(N_BINS)
    jac = np.zeros((N_BINS, n_inputs))
    for b in range(N_BINS):
        Xb, th, amp2, gamma = gp_train_x[b], gp_lengthscale[b], gp_amplitude2[b], gp_weights[b]
        if amp2 == 0.0:
            mu[b] = gp_trend[b]
            continue
        u  = (theta_scaled[None, :] - Xb) / th[None, :]
        r  = np.sqrt(np.sum(u * u, axis=1))
        kv = matern_kernel(r, amp2)
        mu[b] = gp_trend[b] + kv @ gamma
        coef = matern_kernel_derivative(r, amp2)
        jac[b, :] = gamma @ ((coef[:, None] * u) / th[None, :])
    return mu, jac

class GPRSurrogateOp(pt.Op):
    """Maps physical parameters θ → predicted log-std flux vector (N_BINS,)."""
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def perform(self, node, inputs, outputs):
        theta_scaled = inputs[0] * input_scale + input_shift
        mu, _ = gp_predict(theta_scaled)
        outputs[0][0] = mu

    def grad(self, inputs, g):
        return [pt.dot(g[0], gpr_grad_op(inputs[0]))]

class GPRGradientOp(pt.Op):
    """Jacobian of the GPR surrogate: d(y_b)/d(theta_j) for all bins and inputs."""
    itypes = [pt.dvector]
    otypes = [pt.dmatrix]

    def perform(self, node, inputs, outputs):
        theta_scaled = inputs[0] * input_scale + input_shift
        _, jac = gp_predict(theta_scaled)          # d(mu)/d(x_scaled)
        outputs[0][0] = jac * input_scale[None, :]      # chain rule → d(mu)/d(theta)

gpr_op      = GPRSurrogateOp()
gpr_grad_op = GPRGradientOp()

with pm.Model() as model:

    bound_min = np.array([prior_bounds[i][0] for i in range(len(INPUT_COLS))])
    bound_max = np.array([prior_bounds[i][1] for i in range(len(INPUT_COLS))])

    gpr_inputs = []

    # x1
    u_x1 = pm.Uniform("x1_u", lower=0.0, upper=1.0)
    x1   = pm.Deterministic("x1", bound_min[0] + u_x1 * (bound_max[0] - bound_min[0]))
    gpr_inputs.append(x1)

    cum_bounds = [x1]
    for i in range(1, 1 + N_THICKNESS_INPUTS):
        u_t   = pm.Uniform(f"t{i}_u", lower=0.0, upper=1.0)
        t_var = pm.Deterministic(f"t{i}", bound_min[i] + u_t * (bound_max[i] - bound_min[i]))
        gpr_inputs.append(t_var)
        cum_bounds.append(pm.Deterministic(f"x{i + 1}", cum_bounds[-1] + t_var))

    L_idx = INPUT_COLS.index("L")
    u_L   = pm.Uniform("L_u", lower=0.0, upper=1.0)
    L_var = pm.Deterministic("L", bound_min[L_idx] + u_L * (bound_max[L_idx] - bound_min[L_idx]))
    gpr_inputs.append(L_var)

    theta_vec = pt.stack(gpr_inputs)

    mu  = gpr_op(theta_vec)
    bin_weights = pt.as_tensor_variable(np.asarray(BIN_WEIGHTS, dtype=float))
    resid_z = (mu - Y_obs_norm_z) / Sigma_obs
    pm.Potential("obs", -0.5 * pt.sum(bin_weights * resid_z ** 2))

    print("\n[Sampling] Starting Monte Carlo Sampling")
    idata = pm.sample_smc(draws=SAMPLING_DRAWS, chains=SAMPLING_CHAINS, cores=SAMPLING_CORES, random_seed=None)
    #idata = pm.sample(
     #   SAMPLING_DRAWS,
      #  tune=SAMPLING_TUNE,
       # chains=SAMPLING_CHAINS,
        #cores=SAMPLING_CORES,
        #target_accept=0.97,
        #init="jitter+adapt_diag_grad",
        #random_seed=None,
    #)



print("\n" + "="*30)
print("INFERENCE SUMMARY")
print("="*30)
summary = az.summary(idata, var_names=INPUT_COLS)
print(summary)

# PAIR PLOT (Corner Plot)
# CORNER PLOT (scatter + contours + marginal distributions)
print("\n[Plotting] Generating Posterior Corner Plot...")
_stacked_tmp = idata.posterior.stack(draws=["chain", "draw"])
samples = np.column_stack([_stacked_tmp[c].values for c in INPUT_COLS])

ndim = len(INPUT_COLS)
fig_corner = corner.corner(
        samples, labels=INPUT_COLS,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_fmt=".2f", color="#2c6e9c",
        label_kwargs={"fontsize": 18},
        title_kwargs={"fontsize": 16},
        plot_datapoints=True, plot_density=True, fill_contours=True,
        levels=(0.39, 0.86, 0.989),          # 1σ, 2σ, 3σ in 2-D
        hist_kwargs={"density": True, "color": "#2c6e9c"},
        contour_kwargs={"colors": "#34495e"},
    )
# [cm] on axis labels only — titles keep bare parameter names
axes_c = np.array(fig_corner.axes).reshape((ndim, ndim))
for i, col in enumerate(INPUT_COLS):
    axes_c[-1, i].set_xlabel(f"{col} [cm]", fontsize=18)
    if i > 0:
        axes_c[i, 0].set_ylabel(f"{col} [cm]", fontsize=18)
for ax in fig_corner.get_axes():
    ax.tick_params(axis="both", labelsize=16)
fig_corner.savefig("calibration_pair_plot.png", dpi=300)
plt.close(fig_corner)

print("[Result] Corner plot saved as 'calibration_pair_plot.png'")

# Same corner plot without x1 (poster-friendly subset of parameters)
_cols_no_x1 = [c for c in INPUT_COLS if c != "x1"]
_idx_no_x1  = [INPUT_COLS.index(c) for c in _cols_no_x1]
samples_no_x1 = samples[:, _idx_no_x1]

print("\n[Plotting] Generating Posterior Corner Plot (without x1)...")
fig_corner_no_x1 = corner.corner(
        samples_no_x1, labels=_cols_no_x1,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_fmt=".2f", color="#2c6e9c",
        label_kwargs={"fontsize": 18},
        title_kwargs={"fontsize": 16},
        plot_datapoints=True, plot_density=True, fill_contours=True,
        levels=(0.39, 0.86, 0.989),          # 1σ, 2σ, 3σ in 2-D
        hist_kwargs={"density": True, "color": "#2c6e9c"},
        contour_kwargs={"colors": "#34495e"},
    )
# [cm] on axis labels only — titles keep bare parameter names
_ndim_no_x1 = len(_cols_no_x1)
axes_nx = np.array(fig_corner_no_x1.axes).reshape((_ndim_no_x1, _ndim_no_x1))
for i, col in enumerate(_cols_no_x1):
    axes_nx[-1, i].set_xlabel(f"{col} [cm]", fontsize=18)
    if i > 0:
        axes_nx[i, 0].set_ylabel(f"{col} [cm]", fontsize=18)
for ax in fig_corner_no_x1.get_axes():
    ax.tick_params(axis="both", labelsize=16)
fig_corner_no_x1.savefig("calibration_pair_plot_no_x1.png", dpi=300)
plt.close(fig_corner_no_x1)

print("[Result] Corner plot (no x1) saved as 'calibration_pair_plot_no_x1.png'")

# Trace plot (to check for convergence)
az.plot_trace(idata, var_names=INPUT_COLS)
plt.savefig("calibration_trace_plot.png", dpi=300)

print("\n[Validation] Finding best MCMC sample based on mean log-relative error...")

stacked = idata.posterior.stack(draws=["chain", "draw"])
posterior_draws  = np.column_stack([stacked[col].values for col in INPUT_COLS])  # (N_draws, n_inputs)
Y_obs_norm_orig = Y_obs_norm
eps = 1e-14

posterior_scaled = posterior_draws * input_scale + input_shift          # (N_draws, n_inputs)
posterior_sample     = ot.Sample(posterior_scaled.tolist())        # ot.Sample of all draws

pred_z_all    = np.empty((len(posterior_draws), N_BINS))
for b in range(N_BINS):
    pred_z_all[:, b] = np.array(gpr_models[b](posterior_sample)).flatten()

pred_orig_all = np.empty_like(pred_z_all)
for b in range(N_BINS):
    pred_orig_all[:, b] = y_transformers[b].inverse_transform(
        pred_z_all[:, b].reshape(-1, 1)
    ).flatten()

resid_z     = pred_z_all - Y_obs_norm_z[None, :]            # (N_draws, N_BINS)
chi2_draws  = np.sum(BIN_WEIGHTS[None, :] * (resid_z / Sigma_obs[None, :]) ** 2, axis=1)   # (N_draws,)
red_chi2    = chi2_draws / N_BINS                            # reduced χ² (≈1 is ideal)

best_idx    = int(np.argmin(chi2_draws))
best_draw      = posterior_draws[best_idx]
best_chi2   = float(chi2_draws[best_idx])
best_redchi = float(red_chi2[best_idx])

# Keep the old log-relative-error number too, purely as a human-readable
# diagnostic of the chosen draw (NOT used for selection anymore).
log_diff    = np.log(np.maximum(pred_orig_all, eps)) - np.log(np.maximum(Y_obs_norm_orig, eps))
mcmc_errors = np.mean(np.abs(log_diff), axis=1) * 100       # shape (N_draws,)
best_err    = float(mcmc_errors[best_idx])

print("\n" + "=" * 70)
print(f"BEST CONFIGURATION FOUND IN MCMC SAMPLING (min χ² in Z-space, Y_TRANSFORM={Y_TRANSFORM}):")
best_config_str = ", ".join([f"{col} = {best_draw[i]:.2f}" for i, col in enumerate(INPUT_COLS)])
print(best_config_str)
print(f"χ² = {best_chi2:.2f}   reduced χ² = {best_redchi:.2f}  (over {N_BINS} bins)")
print(f"Mean Log-Relative Error of this draw = {best_err:.2f}%  (diagnostic only)")
print("=" * 70 + "\n")

# Predictions of the best sample — already available from the vectorized sweep
best_scaled = best_draw * input_scale + input_shift
best_point  = ot.Point(best_scaled.tolist())
best_pred_z = pred_z_all[best_idx]           # already computed above
emu_mean    = pred_orig_all[best_idx].copy()

# best_draw is (x1, t1, ..., t_{N-1}, L) — split off L (last entry, see
# INPUT_COLS) before reconstructing cumulative X boundaries for OpenMC.
l_index  = INPUT_COLS.index("L")
best_L  = float(best_draw[l_index])
best_thk = np.delete(best_draw, l_index)

emu_std = np.array(val_rmse_list, dtype=float)

post_matrix = np.column_stack([stacked[c].values for c in INPUT_COLS])
corr = np.corrcoef(post_matrix, rowvar=False)

fig_c, ax_c = plt.subplots(figsize=(7, 6))
im = ax_c.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax_c.set_xticks(range(len(INPUT_COLS))); ax_c.set_xticklabels(INPUT_COLS, rotation=45)
ax_c.set_yticks(range(len(INPUT_COLS))); ax_c.set_yticklabels(INPUT_COLS)
for i in range(len(INPUT_COLS)):
    for j in range(len(INPUT_COLS)):
        ax_c.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                  fontsize=8, color="#2c3e50")
fig_c.colorbar(im, label="Pearson r")
ax_c.set_title("Posterior parameter correlation", fontweight="bold")
plt.tight_layout()
if SAVE_PLOTS:
    fig_c.savefig("calibration_posterior_correlation.png", dpi=300)
plt.close(fig_c)

_pp_q_lo, _pp_q_hi = (100 - PP_CRED) / 2.0, 100 - (100 - PP_CRED) / 2.0

# (1) GP predictive σ in log-std space at every posterior draw → surrogate term
sigma_gp_z_draws = np.full((len(posterior_draws), N_BINS), np.nan)
for b in range(N_BINS):
    if gpr_results[b] is not None:
        try:
            gpcc  = ot.GaussianProcessConditionalCovariance(gpr_results[b])
            var_b = np.array(gpcc.getConditionalMarginalVariance(posterior_sample)).flatten()
            sigma_gp_z_draws[:, b] = np.sqrt(np.maximum(var_b, 0.0))
        except Exception as e:
            print(f"        Bin {b:2d}  cond. variance failed: {e}")

# (2) Monte Carlo mixture + exact back-transform, per bin
_rng_pp = np.random.default_rng(RANDOM_SEED)
pp_lo  = np.zeros(N_BINS)
pp_med = np.zeros(N_BINS)
pp_hi  = np.zeros(N_BINS)
for b in range(N_BINS):
    mu_b = pred_z_all[:, b]
    sd_b = np.nan_to_num(sigma_gp_z_draws[:, b], nan=0.0)
    z_samps = mu_b[:, None] + sd_b[:, None] * _rng_pp.standard_normal(
        (len(mu_b), 50))
    y_samps = y_transformers[b].inverse_transform(
        z_samps.reshape(-1, 1)).flatten()
    pp_lo[b]  = np.percentile(y_samps, _pp_q_lo)
    pp_med[b] = np.percentile(y_samps, 50.0)
    pp_hi[b]  = np.percentile(y_samps, _pp_q_hi)

# (3) Variance decomposition (log-std space) — quantitative result for the paper
var_param_z = np.var(pred_z_all, axis=0)
var_surr_z  = np.nanmean(np.nan_to_num(sigma_gp_z_draws, nan=0.0) ** 2, axis=0)
frac_surr   = var_surr_z / np.maximum(var_param_z + var_surr_z, 1e-30)
print(f"\n[Predictive] Combined {PP_CRED}% band: parameter + surrogate uncertainty")
print(f"             Surrogate share of total variance (Z-space, Y_TRANSFORM={Y_TRANSFORM}):")
print(f"               mean={100*frac_surr.mean():.1f}%  "
      f"min={100*frac_surr.min():.1f}%  max={100*frac_surr.max():.1f}%")
print(f"             (high → emulator error dominates; low → parameter posterior dominates)")

in_band  = (Y_obs_norm >= pp_lo) & (Y_obs_norm <= pp_hi)
coverage = float(in_band.mean())
print(f"\n[PPC] Target inside {PP_CRED}% predictive band: "
      f"{in_band.sum()}/{N_BINS} bins ({100*coverage:.1f}%)  ")

e_x = ENERGY_MID_EV
fig_pp, ax_pp = plt.subplots(figsize=(13, 5))
ax_pp.fill_between(e_x, pp_lo, pp_hi, alpha=0.25, color="#16a085",
                   step="mid", label=f"{PP_CRED}% posterior predictive")
ax_pp.step(e_x, pp_med, where="mid", color="#16a085", lw=2,
           label="Predictive median")
ax_pp.scatter(e_x[in_band],  Y_obs_norm[in_band],  s=45, color="#34495e",
              zorder=5, label="Target (in band)")
ax_pp.scatter(e_x[~in_band], Y_obs_norm[~in_band], s=60, color="#e67e22",
              marker="X", zorder=6, label="Target (outside band)")
ax_pp.set_xscale("log")
ax_pp.set_yscale("log")
ax_pp.set_xlabel("Energy [eV]"); ax_pp.set_ylabel("Normalized flux")
ax_pp.set_title(f"Posterior Predictive Check  (coverage={100*coverage:.0f}%)",
                fontweight="bold")
ax_pp.legend(fontsize=8); ax_pp.grid(alpha=0.2, ls=":")
plt.tight_layout()
if SAVE_PLOTS:
    fig_pp.savefig("calibration_posterior_predictive.png", dpi=300)
plt.close(fig_pp)

fig_vd, ax_vd = plt.subplots(figsize=(13, 4))
ax_vd.fill_between(e_x, 0, 100*(1-frac_surr), step="mid",
                   color="#2c6e9c", alpha=0.7, label="Parameter uncertainty")
ax_vd.fill_between(e_x, 100*(1-frac_surr), 100, step="mid",
                   color="#e67e22", alpha=0.7, label="Surrogate uncertainty")
ax_vd.set_xscale("log")
ax_vd.set_ylim(0, 100)
ax_vd.set_xlabel("Energy [eV]"); ax_vd.set_ylabel("Share of total variance [%]")
ax_vd.set_title("Predictive variance decomposition", fontweight="bold")
ax_vd.legend(fontsize=8, loc="upper right"); ax_vd.grid(alpha=0.2, ls=":")
plt.tight_layout()
if SAVE_PLOTS:
    fig_vd.savefig("calibration_variance_decomposition.png", dpi=300)
plt.close(fig_vd)

best_bounds_cum = thicknesses_to_bounds(best_thk.reshape(1, -1))[0]
val_bounds      = [float(r) for r in best_bounds_cum]

print("\n[Val Phase 1] Generating weight windows for best configuration …")
print(f"  x_bounds = {[f'{r:.2f}' for r in val_bounds]}  L = {best_L:.2f} cm")
ww_path, *_ = generate_weight_windows(
    x_bounds=val_bounds,
    materials=list(MATERIALS),
    ww_dir=VAL_WW_DIR,
    rr_batches=WW_BATCHES,
    rr_inactive_batches=WW_INACTIVE_BATCHES,
    rr_particles=WW_PARTICLES,
    mgxs_particles=MGXS_PARTICLES,
    output=False,
    y_dim=best_L,
    z_dim=best_L,
)
print(f"[Val Phase 1] Weight windows written to {ww_path}")

print("\n[Val Phase 2] Running CE production simulation …")
sp_test = run_with_weight_windows(
    x_bounds=val_bounds,
    materials=list(MATERIALS),
    ww_path=ww_path,
    ce_dir=VAL_CE_DIR,
    batches=VAL_CE_BATCHES,
    particles=VAL_CE_PARTICLES,
    use_angle_bias=True,   # angle biasing can cause weird spectrum artifacts; we want a clean comparison
    max_lower_bound_ratio=1.0,
    output=False,
    y_dim=best_L,
    z_dim=best_L,
)
print(f"[Val Phase 2] Statepoint written to {sp_test}")

postprocess_statepoint(sp_test, TARGET_VALUES)

print("[Extraction] Reading OpenMC validation data...")
try:
    y_openmc_all, _rel_unc_pct = extract_flux(sp_test)
    y_openmc_std_all = y_openmc_all * (_rel_unc_pct / 100.0)
    print(f"[Extraction] Successfully extracted {len(y_openmc_all)} energy bins from OpenMC")
except Exception as e:
    print(f"[Warning] Failed to extract OpenMC data: {e}")
    y_openmc_all = None
    y_openmc_std_all = None

def expand_active(arr_active):
    full = np.full(N_BINS_FULL, np.nan)
    full[ACTIVE_IDX] = arr_active
    return full

Y_obs_norm_display = Y_obs_norm_full.copy()
Y_obs_norm_display[~ACTIVE_MASK] = np.nan
pp_med_full = expand_active(pp_med)
pp_lo_full  = expand_active(pp_lo)
pp_hi_full  = expand_active(pp_hi)

if y_openmc_all is not None:
    y_openmc_all     = y_openmc_all.copy()
    y_openmc_std_all = y_openmc_std_all.copy()
    y_openmc_all[~ACTIVE_MASK]     = np.nan
    y_openmc_std_all[~ACTIVE_MASK] = np.nan

np.savez(
    "mcmc_validation_band.npz",
    energy_edges_ev=ENERGY_EDGES_FULL,
    y_target=Y_obs_norm_display,
    pp_med=pp_med_full,
    pp_lo=pp_lo_full,
    pp_hi=pp_hi_full,
    pp_cred=PP_CRED,
)
print("[Result] Calibrated posterior-predictive band exported to 'mcmc_validation_band.npz'")
plot_mcmc_validation_comparison(sp_test)

for f in (sp_test, "summary.h5"):
    if f and os.path.exists(f):
        os.remove(f)

# Energy information from base_plates.ENERGY_EDGES
energy_edges_ev = np.array(ENERGY_EDGES)  # in eV
energy_edges_mev = energy_edges_ev * 1e-6      # convert to MeV
bin_centers_mev = (energy_edges_mev[:-1] + energy_edges_mev[1:]) / 2

# Find indices for 2-15 MeV range
mask_energy = (bin_centers_mev >= 2.0) & (bin_centers_mev <= 20.0)
bin_centers_filtered = bin_centers_mev[mask_energy]
idx_filtered = np.where(mask_energy)[0]

# Filter data to this energy range (skipped bins show up as NaN gaps)
y_obs_filtered = Y_obs_norm_display[idx_filtered]
pp_med_filtered = pp_med_full[idx_filtered]
pp_lo_filtered  = pp_lo_full[idx_filtered]
pp_hi_filtered  = pp_hi_full[idx_filtered]

# Use the OpenMC data we extracted and stored earlier
if y_openmc_all is not None:
    y_openmc_filtered = y_openmc_all[idx_filtered]
    y_openmc_std_filtered = y_openmc_std_all[idx_filtered]
else:
    y_openmc_filtered = None
    y_openmc_std_filtered = None

# Create linear-scale plot
fig_linear, ax_lin = plt.subplots(figsize=(12, 6))

# Plot OpenMC if available with stair plot
if y_openmc_filtered is not None:
    ax_lin.step(bin_centers_filtered, y_openmc_filtered, where='mid', color='#3498db',
                linewidth=2.5, label='OpenMC', zorder=3)
    ax_lin.fill_between(bin_centers_filtered,
                         y_openmc_filtered - y_openmc_std_filtered,
                         y_openmc_filtered + y_openmc_std_filtered,
                         step='mid', alpha=0.2, color='#3498db', label='OpenMC +/- 1σ', zorder=1)

# Plot user target values with stair plot
ax_lin.step(bin_centers_filtered, y_obs_filtered, where='mid', color='#e74c3c',
            linewidth=2.5, linestyle='--', label='Target values (Integral norm)', zorder=3)

# Plot emulator predictions with stair plot
ax_lin.step(bin_centers_filtered, pp_med_filtered, where='mid', color='#16a085',
            linewidth=2.5, label='Emulator (predictive median)', zorder=3)
ax_lin.fill_between(bin_centers_filtered,
                     pp_lo_filtered, pp_hi_filtered,
                     step='mid', alpha=0.2, color='#16a085',
                     label=f'Emulator {PP_CRED}% predictive band', zorder=1)
# Formatting
ax_lin.set_yscale('log')
ax_lin.set_xlabel('Energy [MeV]', fontsize=12)
ax_lin.set_ylabel('Normalized Flux [-]', fontsize=12)
ax_lin.set_title('Neutron Spectrum (Integral Normalized) - Linear E - Comparison',
                 fontsize=13, fontweight='bold')
ax_lin.set_xlim(1.5, 15.0)
ax_lin.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
ax_lin.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig("em_linear_2_15mev.png", dpi=300, bbox_inches='tight')
plt.close(fig_linear)

fig_log, ax_log = plt.subplots(figsize=(12, 7))

# Helper to plot step functions using edges
def plot_step(ax, edges, values, **kwargs):
    # Repeat the last value for 'post' step plot
    ax.step(edges, np.append(values, values[-1]), where='post', **kwargs)

def fill_between_step(ax, edges, v_lo, v_hi, **kwargs):
    ax.fill_between(edges, np.append(v_lo, v_lo[-1]), np.append(v_hi, v_hi[-1]), step='post', **kwargs)

# Plot OpenMC if available
if y_openmc_all is not None:
    plot_step(ax_log, energy_edges_mev, y_openmc_all, color='#3498db',
              linewidth=2.5, label='OpenMC', zorder=3)
    fill_between_step(ax_log, energy_edges_mev,
                      np.maximum(y_openmc_all - y_openmc_std_all, 1e-12),
                      y_openmc_all + y_openmc_std_all,
                      alpha=0.2, color='#3498db', label='OpenMC +/- 1sigma', zorder=1)

# Plot user target values (skipped/zero bins show as gaps)
plot_step(ax_log, energy_edges_mev, Y_obs_norm_display, color='#e74c3c',
          linewidth=2.5, linestyle='--', label='Target values', zorder=4)

# Plot emulator predictions
plot_step(ax_log, energy_edges_mev, pp_med_full, color='#16a085',
          linewidth=2.5, linestyle='-.', label='Emulator (predictive median)', zorder=3)
fill_between_step(ax_log, energy_edges_mev,
                  np.maximum(pp_lo_full, 1e-10),
                  pp_hi_full,
                  alpha=0.15, color='#16a085',
                  label=f'Emulator {PP_CRED}% predictive band', zorder=1)

# Formatting
ax_log.set_xscale('log')
ax_log.set_yscale('log')
ax_log.set_xlabel('Energy [MeV]', fontsize=18)
ax_log.set_ylabel('Normalized Flux [-]', fontsize=18)
ax_log.tick_params(axis='both', which='major', labelsize=14)
ax_log.tick_params(axis='both', which='minor', labelsize=12)
ax_log.set_xlim(energy_edges_mev[0] if energy_edges_mev[0] > 0 else 1e-8, energy_edges_mev[-1])

# Dynamic Y limits based on data range with padding (excluding +/- 1 sigma).
# Inactive / skipped bins are NaN gaps — ignore them when setting limits.
y_plot_data = [Y_obs_norm_display, pp_med_full]
if y_openmc_all is not None:
    y_plot_data.append(y_openmc_all)

all_y = np.concatenate([np.atleast_1d(y) for y in y_plot_data])
positive = all_y[np.isfinite(all_y) & (all_y > 0)]
if positive.size:
    ax_log.set_ylim(np.min(positive) * 0.5, np.max(positive) * 2.0)

ax_log.set_xlim((energy_edges_mev[0] + energy_edges_mev[1]) / 2, energy_edges_mev[-1])
ax_log.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
ax_log.legend(fontsize=14, loc='best')

plt.tight_layout()
plt.savefig("spectrum_comparison_log.png", dpi=300, bbox_inches='tight')
print("[Result] Log-log comparison plot saved as 'spectrum_comparison_log.png'")
plt.close(fig_log)
