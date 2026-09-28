import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import bisect
import pickle
import random
import warnings
from functools import lru_cache

import numpy as np
import optuna  # type: ignore
from optuna.exceptions import ExperimentalWarning  # type: ignore
from optuna.samplers import TPESampler  # type: ignore

from config import (
    LOG_EPS,
    PHI_TARGET_IN_LETHARGY,
    TARGET_ENERGY_EDGES,
    TARGET_VALUES,
    WEIGHT_MODE,
)
from config_detector import DETECTOR_GEOMETRY
from config_materials import COST_PER_KG, build_material_library
from lethargy_converter import lethargy_density_to_flux
from rebin_target_spectrum import rebin_spectrum
from transfer_matrix import (
    ENERGY_EDGES,
    N_MU,
    THICKNESSES,
    fine_to_target_matrix,
    get_source_spec,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  
# ══════════════════════════════════════════════════════════════════

MATERIALS = [
    "PE_BO", "graphite", "iron", "Pb", "concrete",
    "cast_iron", "HDPE", "SS_304", "tungsten", "water",
    "cadmium", "B4C", "Al_6061", "Nickel",
    "Copper", "Zirconium", "brass", "PTFE", "Mg", "Bi",
]

CONSTRAINTS = {
    "max_total_thickness":     160.0,   # cm
    "max_layers":               9,
    "max_thickness_per_layer":  130.0,   # cm
}

WATER_CLAD_MATERIAL = "Al_6061"
WATER_CLAD_THICKNESS = 1.0   # cm, each face

MIN_THICKNESS  = 0.2         # cm
THICKNESS_STEP = 0.1         # cm

DISTANCE_NORM: str = "frobenius"  # "induced1" | "spectral" | "frobenius"

ERROR_REF: float = 0.3     # log-RMSE that counts as 1 unit of spectral error
COST_REF: float = 2.0      # USD/cm² that counts as 1 unit of material cost
HV_REF: tuple[float, float] = (10.0, 10.0)  # nadir in the (f_error, f_cost) plane

N_RESTARTS     = 42
RESTART_WORKERS: int | None = max(1, (os.cpu_count()) // 2 - 1)  # None = all cores
N_TRIALS_EACH  = 4400
N_STARTUP_EACH = 200
PATIENCE       = 400
OPERATOR_LRU_SIZE = 80
TOP_N_CONFIGURATIONS = 10
COST_ERROR_YLIM_TOP = 1.5

RESTART_SEEDS = random.sample(range(1, 30000), N_RESTARTS)

# ══════════════════════════════════════════════════════════════════

OPT_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_database")
N_E = len(ENERGY_EDGES) - 1
N_E_TARGET = len(TARGET_ENERGY_EDGES) - 1
N_STATE = N_MU * N_E
_FINE2TARGET = fine_to_target_matrix(ENERGY_EDGES, TARGET_ENERGY_EDGES)
SOURCE_MU_INDEX = N_MU - 1


def _compute_fixed_trailing_layers() -> list[tuple[str, float]]:
    regions = DETECTOR_GEOMETRY.regions
    active_idx = next((i for i, r in enumerate(regions) if r.active), None)
    if active_idx is None or active_idx == 0:
        return []

    layers = []
    for i in range(active_idx):
        thickness = regions[i].half_extent("x") - regions[i + 1].half_extent("x")
        if thickness > 1e-9:
            mat = regions[i].material
            if not isinstance(mat, str):
                raise TypeError(
                    f"DETECTOR_GEOMETRY region {regions[i].name!r} uses a "
                    f"custom openmc.Material instance, not a MATERIALS name -- "
                    f"can't embed it as a fixed optimizer layer."
                )
            layers.append((mat, float(thickness)))
    return layers


FIXED_TRAILING_LAYERS: list[tuple[str, float]] = _compute_fixed_trailing_layers()


def with_water_cladding(stack):
    """Insert WATER_CLAD walls immediately before and after every water layer."""
    out: list[tuple[str, float]] = []
    for mat, d in stack:
        if mat == "water":
            out.append((WATER_CLAD_MATERIAL, WATER_CLAD_THICKNESS))
            out.append((mat, float(d)))
            out.append((WATER_CLAD_MATERIAL, WATER_CLAD_THICKNESS))
        else:
            out.append((mat, float(d)))
    return out


# ══════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════
def load_transfer_matrices():
    """Load and *validate* S_N tensors into T_lib[material][thickness]."""
    T_lib = {}
    mats = list(dict.fromkeys(
        [*MATERIALS, WATER_CLAD_MATERIAL, *(m for m, _ in FIXED_TRAILING_LAYERS)]
    ))
    for mat in mats:
        T_lib[mat] = {}
        for thk in THICKNESSES:
            fname = os.path.join(OPT_DB_DIR, f"sn_tensor_{mat}_{thk}cm.npy")
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Missing S_N tensor file: {fname}")
            T_sn = np.load(fname)
            if T_sn.shape != (N_MU, N_E, N_MU, N_E):
                raise ValueError(
                    f"{fname}: expected shape {(N_MU, N_E, N_MU, N_E)}, got {T_sn.shape}"
                )
            T_lib[mat][thk] = T_sn.reshape(N_STATE, N_STATE).astype(np.float64)
    return T_lib


def check_operator_dim(T_lib) -> None:
    found = False
    for mat_dict in T_lib.values():
        for T in mat_dict.values():
            if T is None:
                continue
            found = True
            if T.shape != (N_STATE, N_STATE):
                raise RuntimeError(
                    f"Operator shape {T.shape} != ({N_STATE}, {N_STATE})"
                )
    if not found:
        raise RuntimeError("No transfer matrices loaded.")


# ══════════════════════════════════════════════════════════════════
#  PHYSICS
# ══════════════════════════════════════════════════════════════════
def normalize_spectrum(phi):
    s = np.sum(phi)
    return phi / s if s != 0 and np.isfinite(s) else phi


_BASES_CACHE: dict = {}


def _interpolate_operator(mat: str, thickness: float, T_lib: dict) -> np.ndarray:
    bases = _BASES_CACHE[mat]
    for b in bases:
        if abs(b - thickness) < 1e-9:
            return T_lib[mat][b]

    idx = bisect.bisect_left(bases, thickness)
    d1, d2 = bases[idx - 1], bases[idx]
    T1, T2 = T_lib[mat][d1], T_lib[mat][d2]
    alpha = (thickness - d1) / (d2 - d1)
    log_floor = np.float64(1e-30)
    return np.exp(
        (1.0 - alpha) * np.log(np.maximum(T1, log_floor))
        + alpha * np.log(np.maximum(T2, log_floor))
    )


def fractional_matrix_power(T: np.ndarray, f: float,
                            threshold: float = 1e-10) -> np.ndarray:
    """T^f for a substochastic S_N operator.

    Integer ``f`` uses repeated squaring. Fractional ``f`` uses
    eigendecomposition; dead (|λ|<threshold) and Re(λ)<0 channels are zeroed.
    """
    n_int = int(round(f))
    if abs(f - n_int) < 1e-9 and n_int >= 0:
        return np.linalg.matrix_power(T.astype(np.float64), n_int)

    T_clean = np.clip(T.astype(np.float64), 0.0, None)
    eigvals, eigvecs = np.linalg.eig(T_clean)
    dead_or_bad = (np.abs(eigvals) < threshold) | (np.real(eigvals) < 0)
    eigvals_f = np.where(dead_or_bad, 0.0 + 0.0j, eigvals.astype(complex) ** f)

    try:
        T_f = eigvecs @ np.diag(eigvals_f) @ np.linalg.inv(eigvecs)
    except np.linalg.LinAlgError:
        T_f = eigvecs @ np.diag(eigvals_f) @ np.linalg.pinv(eigvecs)

    T_f = np.real(T_f)
    np.clip(T_f, 0.0, None, out=T_f)

    col_sums = T_f.sum(axis=0)
    if not np.all(np.isfinite(col_sums)):
        raise ValueError("Non-finite column sums after matrix power.")
    bad_cols = col_sums > 1.0 + 1e-6
    if np.any(bad_cols):
        raise ValueError(
            f"Column sums exceed 1 after matrix power: {col_sums[bad_cols]}"
        )
    return T_f


_T_LIB: dict = {}


def init_operator_cache(T_lib) -> None:
    """Publish DB tensors for on-demand operators; call once before forking."""
    global _T_LIB
    _T_LIB = T_lib
    _BASES_CACHE.clear()
    _BASES_CACHE.update({mat: sorted(T_lib[mat].keys()) for mat in T_lib})
    _operator.cache_clear()


@lru_cache(maxsize=OPERATOR_LRU_SIZE)
def _operator(mat: str, thickness: float) -> np.ndarray:
    """S_N operator for one layer. Do not mutate the returned array (LRU-shared)."""
    bases = _BASES_CACHE[mat]
    db_min, db_max = bases[0], bases[-1]
    if db_min - 1e-9 <= thickness <= db_max + 1e-9:
        return _interpolate_operator(mat, thickness, _T_LIB)
    base = db_min if thickness < db_min else db_max
    return fractional_matrix_power(_T_LIB[mat][base], thickness / base)


def evaluate_stack(stack, m_source):
    if not stack:
        return m_source.astype(np.float64, copy=True)
    m = m_source
    for mat, d in stack:
        d_key = round(float(d) / THICKNESS_STEP) * THICKNESS_STEP
        m = _operator(mat, d_key) @ m
    return m


_TARGET_CACHE: dict = {}


def _log_target(phi_target):
    key = id(phi_target)
    cached = _TARGET_CACHE.get(key)
    if cached is None:
        tn = normalize_spectrum(np.asarray(phi_target, dtype=np.float64))
        cached = (tn, np.log(tn + LOG_EPS))
        _TARGET_CACHE[key] = cached
    return cached


def target_nonzero_mask(phi_target: np.ndarray) -> np.ndarray:
    """True where the target has actual (non-zero) content."""
    return np.asarray(phi_target, dtype=np.float64) != 0.0


def build_objective_weights(
    phi_target: np.ndarray,
    mode: str | None = None,
) -> np.ndarray:
    """Per-bin log-MSE weights; zero-target bins are excluded, then renormalized."""
    from bin_weights import build_bin_weights

    phi_target = np.asarray(phi_target, dtype=np.float64)
    nonzero = target_nonzero_mask(phi_target)
    if not np.any(nonzero):
        raise ValueError("phi_target is all-zero; no bins to build weights for.")
    return build_bin_weights(mode, mask=nonzero)


def evaluate_output_spectrum(stack, phi_s_shape: np.ndarray) -> np.ndarray:
    v = np.zeros(N_STATE, dtype=np.float64)
    base = SOURCE_MU_INDEX * N_E
    v[base:base + N_E] = np.asarray(phi_s_shape, dtype=np.float64)
    fine = np.asarray(evaluate_stack(stack, v), dtype=np.float64).reshape(N_MU, N_E).sum(axis=0)
    np.maximum(fine, 0.0, out=fine)
    return _FINE2TARGET @ fine


def compute_log_mse(
    stack,
    phi_s_shape,
    phi_target,
    bin_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    phi_out = evaluate_output_spectrum(stack, phi_s_shape)
    pn = normalize_spectrum(phi_out)
    tn, log_tn = _log_target(phi_target)
    diff = np.log(pn + LOG_EPS) - log_tn
    if bin_weights is not None:
        mse = float(np.dot(bin_weights, diff * diff))
    else:
        nonzero = target_nonzero_mask(phi_target)
        mse = float(np.mean(diff[nonzero] * diff[nonzero]))
    return mse, float(np.dot(phi_out, tn))


# ══════════════════════════════════════════════════════════════════
#  COST MODEL
# ══════════════════════════════════════════════════════════════════

_DENSITY_G_CM3: dict[str, float] = {
    name: float(mat.density) for name, mat in build_material_library().items()
}


def stack_areal_cost(stack) -> float:
    """USD per cm² of frontal plate area (search stack + water-can walls)."""
    return float(sum(
        _DENSITY_G_CM3[mat] * d * COST_PER_KG[mat] / 1000.0
        for mat, d in stack
    ))


def normalize_objectives(log_rmse: float, areal_cost: float) -> tuple[float, float]:
    return float(log_rmse) / ERROR_REF, float(areal_cost) / COST_REF


def equal_weight_distance(log_rmse: float, areal_cost: float) -> float:
    e, c = normalize_objectives(log_rmse, areal_cost)
    return float(np.hypot(e, c))


# ══════════════════════════════════════════════════════════════════
#  CATEGORICAL DISTANCE  (Metric TPE)
# ══════════════════════════════════════════════════════════════════
_NORM_ORD_MAP = {
    "frobenius": "fro",   # ‖A‖_F = sqrt(Σ_ij A_ij²)
    "spectral":  2,       # ‖A‖_2 = σ_max(A)
    "induced1":  1,       # ‖A‖_1 = max_j Σ_i |A_ij|
}


def get_material_distance_func(T_lib, norm: str | None = None, thicknesses=None):
    """Metric-TPE distance over MATERIALS: sum_t ||T_m1(t) - T_m2(t)||."""
    norm = norm if norm is not None else DISTANCE_NORM
    if norm not in _NORM_ORD_MAP:
        raise ValueError(
            f"Unknown distance norm {norm!r}. "
            f"Choose from {sorted(_NORM_ORD_MAP)}."
        )
    ord_ = _NORM_ORD_MAP[norm]

    if thicknesses is None:
        # Intersection: only sum over thicknesses available for ALL materials
        thk_sets = [set(T_lib[m].keys()) for m in MATERIALS]
        common = sorted(set.intersection(*thk_sets)) if thk_sets else []
        if not common:
            raise RuntimeError(
                "No common base thicknesses across all MATERIALS; "
                "pass `thicknesses=` explicitly."
            )
        thicknesses = common
    thicknesses = list(thicknesses)

    n = len(MATERIALS)
    D = np.zeros((n, n), dtype=np.float64)

    # Pre-stack the difference once per thickness to amortise indexing.
    for t in thicknesses:
        # Shape (n, N_STATE, N_STATE) → broadcast-difference along axis 0/1
        T_stack = np.stack([T_lib[m][t] for m in MATERIALS], axis=0)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] += float(np.linalg.norm(T_stack[i] - T_stack[j], ord=ord_))

    # Symmetrise (lower triangle was left at zero above).
    D = D + D.T
    np.fill_diagonal(D, 0.0)

    idx = {m: i for i, m in enumerate(MATERIALS)}

    def distance(m1, m2):
        return float(D[idx[m1], idx[m2]])

    distance.matrix = D                   
    distance.materials = list(MATERIALS)
    distance.norm = norm
    distance.thicknesses = thicknesses
    return distance


# ══════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════
def create_objective(phi_s_shape, phi_target, bin_weights=None):
    max_L             = CONSTRAINTS["max_layers"]
    max_thk_per_layer = float(CONSTRAINTS["max_thickness_per_layer"])
    max_total         = CONSTRAINTS["max_total_thickness"]

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, max_L)
        stack    = []
        for i in range(n_layers):
            mat = trial.suggest_categorical(f"layer_{i}_material", MATERIALS)
            thk = trial.suggest_float(
                f"layer_{i}_thickness", MIN_THICKNESS, max_thk_per_layer,
                step=THICKNESS_STEP,
            )
            stack.append((mat, thk))

        clad_stack = with_water_cladding(stack)
        total_cm = sum(d for _, d in clad_stack)
        mse, twf = compute_log_mse(
            list(clad_stack) + FIXED_TRAILING_LAYERS,
            phi_s_shape, phi_target, bin_weights=bin_weights,
        )
        log_rmse = np.sqrt(mse)
        areal_cost = stack_areal_cost(clad_stack)
        overshoot = max(0.0, total_cm - max_total)

        trial.set_user_attr("total_thickness", total_cm)
        trial.set_user_attr("stack_config", str(stack))
        trial.set_user_attr("log_rmse", float(log_rmse))
        trial.set_user_attr("areal_cost", float(areal_cost))
        trial.set_user_attr("target_weighted_flux", twf)
        trial.set_user_attr("thickness_overshoot", float(overshoot))
        return normalize_objectives(log_rmse, areal_cost)

    return objective


def reconstruct_stack(trial):
    n = trial.params["n_layers"]
    stack = []
    for i in range(n):
        mat = trial.params[f"layer_{i}_material"]
        thk = trial.params[f"layer_{i}_thickness"]
        stack.append((mat, thk))
    return stack


def _constraints_func(trial) -> tuple[float, ...]:
    return (float(trial.user_attrs.get("thickness_overshoot", 0.0)),)


def _is_feasible(trial) -> bool:
    return float(trial.user_attrs.get("thickness_overshoot", 0.0)) <= 0.0


def _trial_raw_objectives(trial) -> tuple[float, float] | None:
    """Raw (log-RMSE, areal cost) of a completed trial, or None if missing."""
    attrs = trial.user_attrs
    if "log_rmse" in attrs and "areal_cost" in attrs:
        return float(attrs["log_rmse"]), float(attrs["areal_cost"])
    values = getattr(trial, "values", None)
    if values is not None and len(values) >= 2:
        return float(values[0]) * ERROR_REF, float(values[1]) * COST_REF
    return None

# ══════════════════════════════════════════════════════════════════
#  STAGNATION CALLBACK  (hypervolume of the MOTPE Pareto front)
# ══════════════════════════════════════════════════════════════════
def _hypervolume_2d(obj: np.ndarray, ref: tuple[float, float] | np.ndarray) -> float:
    """2-objective hypervolume for MINIMIZE/MINIMIZE.

    ``obj`` has shape (n, 2) with columns ``(f_error, f_cost)``. Points that
    do not strictly dominate ``ref`` are dropped; the remaining Pareto front
    is integrated toward the nadir. Empty set → 0.
    """
    if obj is None or len(obj) == 0:
        return 0.0
    ref = np.asarray(ref, dtype=np.float64)
    pts = np.asarray(obj, dtype=np.float64)
    pts = pts[np.all(pts < ref, axis=1)]
    if pts.size == 0:
        return 0.0
    mask = _pareto_front_mask(pts[:, 1], pts[:, 0])   # cost, error
    front = pts[mask]
    order = np.argsort(front[:, 0], kind="stable")    # sort by error ↑
    xs = front[order, 0]
    ys = front[order, 1]
    xs_ext = np.append(xs, ref[0])
    return float(np.sum((xs_ext[1:] - xs_ext[:-1]) * (ref[1] - ys)))


def _trials_normalized_obj(trials) -> np.ndarray:
    """(n, 2) array of normalized (error, cost) for feasible completed trials."""
    rows = []
    for t in trials:
        if getattr(t, "state", optuna.trial.TrialState.COMPLETE) != optuna.trial.TrialState.COMPLETE:
            continue
        raw = _trial_raw_objectives(t)
        if raw is None or not _is_feasible(t):
            continue
        rows.append(normalize_objectives(*raw))
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 2))


def _trials_hypervolume(trials, ref=HV_REF) -> float:
    return _hypervolume_2d(_trials_normalized_obj(trials), ref)


class StagnationCallback:
    """Stop a restart when the Pareto hypervolume stops improving.
    """
    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-6,
                 restart_idx: int = 0, progress_q=None):
        self.patience     = patience
        self.min_delta    = min_delta
        self.restart_idx  = restart_idx
        self.progress_q   = progress_q
        self._best_hv     = -np.inf
        self._counter     = 0
        self._n_done      = 0
        self.triggered    = False
        self.best_hv      = 0.0

    def _report(self, hv: float) -> None:
        q = self.progress_q
        if q is None:
            return
        try:
            q.put_nowait((self.restart_idx, self._n_done, float(hv), self.triggered))
        except Exception:
            pass

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        self._n_done += 1
        hv = _trials_hypervolume(study.trials)
        if hv > self._best_hv + self.min_delta:
            self._best_hv = hv
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience and not self.triggered:
            self.triggered = True
            self.best_hv = self._best_hv
            if self.progress_q is None:
                print(f"\n  [Stagnation] No hypervolume improvement for "
                      f"{self.patience} trials → stopping restart early  "
                      f"(HV={self._best_hv:.6f})")
            study.stop()
        self.best_hv = self._best_hv
        self._report(hv)


# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════
def merge_consecutive_layers(stack):
    """Collapse adjacent layers of the same material, summing thicknesses.
    """
    out: list[tuple[str, float]] = []
    for mat, d in stack:
        d = float(d)
        if out and out[-1][0] == mat:
            out[-1] = (mat, out[-1][1] + d)
        else:
            out.append((mat, d))
    return out


def stack_to_key(stack):
    """Convert a stack to a hashable key."""
    return tuple(mat for mat, _ in merge_consecutive_layers(stack))


def count_arrangement_frequencies(all_completed):
    """
    For every completed trial across ALL restarts, tally how many times
    each unique material arrangement (order matters) was sampled.
    """
    freq = {}
    for t in all_completed:
        try:
            key = stack_to_key(with_water_cladding(reconstruct_stack(t)))
            freq[key] = freq.get(key, 0) + 1
        except Exception:
            pass
    return freq


def feasible_completed(trials):
    """Completed trials that satisfy the thickness budget and carry both objectives."""
    out = []
    for t in trials:
        if getattr(t, "state", optuna.trial.TrialState.COMPLETE) != optuna.trial.TrialState.COMPLETE:
            continue
        if _trial_raw_objectives(t) is None:
            continue
        if not _is_feasible(t):
            continue
        out.append(t)
    return out


def pareto_front_trials(trials):
    """Feasible Pareto front, sorted by areal cost ascending."""
    valid = feasible_completed(trials)
    if not valid:
        return []
    costs  = np.array([t.user_attrs["areal_cost"] for t in valid])
    errors = np.array([t.user_attrs["log_rmse"] for t in valid])
    mask   = _pareto_front_mask(costs, errors)
    front  = [t for t, m in zip(valid, mask) if m]
    front.sort(key=lambda t: t.user_attrs["areal_cost"])
    return front


def knee_trial(trials):
    """Equal-weight recommendation: feasible Pareto point closest to the origin
    in the normalized (log_rmse/ERROR_REF, cost/COST_REF) plane."""
    front = pareto_front_trials(trials)
    if not front:
        return None
    return min(
        front,
        key=lambda t: equal_weight_distance(
            t.user_attrs["log_rmse"], t.user_attrs["areal_cost"]
        ),
    )


def _arrangement_tags(front):
    """Canonical labels keyed by material sequence (order only).
    """
    if not front:
        return {}
    knee = min(
        front,
        key=lambda t: equal_weight_distance(
            t.user_attrs["log_rmse"], t.user_attrs["areal_cost"]
        ),
    )
    min_err = min(front, key=lambda t: t.user_attrs["log_rmse"])
    tags: dict[tuple, str] = {}
    for t, label in (
        (min_err, "MIN-ERROR"),
        (knee,    "EQUAL-WEIGHT KNEE"),
    ):
        try:
            k = stack_to_key(with_water_cladding(reconstruct_stack(t)))
        except Exception:
            continue
        prev = tags.get(k)
        tags[k] = f"{prev} + {label}" if prev else label
    return tags


def _trial_merged_stack(t):
    """Reconstruct, add water cladding, merge consecutive same-material layers."""
    return merge_consecutive_layers(with_water_cladding(reconstruct_stack(t)))


def get_unique_top_configs(all_completed, arrangement_freq, total_completed,
                           top_n=TOP_N_CONFIGURATIONS):
    """One representative per unique material sequence, up to ``top_n``.
    Thickness is ignored (consecutive same-material layers are merged). Pareto
    sequences come first, ranked lowest error → highest. If those are fewer
    than ``top_n``, remaining slots are filled with unique sequences that are
    *not* on the front (picked by equal-weight distance, then ranked lowest
    error → highest).
    """
    def _dist(t):
        return equal_weight_distance(
            t.user_attrs["log_rmse"], t.user_attrs["areal_cost"]
        )

    def _best_by_key(trials):
        best: dict = {}
        for t in trials:
            try:
                stack = _trial_merged_stack(t)
            except Exception:
                continue
            key = stack_to_key(stack)
            d = _dist(t)
            prev = best.get(key)
            if prev is None or d < prev[0]:
                best[key] = (d, t, stack)
        return best

    front = pareto_front_trials(all_completed)
    result = []
    blocked = set()
    knee = min_err = None

    if front:
        best = _best_by_key(front)
        knee     = min(front, key=_dist)
        min_err  = min(front, key=lambda t: t.user_attrs["log_rmse"])
        min_cost = front[0]
        keep_keys = set()
        for special in (knee, min_err):
            try:
                stack = _trial_merged_stack(special)
            except Exception:
                continue
            key = stack_to_key(stack)
            best[key] = (_dist(special), special, stack)
            keep_keys.add(key)
        try:
            min_cost_key = stack_to_key(_trial_merged_stack(min_cost))
            blocked.add(min_cost_key)
            if min_cost_key not in keep_keys:
                best.pop(min_cost_key, None)
        except Exception:
            pass

        unique = sorted(
            ((t, stack, key) for key, (_, t, stack) in best.items()),
            key=lambda u: u[0].user_attrs["log_rmse"],
        )
        n = len(unique)
        if n <= top_n:
            idxs = list(range(n))
        else:
            idxs = {int(round(i)) for i in np.linspace(0, n - 1, top_n)}
            by_id = {id(u[0]): i for i, u in enumerate(unique)}
            for special in (knee, min_err):
                i = by_id.get(id(special))
                if i is not None:
                    idxs.add(i)
            idxs = sorted(idxs)

        for i in idxs:
            t, stack, key = unique[i]
            blocked.add(key)
            count = arrangement_freq.get(key, 0)
            result.append((
                t, float(t.user_attrs["log_rmse"]), stack, count,
                100.0 * count / max(total_completed, 1), True,
            ))

    need = top_n - len(result)
    if need > 0:
        extras = _best_by_key(feasible_completed(all_completed))
        ranked = sorted(
            ((d, t, stack, key) for key, (d, t, stack) in extras.items()
             if key not in blocked),
            key=lambda x: x[0],
        )
        picked = ranked[:need]
        picked.sort(key=lambda x: float(x[1].user_attrs["log_rmse"]))
        for _, t, stack, key in picked:
            count = arrangement_freq.get(key, 0)
            result.append((
                t, float(t.user_attrs["log_rmse"]), stack, count,
                100.0 * count / max(total_completed, 1), False,
            ))
    return result


# ══════════════════════════════════════════════════════════════════
#  PLOTS  (design-choice aid: cost vs. spectral error)
# ══════════════════════════════════════════════════════════════════
def _pareto_front_mask(costs: np.ndarray, errors: np.ndarray) -> np.ndarray:

    order    = np.argsort(costs, kind="stable")
    mask     = np.zeros(len(costs), dtype=bool)
    best_err = np.inf
    for i in order:
        if errors[i] < best_err - 1e-12:
            best_err = errors[i]
            mask[i]  = True
    return mask


def plot_cost_error_tradeoff(
    all_completed,
    savefig: str = "cost_error_tradeoff.png",
    highlight=None,
):
    """Scatter every completed trial in the (areal cost, log-RMSE) plane.

    Draws the MOTPE Pareto front, equal-weight iso-distance contours
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    completed = [
        t for t in all_completed
        if "areal_cost" in t.user_attrs and "log_rmse" in t.user_attrs
    ]
    if not completed:
        print("  [plot] no trials carry cost/error attrs; skipping trade-off plot.")
        return

    costs  = np.array([t.user_attrs["areal_cost"] for t in completed])
    errors = np.array([t.user_attrs["log_rmse"] for t in completed])
    feas   = np.array([_is_feasible(t) for t in completed])
    front_trials = pareto_front_trials(completed)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.scatter(costs[~feas], errors[~feas], s=8, c="0.85", alpha=0.4,
               label=f"infeasible (n={int((~feas).sum())})")
    ax.scatter(costs[feas], errors[feas], s=8, c="0.65", alpha=0.45,
               label=f"feasible trials (n={int(feas.sum())})")

    if front_trials:
        fc = np.array([t.user_attrs["areal_cost"] for t in front_trials])
        fe = np.array([t.user_attrs["log_rmse"] for t in front_trials])
        nlay = np.array([int(t.params.get("n_layers", len(reconstruct_stack(t))))
                         for t in front_trials])
        ax.plot(fc, fe, "-", color="crimson", lw=1.6, zorder=3,
                label=f"Pareto front (n={len(front_trials)})")
        nlay_max = max(int(nlay.max()), 1)
        sc = ax.scatter(fc, fe, c=nlay, cmap="viridis", s=28, zorder=4,
                        edgecolor="k", linewidths=0.3, vmin=1, vmax=nlay_max)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, ticks=range(1, nlay_max + 1))
        cbar.set_label("number of layers", fontsize=15)
        cbar.ax.tick_params(labelsize=15)

        knee = min(
            front_trials,
            key=lambda t: equal_weight_distance(
                t.user_attrs["log_rmse"], t.user_attrs["areal_cost"]
            ),
        )
        r_knee = equal_weight_distance(
            knee.user_attrs["log_rmse"], knee.user_attrs["areal_cost"]
        )
        theta = np.linspace(0.0, 0.5 * np.pi, 256)
        ax.plot(
            COST_REF * r_knee * np.cos(theta),
            ERROR_REF * r_knee * np.sin(theta),
            "--", color="goldenrod", lw=1.1, alpha=0.7, zorder=2,
            scalex=False, scaley=False,
        )

        markers = [
            (min(front_trials, key=lambda t: t.user_attrs["log_rmse"]),
             "MIN-ERROR", "^", "royalblue", 120),
            (knee, "EQUAL-WEIGHT KNEE", "*", "gold", 220),
        ]
        drawn = set()
        for t, label, mk, col, size in markers:
            if label in drawn:
                continue
            drawn.add(label)
            ax.scatter(
                [t.user_attrs["areal_cost"]], [t.user_attrs["log_rmse"]],
                marker=mk, s=size, color=col,
                edgecolor="k", linewidths=0.7, zorder=6, label=label,
            )

    if highlight:
        for label, c, e in highlight:
            ax.annotate(label, (c, e), textcoords="offset points",
                        xytext=(7, 5), fontsize=13)

    ax.set_xlabel("Areal material cost  [USD / cm²]", fontsize=15)
    ax.set_ylabel("Spectral error  (detector-weighted log-RMSE)", fontsize=15)
    ax.set_title(
        "MOTPE cost vs. spectral-error Pareto front  "
        f"(y capped at {COST_ERROR_YLIM_TOP:g} log-RMSE)\n"
        f"(1 unit error = {ERROR_REF:g} log-RMSE,  "
        f"1 unit cost = {COST_REF:g} USD/cm²)",
        fontsize=17,
    )
    ax.tick_params(axis="both", which="both", labelsize=15)
    if front_trials:
        fc = np.array([t.user_attrs["areal_cost"] for t in front_trials])
        fe = np.array([t.user_attrs["log_rmse"] for t in front_trials])
        cspan = max(float(fc.max() - fc.min()), 1e-9)
        pad_c = max(0.10 * cspan, 0.04 * float(fc.max()), 1e-3)
        pad_e = max(0.08 * COST_ERROR_YLIM_TOP, 0.02)
        ax.set_xlim(max(0.0, float(fc.min()) - pad_c), float(fc.max()) + pad_c)
        ax.set_ylim(max(0.0, float(fe.min()) - pad_e), COST_ERROR_YLIM_TOP)
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(0.0, COST_ERROR_YLIM_TOP)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    extra = [Line2D([0], [0], ls="--", color="goldenrod",
                    label="equal-weight iso-distance")]
    ax.legend(handles + extra, labels + ["equal-weight iso-distance"],
              loc="upper right", fontsize=13)
    fig.tight_layout()
    fig.savefig(savefig, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved cost/error trade-off → {savefig}")


def plot_pareto_designs(
    designs,
    savefig: str = "pareto_designs.png",
    tags=None,
):
    """Horizontal stacked bars: total length is log-RMSE
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not designs:
        print("  [plot] no Pareto designs to draw; skipping stack plot.")
        return

    tags = tags or {}
    cmap = plt.get_cmap("tab20")
    names = list(dict.fromkeys(
        list(MATERIALS) + [WATER_CLAD_MATERIAL] + [m for m, _ in FIXED_TRAILING_LAYERS]
    ))
    colors = {m: cmap(i % 20) for i, m in enumerate(names)}
    n = len(designs)
    fig, ax = plt.subplots(figsize=(12.5, max(3.8, 0.48 * n + 2.0)))

    used_mats = []
    saw_off_front = False
    n_on = 0
    xmax = 0.0
    for row, (t, stack) in enumerate(designs):
        rmse = float(t.user_attrs["log_rmse"])
        xmax = max(xmax, rmse)
        total_cm = sum(float(d) for _, d in stack) or 1.0
        tag = tags.get(id(t), "")
        off_front = "NOT ON PARETO FRONT" in tag
        if off_front:
            saw_off_front = True
        else:
            n_on += 1
        x = 0.0
        for mat, d in stack:
            w = rmse * (float(d) / total_cm)
            ax.barh(
                row, w, left=x, height=0.72,
                color=colors.get(mat, "0.7"),
                edgecolor="k", linewidth=0.35,
                alpha=0.45 if off_front else 1.0,
                hatch="///" if off_front else None,
            )
            if mat not in used_mats:
                used_mats.append(mat)
            x += w
        thk = " + ".join(f"{float(d):.2f}" for _, d in stack)
        cost = float(t.user_attrs["areal_cost"])
        tag_s = f"  [{tag}]" if tag else ""
        ax.text(
            rmse, row,
            f"  {thk} cm   {cost:.3f} USD/cm²{tag_s}",
            va="center", fontsize=11,
        )

    if saw_off_front and 0 < n_on < n:
        ax.axhline(n_on - 0.5, color="0.35", ls="--", lw=1.0, zorder=2)

    yticklabels = []
    for t, stack in designs:
        arr = " → ".join(m for m, _ in stack)
        if len(arr) > 42:
            arr = arr[:40] + "…"
        yticklabels.append(arr)
    ax.set_yticks(range(n))
    ax.set_yticklabels(yticklabels, fontsize=11)
    ax.set_xlabel("Spectral error  (detector-weighted log-RMSE)", fontsize=13)
    ax.tick_params(axis="x", which="both", labelsize=13)
    ax.set_xlim(0.0, xmax * 1.42 if xmax > 0 else 1.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    title = ("Unique material sequences  "
             "(bar length = error; labels = thickness; lowest error at the top)")
    if saw_off_front:
        title += "\nHatched / faded bars are NOT on the Pareto front"
    ax.set_title(title, fontsize=15)
    ax.grid(True, axis="x", alpha=0.3)

    legend_handles = [
        Patch(facecolor=colors[m], edgecolor="k", label=m) for m in used_mats
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11,
              title="material", title_fontsize=13, framealpha=0.92, ncol=2)
    fig.tight_layout()
    fig.savefig(savefig, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved Pareto recipes → {savefig}")


def plot_design_choice_figures(all_completed, unique, save_prefix: str = ""):
    """Build the design-choice figures from a finished run."""
    front = pareto_front_trials(all_completed)
    arr_tags = _arrangement_tags(front)
    tags = {}
    for t, _, stack, *rest in unique:
        on_front = rest[-1] if rest else True
        label = arr_tags.get(stack_to_key(stack), "")
        if not on_front:
            extra = "NOT ON PARETO FRONT"
            label = f"{label} + {extra}" if label else extra
        tags[id(t)] = label
    plot_cost_error_tradeoff(
        all_completed,
        savefig=f"{save_prefix}cost_error_tradeoff.png",
    )
    on = sorted((u for u in unique if u[-1]), key=lambda u: float(u[1]))
    off = sorted((u for u in unique if not u[-1]), key=lambda u: float(u[1]))
    designs = [(t, stack) for t, _, stack, *_ in (*on, *off)]
    plot_pareto_designs(
        designs,
        savefig=f"{save_prefix}pareto_designs.png",
        tags=tags,
    )


# ══════════════════════════════════════════════════════════════════
#  PERSIST
# ══════════════════════════════════════════════════════════════════
def save_all_studies(all_studies, path="study_results_multistart.pkl"):
    data = {
        "n_restarts": len(all_studies),
        "restart_seeds": [s for s, _ in all_studies],
        "studies": [
            {
                "restart_idx": idx,
                "seed":        seed,
                "best_value":  study.best_value,
                "trials":      study.trials,
                "direction":   study.direction,
                "directions":  getattr(study, "directions", study.direction),
                "n_pareto":    getattr(study, "n_pareto", 0),
                "hypervolume": getattr(study, "hypervolume", 0.0),
            }
            for idx, (seed, study) in enumerate(all_studies)
        ],
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    total = sum(len(s.trials) for _, s in all_studies)
    print(f"  Saved {total} total trials ({len(all_studies)} restarts) → {path}")


# ══════════════════════════════════════════════════════════════════
#  MULTI-START RUNNER
# ══════════════════════════════════════════════════════════════════
class StudyResult:
    __slots__ = (
        "best_value", "trials", "direction", "directions",
        "n_pareto", "hypervolume",
    )

    def __init__(self, best_value, trials, direction, directions=None,
                 n_pareto=0, hypervolume=0.0):
        self.best_value  = best_value
        self.trials      = trials
        self.direction   = direction
        self.directions  = (
            directions if directions is not None
            else (direction if isinstance(direction, (list, tuple))
                  else ("minimize", "minimize"))
        )
        self.n_pareto    = n_pareto
        self.hypervolume = hypervolume


# Set in the parent before fork so workers inherit them copy-on-write.
_W_PHI_S_SHAPE = None
_W_PHI_TARGET = None
_W_BIN_WEIGHTS = None
_W_CAT_DIST_FUNC = None
_W_PROGRESS_Q = None

def _run_one_restart(restart_idx: int):
    """Run a single independent restart end-to-end (one process, n_jobs=1).

    Reads its inputs from the module-level worker globals (fork-inherited) so the
    only thing crossing the process boundary is the integer ``restart_idx`` in
    and a picklable :class:`StudyResult` out.
    """
    seed = RESTART_SEEDS[restart_idx % len(RESTART_SEEDS)]

    objective = create_objective(
        _W_PHI_S_SHAPE, _W_PHI_TARGET, bin_weights=_W_BIN_WEIGHTS
    )

    sampler = TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constant_liar=False,
        n_startup_trials=N_STARTUP_EACH,
        gamma=lambda x: max(1, int(np.ceil(0.15 * x))),
        n_ei_candidates=60,
        consider_prior=True,
        consider_magic_clip=True,
        warn_independent_sampling=False,
        constraints_func=_constraints_func,
        categorical_distance_func=_W_CAT_DIST_FUNC,
    )

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name=f"restart_{restart_idx+1}_seed_{seed}",
    )

    stag_cb = StagnationCallback(
        patience=PATIENCE,
        restart_idx=restart_idx,
        progress_q=_W_PROGRESS_Q,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS_EACH,
        n_jobs=1,
        callbacks=[stag_cb],
        show_progress_bar=False,
    )

    n_done = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE)
    knee = knee_trial(study.trials)
    best_value = (
        equal_weight_distance(
            knee.user_attrs["log_rmse"], knee.user_attrs["areal_cost"]
        ) if knee is not None else float("inf")
    )
    result = StudyResult(
        best_value,
        study.trials,
        tuple(study.directions),
        directions=tuple(study.directions),
        n_pareto=len(pareto_front_trials(study.trials)),
        hypervolume=_trials_hypervolume(study.trials),
    )
    return restart_idx, seed, result, stag_cb.triggered, n_done


def _progress_monitor(q, n_restarts: int, n_trials: int) -> None:
    from queue import Empty
    from tqdm.auto import tqdm

    bars = [
        tqdm(
            total=n_trials,
            desc=f"Restart {i+1:>2}/{n_restarts}",
            position=i,
            leave=True,
            dynamic_ncols=True,
            unit="trial",
            mininterval=0.2,
            smoothing=0.05,
        )
        for i in range(n_restarts)
    ]
    seen_stag: set[int] = set()

    def _apply(msg) -> bool:
        if msg is None:
            return True
        idx, n_done, hv, triggered = msg
        bar = bars[idx]
        n_done = int(min(max(n_done, 0), n_trials))
        delta = n_done - bar.n
        if delta > 0:
            bar.update(delta)
        postfix = {"HV": f"{hv:.4f}"}
        if triggered:
            postfix["stop"] = "stag"
            if idx not in seen_stag:
                seen_stag.add(idx)
                tqdm.write(
                    f"  [Stagnation] Restart {idx+1}: no HV improvement for "
                    f"{PATIENCE} trials → stopping early  (HV={hv:.6f})"
                )
        bar.set_postfix(postfix, refresh=False)
        bar.refresh()
        return False

    try:
        while True:
            msg = q.get()
            if _apply(msg):
                break
            while True:
                try:
                    nxt = q.get_nowait()
                except Empty:
                    break
                if _apply(nxt):
                    return
    finally:
        for bar in bars:
            bar.close()
        print(flush=True)


def run_multi_start(
    T_lib,
    phi_s_shape,
    phi_target,
    bin_weights: np.ndarray | None = None,
):
    """
    Run N_RESTARTS independent TPE studies with different seeds,
    executing up to RESTART_WORKERS of them concurrently as
    separate processes.
    """
    import multiprocessing as mp
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm.auto import tqdm

    global _W_PHI_S_SHAPE, _W_PHI_TARGET, _W_BIN_WEIGHTS
    global _W_CAT_DIST_FUNC, _W_PROGRESS_Q
    _W_PHI_S_SHAPE = phi_s_shape
    _W_PHI_TARGET = phi_target
    _W_BIN_WEIGHTS = bin_weights
    mat_dist = get_material_distance_func(T_lib)
    _W_CAT_DIST_FUNC = {
        f"layer_{i}_material": mat_dist
        for i in range(CONSTRAINTS["max_layers"])
    }

    n_cores = os.cpu_count() or 1
    workers = RESTART_WORKERS if RESTART_WORKERS is not None else n_cores
    workers = max(1, min(int(workers), N_RESTARTS))

    ctx = mp.get_context("fork")
    progress_q = ctx.Queue()
    _W_PROGRESS_Q = progress_q

    print(f"\n{'─'*62}")
    print(f"  Launching {N_RESTARTS} restarts  │  {workers} concurrent "
          f"process(es)  │  n_jobs=1/study")
    print(f"{'─'*62}", flush=True)

    monitor = threading.Thread(
        target=_progress_monitor,
        args=(progress_q, N_RESTARTS, N_TRIALS_EACH),
        name="restart-progress",
        daemon=True,
    )
    monitor_started = False

    results: list = [None] * N_RESTARTS

    def _record(restart_idx, seed, result, triggered, n_done):
        results[restart_idx] = (seed, result)
        tqdm.write(
            f"  ✓ Restart {restart_idx+1:>2}/{N_RESTARTS}"
            f"  │  seed={seed:<4}"
            f"  │  completed={n_done}/{len(result.trials)}"
            f"  │  pareto={result.n_pareto:<3}"
            f"  │  HV={result.hypervolume:.5f}"
            f"  │  knee_d={result.best_value:.5f}"
            f"  │  early_stop={triggered}"
        )

    try:
        if workers == 1:
            tqdm.set_lock(threading.RLock())
            monitor.start()
            monitor_started = True
            for restart_idx in range(N_RESTARTS):
                _record(*_run_one_restart(restart_idx))
        else:
            # Force 'fork' so the matpow cache and worker globals are inherited
            # without pickling.  Start the tqdm monitor AFTER submit so workers
            # are forked before this parent thread exists.
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_run_one_restart, idx): idx
                    for idx in range(N_RESTARTS)
                }
                tqdm.set_lock(threading.RLock())
                monitor.start()
                monitor_started = True
                for fut in as_completed(futures):
                    _record(*fut.result())
    finally:
        if monitor_started:
            try:
                progress_q.put(None)
            except Exception:
                pass
            monitor.join(timeout=5)

    return results



def _print_one_design(rank, t, stack, count, freq_pct, tag, total_completed):
    arr      = " → ".join(m for m, _ in stack)
    total_cm = sum(d for _, d in stack)
    efficiency = t.user_attrs.get("target_weighted_flux")
    rmse = float(t.user_attrs["log_rmse"])
    cost = float(t.user_attrs["areal_cost"])
    dist = equal_weight_distance(rmse, cost)
    if tag and "NOT ON PARETO FRONT" in tag:
        tag_s = f"  † {tag}"
    elif tag:
        tag_s = f"  ★ {tag}"
    else:
        tag_s = ""
    print(f"\n  #{rank}  {arr}{tag_s}")
    print(f"       Trial #{t.number}  |  Log-RMSE={rmse:.8f}  "
          f"Cost={cost:.4f} USD/cm²  equal-weight d={dist:.4f}")
    if efficiency is not None:
        print(f"       Target-Weighted Flux={float(efficiency):.4e}")
    print(f"       Sequence sampled : {count} / {total_completed} trials  "
          f"({freq_pct:.2f}%)")
    for i, (mat, d) in enumerate(stack, 1):
        note = ""
        if mat == WATER_CLAD_MATERIAL:
            prev_mat = stack[i - 2][0] if i >= 2 else None
            next_mat = stack[i][0] if i < len(stack) else None
            if prev_mat == "water" or next_mat == "water":
                note = "  (water can, not optimized)"
        print(f"         Layer {i:2d}:  {mat:12s}  {d:6.2f} cm{note}")
    if FIXED_TRAILING_LAYERS:
        for mat, d in FIXED_TRAILING_LAYERS:
            print(f"         Fixed   :  {mat:12s}  {d:6.2f} cm  "
                  f"(detector-embedding, not optimized)")
    print(f"       Total: {total_cm:.2f} cm"
          + (f"  (+ {sum(d for _, d in FIXED_TRAILING_LAYERS):.2f} cm fixed)"
             if FIXED_TRAILING_LAYERS else ""))


def report_optimization_results(
    all_studies,
    save_plots: bool = True,
    top_n: int = TOP_N_CONFIGURATIONS,
):
    all_completed = []
    for _, study in all_studies:
        all_completed.extend(
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
    total_completed = len(all_completed)
    total_trials    = sum(len(s.trials) for _, s in all_studies)
    combined_hv     = _trials_hypervolume(all_completed)
    combined_front  = pareto_front_trials(all_completed)
    combined_knee   = knee_trial(all_completed)

    hvs = [getattr(s, "hypervolume", _trials_hypervolume(s.trials))
           for _, s in all_studies]
    knees = [s.best_value for _, s in all_studies]
    best_hv = max(hvs) if hvs else 0.0
    best_knee = min(knees) if knees else float("inf")

    print(f"\n{'═'*62}")
    print("  OPTIMISATION COMPLETE  -  MULTI-OBJECTIVE TPE SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'Restart':<10} {'Seed':>6}  {'Trials':>8}  {'Completed':>10}  "
          f"{'Pareto':>6}  {'HV':>10}  {'Knee d':>10}  {'Note'}")
    print(f"  {'─'*78}")
    for idx, (seed, study) in enumerate(all_studies):
        n_done = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
        n_par  = getattr(study, "n_pareto", len(pareto_front_trials(study.trials)))
        hv     = getattr(study, "hypervolume", _trials_hypervolume(study.trials))
        notes  = []
        if hv == best_hv:
            notes.append("◄ BEST HV")
        if study.best_value == best_knee:
            notes.append("◄ BEST KNEE")
        print(f"  {idx+1:<10} {seed:>6}  {len(study.trials):>8}  "
              f"{n_done:>10}  {n_par:>6}  {hv:>10.5f}  "
              f"{study.best_value:>10.5f}  {' '.join(notes)}")
    print(f"  {'─'*78}")
    print(f"  {'ALL':>10}         {total_trials:>8}  "
          f"{total_completed:>10}  {len(combined_front):>6}  "
          f"{combined_hv:>10.5f}  "
          f"{(equal_weight_distance(combined_knee.user_attrs['log_rmse'], combined_knee.user_attrs['areal_cost']) if combined_knee is not None else float('nan')):>10.5f}"
          f"  ← combined front")
    print(f"{'═'*62}")

    if combined_knee is not None:
        k_rmse = combined_knee.user_attrs["log_rmse"]
        k_cost = combined_knee.user_attrs["areal_cost"]
        print(f"\n  Equal-weight recommendation (knee of combined Pareto front):")
        print(f"    log-RMSE = {k_rmse:.6f}    cost = {k_cost:.4f} USD/cm²"
              f"    d = {equal_weight_distance(k_rmse, k_cost):.4f}")
        print(f"    (both objectives scaled by ERROR_REF={ERROR_REF:g}, "
              f"COST_REF={COST_REF:g} so they share the same weight)")

    arrangement_freq = count_arrangement_frequencies(all_completed)
    unique = get_unique_top_configs(
        all_completed, arrangement_freq, total_completed,
        top_n=top_n,
    )
    arr_tags = _arrangement_tags(combined_front)
    tags = {}
    n_front = 0
    for t, _, stack, *rest in unique:
        on_front = rest[-1] if rest else True
        if on_front:
            n_front += 1
        label = arr_tags.get(stack_to_key(stack), "")
        if not on_front:
            extra = "NOT ON PARETO FRONT"
            label = f"{label} + {extra}" if label else extra
        tags[id(t)] = label
    n_off = len(unique) - n_front

    print(f"\n{'═'*62}")
    print(f"  DESIGN CHOICES  ({len(unique)} unique material sequences"
          f"  |  {n_front} on a {len(combined_front)}-point Pareto front"
          + (f", {n_off} not on the front" if n_off else "")
          + ")")
    print(f"  lowest error → highest on the front"
          + ("; then off-front, same order" if n_off else "")
          + "; ★ knee / min-error"
          + ("; † not on Pareto front" if n_off else ""))
    print(f"{'═'*62}")

    on_items  = sorted((u for u in unique if u[-1]),
                       key=lambda u: float(u[1]))
    off_items = sorted((u for u in unique if not u[-1]),
                       key=lambda u: float(u[1]))
    rank = 0
    if on_items:
        print(f"\n  ── ON PARETO FRONT  ({len(on_items)}) ──")
        for t, mse, stack, count, freq_pct, *_ in on_items:
            rank += 1
            _print_one_design(
                rank, t, stack, count, freq_pct, tags.get(id(t), ""),
                total_completed,
            )
    if off_items:
        print(f"\n  ── NOT ON PARETO FRONT  ({len(off_items)}) ──")
        for t, mse, stack, count, freq_pct, *_ in off_items:
            rank += 1
            _print_one_design(
                rank, t, stack, count, freq_pct, tags.get(id(t), ""),
                total_completed,
            )

    if save_plots:
        print()
        plot_design_choice_figures(all_completed, unique)

    print(f"\n{'═'*62}")
    print("  DONE")
    print(f"{'═'*62}\n")
    return unique


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    T_lib = load_transfer_matrices()
    check_operator_dim(T_lib)
    init_operator_cache(T_lib)

    src_edges, src_content = get_source_spec()
    grp = rebin_spectrum(src_edges, src_content, ENERGY_EDGES)
    s = grp.sum()
    if s <= 0:
        raise ValueError("Source has no content inside the simulated grid.")
    phi_s_shape = grp / s

    phi_target = np.asarray(TARGET_VALUES, dtype=float)
    assert len(phi_target) == N_E_TARGET, (
        f"phi_target size {len(phi_target)} ≠ N_E_TARGET {N_E_TARGET}"
    )
    if PHI_TARGET_IN_LETHARGY:
        phi_target = lethargy_density_to_flux(phi_target, TARGET_ENERGY_EDGES)

    bin_weights = build_objective_weights(phi_target)
    n_ignored = int(np.sum(~target_nonzero_mask(phi_target)))
    if n_ignored:
        print(f"  [info] {n_ignored}/{len(phi_target)} target bins are exactly "
              f"0 -> excluded from the log-MSE objective.")

    print(f"\n{'═'*62}")
    print("  MULTI-OBJECTIVE SPECTRUM OPTIMISATION  -  MOTPE MULTI-START")
    print(f"{'═'*62}")
    n_thk_choices = int(round((CONSTRAINTS["max_thickness_per_layer"] - MIN_THICKNESS) / THICKNESS_STEP)) + 1
    w_min, w_max = bin_weights.min(), bin_weights.max()
    det_str = f"mode={WEIGHT_MODE}  w∈[{w_min:.2e}, {w_max:.2e}]"
    print(f"  Objectives (↓,↓): detector-weighted log-RMSE  AND  areal material cost")
    print(f"  Sampler          : Metric-TPE (MOTPE)  multivariate+group, "
          f"equal-weight hypervolume")
    print(f"  Equal weight     : f = (logRMSE/{ERROR_REF:g},  cost/{COST_REF:g} USD·cm⁻²)")
    print(f"  Recommendation   : knee of the Pareto front (min ‖f‖₂ in that plane)")
    print(f"  Bin weighting    : {det_str}")
    print(f"  Restarts         : {N_RESTARTS}")
    print(f"  Trials / restart : {N_TRIALS_EACH}  (startup={N_STARTUP_EACH})")
    print(f"  Stagnation stop  : {PATIENCE} trials without hypervolume improvement")
    print(f"  Max layers       : {CONSTRAINTS['max_layers']}")
    print(f"  Max total cm     : {CONSTRAINTS['max_total_thickness']}")
    print(f"  Thickness grid   : [{MIN_THICKNESS},..., {CONSTRAINTS['max_thickness_per_layer']}] cm  step={THICKNESS_STEP} cm  ({n_thk_choices} choices)")
    print(f"  Material distance: {DISTANCE_NORM} (summed over base thicknesses, full S_N operators)")
    if FIXED_TRAILING_LAYERS:
        fixed_str = " → ".join(f"{mat} {d:.2f}cm" for mat, d in FIXED_TRAILING_LAYERS)
        print(f"  Fixed layer(s)   : {fixed_str}  (detector-embedding, appended after every trial's stack, not optimized)")
    print(f"  Operator cache   : on-demand interpolation, LRU={OPERATOR_LRU_SIZE}/process  ({N_STATE}*{N_STATE} operators)")
    print(f"{'═'*62}")

    all_studies = run_multi_start(T_lib, phi_s_shape, phi_target,
                                  bin_weights=bin_weights)

    print(f"\n{'─'*62}")
    save_all_studies(all_studies)

    report_optimization_results(all_studies, save_plots=True)
