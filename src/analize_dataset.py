import ast
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.preprocessing import PowerTransformer

from base_plates import TARGET_VALUES, ENERGY_EDGES, PHI_TARGET_IN_LETHARGY
from lethargy_converter import convert_if_lethargy

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

CSV_PATH = "../examples/demo_hcpb_vv.csv"    # relative to this script
UNC_LIMITS = [0.30, 1.00]        # extra sigma/y cuts; calibration UNC_LIMIT is always drawn
PLOT_AS_FRACTION = True          # False -> survival counts instead of fractions
OUT_PATH = None                     # survival PNG; None -> same folder as the dataset
N_DETAIL_BINS = 8
DETAIL_BINS = None               # or a list of bin indices, e.g. [10, 40, 102]

# ==================================================================

TRANSFORM_NAMES = (
    "identity",
    "sqrt",
    "cbrt",
    "log",
    "log_eps",
    "asinh_scaled",
    "boxcox",
    "yeo-johnson",
)
TRANSFORM_COLORS = {
    "identity": "#7f7f7f",
    "sqrt": "#8c564b",
    "cbrt": "#e377c2",
    "log": "#1f77b4",
    "log_eps": "#17becf",
    "asinh_scaled": "#2ca02c",
    "boxcox": "#d62728",
    "yeo-johnson": "#ff7f0e",
}
DETAIL_TRANSFORMS = ("identity", "log", "log_eps", "asinh_scaled", "boxcox", "yeo-johnson")


# -- Dataset -------------------------------------------------------------

def calibration_constant(name, cal_path=None):
    """A numeric assignment from calibration.py. That module runs the fit on import."""
    path = cal_path or Path(__file__).resolve().parent / "calibration.py"
    if not path.exists():
        sys.exit(f"calibration.py not found at {path}")

    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    return float(ast.literal_eval(node.value))
                except (ValueError, TypeError) as exc:
                    sys.exit(f"Could not parse {name} in {path}: {exc}")

    sys.exit(f"{name} assignment not found in {path}")


def flux_columns(df):
    """Matching y_/sigma_ pairs, sorted by bin index."""
    y_pattern = re.compile(r"^y_(\d+)$")
    sigma_pattern = re.compile(r"^sigma_(\d+)$")
    y_cols, sigma_cols = {}, {}
    for col in df.columns:
        match = y_pattern.match(col)
        if match:
            y_cols[int(match.group(1))] = col
        match = sigma_pattern.match(col)
        if match:
            sigma_cols[int(match.group(1))] = col

    bin_indices = sorted(set(y_cols) & set(sigma_cols))
    if not bin_indices:
        sys.exit("No matching y_XX / sigma_XX column pairs found in CSV.")
    return (
        bin_indices,
        [y_cols[b] for b in bin_indices],
        [sigma_cols[b] for b in bin_indices],
    )


def active_bin_mask(n_csv_bins):
    """True where the target spectrum is non-zero."""
    target = np.asarray(
        convert_if_lethargy(TARGET_VALUES, ENERGY_EDGES, PHI_TARGET_IN_LETHARGY),
        dtype=float,
    )
    if len(target) != n_csv_bins:
        sys.exit(
            f"Bin count mismatch: CSV has {n_csv_bins} y_/sigma_ bins, "
            f"but TARGET_VALUES has {len(target)} entries."
        )
    return target != 0.0


def resolve_path(path_like, script_dir, default_name=None):
    if path_like is None:
        if default_name is None:
            raise ValueError("path_like is None and no default_name given")
        return script_dir / default_name
    path = Path(path_like)
    if not path.is_absolute():
        path = script_dir / path
    return path


# -- Survival ------------------------------------------------------------

def survival_by_bin(df, y_cols, sigma_cols, thresholds):
    """Per-bin counts with sigma/y <= each threshold. Non-positive y does not survive."""
    y_matrix = df[y_cols].to_numpy(dtype=float)
    sigma_matrix = df[sigma_cols].to_numpy(dtype=float)
    rel_unc = np.full_like(y_matrix, np.nan, dtype=float)
    np.divide(sigma_matrix, y_matrix, out=rel_unc, where=y_matrix > 0)

    counts = {t: (rel_unc <= t).sum(axis=0) for t in thresholds}
    return counts, sigma_matrix.shape[0]


def plot_survival(bin_indices, counts, n_samples, thresholds, cal_limit, as_fraction, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in thresholds:
        y = counts[t] / n_samples if as_fraction else counts[t]
        from_cal = abs(t - cal_limit) < 1e-9
        ax.plot(
            bin_indices, y,
            marker="o", markersize=4 if from_cal else 3,
            linewidth=2.4 if from_cal else 1.2,
            color="k" if from_cal else None,
            zorder=3 if from_cal else 2,
            label=f"calibration.py ({t:.0%})" if from_cal else f"{t:.0%}",
        )

    ax.set_xlabel("Bin index", fontsize=15)
    ax.set_ylabel("Surviving sample fraction" if as_fraction else "Surviving sample count", fontsize=15)
    ax.set_title("Bin-by-bin sample survival vs. relative uncertainty cut", fontsize=17)
    ax.tick_params(axis="both", which="both", labelsize=15)
    ax.set_ylim(0, 1.05 if as_fraction else n_samples * 1.05)
    ax.axhline(
        1.0 if as_fraction else n_samples,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"total samples (n={n_samples})",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# -- Transforms ----------------------------------------------------------

def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if not np.isfinite(sd) or sd < 1e-15:
        return np.zeros_like(x)
    return (x - mu) / sd


def apply_transform(name, y, log_floor):
    """f(Y) for one bin. (values, info) or (None, info) when it cannot be applied."""
    y = np.asarray(y, dtype=float)
    info = {"lambda": np.nan, "n_used": 0, "n_dropped": 0, "ok": False}

    if name == "identity":
        x = y.copy()
    elif name == "sqrt":
        if np.any(y < 0):
            return None, info
        x = np.sqrt(np.maximum(y, 0.0))
    elif name == "cbrt":
        x = np.cbrt(y)
    elif name == "log":
        x = np.log(np.maximum(y, log_floor))
    elif name == "log_eps":
        pos = y[y > 0]
        if pos.size == 0:
            return None, info
        eps = max(float(pos.min()) / 10.0, 1e-12 * float(np.median(pos)))
        info["lambda"] = eps
        x = np.log(np.maximum(y, 0.0) + eps)
    elif name == "asinh_scaled":
        pos = y[y > 0]
        scale = float(np.median(pos)) if pos.size else 1.0
        if scale <= 0:
            scale = 1.0
        info["lambda"] = scale
        x = np.arcsinh(y / scale)
    elif name == "boxcox":
        pos = y[y > 0]
        n_nonpos = int((y <= 0).sum())
        info["n_dropped"] = 0
        if pos.size < 8:
            return None, info
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _zt, lam = stats.boxcox(pos)
            lam = float(lam)
            info["lambda"] = lam
            y_floor = max(float(pos.min()) / 10.0, float(log_floor))
            y_clip = np.maximum(y, y_floor)
            if abs(lam) < 1e-8:
                x = np.log(y_clip)
            else:
                x = (np.power(y_clip, lam) - 1.0) / lam
            if n_nonpos:
                info["n_dropped"] = n_nonpos
        except Exception:
            return None, info
    elif name == "yeo-johnson":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power = PowerTransformer(method="yeo-johnson", standardize=False)
                x = power.fit_transform(y.reshape(-1, 1)).ravel()
            info["lambda"] = float(power.lambdas_[0])
        except Exception:
            return None, info
    else:
        raise ValueError(f"Unknown transform: {name}")

    finite = np.isfinite(x)
    if finite.sum() < 8:
        return None, info
    info["n_used"] = int(finite.sum())
    if name != "boxcox":
        info["n_dropped"] = int((~finite).sum())
    info["ok"] = True
    return x[finite], info


def normality_pvalue(z):
    """D'Agostino K2 when n >= 20, otherwise Shapiro-Wilk."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 8:
        return np.nan
    try:
        if z.size >= 20:
            return float(stats.normaltest(z).pvalue)
        return float(stats.shapiro(z).pvalue)
    except Exception:
        return np.nan


def qq_correlation(z):
    """Correlation of sample quantiles with a normal reference (Filliben positions)."""
    z = np.asarray(z, dtype=float)
    z = np.sort(z[np.isfinite(z)])
    n = z.size
    if n < 8:
        return np.nan
    if n == 1:
        return np.nan
    p = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    theo = stats.norm.ppf(p)
    if np.std(theo) < 1e-15 or np.std(z) < 1e-15:
        return np.nan
    return float(np.corrcoef(theo, z)[0, 1])


def _blank_metrics(bin_index, name, n_dropped, y_stats):
    return {
        "bin": bin_index,
        "transform": name,
        "n_used": 0,
        "n_dropped": int(n_dropped),
        "lambda": np.nan,
        "skew": np.nan,
        "kurtosis": np.nan,
        "abs_skew": np.nan,
        "abs_kurt": np.nan,
        "normal_pvalue": np.nan,
        "qq_r": np.nan,
        **y_stats,
        "ok": False,
    }


def analyze_transforms(y_matrix, bin_indices, log_floor):
    """One row per bin and transform: skew, kurtosis, normality p, QQ correlation on Z."""
    rows = []
    n_samples = y_matrix.shape[0]

    for j, bin_index in enumerate(bin_indices):
        y = y_matrix[:, j]
        pos = y[y > 0]
        y_stats = {
            "frac_zero": float((y <= 0).mean()),
            "cv": float(pos.std() / pos.mean()) if pos.size and pos.mean() > 0 else np.nan,
            "dynamic_range_decades": (
                float(np.log10(pos.max() / pos.min())) if pos.size and pos.min() > 0 else np.nan
            ),
            "log_floor_outlier_frac": float((y <= log_floor).mean()) if log_floor > 0 else 0.0,
        }

        for name in TRANSFORM_NAMES:
            x, info = apply_transform(name, y, log_floor)
            if x is None:
                rows.append(_blank_metrics(bin_index, name, info.get("n_dropped", n_samples), y_stats))
                continue

            z = zscore(x)
            skew = float(stats.skew(z, bias=False))
            kurt = float(stats.kurtosis(z, fisher=True, bias=False))
            rows.append({
                "bin": bin_index,
                "transform": name,
                "n_used": int(info["n_used"]),
                "n_dropped": int(info["n_dropped"]),
                "lambda": info["lambda"],
                "skew": skew,
                "kurtosis": kurt,
                "abs_skew": abs(skew),
                "abs_kurt": abs(kurt),
                "normal_pvalue": normality_pvalue(z),
                "qq_r": qq_correlation(z),
                **y_stats,
                "ok": True,
            })

    return pd.DataFrame(rows)


def summarize_transforms(metrics):
    """Median normality metrics across bins, ranked by score."""
    ok = metrics[metrics["ok"]].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby("transform", sort=False)
        .agg(
            n_bins=("bin", "count"),
            median_abs_skew=("abs_skew", "median"),
            mean_abs_skew=("abs_skew", "mean"),
            median_abs_kurt=("abs_kurt", "median"),
            mean_abs_kurt=("abs_kurt", "mean"),
            median_qq_r=("qq_r", "median"),
            frac_normal_p05=("normal_pvalue", lambda s: float((s > 0.05).mean())),
            frac_abs_skew_lt_0_5=("abs_skew", lambda s: float((s < 0.5).mean())),
            median_lambda=("lambda", "median"),
        )
        .reset_index()
    )
    summary["score"] = (
        summary["median_qq_r"].fillna(0)
        + summary["frac_normal_p05"].fillna(0)
        + summary["frac_abs_skew_lt_0_5"].fillna(0)
        - 0.15 * summary["median_abs_skew"].fillna(10)
        - 0.05 * summary["median_abs_kurt"].fillna(10)
    )
    return summary.sort_values("score", ascending=False).reset_index(drop=True)


def pick_detail_bins(metrics, n, forced):
    """A spread of bins: mid-spectrum, worst and best log, where Box-Cox helps, many zeros."""
    if forced:
        return sorted(set(int(b) for b in forced))

    log_m = metrics[metrics["transform"] == "log"].set_index("bin")
    box_m = metrics[metrics["transform"] == "boxcox"].set_index("bin")
    bins = list(log_m.index)
    if not bins:
        return []

    chosen = []

    def add(bin_index):
        if bin_index in log_m.index and bin_index not in chosen:
            chosen.append(int(bin_index))

    add(bins[len(bins) // 8])
    add(bins[len(bins) // 2])
    add(bins[3 * len(bins) // 4])
    add(bins[-1])

    order_skew = log_m["abs_skew"].sort_values(ascending=False)
    for bin_index in order_skew.index[:2]:
        add(bin_index)
    for bin_index in order_skew.index[::-1][:1]:
        add(bin_index)

    if not box_m.empty:
        common = log_m.index.intersection(box_m.index)
        gap = (log_m.loc[common, "abs_skew"] - box_m.loc[common, "abs_skew"]).sort_values(ascending=False)
        for bin_index in gap.index[:2]:
            add(bin_index)

    for bin_index in log_m["frac_zero"].sort_values(ascending=False).index[:2]:
        add(bin_index)
    for bin_index in log_m["dynamic_range_decades"].sort_values(ascending=False).index[:1]:
        add(bin_index)

    return chosen[:n]


# -- Plots ---------------------------------------------------------------

def plot_metric_vs_bin(metrics, out_path):
    transforms = [name for name in TRANSFORM_NAMES if (metrics["transform"] == name).any()]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = [
        (axes[0, 0], "abs_skew", r"|skewness| (lower better)", False),
        (axes[0, 1], "abs_kurt", r"|excess kurtosis| (lower better)", False),
        (axes[1, 0], "qq_r", r"QQ correlation with Normal (higher better)", False),
        (axes[1, 1], "normal_pvalue", r"normality p-value (higher better)", True),
    ]

    for ax, col, title, log_y in panels:
        for name in transforms:
            sub = metrics[metrics["transform"] == name].sort_values("bin")
            if sub.empty:
                continue
            ax.plot(sub["bin"], sub[col], label=name, color=TRANSFORM_COLORS.get(name), linewidth=1.2, alpha=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Bin index")
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
            ax.axhline(0.05, color="k", linestyle="--", linewidth=0.8, label="p=0.05")
        if col == "abs_skew":
            ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, label="|skew|=0.5")

    axes[0, 0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("Per-bin Gaussianity after candidate transforms (then z-scored)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_summary_bars(summary, out_path):
    if summary.empty:
        return

    order = list(summary["transform"])
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    def bars(ax, col, ylabel):
        vals = [float(summary.loc[summary["transform"] == name, col].iloc[0]) for name in order]
        colors = [TRANSFORM_COLORS.get(name, "C0") for name in order]
        ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(True, axis="y", alpha=0.3)

    bars(axes[0], "median_abs_skew", "median |skew|  (↓ better)")
    bars(axes[1], "median_qq_r", "median QQ-R  (↑ better)")
    bars(axes[2], "frac_normal_p05", "fraction bins with p>0.05  (↑ better)")

    fig.suptitle("Transform ranking across all active bins", fontsize=17)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_lambda_vs_bin(metrics, out_path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, label in (("boxcox", "Box-Cox λ"), ("yeo-johnson", "Yeo-Johnson λ")):
        sub = metrics[metrics["transform"] == name].sort_values("bin")
        ax.plot(
            sub["bin"], sub["lambda"],
            marker="o", markersize=2.5, linewidth=1.1,
            label=label, color=TRANSFORM_COLORS[name],
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.9, label="λ=0 (≡ log for Box-Cox)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="λ=0.5 (≈ sqrt)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="λ=1 (identity)")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Fitted λ")
    ax.set_title("Optimal power-transform λ per bin")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _annotation(name, info):
    lam = info.get("lambda", np.nan)
    if name == "boxcox" and np.isfinite(lam):
        return f" λ={lam:.2f}"
    if name == "log_eps" and np.isfinite(lam):
        return f" ε={lam:.1e}"
    if name == "yeo-johnson" and np.isfinite(lam):
        return f" λ={lam:.2f}"
    return ""


def plot_bin_details(y_matrix, bin_indices, detail_bins, log_floor, out_path):
    """Histograms in Z for the detail bins."""
    bin_to_col = {b: j for j, b in enumerate(bin_indices)}
    detail_bins = [b for b in detail_bins if b in bin_to_col]
    if not detail_bins:
        print("[Transform] No detail bins available — skipping detail plot.")
        return

    n_rows, n_cols = len(detail_bins), len(DETAIL_TRANSFORMS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.4 * n_rows), squeeze=False)

    for r, bin_index in enumerate(detail_bins):
        y = y_matrix[:, bin_to_col[bin_index]]
        for c, name in enumerate(DETAIL_TRANSFORMS):
            ax = axes[r, c]
            x, info = apply_transform(name, y, log_floor)
            if x is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, fontsize=15)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                z = zscore(x)
                ax.hist(
                    z,
                    bins=min(25, max(8, z.size // 8)),
                    color=TRANSFORM_COLORS.get(name, "C0"),
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.text(
                    0.02, 0.98,
                    f"|sk|={abs(stats.skew(z, bias=False)):.2f}\nQQ={qq_correlation(z):.3f}{_annotation(name, info)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
                ax.tick_params(axis="x", labelsize=14)
                ax.tick_params(axis="y", labelsize=13)

            if r == 0:
                ax.set_title(name, fontsize=15)
            if c == 0:
                ax.set_ylabel(f"y_{bin_index:02d}", fontsize=13)
            if r == n_rows - 1:
                ax.set_xlabel("Z [-]", fontsize=13)

    fig.suptitle(r"Selected bins: histograms in Z-space  ($Z=(f(Y)-\mu)/\sigma$)", fontsize=17, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_qq_grid(y_matrix, bin_indices, detail_bins, log_floor, out_path):
    bin_to_col = {b: j for j, b in enumerate(bin_indices)}
    detail_bins = [b for b in detail_bins if b in bin_to_col]
    if not detail_bins:
        return

    n_rows, n_cols = len(detail_bins), len(DETAIL_TRANSFORMS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.4 * n_rows), squeeze=False)

    for r, bin_index in enumerate(detail_bins):
        y = y_matrix[:, bin_to_col[bin_index]]
        for c, name in enumerate(DETAIL_TRANSFORMS):
            ax = axes[r, c]
            x, _info = apply_transform(name, y, log_floor)
            if x is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                stats.probplot(zscore(x), dist="norm", plot=ax)
                ax.get_lines()[0].set_markersize(3)
                ax.get_lines()[0].set_markerfacecolor(TRANSFORM_COLORS.get(name, "C0"))
                ax.get_lines()[0].set_markeredgewidth(0)
                ax.get_lines()[1].set_color("k")
                ax.get_lines()[1].set_linewidth(1.0)
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"y_{bin_index:02d}", fontsize=8)

    fig.suptitle("Selected bins: Normal QQ plots after transform", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_best_transform_map(metrics, out_path):
    ok = metrics[metrics["ok"]].copy()
    if ok.empty:
        return

    ok["rank_score"] = ok["qq_r"].fillna(0) - 0.5 * ok["abs_skew"].fillna(10)
    winners = ok.loc[
        ok.groupby("bin")["rank_score"].idxmax(),
        ["bin", "transform", "abs_skew", "qq_r", "lambda"],
    ].sort_values("bin")

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 2]})
    ax = axes[0]
    ax.bar(
        winners["bin"], np.ones(len(winners)),
        color=[TRANSFORM_COLORS.get(name, "C0") for name in winners["transform"]],
        width=1.0, edgecolor="none",
    )
    ax.set_yticks([])
    ax.set_ylabel("best\ntransform")
    ax.set_title("Per-bin winning transform (score = QQ-R − 0.5·|skew|)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=TRANSFORM_COLORS[name], markersize=10, label=name)
            for name in TRANSFORM_NAMES
            if name in set(winners["transform"])
        ],
        loc="upper left", ncol=4, fontsize=8, framealpha=0.9,
    )

    ax2 = axes[1]
    for name in ("log", "log_eps", "boxcox", "yeo-johnson", "asinh_scaled"):
        sub = metrics[metrics["transform"] == name].sort_values("bin")
        ax2.plot(sub["bin"], sub["abs_skew"], label=name, color=TRANSFORM_COLORS[name], lw=1.2)
    ax2.set_xlabel("Bin index")
    ax2.set_ylabel("|skewness|")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axhline(0.5, color="k", ls=":", lw=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return winners


# -- Run -----------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = resolve_path(CSV_PATH, script_dir)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}\nEdit CSV_PATH at the top of this script.")

    cal_path = script_dir / "calibration.py"
    cal_limit = calibration_constant("UNC_LIMIT", cal_path)
    log_floor = calibration_constant("LOG_FLOOR", cal_path)
    thresholds = [cal_limit] + [t for t in UNC_LIMITS if abs(t - cal_limit) >= 1e-9]
    print(f"Using UNC_LIMIT={cal_limit:g} from calibration.py")
    print(f"Using extra UNC_LIMITS={list(UNC_LIMITS)}")
    print(f"Using LOG_FLOOR={log_floor:g} from calibration.py")

    df = pd.read_csv(csv_path)
    bin_indices, y_cols, sigma_cols = flux_columns(df)

    active = active_bin_mask(len(bin_indices))
    n_skipped = int((~active).sum())
    if n_skipped:
        skipped = [bin_indices[i] for i in range(len(bin_indices)) if not active[i]]
        print(f"[Active bins] Skipping {n_skipped} zero-TARGET_VALUES bin(s): {skipped}")
    else:
        print("[Active bins] No zero-valued TARGET_VALUES bins — keeping all.")

    bin_indices = [b for b, keep in zip(bin_indices, active) if keep]
    y_cols = [c for c, keep in zip(y_cols, active) if keep]
    sigma_cols = [c for c, keep in zip(sigma_cols, active) if keep]
    if not bin_indices:
        sys.exit("No active bins left after TARGET_VALUES filter.")

    counts, n_samples = survival_by_bin(df, y_cols, sigma_cols, thresholds)
    print(f"\nSamples: {n_samples}  |  active bins: {len(bin_indices)}")
    for t in thresholds:
        frac = counts[t] / n_samples
        source = "  (calibration.py)" if abs(t - cal_limit) < 1e-9 else ""
        print(
            f"  UNC_LIMIT={t:.0%}{source}: survival fraction "
            f"min={frac.min():.3f}  median={np.median(frac):.3f}  max={frac.max():.3f}"
        )

    out_path = resolve_path(OUT_PATH, script_dir, default_name="survival_by_bin.png")
    if OUT_PATH is None:
        out_path = csv_path.parent / "survival_by_bin.png"
    plot_survival(bin_indices, counts, n_samples, thresholds, cal_limit, PLOT_AS_FRACTION, out_path)

    print("\n" + "=" * 72)
    print("TARGET TRANSFORM NORMALITY COMPARISON")
    print("=" * 72)
    print(
        "Goal: find a per-bin map Y → Z that is closer to Gaussian, so the\n"
        "GPR prior / MLE noise model in calibration.py is better matched."
    )

    y_matrix = df[y_cols].to_numpy(dtype=float)
    metrics = analyze_transforms(y_matrix, bin_indices, log_floor)
    summary = summarize_transforms(metrics)

    if not summary.empty:
        print(f"\n→ Best overall on this dataset: {summary.iloc[0]['transform']}")

    plot_metric_vs_bin(metrics, csv_path.parent / "transform_metric_vs_bin.png")
    plot_summary_bars(summary, csv_path.parent / "transform_ranking.png")
    plot_lambda_vs_bin(metrics, csv_path.parent / "transform_lambda_vs_bin.png")
    winners = plot_best_transform_map(metrics, csv_path.parent / "transform_best_per_bin.png")
    if winners is not None:
        print("\nWins per transform:")
        for name, n in winners["transform"].value_counts().items():
            print(f"  {name:12s}  {n:4d} / {len(winners)}")

    detail = pick_detail_bins(metrics, N_DETAIL_BINS, DETAIL_BINS)
    print(f"\nDetail bins for hist/QQ panels: {detail}")
    plot_bin_details(y_matrix, bin_indices, detail, log_floor, csv_path.parent / "transform_hist_detail.png")
    plot_qq_grid(y_matrix, bin_indices, detail, log_floor, csv_path.parent / "transform_qq_detail.png")

    print("\nDone. Key figures:")
    print("  transform_ranking.png          — overall transform comparison")
    print("  transform_metric_vs_bin.png    — |skew|/kurt/QQ/p vs bin")
    print("  transform_lambda_vs_bin.png    — fitted Box-Cox / YJ λ")
    print("  transform_best_per_bin.png     — which transform wins per bin")
    print("  transform_hist_detail.png      — histograms in Z-space")
    print("  transform_qq_detail.png        — QQ plots for selected bins")


if __name__ == "__main__":
    main()
