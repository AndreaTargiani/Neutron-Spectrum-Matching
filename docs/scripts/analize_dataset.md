# `analize_dataset.py` — Checks on the Sobol CSV

## Purpose

This script is **optional**. Run it on the CSV from [`dataset_creation.py`](dataset_creation.md) before [`calibration.py`](calibration.md). It does not refit the GPR and it does not change the CSV.

Two checks:

1. **Survival.** For each energy bin, the fraction of samples whose Monte Carlo relative uncertainty \(\sigma / y\) stays under a cut. The plot always includes `UNC_LIMIT` from `calibration.py`, plus any extra cuts in `UNC_LIMITS`.
2. **Target transform.** Which map \(f(Y)\) makes each bin closer to Gaussian. The GPR is fit in that transformed space. The ranking is what you look at when you set `Y_TRANSFORM`.

Bins whose target spectrum entry is zero are dropped, the same active-bin rule as `calibration.py`.

---

## Configuration

```python
CSV_PATH = "../sobol_dataset.csv"   # relative paths resolve next to this script
UNC_LIMITS = [0.30, 1.00]           # extra sigma/y cuts; calibration UNC_LIMIT is always drawn
PLOT_AS_FRACTION = True             # survival plot: fraction (True) or count (False)
OUT_PATH = None                     # survival PNG; None -> same folder as the dataset
N_DETAIL_BINS = 8                   # bins in the histogram and QQ grids
DETAIL_BINS = None                  # or a list of bin indices, e.g. [10, 40, 102]
```

| Variables | Role |
|---|---|
| `CSV_PATH` | Sobol CSV. A relative path is resolved next to `analize_dataset.py`, so from a launch in the repository root use `../your_file.csv` |
| `UNC_LIMITS` | Extra relative-uncertainty cuts, one curve each. `UNC_LIMIT` from `calibration.py` is added automatically and drawn in black |
| `PLOT_AS_FRACTION` | Survival figure in fraction of samples, or in raw counts |
| `OUT_PATH` | Where to write the survival PNG. `None` puts `survival_by_bin.png` beside the dataset |
| `N_DETAIL_BINS` | How many bins get a histogram and a QQ panel |
| `DETAIL_BINS` | Force those bins. `None` picks a spread: mid-spectrum, worst and best `log`, where Box–Cox helps most, and bins with many zeros |

---

## Survival

Relative uncertainty is \(\sigma / y\), with \(\sigma\) the absolute \(1\sigma\) stored in `sigma_XX`. A sample survives bin \(b\) at cut \(t\) when that ratio is at most \(t\). Non-positive \(y\) and NaNs do not survive.

The figure `survival_by_bin.png` has one curve per cut, against bin index. The `calibration.py` cut is the thick black line, labelled `calibration.py`. The terminal marks that same cut and prints the min, median, and max survival fraction at each cut.

A bin that survives poorly is a bin whose GPR will be trained on few points. I personally suggest raising the CE particle count in `dataset_creation.py` and regenerating, rather than calibrating on a bin that keeps only a small slice of the design.

---

## Transforms

Each candidate is applied per bin, then z-scored, \(Z = (f(Y) - \mu) / \sigma\). Skewness, excess kurtosis, a normality p-value, and the QQ correlation with a normal are all computed on \(Z\).

| Name | \(f(Y)\) |
|---|---|
| `identity` | \(Y\) |
| `sqrt`, `cbrt` | Mild compression |
| `log` | \(\log(\max(Y, \texttt{LOG_FLOOR}))\) |
| `log_eps` | \(\log(Y + \varepsilon)\) with \(\varepsilon\) set from the positive samples |
| `asinh_scaled` | \(\operatorname{arsinh}(Y / \operatorname{median}_{Y>0})\) |
| `boxcox` | Per-bin \(\lambda\), with non-positive samples floored |
| `yeo-johnson` | Per-bin \(\lambda\), defined at zero as well |

`calibration.py` accepts `boxcox`, `yeojohnson`, or `log_eps`. The other rows are there so you can see whether the winner is one of those three or only close to it. Set `Y_TRANSFORM` to the supported name the ranking prefers.

The ranking (higher QQ correlation and normality rate, lower \(|\mathrm{skew}|\) and \(|\mathrm{kurtosis}|\)) is the bar chart `transform_ranking.png`. The terminal prints only the winning transform.

`LOG_FLOOR`, the hard floor in the legacy `log` transform, is read from [`calibration.py`](calibration.md).

---

## Plots

All of these are written next to the CSV.

| File | What it shows |
|---|---|
| `survival_by_bin.png` | Surviving fraction (or count) versus bin, one line per uncertainty cut |
| `transform_ranking.png` | Three bars per transform, across all active bins: median \(\lvert\mathrm{skew}\rvert\), median QQ correlation, fraction of bins with normality \(p > 0.05\) |
| `transform_metric_vs_bin.png` | Those same metrics, plus excess kurtosis, as a curve versus bin index |
| `transform_lambda_vs_bin.png` | Fitted Box–Cox and Yeo–Johnson \(\lambda\). For Box–Cox, \(\lambda = 0\) is a logarithm and \(\lambda = 0.5\) is a square root |
| `transform_best_per_bin.png` | A colour strip of the winning transform in each bin (score \(=\) QQ correlation \(-\, 0.5\,\lvert\mathrm{skew}\rvert\)), and \(\lvert\mathrm{skew}\rvert\) for `log`, `log_eps`, `boxcox`, `yeo-johnson`, and `asinh_scaled` |
| `transform_hist_detail.png` | Histograms in \(Z\) for the detail bins, one column per transform |
| `transform_qq_detail.png` | Normal QQ plots for the same bins and transforms. A straight line is the Gaussian reference |

---

## Running

```bash
python src/analize_dataset.py
```

Set `CSV_PATH` first. Figures land next to that CSV.
