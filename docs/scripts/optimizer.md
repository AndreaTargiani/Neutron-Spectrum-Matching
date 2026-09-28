# `optimizer.py` — Material Arrangement Optimizer

## Purpose

`optimizer.py` loads the precomputed \(S_N\) transfer-tensor database, evaluates candidate plate stacks by composing those operators, and runs a **multi-start TPE search in multi-objective mode**. The two objectives are spectral error (detector-weighted log-RMSE) and areal material cost. The result is a Pareto front of non-dominated designs, not a single scalar winner.

See [Material Arrangement Optimization Theory](../theory/optimization.md) for the mathematical background. This page is the script: the knobs at the top of `optimizer.py`, how a run is launched, and what it prints and writes.

!!! info "No Monte Carlo at evaluation time"
    Once the tensor database is built, every candidate is a product of preloaded operators. No OpenMC runs happen during optimization.

---

## Contents

| Section | Variables |
|---|---|
| [Materials](#materials) | `MATERIALS` |
| [Geometric constraints](#geometric-constraints) | `CONSTRAINTS`, `MIN_THICKNESS`, `THICKNESS_STEP` |
| [Water cans](#water-cans) | `WATER_CLAD_MATERIAL`, `WATER_CLAD_THICKNESS` |
| [Passive shells](#passive-shells-in-the-optimizer) | `FIXED_TRAILING_LAYERS` (derived) |
| [Cost](#cost) | `COST_PER_KG` (in `config_materials.py`) |
| [Equal-weight MOTPE](#equal-weight-motpe) | `ERROR_REF`, `COST_REF`, `HV_REF` |
| [Material distance](#material-distance-metric) | `DISTANCE_NORM` |
| [Hyperparameters](#optimization-hyperparameters) | `N_RESTARTS`, `RESTART_WORKERS`, `N_TRIALS_EACH`, … |

---

## Materials

```python
MATERIALS = [
    "PE_BO", "graphite", "iron",  # ... any keys from config_materials.py
]
```

Materials the optimizer is allowed to pick. Each entry must already have `sn_tensor_{name}_{t}cm.npy` files in `opt_database/` for every thickness in `transfer_matrix.py`'s `THICKNESSES`. A subset of the database is fine if you want to shrink the search.

Compositions themselves live in [`config_materials.py`](config_materials.md). Adding a material there does nothing here until it has tensors **and** is listed in this `MATERIALS`.

The search also loads tensors for [`WATER_CLAD_MATERIAL`](#water-cans) even if that name is not in `MATERIALS`, so the water-can walls can be applied.

!!! warning "Every material in the beam needs a tensor"
    Include all of them in `MATERIALS`: the plates you want to search, the water-can wall materials **and** the detector materials (passive shells). Each name needs `.npy` files in `opt_database/`.

---

## Geometric constraints

```python
CONSTRAINTS = {
    "max_total_thickness":     ...,   # cm  - total budget (soft: see below)
    "max_layers":               ...,  # integer cap
    "max_thickness_per_layer":  ...,  # cm  - hard upper bound per layer
}
MIN_THICKNESS  = 0.2   # cm
THICKNESS_STEP = 0.1   # cm
```

`max_layers` and `max_thickness_per_layer` are **hard bounds** of the search space: Optuna never proposes more layers or a thicker single plate.

`max_total_thickness` is a **soft bound**. Layer thicknesses are sampled independently, so their sum can overshoot. That overshoot is reported to constrained TPE instead of being folded into the two objectives (which would warp the Pareto front). Infeasible trials are down-weighted, not deleted. See [Constraints](../theory/optimization.md#constraints).

The total that is checked is the search stack plus water-can walls, detector shells don't count toward the constraints.

`THICKNESS_STEP` is the resolution of the thickness grid. Every proposed thickness is snapped to this step before the operator is built, so two trials that differ by a fraction of a millimetre are the same design.

!!! tip "Cover the search with the database"
    I personally suggest that `THICKNESSES` in [`transfer_matrix.py`](transfer_matrix.md) includes `MIN_THICKNESS` and `max_thickness_per_layer`. Inside that range the optimizer interpolates; outside it extrapolates with a fractional matrix power, which can break positivity or the sub-stochastic column-sum bound. See [Evaluating a stack](../theory/optimization.md#evaluating-a-stack).

---

## Water cans

Water in this model is stored in a can: a fixed wall of `WATER_CLAD_MATERIAL` (default `"Al_6061"`) of thickness `WATER_CLAD_THICKNESS` (default `1.0` cm) is inserted on each face of every `water` layer.

That happens after Optuna has chosen the search stack, and it affects physics, cost, the total-thickness check. The walls are labelled `(water can, not optimized)` in the report. Consecutive same-material layers are still merged for uniqueness, so `Al_6061` can walls next to an `Al_6061` plate become one thicker aluminium layer in the displayed sequence.

If `water` is not in `MATERIALS`, none of this applies.

---

## Passive shells in the optimizer

The plate stack sits directly in front of the detector. If the active volume is embedded in a passive shell, that shell attenuates the beam like an extra plate, one the optimizer did not choose and cannot change.

For every passive region **in front of the first** `active=True` region, the optimizer appends a fixed trailing layer whose thickness is the difference of beam-axis (`"x"`) half-extents. Those layers are used for the physics evaluation only: they are not part of the search space, they do not count against `max_layers` / `max_total_thickness`, and they are omitted from the cost. If the outermost region is itself active, the list is empty and the optimizer sees plates only.

The startup banner prints the fixed stack when it is non-empty. Those shell names also need tensors, see the warning under [Materials](#materials).

---

## Bin weighting

The detector body, the \((E, R)\) table, and the log-log polynomial fit all live in [`config_detector.py`](config_detector.md). Which weighting those numbers feed is `WEIGHT_MODE` in [`config.py`](config.md#per-bin-weighting). [`bin_weights.py`](bin_weights.md) is the optional plot for choosing `ICRP` versus `dose`.

---

## Cost

Prices live in [`COST_PER_KG`](config_materials.md#cost). Densities come from the material library. The optimizer turns them into an areal cost (USD per cm$^2$ of plate face), independent of plate size \(L\):

\[
\text{areal cost} = \sum_{\text{layers}} \rho\,[\mathrm{g/cm}^3]\cdot d\,[\mathrm{cm}]\cdot \text{price}\,[\mathrm{USD/kg}]\,/\,1000
\]

Water-can walls are counted. Detector-embedding shells are not, they are not a design choice, so they would only add a constant offset.

---

## Equal-weight MOTPE

Raw log-RMSE and USD/cm$^2$ live on different scales. MOTPE ranks trials by hypervolume, which is scale-dependent, so the two values returned to Optuna are

\[
f_\text{error} = \frac{\mathcal{L}}{\texttt{ERROR_REF}}
\qquad
f_\text{cost} = \frac{\text{areal cost}}{\texttt{COST_REF}}
\]

```python
ERROR_REF  = 0.3    # log-RMSE that counts as "1 unit" of spectral error
COST_REF   = 2.0    # USD/cm² that counts as "1 unit" of material cost
HV_REF     = (10.0, 10.0)   # nadir in the (f_error, f_cost) plane
```

A design that is \(1\times\) `ERROR_REF` away in error then costs the hypervolume exactly as much as one that is \(1\times\) `COST_REF` away in money. Tune the two refs together so a typical good design sits near \(O(1)\) on both axes. Raw (unscaled) error and cost are stored on every trial for the terminal report and the plots.

`HV_REF` is the nadir used to compute 2-objective hypervolume in that same normalised plane. A trial that does not strictly dominate it contributes nothing. Early stopping watches this area, not a single “best” log-RMSE. Details, including how to read the knee, are on the [theory page](../theory/optimization.md#equal-weight-motpe).

=== "You care about the shape"

    After the run, start from the **MIN-ERROR** tag on the front.

=== "You want a balanced pick"

    Use the **EQUAL-WEIGHT KNEE**. That is the recommendation printed at the end of the run (feasible Pareto point closest to the origin in the \((f_\text{error},\, f_\text{cost})\) plane).

=== "The front looks wrong (everything piled in one corner)"

    `ERROR_REF` / `COST_REF` are off. Rescale until both axes of a decent design are \(O(1)\).

---

## Material distance metric

```python
DISTANCE_NORM: str = "frobenius"   # "induced1", "spectral", or "frobenius"
```

Controls the operator norm used to compute pairwise material distances for Metric-TPE. The metric is a sum of \(\|T_{m_1}(t) - T_{m_2}(t)\|_*\) over every base thickness that exists for all search materials, on the full angle-resolved operators. See [Sampler](../theory/optimization.md#sampler).

| Option | Reads as |
|---|---|
| `frobenius` | Entry-wise RMS disagreement |
| `spectral` | Worst-case amplification over any input |
| `induced1` | Largest L1 discrepancy for a single input bin |

---

## Optimization hyperparameters

| Constant | Role |
|---|---|
| `N_RESTARTS` | How many independent searches to run in total (different seeds) |
| `RESTART_WORKERS` | How many of those searches run **at the same time**. `1` = one after another. `4` with `N_RESTARTS = 16` = four at a time until all 16 finish. `None` = one per CPU core (never more than `N_RESTARTS`) |
| `N_TRIALS_EACH` | Trial cap per restart |
| `N_STARTUP_EACH` | Random warm-up per restart: TPE samples at random until this many trials have finished, then the model takes over ([`n_startup_trials`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)) |
| `PATIENCE` | **Early-stop** a restart when the Pareto hypervolume has not improved for this many trials |
| `OPERATOR_LRU_SIZE` | How many plate operators to keep in memory per restart (the last ones used). Building an operator is work; the search asks for the same plates over and over, so those results are reused. A cap keeps memory from growing without bound |
| `TOP_N_CONFIGURATIONS` | How many unique material sequences to list in the terminal printout |
| `RESTART_SEEDS` | One seed per restart |

There is no precise guideline for `N_RESTARTS`. It depends on the time available and on how rough the landscape is. This is an inverse problem: there is no guarantee the sampler finds a global optimum. A configuration whose log-RMSE is around `ERROR_REF` is generally a usable starting point for the Bayesian calibration step.

I personally suggest setting `N_TRIALS_EACH` high enough that most restarts are stopped by `PATIENCE` rather than hitting the trial cap. If restarts hit the cap, the front may still have been moving. `N_STARTUP_EACH` matters mcuh less

Sampler flags that are not exposed as one-line knobs (`multivariate`, `group`, \(\gamma\), EI candidates, prior / bandwidth floor) follow the TPE papers cited on the [theory page](../theory/optimization.md#sampler). Changing the sampler itself (NSGA-II, GP, …) is a small code change in `optimizer.py`.

---

## Zero target bins 

Some `TARGET_VALUES` entries can legitimately be `0` — groups outside the tabulated range, or padding left over from rebinning.  Those bins weight `0` and the optimizer is never penalised for whatever flux it transmits there. At startup the banner reports how many bins were excluded:

```
  [info] k/N target bins are exactly 0 -> excluded from the log-MSE objective.
```

---

## How a trial is evaluated

A trial is `n_layers` pairs `(material, thickness)` during the search. Consecutive same-material layers are collapsed only when ranking unique sequences and when printing recipes (`iron 2 cm + iron 5 cm` → one 7 cm iron layer). Uniqueness is the **material order** after that merge (and after inserting water cans); thicknesses are ignored, so `iron → iron → Pb` and `iron → Pb` are the same arrangement.

What actually sits in the beam is

1. the search stack,
2. then water-can walls if a layer is `water`,
3. then the fixed detector-embedding shells.

Operators are not precomputed on a fine thickness grid. The resident data are the database tensors. A requested thickness inside the database range is a log-linear interpolation between the bracketing bases; outside it is a fractional matrix power of the nearest boundary. Results are memoised per process (`OPERATOR_LRU_SIZE`). Formulae: [Fractional thicknesses](../theory/sn_transfer_operator.md#fractional-thicknesses).

---

## Running the optimizer

```bash
python src/optimizer.py
```

Needs a complete `opt_database/` for every name it will load. The working directory is wherever you launch it: plots and the pickle are written there.

The run prints a startup banner (objectives, equal-weight refs, weighting mode, restart budget, thickness grid, distance norm, fixed layers if any), then one bar per restart. When a restart stagnates you get a line like this (`PATIENCE` and the restart count are whatever you set):

```
  [Stagnation] Restart 3: no HV improvement for 400 trials → stopping early  (HV=...)
  ✓ Restart  3/16  │  seed=....  │  completed=.../...  │  pareto=...  │  HV=...  │  knee_d=...  │  early_stop=True
```

`knee_d` is the equal-weight distance of that restart's own knee, not a log-RMSE.

---

## Outputs

### Pickle

All restarts are saved to `study_results_multistart.pkl`. Frozen trials, seeds, per-restart Pareto size and hypervolume go in the file. To reprint the same report and regenerate the plots without re-optimising, run [`read_opt.py`](read_opt.md) (optional path to the pickle).

### Terminal report

A per-restart table (trials, completed, Pareto size, HV, knee distance), then the **combined front** across restarts. The equal-weight recommendation (knee of that combined front) is printed in raw units.

Then **design choices**: up to `TOP_N_CONFIGURATIONS` unique material sequences.

- Sequences that appear on the combined Pareto front come first, ranked lowest error → highest. The knee and the min-error endpoint are always included and tagged `★`.
- If that is fewer than `TOP_N_CONFIGURATIONS`, remaining slots are unique sequences **not** on the front (picked by equal-weight distance), listed under `NOT ON PARETO FRONT` and tagged `†`.
- Each entry shows log-RMSE, areal cost, equal-weight distance, target-weighted flux, how often that sequence was sampled, the merged layer recipe (with water-can notes), and the fixed detector shells.

Alongside the log-RMSE, the target-weighted flux for a configuration is

\[
\eta = \sum_{i} \hat{\phi}^{\text{target}}_i \cdot \phi^{\text{out}}_i
\]

where \(\hat{\phi}^{\text{target}}_i\) is the normalised target (used as per-bin weights) and \(\phi^{\text{out}}_i\) is the unnormalised transmitted flux. It is not a physical detector count. It discriminates stacks that put flux where the target has support from stacks that transmit in irrelevant groups. Higher usually means better detector statistics (or a shorter experiment).

### Plots

| File | What it shows |
|---|---|
| `cost_error_tradeoff.png` | Every trial in the (USD/cm$^2$, log-RMSE) plane, the Pareto staircase, equal-weight iso-distance contours, **MIN-ERROR** and **EQUAL-WEIGHT KNEE**. The y-axis is capped at `COST_ERROR_YLIM_TOP` so the cheap/high-error tail does not squash the accurate plateau |
| `pareto_designs.png` | One horizontal bar per unique sequence. Bar length is log-RMSE; coloured segments share that length in proportion to layer thickness; centimetres and cost are written to the right. Hatched / faded bars are off the front |

Take the tagged sequences from the report (or the plots), not a single automatic “winner”, and check them in 3-D Monte Carlo with [`base_plates.py`](base_plates.md) before spending a dataset on it.
