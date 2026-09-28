# `dataset_creation.py` — Sobol Dataset Generator

## Purpose

`dataset_creation.py` builds the training dataset for the GPR surrogate. For each point in a Sobol design over one shield stack, it runs the two-phase OpenMC workflow from [`base_plates.py`](base_plates.md) and writes the integral-normalised flux spectrum and its per-bin Monte Carlo uncertainty to a CSV file.

The material order comes from [`optimizer.py`](optimizer.md). Check that recipe in 3-D with `base_plates.py` before spending the sweep on it. This script then varies the source gap, the layer thicknesses, and the square plate size \(L\). One CSV is one stack. [`calibration.py`](calibration.md) reads that file directly.

!!! info "One-time cost, resumable"
    Generating the full dataset is one of the most computationally expensive step in the pipeline. The script skips rows already present in the CSV, reuses one multigroup cross-section library for the whole sweep, and continues if a single geometry fails.

---

## Contents

| Section | Variables |
|---|---|
| [Material stack](#material-stack) | `MATERIALS` |
| [Geometry bounds](#geometry-bounds) | `X_START_BOUNDS`, `THICKNESS_BOUNDS`, `L_BOUNDS` |
| [Sampling](#sampling) | `N_SAMPLES`, `SOBOL_SEED` |
| [Phase 1 (weight windows)](#mc-run-parameters--phase-1-weight-windows) | `RR_*`, `MGXS_PARTICLES` |
| [Phase 2 (CE production)](#mc-run-parameters--phase-2-ce-production) | `CE_*`, `MAX_LOWER_BOUND_RATIO`, `MAX_SPLIT`, `CE_TIMEOUT_S`, `USE_ANGLE_BIAS` |
| [Paths](#paths) | `SAMPLE_WORK_DIR`, `MGXS_PATH`, `OUTPUT_CSV` |
| [Output CSV](#output-csv) | columns read by `calibration.py` |

`HPC`, the MPI ranks, and the OpenMP thread counts are set once in [`base_plates.py`](base_plates.md#parallelism). This script imports them, and the startup banner prints the backend it picked up.

---

## Material stack

Paste the arrangement you kept from the optimizer report, the same list you checked in `base_plates.py`:

```python
MATERIALS: list[str] = ['graphite', 'iron', 'Pb']   # keys from config_materials.py
```

The number of entries is the number of layers, and it must equal the number of `THICKNESS_BOUNDS` pairs. The script raises `ValueError` at import when they differ. Names must be keys in [`config_materials.py`](config_materials.md).
 
if you want to try a second arrangement, it should be a second file: change `MATERIALS` and `OUTPUT_CSV`, then run again.

---

## Geometry bounds

```python
X_START_BOUNDS: tuple[float, float] = (0.1, 0.1)   # cm — source → first face

THICKNESS_BOUNDS: list[tuple[float, float]] = [
    (5.0,  25.0),   # plate 1
    (2.0,  16.0),   # plate 2
    (1.0,   8.0),   # plate 3
]

L_BOUNDS: tuple[float, float] = (40.0, 80.0)       # cm — square side L
```

| Bound | Meaning |
|---|---|
| `X_START_BOUNDS` | Gap between the point source (the origin) and the upstream face of the first plate |
| `THICKNESS_BOUNDS` | One `(min, max)` pair per layer, same order as `MATERIALS` |
| `L_BOUNDS` | Square transverse size |

Centre `THICKNESS_BOUNDS` on the thicknesses that survived the `base_plates.py` check, wide enough for the posterior to move. Centre `L_BOUNDS` on the `PLATE_DIM` you settled on there. That single check size is one point; the sweep has to cover the sizes you might actually build for both thicknesses and plate transversal sizes.

The last plate face plus the detector must stay inside `AIR_SPHERE_RADIUS` in `base_plates.py`. A sample that does not is marked `FAILED` and is omitted from the CSV. Keep the upper end of the gap plus the upper ends of the thicknesses under that radius, with room for the detector body.

!!! tip "The gap is usually not the parameter you want to calibrate"
    I personally suggest fixing `x_start` to a small value (`X_START_BOUNDS = (0.1, 1.0)`, the same gap as the 3-D check) when the shield will be built. Good stacks stay good across a wide standoff, and a short gap puts more source neutrons on the first plate. Open `X_START_BOUNDS` into a real interval when the standoff itself is a design choice.

---

## Sampling

```python
N_SAMPLES: int  = 256    # new design points this run
SOBOL_SEED: int = 42
```

`N_SAMPLES` is the number of new points this invocation will try. `SOBOL_SEED` scrambles the sequence. The same seed and the same bounds always produce the same points in the same order.

The sampler is the [SciPy Sobol implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html). It draws

\[
d = N_{\text{plates}} + 2
\]

independent uniform variates, one for \(x_\text{start}\), one for \(L\), and one thickness per plate:

\[
\mathbf{u} = (x_\text{start},\; L,\; t_1,\; \ldots,\; t_{N}) \in [0,1]^{d}
\]

Those are scaled onto `X_START_BOUNDS`, `L_BOUNDS`, and `THICKNESS_BOUNDS`. `thicknesses_to_x_bounds` then builds the faces OpenMC receives:

\[
x_1 = x_\text{start}, \qquad x_{k+1} = x_k + t_k, \quad k = 1,\ldots,N_{\text{plates}}
\]

The CSV stores the cumulative boundaries \((x_1,\ldots,x_{N+1})\) and \(L\). `calibration.py` recovers the independent inputs \((x_1, t_1, \ldots, t_N, L)\).

Sobol coverage is even when `N_SAMPLES` is a power of two (128, 256, 512, ...) ([Sobol' 1967](https://doi.org/10.1016/0041-5553(67)90144-9)). Any count runs; a count that is not \(2^k\) leaves parts of the hypercube thinner than the rest. Pick the size from how long one FW-CADIS pair takes on your machine. [`--refine`](#running) appends another block with the same seed, so the first run does not have to be the final size.

---

## MC run parameters — Phase 1 (weight windows)

These override the single-check counts in `base_plates.py` for the sweep. The mesh, `WW_ENERGY_EDGES`, the detector, and the thread layout stay in that file.

| Constant | Default | Description |
|---|---|---|
| `RR_BATCHES` | 100 | Random-ray batches for the adjoint solve |
| `RR_INACTIVE_BATCHES` | 50 | Inactive batches |
| `RR_PARTICLES` | 20 000 | Rays per batch |
| `MGXS_PARTICLES` | 20 000 | Particles for the stochastic-slab MGXS library |

The library is built once per sweep and then reused. See [Paths](#paths).

---

## MC run parameters — Phase 2 (CE production)

| Constant | Default | Description |
|---|---|---|
| `CE_PARTICLES` | 90 000 | Particles per batch |
| `CE_BATCHES` | 100 | Number of batches |
| `MAX_LOWER_BOUND_RATIO` | 1.0 | Weight-window lower-bound scaling |
| `MAX_SPLIT` | 10 | Maximum splits per weight-window crossing |
| `CE_TIMEOUT_S` | 600 s | Wall-time cap on one CE run |
| `USE_ANGLE_BIAS` | `True` | Source angular bias toward the plate |

`CE_TIMEOUT_S` aborts a CE run that stalls on splitting. The scratch directory is removed, the terminal prints `FAILED`, and that point is left out of the CSV. The sweep continues with the next Sobol point.

`USE_ANGLE_BIAS` (recommended on) is the same piecewise-linear bias on \(\mu = \cos\theta\) as in [`base_plates.py`](base_plates.md#mc-run-parameters--phase-2-ce-production), rebuilt from this sample's \(x_\text{start}\) and \(L\). The value used for the row is written to the CSV.

`MAX_SPLIT` is lower here than in a one-off `base_plates.py` check, because a single slow geometry is paid for on every remaining sample. If CE runs time out, lower it further before raising the particle count.

!!! tip "Copy the counts from the 3-D check"
    I personally suggest running the chosen stack in `base_plates.py` until `rel_unc_%` on the bins you care about looks usable, then copying those batch and particle counts into this file. The sweep repeats that pair once per Sobol point.

---

## Paths

```python
SAMPLE_WORK_DIR: str = 'sample_workdir'
MGXS_PATH: Path      = Path(SAMPLE_WORK_DIR) / 'mgxs.h5'
OUTPUT_CSV: Path     = Path('sobol_dataset.csv')
```

`SAMPLE_WORK_DIR/run_XXXX/` holds the OpenMC files for one point (`ww/` and `ce/`). It is deleted after the flux is written, and after a failure. `OUTPUT_CSV` is the file you keep. Paths are relative to the directory you launch from; point `CSV_PATH` in `calibration.py` at the same file.

`MGXS_PATH` sits outside the `run_*` trees, so cleanup does not remove it. Materials and the weight-window energy grid are fixed across the sweep, so the stochastic-slab library is generated on the first sample and reused. The banner reports `generate once, then reuse` or `reuse`.

`--regen` deletes `mgxs.h5` before the first sample. `--refine` keeps it. A library whose group structure differs from `WW_ENERGY_EDGES` is regenerated inside `generate_weight_windows` either way.

!!! warning "A material change needs a new library"
    Reuse checks the energy grid. It does not check whether `MATERIALS` changed. After editing the stack, pass `--regen` or delete `mgxs.h5` yourself. Otherwise an error will be raised.

---

## How it works

### Resumption

| Invocation | Behaviour |
|---|---|
| CSV absent | Writes the header and starts at Sobol index 0 |
| CSV present, no flag | Prints a hint and exits without writing |
| `--regen` | Replaces the CSV, deletes `mgxs.h5` if it exists, restarts at index 0 |
| `--refine` | Keeps the CSV and the MGXS library, fast-forwards the Sobol engine by the number of existing rows, appends `N_SAMPLES` new attempts |

!!! warning "The skip count is the number of CSV rows"
    `--refine` does not remember which Sobol points were already drawn. It fast-forwards by the number of rows in the file. A `FAILED` sample took a point from the sequence and wrote nothing, so that count is smaller than the number of points already used.

Take a first run of 10 points in which Sobol indices 2 and 7 fail. Indices 0 through 9 have all been drawn. The CSV has 8 rows.

| Sobol index | Result | In the CSV |
|---|---|---|
| 0, 1 | done | yes |
| 2 | `FAILED` | no |
| 3–6 | done | yes |
| 7 | `FAILED` | no |
| 8, 9 | done | yes |

The next `--refine` skips 8 and starts at index 8. Indices 8 and 9 are already in the file, so those two geometries are simulated again and you get duplicate rows. Indices 2 and 7 stay missing: they sit inside the 8 points that are skipped, and nothing goes back to fill them. A failed geometry is drawn again only when it lies in that overlap at the end of the previous block.

---

## Output CSV

| Column group | Columns | Description |
|---|---|---|
| Run metadata | `run_idx` | Attempt index. Failed attempts leave a gap |
| Materials | `mat_1`, `mat_2`, ... | The fixed stack, repeated on every row |
| Geometry | `x1` ... `x{N+1}` | Cumulative plate-boundary \(X\) coordinates (cm) |
| Plate size | `L` | Square side (cm) |
| Run settings | `use_angle_bias`, `ce_particles`, `ce_batches` | Settings used for this row |
| Wall times | `wall_time_ww_s`, `wall_time_ce_s`, `wall_time_s` | Phase 1, Phase 2, and total (s) |
| Flux | `y_00`, `y_01`, ... | Integral-normalised flux, one column per target bin |
| Uncertainty | `sigma_00`, `sigma_01`, ... | Absolute \(1\sigma\) on that normalised flux |

---

## Running

```bash
# Start a fresh dataset (OUTPUT_CSV must not exist yet)
python src/dataset_creation.py

# Replace the CSV and the shared mgxs.h5, restart the sequence
python src/dataset_creation.py --regen

# Append N_SAMPLES more rows to an existing CSV
python src/dataset_creation.py --refine
```

The script prints a startup banner and one progress line per sample:

```
Sobol refinement - N-layer plate FW-CADIS sampling
  layers          : 3
  stack           : graphite | iron | Pb
  x_start bounds  : (0.1, 0.1) cm
  t1 bounds       : (5.0, 25.0) cm  [graphite]
  t2 bounds       : (2.0, 16.0) cm  [iron]
  t3 bounds       : (1.0, 8.0) cm  [Pb]
  L bounds        : (40.0, 80.0) cm  (square plate transverse size)
  n_samples       : 256
  existing rows   : 0  (Sobol sequence fast-forwarded by 0)
  mgxs            : sample_workdir/mgxs.h5  (generate once, then reuse)
  CE run          : particles=90000, batches=100
  angle bias      : True
  backend         : local threads (…)
  output          : sobol_dataset.csv  (mode='w')

[   1/ 256] idx=   1  x=(  0.1,  12.4,  18.0,  21.2)  L=  58.0 ... [ww] 140s [ce] done ( 210.0 s  |  ww 140.0 s, ce  70.0 s)
```

On HPC the backend lines list the Phase 1 OpenMP count and the Phase 2 MPI x OpenMP layout inherited from `base_plates.py`. A failed sample ends the progress line with `FAILED` and the exception, in place of `done`.

!!! tip "Check the CSV before calibrating"
    `analize_dataset.py` reads this CSV and reports two things you want before `calibration.py`: the per-bin survival fraction at `UNC_LIMIT`, and which target transform is closer to Gaussian. Use that comparison when you set `Y_TRANSFORM`. See [`analize_dataset.py`](analize_dataset.md).

    ```bash
    python src/analize_dataset.py
    ```

---

## References

- I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals", *USSR Computational Mathematics and Mathematical Physics* 7(4), 86–112 (1967). [doi:10.1016/0041-5553(67)90144-9](https://doi.org/10.1016/0041-5553(67)90144-9)
