# `transfer_matrix.py` — $S_N$ Transfer Tensor Generator

## Purpose

`transfer_matrix.py` is the database builder for the optimization pipeline.  
It probes every *(material, thickness)* pair in a predefined grid with OpenMC Monte Carlo, constructing the angle resolved, energy resolved $S_N$ transfer tensor for each combination. The results are saved as NumPy arrays.
For the physics and mathematical background  see [$S_N$ Transfer Operator](../theory/sn_transfer_operator.md).

!!! info "One-time cost"
    Generating the full grid is expensive (about some hours on an HPC cluster with default values), but it only needs to be done once. If you calibrate multiple spectra on the same energy bin structure, the same tensor database can be reused.

The energy grids and source spectrum are **not** set here, they live in [`config.py`](config.md). The knobs below are local to this script.

---

## Contents

| Section | Variables |
|---|---|
| [Materials](#material-library) | `MATERIALS` |
| [Thicknesses](#thickness-grid) | `THICKNESSES` |
| [Angular quadrature](#angular-quadrature) | `N_MU` |
| [MC run](#mc-run-parameters) | `SN_BATCHES`, `SN_PARTICLES_INITIAL`, `SN_PARTICLES_MAX`, `SN_THRESHOLD`, `SN_REL_ERR` |
| [Overwrite](#overwrite) | `OVERWRITE` |
| [Parallelism](#parallelism) | `HPC`, `SN_WORKERS`, `SN_THREADS` |

---

## Material library

Compositions live in [`config_materials.py`](config_materials.md). This script only chooses **which of those keys get an \(S_N\) tensor**. `MATERIALS` can be the full library or any subset.

```python
MATERIALS = {
    name: _MATERIAL_LIBRARY[name]
    for name in (
        "PE_BO", "concrete", "HDPE",  # ... any keys from config_materials.py
    )
}
```

Every name must be a key from `build_material_library()`. Drop materials you do not want in the database.

!!! warning "A catalog entry is not a tensor"
    Adding a material in `config_materials.py` does nothing here until you list it in `MATERIALS` and regenerate. The optimizer can only search names that already have `.npy` files in `opt_database/`.

## Thickness grid

```python
THICKNESSES = [0.2, 0.8, 3.0, 10.0, 28.0, 75.0, 150.0]  # cm
```
The user can replace this list with any thickness values. For best coverage of the design space, a uniform logarithmic spacing between the minimum and maximum thickness is recommended; see [Fractional thicknesses](../theory/sn_transfer_operator.md#fractional-thicknesses) and [Evaluating a stack](../theory/optimization.md#evaluating-a-stack).

## Angular quadrature

The only user setting is `N_MU`. Nodes and tally-bin edges are derived from it.

```python
N_MU = 5
```

`optimizer.py` imports `N_MU` from here so the tensor database and the search stay on the same angular grid.

We only treat the **forward hemisphere** (transmission). \(\mu = \cos\theta\) from the slab normal, so \(\mu \in (0, 1]\): \(\mu = 1\) is normal incidence, \(\mu \to 0\) is grazing.

Half-range Gauss–Legendre: standard nodes on \([-1, 1]\) remapped by \(s \to \mu = (s+1)/2\). From that the script builds:

- `MU_NODES` — the \(N_\mu\) ordinates (incident beam directions)
- `MU_EDGES` — \(N_\mu+1\) edges for `openmc.MuSurfaceFilter` (`0`, midpoints between nodes, `1`), so each tally bin contains exactly one ordinate

See [Angular quadrature details](../theory/sn_transfer_operator.md#angular-quadrature-details) for the current node table.

!!! danger "Changing the quadrature order"
    To increase angular resolution, change `N_MU` and regenerate the entire tensor database.

    ```python
    N_MU = 8
    ```
    All `.npy` files in `opt_database/` must be deleted and regenerated whenever `N_MU` changes.

!!! tip "Default \(N_\mu = 5\) is usually enough"
    I personally suggest leaving `N_MU` at 5 unless you have a strong reason to spend the extra generation cost. Generation cost scales with the number of incident beams (\(N_\mu \times N_E\)).

## MC run parameters

| Constant | Default | Description |
|---|---|---|
| `SN_BATCHES` | 100 | Number of batches |
| `SN_PARTICLES_INITIAL` | 16 000 | Particles per batch (first attempt) |
| `SN_PARTICLES_MAX` | 100 000 | Particle cap for the adaptive gate |
| `SN_THRESHOLD` | $8 \times 10^{-6}$ | Noise floor: a bin below this is kept only if it is statistically solid; otherwise it is zeroed and does not trigger a rerun |
| `SN_REL_ERR` | 0.15 (15 %) | Maximum relative error \(\sigma/\mu\) on bins at or above `SN_THRESHOLD` |

## Overwrite

```python
OVERWRITE = False
```

The grid is resumable: existing `sn_tensor_*.npy` files are skipped. Set `OVERWRITE = True` to regenerate them (for example after changing `N_MU` or a material composition).

## Parallelism

One pair of knobs: **workers** (how many incident-beam columns run at once) x **threads** (OpenMC OpenMP threads inside each column).

```python
HPC = False
```

=== "Local (`HPC=False`)"

    ```python
    SN_WORKERS = 1
    SN_THREADS = max(1, (os.cpu_count() or 2) // 2 - 1)
    ```

    Default is one worker so a single OpenMC run can use most of the machine. Raise `SN_WORKERS` only if you want several columns in flight on a big workstation.

=== "HPC (`HPC=True`)"

    ```python
    SN_WORKERS = 48
    SN_THREADS = 4
    ```

    Those are **fallbacks**. On a real submission the sbatch script owns the layout, for example:

    ```bash
    #SBATCH -N 1                      # single node
    #SBATCH -n 24                     # workers  →  SLURM_NTASKS
    #SBATCH --ntasks-per-node=24      # pack all workers on that node
    #SBATCH --cpus-per-task=8         # threads  →  SLURM_CPUS_PER_TASK
    export OMP_NUM_THREADS=8          # match --cpus-per-task
    ```

    `-n` / `--ntasks` is the worker count; `--ntasks-per-node` should match when you stay on one node. `--cpus-per-task` and `OMP_NUM_THREADS` should agree and are the OpenMC threads inside each worker. The Python file does not need to repeat those numbers: `SLURM_NTASKS` / `SLURM_CPUS_PER_TASK` override `SN_WORKERS` / `SN_THREADS` whenever they are set.

    !!! warning "HPC layout is machine-specific"
        The right `-N`, `-n`, `--ntasks-per-node`, `--cpus-per-task`, and `OMP_NUM_THREADS` depend heavily on the cluster (node size, queue limits, how the site maps `--ntasks` vs `--cpus-per-task`). Treat any numbers here as an example, not a recipe.

| Mode | Workers | Threads / worker |
|---|---|---|
| Local | `SN_WORKERS` | `SN_THREADS` |
| HPC (SLURM) | `#SBATCH --ntasks-per-node` | `#SBATCH --cpus-per-task` |

I personally suggest `workers × threads` = the cores you were allocated. Too many workers with one thread each thrashes OpenMC startup; one worker with all the threads leaves columns serial.

---

## Energy grids

The script maintains two distinct energy grids: `TARGET_ENERGY_EDGES` (imported from `config.py`), the target spectrum grid, and `ENERGY_EDGES`, the fine generation grid built here by refining `TARGET_ENERGY_EDGES` around the source spectrum.

See [Source spectrum and the fine energy grid](../theory/sn_transfer_operator.md#source-spectrum-and-the-fine-energy-grid) for theory details.

The OpenMC simulations, the tally bins, and the stored tensor all use `ENERGY_EDGES`. The tensor shape is therefore `(N_MU, N_E_fine, N_MU, N_E_fine)` where `N_E_fine = len(ENERGY_EDGES) - 1`.

!!! info "Cost scale linearly with the number of energy bins"
    computational cost scales almost linearly with the number of energy bins. Do not use an unnecessarily fine energy grid for the source definition.

---

## How it works

### Quality gate (per incident column)

Each OpenMC run is one incident beam \((\mu_\text{in}, E_\text{in})\). The gate is applied to that column's transmitted **current**, bin by bin, before the next column starts.

A bin is statistically solid when

\[
\frac{\sigma}{\text{Mean}} \le \texttt{SN_REL_ERR}
\]

 What happens next depends on the mean:

| Mean | Solid? | Action |
|---|---|---|
| any | yes | **Keep** including values below `SN_THRESHOLD`. A well-resolved small current is signal, not noise. |
| \(\text{Mean} \ge\) `SN_THRESHOLD` | no | **Fail the run** — double `particles` and rerun, up to `SN_PARTICLES_MAX`. |
| \(\text{Mean} <\) `SN_THRESHOLD` | no | **Zero that bin** and continue. This never triggers a rerun. |

If the particle cap is hit and a large bin is still noisy, the cleaned column is **accepted anyway** (`cap reached`) so that it doesn't stall the grid forever.

!!! info "Only large noisy bins cost more particles"
    `SN_THRESHOLD` is a speed/accuracy trade: tiny, poorly sampled tails are dropped instead of refining the whole column for them.

### Transmission factor (per-column log)

Each finished OpenMC column prints a line like:

```
[ 12/ 55] mu_in=0.9531 E_in= 3  T=8.412e-01  particles=8000  (Validation passed.)
```

`T` is the **transmission factor** for that incident beam \((\mu_\text{in}, E_\text{in})\): the sum of the gate-cleaned outgoing current over every \((\mu_\text{out}, E_\text{out})\) bin. The source strength is 1, the tally is surface current on the exit face (forward hemisphere only), and OpenMC reports a fixed-source score per source particle, so

\[
T(\mu_\text{in}, E_\text{in})
= \sum_{\mu_\text{out},\, E_\text{out}}
  J(\mu_\text{out}, E_\text{out} \leftarrow \mu_\text{in}, E_\text{in}).
\]

That is the same sum as the stored tensor column. It should lie in \([0, 1]\) for a purely absorbing/scattering slab. Values above 1 mean multiplication (for example \((n,2n)\)) is winning.

### Grid execution and output
The top-level driver iterates over all `(material, thickness)` pairs. Each pair is \(N_\mu \times N_E\) gated columns, then two `.npy` files:

| File | Shape | Contents |
|---|---|---|
| `sn_tensor_{mat}_{t}cm.npy` | `(N_MU, N_E, N_MU, N_E)` | Monte Carlo mean of \(T\) |
| `sn_tensor_{mat}_{t}cm_std.npy` | `(N_MU, N_E, N_MU, N_E)` | Monte Carlo standard deviation |

Loading a tensor:

```python
import numpy as np
T = np.load("opt_database/sn_tensor_graphite_10.0cm.npy")
```

##  Usage

```bash
python src/transfer_matrix.py
```

