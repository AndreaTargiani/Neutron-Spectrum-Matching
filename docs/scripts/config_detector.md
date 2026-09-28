# `config_detector.py` — Detector Geometry and Energy Response

## Purpose

`config_detector.py` is the single place where the detector is declared: which part of that detector body is tallied, and (optionally) the detector's energy-response curve.

---

## Contents

| Section | Variables |
|---|---|
| [Geometry](#geometry-and-materials) | `DETECTOR_NAME`, `DETECTOR_REGIONS` |
| [Energy response](#energy-response) | `DETECTOR_RESPONSE_ENERGIES`, `DETECTOR_RESPONSE_VALUES`, `MAX_POLY_DEGREE` |

---

## Geometry and materials

```python
DETECTOR_NAME = "..."
DETECTOR_REGIONS = [ ... ]   # outermost first
```

`DETECTOR_REGIONS` is a list of nested regions, **outermost first**. Each dict becomes one `DetectorRegion`. The framework turns that list into `DETECTOR_GEOMETRY` for the OpenMC model.

This is a nested-shape description, not a full CSG editor. It covers "a body around a smaller active volume" (with as many concentric shells as you like). Mix shapes freely.

=== "Single region"

    The minimum detector is one region, and it must be active (otherwise nothing would be tallied):

    ```python
    DETECTOR_NAME = "bare spherical detector"
    DETECTOR_REGIONS = [
        dict(name="active_volume", material="air",
             shape="sphere", diameter=4.0, active=True),
    ]
    ```

=== "Nested (moderator + active volume)"

    List the outer body first, then carve the inner volume out of it:

    ```python
    DETECTOR_NAME = "HDPE-moderated spherical detector"
    DETECTOR_REGIONS = [
        dict(name="moderator", material="HDPE",
             shape="sphere", diameter=30.0, active=False),
        dict(name="active_volume", material="air",
             shape="cylinder", axis="x",
             diameter=2.0, height=12.0, active=True),
    ]
    ```

    Region *i* is that region's solid minus region *i+1*'s solid. The inner region must fit strictly inside the outer one along X, Y, and Z.

`base_plates.py` places the detector immediately downstream of the last plate, on the beam axis. The shared centre of every region is at

\[
x_\text{centre} = x_\text{last plate face} + \tfrac{1}{2} L_x
\]

where \(L_x\) is the outermost region's full extent along X. The upstream face of that outer body therefore sits on the last plate.

---

### Region fields

| Field | Shapes | Meaning |
|---|---|---|
| `name` | all | Label used in logs and in the OpenMC cell name (`detector_<name>`) |
| `material` | all | A [`config_materials.py`](config_materials.md) key, or an `openmc.Material` you built yourself |
| `shape` | all | `"sphere"`, `"cylinder"`, or `"box"` |
| `active` | all | `True` → tallied (sensitive volume). `False` → passive structure |
| `diameter` | sphere, cylinder | Full diameter (cm) |
| `height`, `axis` | cylinder | Full length (cm) and axis `"x"` / `"y"` / `"z"` (`"x"` = along the beam, the default) |
| `dx`, `dy`, `dz` | box | Full extents (cm) along the global axes |

=== "sphere"

    ```python
    dict(name="moderator", material="HDPE",
         shape="sphere", diameter=20.0, active=False)
    ```

=== "cylinder"

    ```python
    dict(name="active_volume", material="air",
         shape="cylinder", axis="x",
         diameter=2.0, height=12.0, active=True)
    ```

=== "box"

    ```python
    dict(name="housing", material="Al_6061",
         shape="box", dx=10.0, dy=8.0, dz=8.0, active=False)
    ```

All sizes are centimetres (OpenMC convention).

---

### Nesting and the sensitive volume

- List regions **outermost first**. Every region after the first is carved out of the one before it.
- Exactly the regions with `active=True` are tallied (summed together if there is more than one). Everything else is local structure / shielding.
- There must be at least one region, and at least one of them must be `active=True`.
- Each inner region must fit inside the outermost body along X, Y, and Z.

---

### Geometry sanity check

```bash
python src/config_detector.py
```

Prints a one-block summary: each region (shape, material, volume, active vs passive), the outer footprint, and the total active volume. If an energy-response curve is configured, it also writes `detector_response_fit.png` on the working target grid. No OpenMC run is launched.

---

## Energy response

```python
MAX_POLY_DEGREE = 8

DETECTOR_RESPONSE_ENERGIES = np.array([...], dtype=np.float64)
DETECTOR_RESPONSE_VALUES   = np.array([...], dtype=np.float64)
```

The measured or tabulated energy-response pairs for the detector, on **any** grid (any number of points, any spacing). The two arrays must have the same length, at least two points, strictly positive energies, and non-negative values.

Units of `DETECTOR_RESPONSE_VALUES` are arbitrary: only the **shape** (relative ratios) matters, because every consumer renormalises the weights. Energies must be in eV.

This curve is **not** `WEIGHT_MODE`. The mode is set once in [`config.py`](config.md#per-bin-weighting). This file only supplies \(R(E)\) for the modes that need it. Running this file writes `detector_response_fit.png`, which is the picture if `WEIGHT_MODE = "response"`.

=== "Tabulated curve"

    Paste your \((E, R)\) pairs. Callers fit a polynomial in log-log space and evaluate it on their energy midpoints:

    1. Fit degrees \(1, 2, \ldots, \min(n-1, D)\) with closed form leave-one-out cross-validation (LOO-CV), where \(D\) is `MAX_POLY_DEGREE`; keep the degree with the smallest LOO-RMSE.
    2. Evaluate the winning polynomial at the requested bin midpoints. Energies outside the tabulated range are **clamped** to the nearest endpoint.

=== "No curve (flat)"

    ```python
    DETECTOR_RESPONSE_ENERGIES = None
    DETECTOR_RESPONSE_VALUES   = None
    ```

    The fit is skipped. `"uniform"` and `"ICRP"` still work: they do not need \(R(E)\).

!!! warning "`response` / `dose` need a curve"
    If you switch [`WEIGHT_MODE`](config.md#per-bin-weighting) to `"response"` or `"dose"` but leave both arrays as `None` you will encounter errors down the pipeline.

