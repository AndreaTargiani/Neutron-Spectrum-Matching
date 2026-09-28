# `config.py` — Central Configuration

## Purpose

`config.py` is the single place where the framework's inputs are declared: cross-section library paths, the neutron source spectrum, the target energy grid, the target spectrum, and how energy bins are weighted when matching that spectrum. Editing one parameter in `config.py` propagates consistently everywhere it is used.

It is not the only configuration file. Materials live in [`config_materials.py`](config_materials.md) and the detector (geometry + energy response) lives in [`config_detector.py`](config_detector.md). This page covers only `config.py`.

!!! warning "`HPC` is intentionally NOT here"
    Each script (`transfer_matrix.py`, `base_plates.py`) keeps its **own** `HPC` on/off switch at the top of the file. `config.py` only supplies the two path sets (`"local"` / `"hpc"`) for cross section data; each script's own `HPC` flag decides which one it uses.

---

## Contents

| Section | Variables |
|---|---|
| [Cross sections](#cross-sections-nuclear-data) | `CROSS_SECTIONS`, `CHAIN_FILE` |
| [Source spectrum](#source-spectrum) | `SOURCE_KIND`, `SOURCE_GAUSSIAN`, `SOURCE_HISTOGRAM` |
| [Raw target spectrum](#raw-target-spectrum) | `RAW_TARGET_ENERGY_EDGES`, `RAW_TARGET_VALUES`, `RAW_TARGET_VALUES_ARE_LETHARGY` |
| [Target spectrum](#target-spectrum) | `TARGET_ENERGY_EDGES`, `TARGET_VALUES`, `PHI_TARGET_IN_LETHARGY` |
| [Per-bin weighting](#per-bin-weighting) | `WEIGHT_MODE` |
| [Helpers](#helpers) | `LOG_EPS` |

---

## Cross sections / nuclear data

```python
CROSS_SECTIONS = {
    "local": "/path/to/endfb-viii.0-hdf5/cross_sections.xml",
    "hpc": "~/endfb-viii.0-hdf5/cross_sections.xml",
}
CHAIN_FILE = {
    "local": "/path/to/chain-endf-b8.0.xml",
    "hpc": None,
}
```

Update these paths to point at your own cross-section library, downloaded separately from the [OpenMC data repository](https://openmc.org/data/).

---

## Source spectrum

Selects how the neutron source energy distribution is represented. See [Source spectrum and the fine energy grid](../theory/sn_transfer_operator.md#source-spectrum-and-the-fine-energy-grid) for the underlying theory.

```python
SOURCE_KIND = "gaussian"   # "gaussian" | "edges"
```

=== "gaussian (default)"

    ```python
    SOURCE_GAUSSIAN = {
        "mean": 14.1e6,   # eV — peak energy
        "fwhm": 0.3e6,    # eV — set exactly one of fwhm / std
        "std": None,      # eV
        "n_bins": 5,      # number of energy bins used to discretise the Gaussian
        "n_sigma": 2.0,   # the Gaussian is truncated at +/- this many standard deviations
    }
    ```
    Set exactly one of `"fwhm"` / `"std"` (leave the other as `None`).

=== "edges"

    ```python
    SOURCE_HISTOGRAM = {
        "edges": np.array([5.0e6, 1.28e7, 1.455e7]),  # eV
        "values": np.array([0.4, 0.2]),
        "values_are_density": False,
    }
    ```

    Specify the spectrum of your characterised source on the energy structure you want. If `"values_are_density"` is `True`, values are treated as per-unit-energy densities (multiplied by bin width internally). Otherwise they are bin integrated content.

---

## Raw target spectrum

```python
RAW_TARGET_ENERGY_EDGES = np.array([...])   # (M+1,) eV, must be ascending
RAW_TARGET_VALUES = np.array([...])         # (M,) values, one per bin, same order
RAW_TARGET_VALUES_ARE_LETHARGY: bool = True
```

The target spectrum exactly as it comes from its source (e.g. a benchmark, a measurement), on whatever energy grid it is tabulated on. The framework uses the [working target](#target-spectrum) (`TARGET_ENERGY_EDGES` / `TARGET_VALUES`) instead.

`RAW_TARGET_VALUES_ARE_LETHARGY` declares whether `RAW_TARGET_VALUES` is a flux-per-unit-lethargy density $\phi_u = \mathrm{d}\phi/\mathrm{d}u$ rather than a classical flux.

Rebinning onto a different working grid is a **feature, not a required step**. If you are happy matching on the tabulated structure, the working target is the raw spectrum:

=== "Use the raw grid as-is"

    Copy the three raw constants into the working pair:

    ```python
    TARGET_ENERGY_EDGES = RAW_TARGET_ENERGY_EDGES
    TARGET_VALUES = RAW_TARGET_VALUES
    PHI_TARGET_IN_LETHARGY = RAW_TARGET_VALUES_ARE_LETHARGY
    ```

    (or paste the same arrays).

=== "Rebin onto a different working grid"

    Choose `TARGET_ENERGY_EDGES` as the group structure you actually want (the file ships with VITAMIN-J 175; see [below](#target-spectrum)), and run [`rebin_target_spectrum.py`](rebin_target_spectrum.md) to project the raw spectrum onto it. The result is saved to `target_spectrum_rebinned.npy`, ready to paste into `TARGET_VALUES`.

!!! warning "`RAW_TARGET_VALUES` must have one entry per `RAW_TARGET_ENERGY_EDGES` bin"
    `RAW_TARGET_VALUES` should have exactly `len(RAW_TARGET_ENERGY_EDGES) - 1` entries. If it is shorter, `rebin_target_spectrum.py` silently trims `RAW_TARGET_ENERGY_EDGES` from its high-energy end to match (`n = min(len(edges) - 1, len(values))`) rather than raising an error.

!!! info "Zero entries"
    Exact `0.0` values are allowed. Those bins are ignored throughout the whole framework: wherever the flux is exactly zero, the bin is dropped completely.

---

## Target spectrum

```python
TARGET_ENERGY_EDGES = [..., ...]   # eV, N+1 edges -> N groups
```

The group structure every downstream script actually uses. `transfer_matrix.py` builds its fine tensor-generation grid by refining this grid around the source spectrum (see [Source spectrum and the fine energy grid](../theory/sn_transfer_operator.md#source-spectrum-and-the-fine-energy-grid)); `optimizer.py` and `base_plates.py` both compare their output against `TARGET_VALUES` on this exact grid.

If you are using the raw spectrum as-is, this list is `RAW_TARGET_ENERGY_EDGES`. If you are rebinning, this is the destination group structure (independent of how the raw spectrum was tabulated).

After changing it, the $S_N$ tensor database must be regenerated (`python src/transfer_matrix.py`).

```python
PHI_TARGET_IN_LETHARGY: bool = True

TARGET_VALUES = [...]   # one value per TARGET_ENERGY_EDGES group
```

The reference neutron spectrum every downstream script compares its output against. `len(TARGET_VALUES)` must equal `len(TARGET_ENERGY_EDGES) - 1`.

`PHI_TARGET_IN_LETHARGY` declares whether `TARGET_VALUES` is a flux-per-unit-lethargy density $\phi_u = \mathrm{d}\phi/\mathrm{d}u$ (with lethargy $u = \ln(E_\text{ref}/E)$) rather than a classical flux. If you copied the raw spectrum as-is, this flag must match `RAW_TARGET_VALUES_ARE_LETHARGY`; if you ran the rebinner, the saved array is already in whatever convention this flag declares.

---

## Per-bin weighting

```python
WEIGHT_MODE = "ICRP"   # "uniform" | "ICRP" | "response" | "dose"
```

A single switch controlling how each energy bin is weighted in the framework. To choose between `ICRP` and `dose` for a given target, run [`bin_weights.py`](bin_weights.md) and compare the two dose-rate curves.

The default coefficient \(h(E)\) is ambient dose equivalent H\*(10) from [ICRP Publication 74](#references). The same switch also accepts [ICRP Publication 116](#references), and either ambient or effective dose. Which publication and quantity are active, and what that changes in the weights, is set and explained in [`bin_weights.py`](bin_weights.md).

=== "uniform"

    Every (non-zero) bin is weighted equally.

=== "ICRP"

    \(w_i \propto \phi_i \cdot h_i\), the dose rate the target deposits in bin \(i\). \(h\) is the fluence-to-dose coefficient set in [`bin_weights.py`](bin_weights.md). The default is H\*(10).

=== "response"

    \(w_i \propto R_i\), the detector energy response. The curve and its fit are plotted by [`config_detector.py`](config_detector.md) (`detector_response_fit.png`).

=== "dose"

    \(w_i \propto \phi_i \cdot R_i \cdot h_i\), the dose rate the detector would report in bin \(i\). Same \(h\) as `ICRP`, folded with \(R\) from [`config_detector.py`](config_detector.md).

---

## Helpers

```python
LOG_EPS = 1e-20   # floor inside log(phi + LOG_EPS) for the log-RMSE
```

Shared floor for the detector-weighted log-RMSE. [`optimizer.py`](optimizer.md) and the OpenMC printout in [`base_plates.py`](base_plates.md). See [Spectral error](../theory/optimization.md#spectral-error).

---

A concrete walkthrough of one experiment setup (the DEMO vacuum vessel) is on the [Example](../example/demo_vv.md) page.

## References

- ICRP, *Conversion Coefficients for use in Radiological Protection against External Radiation*, ICRP Publication 74, Annals of the ICRP 26(3–4) (1996). [icrp.org](https://www.icrp.org/publication.asp?id=ICRP%20Publication%2074)
- ICRP, *Conversion Coefficients for Radiological Protection Quantities for External Radiation Exposures*, ICRP Publication 116, Annals of the ICRP 40(2–5) (2010). [icrp.org](https://www.icrp.org/publication.asp?id=ICRP%20Publication%20116)
