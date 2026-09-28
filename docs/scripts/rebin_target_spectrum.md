# `rebin_target_spectrum.py` — Target Spectrum Rebinner

## Purpose

This script is **optional**. Use `rebin_target_spectrum.py` when you *want* a different working group structure than the one the spectrum was tabulated on. `TARGET_VALUES` must then be a flat array of length `len(TARGET_ENERGY_EDGES) - 1`.

The script reads that raw spectrum and, if its energy grid differs from `TARGET_ENERGY_EDGES`, rebins it; the result is saved to `target_spectrum_rebinned.npy`.

!!! tip "What to do, in 3 steps"
    1. In [`config.py`](config.md), paste your spectrum's edges into `RAW_TARGET_ENERGY_EDGES` and its values into `RAW_TARGET_VALUES`, set `RAW_TARGET_VALUES_ARE_LETHARGY` to match how your values are tabulated, and set `TARGET_ENERGY_EDGES` to the destination group structure.
    2. Run `python src/rebin_target_spectrum.py`.
    3. Sanity-check `target_spectrum_rebin_comparison.png`, load `target_spectrum_rebinned.npy`, paste it into `TARGET_VALUES`, and.
   
---

## How it works (physically)

Rebinning a spectrum means moving it from one set of energy bin edges to another without changing how many neutrons it represents. The subtlety is that a flux density (flux per unit energy, or per unit lethargy) isn't additive across bin boundaries — only the bin-integrated content $\Phi_i = \int_{E_i}^{E_{i+1}} \phi(E)\,dE$ (n/cm²/s "in this bin") is. So the script always converts to content first, redistributes that, then converts back:

1. **Undo the lethargy convention (if used).** A per-lethargy density $\phi_u = d\Phi/du$ (with $u = \ln(E_\text{ref}/E)$) is turned back into content via $\Phi_i = \phi_{u,i}\cdot\Delta u_i$, where $\Delta u_i = |\ln(E_{i,\text{high}}/E_{i,\text{low}})|$ is the bin's lethargy width.
2. **Treat each raw bin as flat.** Within an original bin, the flux is assumed uniform in energy: $\rho_i = \Phi_i / w_i$ (linear width $w_i$ in eV). This turns the raw spectrum into a staircase function.
3. **Redistribute by overlap.** For every destination bin, sum up how many eV of overlap it has with each original bin, weighted by that bin's density: $\Phi'_j = \sum_i \rho_i \cdot \text{overlap}(\text{bin}_i, \text{bin}_j)$. This is just re-slicing the staircase at new bin edges, so the total area under the curve (total neutron count) is exactly conserved wherever the two grids overlap.
4. **Re-apply the output convention.** The rebinned content is converted (or left alone) so that it always matches `config.PHI_TARGET_IN_LETHARGY` — see next paragraph.

The script then saves the rebinned spectrum to a `.npy` file, and saves a diagnostic plot comparing the raw and rebinned spectra.

!!! info "What 'a good rebin' actually guarantees — and what it doesn't"
    Step 3 conserves the total $\text{n/cm}^2/\text{s}$ exactly: every rebinned bin's content equals the sum of the raw content it overlaps, and the grand total over the whole range matches to floating-point precision. The script re-verifies this on every run and prints it (see [Usage](#usage) below) — if it ever reports `MISMATCH!`, something is genuinely wrong

    What this does **not** mean is that a raw bin's content and the target bin it falls into will have the same height on a plot. Bin content ($\text{n/cm}^2/\text{s}$ *per group*) is an extensive quantity, it scales with bin width. Comparing two different group structures bin-for-bin only makes sense for an intensive quantity, i.e. a density.

---

## Configuration

All configuration lives in [`config.py`](config.md), not in this script:

```python
RAW_TARGET_ENERGY_EDGES = np.array([...])   # (M+1,) eV, must be ascending
RAW_TARGET_VALUES = np.array([...])         # (M,) values, one per bin, same order
RAW_TARGET_VALUES_ARE_LETHARGY: bool = True
```

This page assumes you chose the rebin path on the [`config.py` raw-target section](config.md#raw-target-spectrum). To rebin a different spectrum, edit those three values in `config.py`, everything downstream requires no other change.

---

## Usage

```bash
python src/rebin_target_spectrum.py
```

Example output:

```
======================================================================
  TARGET SPECTRUM AUTOMATIC REBINNING
======================================================================
  Original spectrum : 614 groups  [1e-05, 1.995e+07] eV  (per-lethargy density)
  Target grid       : 175 groups  [0.01585, 1.964e+07] eV
  Flux conserved  : total n/cm^2/s  raw=1.128493E+10  rebinned=1.128493E+10  (rel. diff 1.69e-16) OK
  Output units    : per-unit-lethargy density  (config.PHI_TARGET_IN_LETHARGY = True)
  Saved array     : src/target_spectrum_rebinned.npy
  Saved plot      : src/target_spectrum_rebin_comparison.png
======================================================================
```

## Outputs

| File | Description |
|---|---|
| `target_spectrum_rebinned.npy` | Rebinned `TARGET_VALUES` on `TARGET_ENERGY_EDGES`, in the units declared by `PHI_TARGET_IN_LETHARGY` |
| `target_spectrum_rebin_comparison.png` | Raw vs. rebinned spectrum, three panels: **left** is flux per unit lethargy, **center** is flux per unit energy (n/cm²/s/eV), **right** is per-group flux (n/cm²/s). The first two are bin-width-independent densities, so the raw and rebinned curves should visually track each other; the right panel is the extensive per-group content the pipeline actually consumes |
