# `bin_weights.py` — Dose vs ICRP weighting

## Purpose

This script is **optional**. Run it after the working target spectrum is in place ([`rebin_target_spectrum.py`](rebin_target_spectrum.md), or a direct paste into `TARGET_VALUES`) and before you commit to `WEIGHT_MODE`.

`uniform` and `response` do not need this script:

- `uniform` weights every non-zero bin the same. There is nothing to compare.
- `response` weights by the detector curve \(R(E)\) alone. That curve, and the log-log fit used everywhere downstream, is already drawn by [`config_detector.py`](config_detector.md) as `detector_response_fit.png`.

The choice this script is for is **`ICRP` vs `dose`**. Both use the fluence-to-dose coefficient \(h(E)\). They differ by whether the detector response is folded in.

!!! tip "What to do, in 3 steps"
    1. Leave `DOSE_DATA_SOURCE`, `DOSE_QUANTITY`, and `DOSE_GEOMETRY` at the top of `bin_weights.py` unless you want a different coefficient.
    2. Run `python src/bin_weights.py`.
    3. Read `target_dose_rate.png` (dose and energy both on logarithmic axes). If the two stairs disagree where you care about the match, pick `dose` when the instrument reading should drive the weights, or `ICRP` when the true dose should. Set that string as `WEIGHT_MODE` in [`config.py`](config.md#per-bin-weighting).

---

## How it works

On each target bin the coefficient \(h\) is sampled at the edge where it is larger. With \(\phi\) the target group flux:

| `WEIGHT_MODE` | Weight | What it emphasises |
|---|---|---|
| `ICRP` | \(\phi \cdot h\) | Bins that carry the true dose of the target spectrum |
| `dose` | \(\phi \cdot R \cdot h\) | Bins that carry the dose the detector would report for that same spectrum |

\(R\) is the detector response from [`config_detector.py`](config_detector.md). Absolute scale does not matter: the weights are renormalised to sum to 1, and bins with zero target flux are dropped.

### Which coefficient

`DOSE_DATA_SOURCE` picks the publication. `DOSE_QUANTITY` picks the quantity. Plots and the terminal label only the quantity, H\*(10) or effective dose.

| Setting | What it is |
|---|---|
| `icrp74` + `ambient` (default) | Ambient dose equivalent H\*(10) per fluence. This is the operational quantity survey meters are calibrated against. |
| `icrp74` + `effective` | Effective dose per fluence, for the irradiation geometry in `DOSE_GEOMETRY` (default antero-posterior, `AP`). A protection quantity: how harmful the field is, not what an instrument displays. |
| `icrp116` + `effective` | The same protection quantity from the 2010 coefficients, again for `DOSE_GEOMETRY`. |

`ambient` is only defined for `icrp74`. `DOSE_GEOMETRY` is ignored for H\*(10).

`dose_coefficients.png` is \(h(E)\) itself, and \(R(E)\,h(E)\) on the same axes. `target_dose_rate.png` is the bin-by-bin dose: dose rate on a logarithmic axis, energy on a logarithmic axis, in uSv/h. The terminal prints the two integrals.

---

## Configuration

The mode the rest of the pipeline reads is in [`config.py`](config.md#per-bin-weighting):

```python
WEIGHT_MODE = "ICRP"   # "uniform" | "ICRP" | "response" | "dose"
```

The coefficient used by `ICRP` and `dose` is local to this script:

```python
DOSE_DATA_SOURCE = "icrp74"    # "icrp74" | "icrp116"
DOSE_QUANTITY = "ambient"      # "effective" | "ambient" (H*(10); icrp74 only)
DOSE_GEOMETRY = "AP"           # irradiation geometry; unused when "ambient"
```

`response` and `dose` also need `DETECTOR_RESPONSE_ENERGIES` / `DETECTOR_RESPONSE_VALUES` in [`config_detector.py`](config_detector.md). `uniform` and `ICRP` do not.

---

## Usage

```bash
python src/bin_weights.py
```

Example output:

```
  Dose coefficients saved -> src/dose_coefficients.png
  Bin dose rate saved -> src/target_dose_rate.png
Total dose rate from target spectrum  [H*(10)]
True dose       : ... uSv/h
Detector reads  : ... uSv/h
```

## Outputs

| File | Description |
|---|---|
| `dose_coefficients.png` | \(h(E)\) and \(R(E)\,h(E)\) versus energy, both axes logarithmic. Independent of the target spectrum |
| `target_dose_rate.png` | Bin-by-bin dose rate of the target, dose and energy both logarithmic: true dose (`ICRP`) and detector reading (`dose`). This is the plot to use when setting `WEIGHT_MODE` |
