import os

import numpy as np
import openmc
from scipy.interpolate import interp1d

from config import (
    PHI_TARGET_IN_LETHARGY,
    TARGET_ENERGY_EDGES,
    TARGET_VALUES,
    WEIGHT_MODE,
    WEIGHT_MODES,
)
from config_detector import (
    DETECTOR_RESPONSE_ENERGIES,
    DETECTOR_RESPONSE_VALUES,
    build_detector_response_on_grid,
)
from lethargy_converter import convert_if_lethargy

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

DOSE_DATA_SOURCE = "icrp74"    # "icrp74" | "icrp116"
DOSE_QUANTITY = "ambient"      # "effective" | "ambient" (H*(10); icrp74 only)
DOSE_GEOMETRY = "AP"           # irradiation geometry; unused when "ambient"

# ==================================================================

# pSv*cm2 -> uSv/h. Cancels in the normalized weights; kept for the report.
_DOSE_UNIT = 1e-6 * 3600.0

EDGES = np.asarray(TARGET_ENERGY_EDGES, dtype=float)
FLUX = convert_if_lethargy(TARGET_VALUES, EDGES, PHI_TARGET_IN_LETHARGY)
MIDPOINTS = 0.5 * (EDGES[:-1] + EDGES[1:])
N_BINS = len(FLUX)

_interpolator = None
_dose_rates_cache = None


def _dose_coeff():
    """Cubic interpolant of the configured fluence-to-dose coefficients, h(E) [pSv*cm2]."""
    global _interpolator
    if _interpolator is None:
        energy, coeff = openmc.data.dose_coefficients(
            particle="neutron",
            geometry=DOSE_GEOMETRY,
            data_source=DOSE_DATA_SOURCE,
            dose_quantity=DOSE_QUANTITY,
        )
        _interpolator = interp1d(
            energy, coeff, kind="cubic", bounds_error=False, fill_value=0.0,
        )
    return _interpolator


def _dose_name():
    if DOSE_QUANTITY == "ambient":
        return "H*(10)"
    return "effective dose"


def _response_on(energies):
    """Detector response R sampled on ``energies`` [eV], or None if unconfigured."""
    if DETECTOR_RESPONSE_ENERGIES is None or DETECTOR_RESPONSE_VALUES is None:
        return None
    response, _ = build_detector_response_on_grid(
        DETECTOR_RESPONSE_ENERGIES, DETECTOR_RESPONSE_VALUES, energies,
    )
    return np.asarray(response, dtype=float)


def dose_coeff_on_grid(energies):
    """h(E) interpolated onto ``energies`` [eV]. Cubic overshoot is clipped at 0."""
    values = np.asarray(_dose_coeff()(np.asarray(energies, dtype=float)), dtype=float)
    np.maximum(values, 0.0, out=values)
    return values


def _coeff_per_bin():
    """h sampled at the bin edge where it is larger."""
    h = _dose_coeff()
    h_lo = h(EDGES[:-1])
    h_hi = h(EDGES[1:])
    edge = np.where(h_hi >= h_lo, EDGES[1:], EDGES[:-1])
    coeff = np.asarray(h(edge), dtype=float)
    np.maximum(coeff, 0.0, out=coeff)
    return coeff, edge


def _dose_rates():
    """Per-bin dose rate [uSv/h]: (true dose, detector reading or None)."""
    global _dose_rates_cache
    if _dose_rates_cache is None:
        coeff, edge = _coeff_per_bin()
        true_dose = FLUX * coeff * _DOSE_UNIT
        response = _response_on(edge)
        detector_dose = None if response is None else true_dose * response
        _dose_rates_cache = (true_dose, detector_dose)
    return _dose_rates_cache


def _raw_weights(mode):
    if mode == "uniform":
        return np.ones(N_BINS, dtype=float)
    if mode == "response":
        response = _response_on(MIDPOINTS)
        if response is None:
            raise ValueError(
                "WEIGHT_MODE='response' needs DETECTOR_RESPONSE_ENERGIES / "
                "DETECTOR_RESPONSE_VALUES in config_detector.py."
            )
        return response
    true_dose, detector_dose = _dose_rates()
    if mode == "ICRP":
        return true_dose
    if mode == "dose":
        if detector_dose is None:
            raise ValueError(
                "WEIGHT_MODE='dose' needs DETECTOR_RESPONSE_ENERGIES / "
                "DETECTOR_RESPONSE_VALUES in config_detector.py."
            )
        return detector_dose
    raise ValueError(f"Unknown weight mode {mode!r}; choose one of {WEIGHT_MODES}.")


def build_bin_weights(mode=None, mask=None):
    """Per-bin weights on the target grid, summing to 1. Zero-mask bins are dropped."""
    mode = WEIGHT_MODE if mode is None else mode
    if mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown WEIGHT_MODE {mode!r}; choose one of {WEIGHT_MODES}.")

    weights = np.asarray(_raw_weights(mode), dtype=float).copy()
    if weights.shape != (N_BINS,):
        raise ValueError(
            f"weight vector for mode {mode!r} has shape {weights.shape}, "
            f"expected {(N_BINS,)}."
        )
    if np.any(weights < 0.0):
        raise ValueError(f"weight vector for mode {mode!r} has negative entries.")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (N_BINS,):
            raise ValueError(
                f"mask shape {mask.shape} does not match weight length {(N_BINS,)}."
            )
        weights = np.where(mask, weights, 0.0)

    total = weights.sum()
    if total < 1e-300:
        raise ValueError(f"weights for mode {mode!r} sum to ~0 after masking.")
    return weights / total


def _plot_coefficients(path):
    import matplotlib.pyplot as plt

    h = _dose_coeff()
    energy, coeff = np.asarray(h.x), np.asarray(h.y)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(energy * 1e-6, coeff, color="#1f77b4", linewidth=2.4,
            label=_dose_name())
    response = _response_on(energy)
    if response is not None:
        ax.plot(energy * 1e-6, coeff * response, color="#d62728", linewidth=2.4,
                label=f"detector x {_dose_name()}")
    ax.set_xlabel("Neutron energy [MeV]", fontsize=14)
    ax.set_ylabel("Dose per unit fluence  [pSv*cm2]", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dose coefficients saved -> {path}")


def _plot_target_dose(path):
    import matplotlib.pyplot as plt

    true_dose, detector_dose = _dose_rates()
    edges_mev = EDGES * 1e-6
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.stairs(true_dose, edges_mev, linewidth=2.2, color="#1f77b4", label=_dose_name())
    if detector_dose is not None:
        ax.stairs(detector_dose, edges_mev, linewidth=2.2, color="#d62728",
                  label="detector reading")
    ax.set_xlabel("Energy [MeV]", fontsize=16)
    ax.set_ylabel("Dose rate [uSv/h]", fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=(edges_mev[0] + edges_mev[1]) / 2)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Bin dose rate saved -> {path}")


def main():
    import matplotlib
    matplotlib.use("Agg")

    here = os.path.dirname(os.path.abspath(__file__))
    _plot_coefficients(os.path.join(here, "dose_coefficients.png"))
    _plot_target_dose(os.path.join(here, "target_dose_rate.png"))

    true_dose, detector_dose = _dose_rates()
    print(f"Total dose rate from target spectrum  [{_dose_name()}]")
    print(f"True dose       : {true_dose.sum():.4g} uSv/h")
    if detector_dose is not None:
        print(f"Detector reads  : {detector_dose.sum():.4g} uSv/h")


if __name__ == "__main__":
    main()
