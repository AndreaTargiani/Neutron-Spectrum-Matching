
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    RAW_TARGET_ENERGY_EDGES,
    RAW_TARGET_VALUES,
    RAW_TARGET_VALUES_ARE_LETHARGY,
    TARGET_ENERGY_EDGES,
    PHI_TARGET_IN_LETHARGY,
)
from lethargy_converter import convert_if_lethargy

def rebin_spectrum(raw_edges, raw_content, target_edges):
    raw_edges = np.asarray(raw_edges, float)
    target_edges = np.asarray(target_edges, float)
    raw_content = np.asarray(raw_content, float)
    w = np.diff(raw_edges)
    dens = np.where(w > 0, raw_content / w, 0.0)
    lo_raw, hi_raw = raw_edges[:-1], raw_edges[1:]
    out = np.zeros(target_edges.size - 1)
    for j in range(out.size):
        lo = np.maximum(target_edges[j], lo_raw)
        hi = np.minimum(target_edges[j + 1], hi_raw)
        out[j] = np.dot(dens, np.clip(hi - lo, 0.0, None))
    return out


def grids_match(edges_a, edges_b, rtol: float = 1e-4) -> bool:
    a = np.asarray(edges_a, dtype=float)
    b = np.asarray(edges_b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=0.0)


def rebin_target_spectrum(
    raw_edges,
    raw_values,
    *,
    values_are_lethargy: bool,
    target_edges,
    output_as_lethargy: bool = True,
):
    raw_edges = np.asarray(raw_edges, dtype=float)
    n = min(len(raw_edges) - 1, len(raw_values))
    edges = raw_edges[: n + 1]
    values = np.asarray(raw_values, dtype=float)[:n]
    content = convert_if_lethargy(values, edges, values_are_lethargy)
    target_edges = np.asarray(target_edges, dtype=float)

    if grids_match(edges, target_edges):
        rebinned_content = content
        was_rebinned = False
    else:
        rebinned_content = rebin_spectrum(edges, content, target_edges)
        was_rebinned = True

    if output_as_lethargy:
        result = rebinned_content / np.abs(np.diff(np.log(target_edges)))
    else:
        result = rebinned_content
    return result, was_rebinned


def energy_range_with_nonzero_flux(edges, values):
    edges = np.asarray(edges, dtype=float)
    nz = np.flatnonzero(np.asarray(values, dtype=float) != 0.0)
    if nz.size == 0:
        return float(np.min(edges)), float(np.max(edges))
    i0, i1 = int(nz[0]), int(nz[-1])
    lo, hi = edges[i0], edges[i1 + 1]
    return float(min(lo, hi)), float(max(lo, hi))


def main():
    print("=" * 70)
    print("  TARGET SPECTRUM AUTOMATIC REBINNING")
    print("=" * 70)

    n_raw = min(len(RAW_TARGET_ENERGY_EDGES) - 1, len(RAW_TARGET_VALUES))
    n_target = len(TARGET_ENERGY_EDGES) - 1
    print(f"  Original spectrum : {n_raw} groups  "
          f"[{RAW_TARGET_ENERGY_EDGES[0]:.4g}, {RAW_TARGET_ENERGY_EDGES[n_raw]:.4g}] eV  "
          f"({'per-lethargy density' if RAW_TARGET_VALUES_ARE_LETHARGY else 'classical flux'})")
    print(f"  Target grid       : {n_target} groups  "
          f"[{TARGET_ENERGY_EDGES[0]:.4g}, {TARGET_ENERGY_EDGES[-1]:.4g}] eV")

    result, _ = rebin_target_spectrum(
        RAW_TARGET_ENERGY_EDGES,
        RAW_TARGET_VALUES,
        values_are_lethargy=RAW_TARGET_VALUES_ARE_LETHARGY,
        target_edges=TARGET_ENERGY_EDGES,
        output_as_lethargy=PHI_TARGET_IN_LETHARGY,
    )

    if RAW_TARGET_VALUES_ARE_LETHARGY != PHI_TARGET_IN_LETHARGY:
        print(f"  -> raw spectrum is given as "
              f"{'a per-lethargy density' if RAW_TARGET_VALUES_ARE_LETHARGY else 'classical flux'}, "
              f"but config.PHI_TARGET_IN_LETHARGY = {PHI_TARGET_IN_LETHARGY}: "
              "converted to match.")

    raw_edges = np.asarray(RAW_TARGET_ENERGY_EDGES, dtype=float)[:n_raw + 1]
    raw_content = convert_if_lethargy(
        np.asarray(RAW_TARGET_VALUES, dtype=float)[:n_raw], raw_edges, RAW_TARGET_VALUES_ARE_LETHARGY
    )
    target_edges = np.asarray(TARGET_ENERGY_EDGES, dtype=float)
    d_u_target = np.abs(np.diff(np.log(target_edges)))
    target_content = result * d_u_target if PHI_TARGET_IN_LETHARGY else result

    total_raw, total_target = raw_content.sum(), target_content.sum()
    rel_err = abs(total_target - total_raw) / total_raw if total_raw > 0 else float("nan")
    ok = abs(total_target - total_raw) < total_raw / 100
    print(f"  Flux conserved  : total n/cm^2/s  raw={total_raw:.6E}  "
          f"rebinned={total_target:.6E}  (rel. diff {rel_err:.2e}) "
          f"{'OK' if ok else 'MISMATCH! -- target range likely clips non-zero raw content'}")

    units = "per-unit-lethargy density" if PHI_TARGET_IN_LETHARGY else "classical n/cm^2/s group flux"
    print(f"  Output units    : {units}  (config.PHI_TARGET_IN_LETHARGY = {PHI_TARGET_IN_LETHARGY})")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(out_dir, "target_spectrum_rebinned.npy"), result)
    print(f"  Saved array     : {os.path.join(out_dir, 'target_spectrum_rebinned.npy')}")

    d_u_raw = np.abs(np.diff(np.log(raw_edges)))
    phi_leth_raw = np.divide(raw_content, d_u_raw, out=np.zeros_like(raw_content), where=d_u_raw > 0)
    if PHI_TARGET_IN_LETHARGY:
        phi_leth_target = result
        flux_target = result * d_u_target
    else:
        flux_target = result
        phi_leth_target = result / d_u_target

    dE_raw = np.abs(np.diff(raw_edges))
    density_raw = np.divide(raw_content, dE_raw, out=np.zeros_like(raw_content), where=dE_raw > 0)
    density_target = flux_target / np.diff(target_edges)

    mask_raw = phi_leth_raw != 0.0
    x_raw = raw_edges[:-1][mask_raw]
    lo_raw, hi_raw = energy_range_with_nonzero_flux(raw_edges, raw_content)
    lo_target, hi_target = energy_range_with_nonzero_flux(target_edges, flux_target)
    x_lo, x_hi = min(lo_raw, lo_target), max(hi_raw, hi_target)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    axes = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
    )
    panels = (
        (phi_leth_raw[mask_raw], phi_leth_target,
         r'Flux per unit lethargy, $\phi/\Delta u$',
         'Per-unit-lethargy comparison'),
        (density_raw[mask_raw], density_target,
         r'Flux per unit energy, $d\phi/dE$ (n/cm$^2$/s/eV)',
         'Classical (per-unit-energy) flux comparison'),
        (raw_content[mask_raw], flux_target,
         r'Neutron flux, $\phi$ (n/cm$^2$/s per group)',
         'Raw per-group content (normal units, NOT bin-width normalised)'),
    )
    for ax, (y_raw, y_target, ylabel, title) in zip(axes, panels):
        ax.step(x_raw, y_raw, where='post',
                color='tab:blue', alpha=0.6, label=f"original ({n_raw} groups)")
        ax.step(target_edges[:-1], y_target, where='post',
                color='tab:red', label=f"rebinned ({n_target} groups)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel('Energy (eV)', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title, fontsize=17)
        ax.tick_params(axis='both', which='both', labelsize=15)
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_spectrum_rebin_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved plot      : {os.path.join(out_dir, 'target_spectrum_rebin_comparison.png')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
