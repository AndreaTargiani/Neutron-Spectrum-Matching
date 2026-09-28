"""Shared lethargy flux conversion utilities.
"""

from __future__ import annotations
import numpy as np

def lethargy_density_to_flux(phi_lethargy, energy_edges) -> np.ndarray:
    phi = np.asarray(phi_lethargy, dtype=np.float64)
    edges = np.asarray(energy_edges, dtype=np.float64)
    if edges.shape[0] != phi.shape[0] + 1:
        raise ValueError(
            f"energy_edges length {edges.shape[0]} must be one more than the "
            f"spectrum length {phi.shape[0]}."
        )
    if np.any(edges <= 0.0):
        raise ValueError(
            "energy_edges must be strictly positive to take a lethargy (log) width."
        )
    delta_u = np.abs(np.diff(np.log(edges)))    
    return phi * delta_u


def convert_if_lethargy(phi, energy_edges, is_lethargy: bool) -> np.ndarray:
    if is_lethargy:
        return lethargy_density_to_flux(phi, energy_edges)
    return np.asarray(phi, dtype=np.float64)
