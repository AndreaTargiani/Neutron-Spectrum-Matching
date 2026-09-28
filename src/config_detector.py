
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import openmc


# =============================================================================
# GEOMETRY
# =============================================================================

DETECTOR_NAME = "spherical detector"
"""
DETECTOR_REGIONS = [
    dict(name="active_volume", material="SS_316", shape="sphere",
         diameter=4.0, active=True),
]
"""
DETECTOR_REGIONS = [
    dict(name="active_volume", material="air", shape="sphere",
         diameter=4.0, active=True),
]


# ============================================================================
# DETECTOR ENERGY RESPONSE
# ===========================================================================

MAX_POLY_DEGREE = 8

DETECTOR_RESPONSE_ENERGIES = np.array([
    2.0e-8, 1.5e-7, 8.0e-7, 1.2e-5, 1.5e-4, 8.0e-4, 1.5e-2,
    0.030, 0.055, 0.075, 0.100, 0.160, 0.500, 1.20,
    3.0, 8.0, 15.0, 18.0,
], dtype=np.float64) * 1e6  # MeV → eV

DETECTOR_RESPONSE_VALUES = np.array([
    7.4e-6, 9.1e-6, 8.2e-6, 7.0e-6, 5.8e-6, 6.5e-6, 1.1e-5,
    1.6e-5, 3.1e-5, 4.5e-5, 5.9e-5, 9.5e-5, 2.1e-4, 2.7e-4,
    3.0e-4, 3.3e-4, 3.6e-4, 3.9e-4,
], dtype=np.float64)

# =============================================================================
# HELPERS
# =============================================================================


# ═══════════════════════════════════════════════════════════════════════════
#  1. DETECTOR GEOMETRY & MATERIALS
# ═══════════════════════════════════════════════════════════════════════════

_VALID_SHAPES = ("sphere", "cylinder", "box")
_VALID_AXES = ("x", "y", "z")


@dataclass
class DetectorRegion:
    name: str
    material: "str | openmc.Material"
    shape: str = "cylinder"
    axis: str = "x"
    diameter: float | None = None
    height: float | None = None
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None
    active: bool = False

    def __post_init__(self) -> None:
        if self.shape not in _VALID_SHAPES:
            raise ValueError(f"Region {self.name!r}: shape must be one of {_VALID_SHAPES}, got {self.shape!r}.")
        if self.shape == "sphere":
            if self.diameter is None:
                raise ValueError(f"Region {self.name!r}: sphere shape needs 'diameter'.")
            if self.diameter <= 0:
                raise ValueError(f"Region {self.name!r}: 'diameter' must be > 0.")
        elif self.shape == "cylinder":
            if self.axis not in _VALID_AXES:
                raise ValueError(f"Region {self.name!r}: axis must be one of {_VALID_AXES}, got {self.axis!r}.")
            if self.diameter is None or self.height is None:
                raise ValueError(f"Region {self.name!r}: cylinder shape needs 'diameter' and 'height'.")
            if self.diameter <= 0 or self.height <= 0:
                raise ValueError(f"Region {self.name!r}: 'diameter' and 'height' must be > 0.")
        else:  # box
            if None in (self.dx, self.dy, self.dz):
                raise ValueError(f"Region {self.name!r}: box shape needs 'dx', 'dy' and 'dz'.")
            if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
                raise ValueError(f"Region {self.name!r}: 'dx', 'dy', 'dz' must be > 0.")

    # -- derived geometric quantities ---------------------------------------

    def half_extent(self, dim: str) -> float:
        if self.shape == "sphere":
            return self.diameter / 2.0
        if self.shape == "cylinder":
            return self.height / 2.0 if dim == self.axis else self.diameter / 2.0
        return {"x": self.dx, "y": self.dy, "z": self.dz}[dim] / 2.0

    def volume(self) -> float:
        if self.shape == "sphere":
            return (4.0 / 3.0) * math.pi * (self.diameter / 2.0) ** 3
        if self.shape == "cylinder":
            return math.pi * (self.diameter / 2.0) ** 2 * self.height
        return self.dx * self.dy * self.dz

@dataclass
class DetectorBuild:
    cells: list          # openmc.Cell, ordered outermost -> innermost
    active_cells: list   # subset of `cells` that should be tallied
    outer_region: "openmc.Region"   # the detector's full footprint 
    materials: set       # openmc.Material instances actually used


@dataclass
class DetectorGeometry:
    regions: list[DetectorRegion]
    name: str = "detector"

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("DetectorGeometry needs at least one region.")
        if not any(r.active for r in self.regions):
            raise ValueError(
                "DetectorGeometry has no 'active=True' region -- nothing "
                "would be tallied. Flag at least one region active."
            )
        outer = self.regions[0]
        for dim in ("x", "y", "z"):
            outer_half = outer.half_extent(dim)
            for inner in self.regions[1:]:
                if inner.half_extent(dim) > outer_half:
                    raise ValueError(
                        f"Region {inner.name!r} does not fit inside "
                        f"{outer.name!r} along {dim} "
                        f"({inner.half_extent(dim):.3g} > {outer_half:.3g} cm)."
                    )

    # -- bounding-box helpers (outermost region only) ------------------------

    def half_extent(self, dim: str) -> float:
        return self.regions[0].half_extent(dim)

    def active_volume(self) -> float:
        return sum(r.volume() for r in self.regions if r.active)

    def active_region_names(self) -> list[str]:
        return [r.name for r in self.regions if r.active]

    def summary(self) -> str:
        lines = [f"Detector {self.name!r}:"]
        for i, r in enumerate(self.regions):
            tag = "ACTIVE/TALLIED" if r.active else "passive"
            mat = r.material if isinstance(r.material, str) else r.material.name
            if r.shape == "sphere":
                shape_txt = f"sphere  d={r.diameter:g} cm"
            elif r.shape == "cylinder":
                shape_txt = f"cylinder  d={r.diameter:g} cm  h={r.height:g} cm  axis={r.axis}"
            else:
                shape_txt = f"box  {r.dx:g}x{r.dy:g}x{r.dz:g} cm"
            lines.append(
                f"  [{i}] {r.name!r:<16} material={mat:<10} {shape_txt:<38} "
                f"volume={r.volume():8.2f} cm^3  ({tag})"
            )
        lines.append(
            f"  footprint (outer region): "
            f"{2.0 * self.half_extent('x'):.2f} x {2.0 * self.half_extent('y'):.2f} x {2.0 * self.half_extent('z'):.2f} cm  (X x Y x Z)"
        )
        lines.append(f"  total active volume: {self.active_volume():.3f} cm^3")
        return "\n".join(lines)

    # -- OpenMC construction --------------------------------------------------

    def _region_solid(self, region: DetectorRegion, center: tuple[float, float, float]) -> "openmc.Region":
        cx, cy, cz = center
        if region.shape == "box":
            surf = openmc.model.RectangularParallelepiped(
                cx - region.dx / 2.0, cx + region.dx / 2.0,
                cy - region.dy / 2.0, cy + region.dy / 2.0,
                cz - region.dz / 2.0, cz + region.dz / 2.0,
            )
            return -surf

        if region.shape == "sphere":
            sph = openmc.Sphere(x0=cx, y0=cy, z0=cz, r=region.diameter / 2.0)
            return -sph

        radius = region.diameter / 2.0
        half = region.height / 2.0
        if region.axis == "x":
            cyl = openmc.XCylinder(y0=cy, z0=cz, r=radius)
            lo, hi = openmc.XPlane(x0=cx - half), openmc.XPlane(x0=cx + half)
        elif region.axis == "y":
            cyl = openmc.YCylinder(x0=cx, z0=cz, r=radius)
            lo, hi = openmc.YPlane(y0=cy - half), openmc.YPlane(y0=cy + half)
        else:  # "z"
            cyl = openmc.ZCylinder(x0=cx, y0=cy, r=radius)
            lo, hi = openmc.ZPlane(z0=cz - half), openmc.ZPlane(z0=cz + half)
        return -cyl & +lo & -hi

    def build_cells(
        self,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        materials: dict[str, "openmc.Material"] | None = None,
    ) -> DetectorBuild:
    
        materials = detector_materials() if materials is None else materials

        solids = [self._region_solid(r, center) for r in self.regions]
        cells, used_materials = [], set()
        for i, region in enumerate(self.regions):
            reg = solids[i]
            if i + 1 < len(self.regions):
                reg = reg & ~solids[i + 1]

            if isinstance(region.material, openmc.Material):
                base_mat = region.material
            else:
                try:
                    base_mat = materials[region.material]
                except KeyError as exc:
                    raise KeyError(
                        f"Detector region {region.name!r} uses material "
                        f"{region.material!r}, which is not in the detector "
                        f"material lookup. Available: {sorted(materials)}"
                    ) from exc
            mat = base_mat.clone()
            mat.name = f"detector_{region.name}"
            used_materials.add(mat)

            cells.append(openmc.Cell(fill=mat, region=reg, name=f"detector_{region.name}"))

        active_cells = [c for c, r in zip(cells, self.regions) if r.active]
        return DetectorBuild(
            cells=cells, active_cells=active_cells,
            outer_region=solids[0], materials=used_materials,
        )

def detector_materials() -> dict[str, openmc.Material]:
    from config_materials import air_material, build_material_library

    lib = dict(build_material_library())
    lib["air"] = air_material()
    return lib

DETECTOR_GEOMETRY = DetectorGeometry(
    name=DETECTOR_NAME,
    regions=[DetectorRegion(**r) for r in DETECTOR_REGIONS],
)

# ═══════════════════════════════════════════════════════════════════════════
#  2. DETECTOR ENERGY RESPONSE FITTING
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DetectorFitInfo:
    best_degree:   int
    loo_rmse:      np.ndarray    # one value per degree tried
    degrees_tried: np.ndarray
    coeffs:        np.ndarray    # polynomial coefficients in log-log space
    u_min:         float         # log(E_min) — left clamp boundary
    u_max:         float         # log(E_max) — right clamp boundary
    n_points:      int
    r_squared:     float         # R² on training data (log-log space)


def _loo_rmse(u: np.ndarray, v: np.ndarray, degree: int) -> float:
    n = len(u)
    if degree >= n:
        return np.inf
    X = np.vander(u, degree + 1, increasing=False)
    try:
        Q, _ = np.linalg.qr(X)
        h = np.sum(Q ** 2, axis=1)               # hat-matrix diagonal
        coeffs = np.linalg.lstsq(X, v, rcond=None)[0]
        resid  = v - X @ coeffs
        denom  = 1.0 - h
        if not np.all(denom > 1e-10):
            return np.inf
        return float(np.sqrt(np.mean((resid / denom) ** 2)))
    except np.linalg.LinAlgError:
        return np.inf


def build_detector_response_on_grid(
    energies: np.ndarray,
    values: np.ndarray,
    energy_mid: np.ndarray,
    max_degree: int = MAX_POLY_DEGREE,
) -> tuple[np.ndarray, DetectorFitInfo]:
   
    energies = np.asarray(energies, dtype=np.float64)
    values   = np.asarray(values,   dtype=np.float64)
    e_mid    = np.asarray(energy_mid, dtype=np.float64)

    if energies.ndim != 1 or values.ndim != 1:
        raise ValueError("energies and values must be 1-D arrays.")
    if len(energies) != len(values):
        raise ValueError(
            f"energies (len {len(energies)}) and values (len {len(values)}) "
            "must have the same length."
        )
    if len(energies) < 2:
        raise ValueError("At least 2 (energy, response) pairs are required.")
    if np.any(energies <= 0):
        raise ValueError("All energies must be strictly positive (eV).")
    if np.any(values < 0):
        raise ValueError("Detector response values must be non-negative.")

    order = np.argsort(energies)
    e_src = energies[order]
    v_src = values[order]

    eps = 1e-300
    u   = np.log(e_src)
    v   = np.log(np.maximum(v_src, eps))

    # ── degree selection ──────────────────────────────────────────────────
    max_deg   = min(max_degree, len(energies) - 1)
    degrees   = np.arange(1, max_deg + 1)
    loo_rmses = np.array([_loo_rmse(u, v, d) for d in degrees])

    best_degree = int(degrees[np.argmin(loo_rmses)])

    # ── fit on full data ──────────────────────────────────────────────────
    X_full = np.vander(u, best_degree + 1, increasing=False)
    coeffs = np.linalg.lstsq(X_full, v, rcond=None)[0]
    v_hat  = X_full @ coeffs
    ss_res = float(np.sum((v - v_hat) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ── evaluate at bin midpoints (clamped) ───────────────────────────────
    u_min, u_max = u[0], u[-1]
    u_mid        = np.clip(np.log(e_mid), u_min, u_max)
    r_grid       = np.exp(np.polyval(coeffs, u_mid))
    np.maximum(r_grid, 0.0, out=r_grid)

    fit_info = DetectorFitInfo(
        best_degree   = best_degree,
        loo_rmse      = loo_rmses,
        degrees_tried = degrees,
        coeffs        = coeffs,
        u_min         = u_min,
        u_max         = u_max,
        n_points      = len(e_src),
        r_squared     = r2,
    )
    return r_grid, fit_info


def plot_detector_response_fit(
    energies: np.ndarray,
    values: np.ndarray,
    energy_mid: np.ndarray,
    response_on_grid: np.ndarray,
    fit_info: DetectorFitInfo,
    energy_edges: np.ndarray | None = None,
    savefig: str = "detector_response_fit.png",
) -> None:
  
    energies = np.asarray(energies,         dtype=np.float64)
    values   = np.asarray(values,           dtype=np.float64)
    e_mid    = np.asarray(energy_mid,       dtype=np.float64)
    r_grid   = np.asarray(response_on_grid, dtype=np.float64)

    # Dense curve for plotting
    e_lo   = max(min(energies.min(), e_mid.min()) * 0.3, 1e-2)
    e_hi   = max(energies.max(), e_mid.max()) * 3.0
    e_plot = np.geomspace(e_lo, e_hi, 800)
    u_plot = np.clip(np.log(e_plot), fit_info.u_min, fit_info.u_max)
    r_plot = np.exp(np.polyval(fit_info.coeffs, u_plot))

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax_top, ax_cv) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.38},
    )

    # ── top panel ─────────────────────────────────────────────────────────
    if energy_edges is not None:
        e_edges = np.asarray(energy_edges, dtype=np.float64)
        for lo, hi, rv in zip(e_edges[:-1], e_edges[1:], r_grid):
            lo = max(lo, 1e-3)
            ax_top.fill_betweenx([0, rv], lo, hi,
                                  color="steelblue", alpha=0.18, linewidth=0)
            ax_top.plot([lo, hi], [rv, rv], color="steelblue",
                        linewidth=0.7, alpha=0.6)

    # Polynomial curve
    ax_top.plot(e_plot, r_plot, color="steelblue", linewidth=1.8,
                label=f"Poly fit  degree={fit_info.best_degree}  R²={fit_info.r_squared:.4f}")

    # Bin midpoint samples
    ax_top.scatter(e_mid, r_grid, color="steelblue", s=30, zorder=4,
                   linewidths=0, label="Bin-midpoint values")

    # Raw user data
    ax_top.scatter(energies, values, color="darkorange", s=55, zorder=5,
                   marker="D", linewidths=0, label="User input")

    # Extrapolation shading
    ax_top.axvspan(e_lo, energies.min(), color="grey", alpha=0.08, linewidth=0)
    ax_top.axvspan(energies.max(), e_hi,  color="grey", alpha=0.08, linewidth=0)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    r_all = np.concatenate([values[values > 0], r_grid[r_grid > 0]])
    ax_top.set_ylim(r_all.min() * 0.05, r_all.max() * 8)
    ax_top.set_xlim(e_lo, e_hi)
    ax_top.set_ylabel("Response [arb. units]")
    ax_top.set_xlabel("Energy [eV]")
    ax_top.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax_top.legend(fontsize=8.5, loc="upper left")
    ax_top.set_title(
        f"Detector response fit — log-log polynomial  "
        f"(degree {fit_info.best_degree}, LOO-CV,  "
        f"{fit_info.n_points} pts → {len(e_mid)} bins)",
        fontsize=9,
    )
    ax_top.grid(True, which="major", linewidth=0.5, linestyle="--", color="#cccccc")
    ax_top.grid(True, which="minor", linewidth=0.25, linestyle=":", color="#dddddd")

    # ── bottom panel: CV curve ─────────────────────────────────────────────
    deg   = fit_info.degrees_tried
    rmse  = fit_info.loo_rmse
    fin   = np.isfinite(rmse)

    ax_cv.plot(deg[fin], rmse[fin], color="#555555", linewidth=1.4,
               marker="o", markersize=4, label="LOO-CV RMSE")

    best_d = fit_info.best_degree
    ax_cv.scatter([best_d], [fit_info.loo_rmse[best_d - 1]],
                  color="darkorange", s=60, zorder=5, label=f"Selected (deg {best_d})")
    ax_cv.axvline(best_d, color="darkorange", linewidth=0.9, linestyle="--", alpha=0.5)

    ax_cv.set_xlabel("Polynomial degree")
    ax_cv.set_ylabel("LOO-CV RMSE")
    ax_cv.set_xticks(deg)
    ax_cv.tick_params(labelsize=8)
    ax_cv.legend(fontsize=8.5, loc="upper right")
    ax_cv.grid(True, which="major", linewidth=0.5, linestyle="--", color="#cccccc")

    fig.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Detector response fit saved → {savefig}")


if __name__ == "__main__":
    print(DETECTOR_GEOMETRY.summary())
    if DETECTOR_RESPONSE_ENERGIES is None or DETECTOR_RESPONSE_VALUES is None:
        print("  No energy-response curve configured.")
    else:
        from config import TARGET_ENERGY_EDGES
        edges = np.asarray(TARGET_ENERGY_EDGES, dtype=np.float64)
        energy_mid = 0.5 * (edges[:-1] + edges[1:])
        r_grid, fit_info = build_detector_response_on_grid(
            DETECTOR_RESPONSE_ENERGIES,
            DETECTOR_RESPONSE_VALUES,
            energy_mid,
        )
        plot_detector_response_fit(
            DETECTOR_RESPONSE_ENERGIES,
            DETECTOR_RESPONSE_VALUES,
            energy_mid,
            r_grid,
            fit_info,
            edges,
            savefig="detector_response_fit.png",
        )
