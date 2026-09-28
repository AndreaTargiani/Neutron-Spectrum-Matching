from __future__ import annotations

import contextlib
import glob
import io
import math
import os
import shutil
import subprocess
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openmc

import config
from config import (
    LOG_EPS, SOURCE_EDGES, SOURCE_VALUES, TARGET_ENERGY_EDGES,
    TARGET_VALUES, PHI_TARGET_IN_LETHARGY, WEIGHT_MODE,
)
from config_detector import (
    DETECTOR_GEOMETRY,
    DETECTOR_RESPONSE_ENERGIES,
    DETECTOR_RESPONSE_VALUES,
    build_detector_response_on_grid,
    detector_materials,
)
from bin_weights import build_bin_weights, dose_coeff_on_grid
from config_materials import air_material, build_material_library
from lethargy_converter import convert_if_lethargy

warnings.filterwarnings('ignore', message='.*IDWarning.*')
warnings.filterwarnings('ignore', message='.*Another Filter instance already exists.*')

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

HPC: bool = False

### HPC / SLURM settings 
MPI_TASKS_HPC: int = 8
OMP_THREADS_RR_HPC: int = 192              # Phase 1: all cores, no MPI
OMP_THREADS_CE_HPC: int = 24               # Phase 2: OMP threads per MPI rank

#### Local settings
THREADS: int = (os.cpu_count() // 2) - 1   # Phase 1 local OpenMP threads
MPI_TASKS_LOCAL: int = 2
OMP_THREADS_CE_LOCAL: int = 4              # uniform per rank (shared_secondary_bank)

# Geometry (detector from config_detector.py)
X_START: float = 0.1                       # source -> first-plate distance [cm]
MATERIALS   = ['cast_iron', 'HDPE']
THICKNESSES = [66.50, 1.20]
PLATE_DIM   = 30                           # square transverse size [cm]
Y_DIM = Z_DIM = PLATE_DIM
AIR_SPHERE_RADIUS: float = 300.0           # vacuum outer boundary [cm]

# Run sizes
WW_BATCHES, WW_INACTIVE_BATCHES, WW_PARTICLES = 100, 50, 50_000
MGXS_PARTICLES = 30_000
CE_BATCHES, CE_PARTICLES = 100, 150_000
MAX_LOWER_BOUND_RATIO, MAX_SPLIT = 1.0, 1000
USE_ANGLE_BIAS = True
SOURCE_STRENGTH = 5e9

# WW mesh: plate cell counts = ceil(span / resolution)
MESH_X_BEFORE_SOURCE = 10.0
SOURCE_X_CELLS = 3
AIR_CELLS = 3                              # coarse air outside plates (X, Y, Z)
PLATE_X_RESOLUTION = 1.5                   # cm/cell along X inside plates
PLATE_YZ_RESOLUTION = 3.0                  # cm/cell along Y and Z inside plates

# Top edge must cover the source max energy (not just TARGET_ENERGY_EDGES[-1])
WW_ENERGY_EDGES: list[float] = np.geomspace(
    TARGET_ENERGY_EDGES[0],
    max(TARGET_ENERGY_EDGES[-1], float(np.max(SOURCE_EDGES)) * (1 + 1e-9)),
    12,
).tolist()

PLOT_WW_BIN = 5                            # 1-indexed int or vector; saved under WW_DIR
PLOT_GEOMETRY = True
WW_DIR, CE_DIR = 'ww_generation', 'ce_run'

# ==================================================================

config.configure_openmc_cross_sections(HPC)

ENERGY_EDGES: list[float] = TARGET_ENERGY_EDGES
N_BINS: int = len(ENERGY_EDGES) - 1


# -- geometry / model ----------------------------------------------

def thicknesses_to_x_bounds(x_start: float, thicknesses: list[float]) -> list[float]:
    """Starting X and plate thicknesses, strictly increasing boundary coordinates."""
    x_bounds, x = [x_start], x_start
    for t in thicknesses:
        if t <= 0:
            raise ValueError(f"Plate thickness must be positive; got {t} cm.")
        x += t
        x_bounds.append(x)
    return x_bounds


def source_angle_bias(x_start: float, y_dim: float) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear mu = cos theta bias: unit weight toward the plate, 1% backward."""
    theta_edge = math.atan((y_dim / 2.0) / x_start)
    theta_30 = min(theta_edge + math.radians(30), math.pi)
    theta_45 = min(theta_edge + math.radians(45), math.pi)

    data = [
        (math.cos(math.pi), 0.01),
        (math.cos(theta_45), 0.01),
        (math.cos(theta_30), 0.1),
        (math.cos(theta_edge), 1.0),
        (math.cos(0.0), 1.0),
    ]
    data.sort(key=lambda pair: pair[0])

    unique = [data[0]]
    for mu, p in data[1:]:
        if abs(mu - unique[-1][0]) > 1e-5:
            unique.append((mu, p))

    mu_x = np.array([pair[0] for pair in unique])
    mu_p = np.array([pair[1] for pair in unique])
    return mu_x, mu_p


def build_source_energy(is_random_ray: bool):
    """OpenMC energy distribution from config.SOURCE_EDGES / SOURCE_VALUES."""
    edges = np.asarray(SOURCE_EDGES, dtype=float)
    content = np.asarray(SOURCE_VALUES, dtype=float)
    content = content / content.sum()
    if is_random_ray:
        return openmc.stats.Discrete(0.5 * (edges[:-1] + edges[1:]), content)

    density = content / np.diff(edges)
    return openmc.stats.Tabular(edges, np.append(density, 0.0), interpolation='histogram')


def build_model(
    x_bounds: list[float],
    materials: list[str],
    *,
    particles: int,
    batches: int,
    use_angle_bias: bool = False,
    mu_x: np.ndarray | None = None,
    mu_bias_probs: np.ndarray | None = None,
    is_random_ray: bool = False,
    y_dim: float | None = None,
    z_dim: float | None = None,
) -> openmc.Model:
    """N-layer plate stack + detector, bounded by a vacuum sphere."""
    openmc.reset_auto_ids()

    y_dim = Y_DIM if y_dim is None else float(y_dim)
    z_dim = Z_DIM if z_dim is None else float(z_dim)

    n_plates = len(materials)
    if len(x_bounds) != n_plates + 1:
        raise ValueError(f"Need {n_plates + 1} x_bounds for {n_plates} plates; got {len(x_bounds)}.")
    if not all(x_bounds[i] < x_bounds[i + 1] for i in range(n_plates)):
        raise ValueError(f"X bounds must be strictly increasing; got {x_bounds}.")
    if x_bounds[-1] >= AIR_SPHERE_RADIUS:
        raise ValueError(
            f"Outermost X boundary {x_bounds[-1]:.2f} cm must be < "
            f"AIR_SPHERE_RADIUS ({AIR_SPHERE_RADIUS} cm)."
        )

    lib = build_material_library()
    air = air_material()
    unknown = [m for m in materials if m not in lib]
    if unknown:
        raise ValueError(f"Unknown material(s) {unknown}. Available: {sorted(lib)}.")
    fills = [lib[m] for m in materials]

    outer = openmc.Sphere(r=AIR_SPHERE_RADIUS, boundary_type='vacuum')

    plate_cells, combined_region = [], None
    for i, fill in enumerate(fills):
        region = openmc.model.RectangularParallelepiped(
            x_bounds[i], x_bounds[i + 1],
            -y_dim / 2.0, y_dim / 2.0,
            -z_dim / 2.0, z_dim / 2.0,
        )
        combined_region = -region if combined_region is None else combined_region | (-region)
        plate_cells.append(openmc.Cell(fill=fill, region=-region, name=f'plate_{i}_{fill.name}'))

    x_outer = x_bounds[-1]
    cx = x_outer + DETECTOR_GEOMETRY.half_extent('x')
    detector_build = DETECTOR_GEOMETRY.build_cells(
        center=(cx, 0.0, 0.0), materials=detector_materials(),
    )

    air_cell = openmc.Cell(
        fill=air,
        region=~combined_region & -outer & ~detector_build.outer_region,
        name='air',
    )
    geometry = openmc.Geometry([*plate_cells, air_cell, *detector_build.cells])
    all_materials = openmc.Materials(set(fills) | {air} | detector_build.materials)

    if use_angle_bias:
        if mu_x is None or mu_bias_probs is None:
            mu_x, mu_bias_probs = source_angle_bias(x_bounds[0], y_dim)
        mu_dist = openmc.stats.Uniform(-1.0, 1.0)
        mu_dist.bias = openmc.stats.Tabular(mu_x, mu_bias_probs, interpolation='linear-linear')
        angle = openmc.stats.PolarAzimuthal(
            mu=mu_dist,
            phi=openmc.stats.Uniform(0.0, 2 * math.pi),
            reference_uvw=(1.0, 0.0, 0.0),
            reference_vwu=(0.0, 0.0, 1.0),
        )
    else:
        angle = openmc.stats.Isotropic()

    source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        angle=angle,
        energy=build_source_energy(is_random_ray),
    )

    settings = openmc.Settings(
        batches=batches,
        particles=particles,
        run_mode='fixed source',
        source=source,
    )
    tally = openmc.Tally(tally_id=6, name='spectrum')
    tally.higher_moments = True
    tally.estimator = 'tracklength'
    tally.filters = [
        openmc.CellFilter(detector_build.active_cells),
        openmc.ParticleFilter('neutron'),
        openmc.EnergyFilter(ENERGY_EDGES),
    ]
    tally.scores = ['flux']

    return openmc.Model(geometry, all_materials, settings, openmc.Tallies([tally]))


# -- FW-CADIS ------------------------------------------------------

def _run_quietly(fn, *, output: bool) -> None:
    if output:
        fn()
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        fn()


def _n_cells_for_span(span: float, resolution: float) -> int:
    if resolution <= 0.0:
        raise ValueError(f"Mesh resolution must be > 0; got {resolution} cm.")
    if span <= 0.0:
        raise ValueError(f"Mesh span must be > 0; got {span} cm.")
    return max(1, int(math.ceil(span / resolution - 1e-12)))


def _rounded_linspace(start: float, stop: float, n_cells: int) -> list[float]:
    return [round(float(v), 8) for v in np.linspace(start, stop, n_cells + 1)]


def _transverse_mesh_grid(
    half_box: float, plate_cells: int, air_cells: int, outer: float,
) -> list[float]:
    pts: set[float] = set()
    pts.update(_rounded_linspace(-outer, -half_box, air_cells))
    pts.update(_rounded_linspace(-half_box, half_box, plate_cells))
    pts.update(_rounded_linspace(half_box, outer, air_cells))
    return sorted(pts)


def _weight_window_mesh(
    x_bounds: list[float], y_dim: float, z_dim: float,
) -> tuple[openmc.RectilinearMesh, list[float], list[float], list[float]]:
    """Fine inside plates, coarse in air; one WW cell across the detector X-span."""
    x_outer = x_bounds[-1]
    x_det_end = x_outer + 2.0 * DETECTOR_GEOMETRY.half_extent('x')
    x_fine_start = -MESH_X_BEFORE_SOURCE

    x_pts: set[float] = set()
    x_pts.update(_rounded_linspace(-AIR_SPHERE_RADIUS, x_fine_start, AIR_CELLS))
    x_pts.update(_rounded_linspace(x_fine_start, x_bounds[0], SOURCE_X_CELLS))
    for i in range(len(x_bounds) - 1):
        n_x = _n_cells_for_span(float(x_bounds[i + 1] - x_bounds[i]), PLATE_X_RESOLUTION)
        x_pts.update(_rounded_linspace(x_bounds[i], x_bounds[i + 1], n_x))
    x_pts.add(round(float(x_det_end), 8))
    x_pts.update(_rounded_linspace(x_det_end, AIR_SPHERE_RADIUS, AIR_CELLS))

    x_grid = sorted(v for v in x_pts if not (x_outer + 1e-5 < v < x_det_end - 1e-5))
    y_grid = _transverse_mesh_grid(
        y_dim / 2.0, _n_cells_for_span(y_dim, PLATE_YZ_RESOLUTION),
        AIR_CELLS, AIR_SPHERE_RADIUS,
    )
    z_grid = _transverse_mesh_grid(
        z_dim / 2.0, _n_cells_for_span(z_dim, PLATE_YZ_RESOLUTION),
        AIR_CELLS, AIR_SPHERE_RADIUS,
    )

    mesh = openmc.RectilinearMesh()
    mesh.x_grid = x_grid
    mesh.y_grid = y_grid
    mesh.z_grid = z_grid
    return mesh, x_grid, y_grid, z_grid


def _mgxs_matches_energy_edges(mgxs_path: str, edges: list[float]) -> bool:
    if not os.path.isfile(mgxs_path):
        return False
    try:
        import h5py
        with h5py.File(mgxs_path, 'r') as f:
            stored = np.asarray(f.attrs['group structure'], dtype=float)
    except Exception:
        return False
    want = np.asarray(edges, dtype=float)
    return stored.shape == want.shape and np.allclose(stored, want, rtol=0.0, atol=0.0)


def _detector_active_cells(model: openmc.Model) -> list:
    active_names = {f'detector_{n}' for n in DETECTOR_GEOMETRY.active_region_names()}
    cells = [c for c in model.geometry.get_all_cells().values() if c.name in active_names]
    if not cells:
        raise RuntimeError(
            "No active detector cells found for the adjoint source "
            f"(looked for names {sorted(active_names)})."
        )
    return cells


def _detector_response_on_midpoints(midpoints: np.ndarray) -> np.ndarray | None:
    if DETECTOR_RESPONSE_ENERGIES is None or DETECTOR_RESPONSE_VALUES is None:
        return None
    strengths, _ = build_detector_response_on_grid(
        DETECTOR_RESPONSE_ENERGIES, DETECTOR_RESPONSE_VALUES, midpoints,
    )
    return np.asarray(strengths, dtype=float)


def _adjoint_energy_strengths(midpoints: np.ndarray) -> tuple[np.ndarray, str]:
    """Group strengths from config.WEIGHT_MODE: ICRP / response / dose / uniform."""
    mode = str(WEIGHT_MODE).strip().lower()
    if mode == 'icrp':
        strengths = dose_coeff_on_grid(midpoints)
        label = 'ICRP flux-to-dose coefficients'
    elif mode == 'response':
        strengths = _detector_response_on_midpoints(midpoints)
        if strengths is None:
            strengths = np.ones(len(midpoints), dtype=float)
            label = 'uniform (no detector response configured)'
        else:
            label = 'detector response'
    elif mode == 'dose':
        response = _detector_response_on_midpoints(midpoints)
        if response is None:
            raise ValueError(
                "WEIGHT_MODE='dose' needs a detector response, but "
                'DETECTOR_RESPONSE_ENERGIES / DETECTOR_RESPONSE_VALUES are None '
                'in config_detector.py.'
            )
        strengths = dose_coeff_on_grid(midpoints) * response
        label = 'ICRP h(E) * detector response'
    else:
        strengths = np.ones(len(midpoints), dtype=float)
        label = f'uniform (WEIGHT_MODE={WEIGHT_MODE!r})'

    strengths = np.asarray(strengths, dtype=float)
    np.maximum(strengths, 0.0, out=strengths)
    if not np.any(strengths > 0.0):
        raise ValueError(f'Adjoint source strengths for {label} are all non-positive.')
    return strengths, label


def build_detector_adjoint_source(
    detector_cells: list,
    energy_edges: list[float] | np.ndarray,
    detector_center: tuple[float, float, float],
) -> openmc.IndependentSource:
    """Local adjoint source in the active detector cells (energy from WEIGHT_MODE)."""
    edges = np.asarray(energy_edges, dtype=float)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    strengths, _ = _adjoint_energy_strengths(midpoints)

    hx = DETECTOR_GEOMETRY.half_extent('x')
    hy = DETECTOR_GEOMETRY.half_extent('y')
    hz = DETECTOR_GEOMETRY.half_extent('z')
    cx, cy, cz = detector_center
    return openmc.IndependentSource(
        energy=openmc.stats.Discrete(x=midpoints, p=strengths),
        space=openmc.stats.Box(
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz + hz],
            only_fissionable=False,
        ),
        constraints={'domains': detector_cells},
        particle='neutron',
    )


@contextlib.contextmanager
def _openmc_mpi_env(*, singleton: bool):
    """Hide SLURM PMI from this OpenMC launch, then restore the job environment.

    A bare ``openmc`` (Phase 1) is one OpenMP process, so Open MPI must init as
    a singleton. ``mpirun`` (Phase 2) keeps its own ranks, but must not see the
    sbatch task layout or it tries to use SLURM PMI.
    """
    saved = {
        key: os.environ.pop(key)
        for key in list(os.environ)
        if key.startswith(('PMI_', 'PMIX_', 'SLURM_'))
    }
    if not saved:
        yield
        return
    previous_ess = os.environ.get('OMPI_MCA_ess')
    try:
        if singleton:
            os.environ['OMPI_MCA_ess'] = 'singleton'
        yield
    finally:
        if previous_ess is None:
            os.environ.pop('OMPI_MCA_ess', None)
        else:
            os.environ['OMPI_MCA_ess'] = previous_ess
        os.environ.update(saved)


def generate_weight_windows(
    x_bounds: list[float],
    materials: list[str],
    ww_dir: str,
    *,
    rr_batches: int = 100,
    rr_inactive_batches: int = 50,
    rr_particles: int = 15_000,
    mgxs_particles: int = 10_000,
    output: bool = True,
    mgxs_path: str | None = None,
    y_dim: float | None = None,
    z_dim: float | None = None,
    ww_energy_edges: list[float] | None = None,
) -> tuple:
    """Phase 1: MGXS + adjoint random-ray solve -> FW-CADIS weight windows.

    Local adjoint source in the detector (WEIGHT_MODE group strengths), not a
    forward-weighted 1/phi target. Existing mgxs.h5 is reused only when its group
    structure matches ``ww_energy_edges`` (defaults to WW_ENERGY_EDGES).
    """
    os.makedirs(ww_dir, exist_ok=True)

    y_dim = Y_DIM if y_dim is None else float(y_dim)
    z_dim = Z_DIM if z_dim is None else float(z_dim)

    ww_energy_edges = list(WW_ENERGY_EDGES) if ww_energy_edges is None else list(ww_energy_edges)
    if len(ww_energy_edges) < 2:
        raise ValueError('ww_energy_edges must contain at least 2 edges.')

    if mgxs_path is None:
        _mgxs_path = os.path.abspath(os.path.join(ww_dir, 'mgxs.h5'))
    else:
        _mgxs_path = os.path.abspath(mgxs_path)
        mgxs_dir = os.path.dirname(_mgxs_path)
        if mgxs_dir:
            os.makedirs(mgxs_dir, exist_ok=True)

    skip_mgxs = _mgxs_matches_energy_edges(_mgxs_path, ww_energy_edges)
    if output:
        if skip_mgxs:
            print(f"  [mgxs] Reusing '{_mgxs_path}' (group structure matches).")
        elif os.path.isfile(_mgxs_path):
            print(
                f"  [mgxs] Existing '{_mgxs_path}' has a different group structure -- regenerating"
            )
        else:
            print(f"  [mgxs] Generating MGXS library -> '{_mgxs_path}'.")

    model = build_model(
        x_bounds, materials,
        batches=rr_batches,
        particles=1,
        use_angle_bias=False,
        is_random_ray=True,
        y_dim=y_dim,
        z_dim=z_dim,
    )

    with _openmc_mpi_env(singleton=True):
        _run_quietly(lambda: model.convert_to_multigroup(
            method='stochastic_slab',
            groups=ww_energy_edges,
            particles=mgxs_particles,
            overwrite_mgxs_library=not skip_mgxs,
            mgxs_path=_mgxs_path,
        ), output=output)
    _run_quietly(lambda: model.convert_to_random_ray(), output=output)

    mesh, x_grid, y_grid, z_grid = _weight_window_mesh(x_bounds, y_dim, z_dim)

    model.settings.random_ray['volume_estimator'] = 'naive'
    model.settings.random_ray['source_shape'] = 'flat'
    model.settings.random_ray['source_region_meshes'] = [
        (mesh, [model.geometry.root_universe])
    ]
    model.settings.batches = rr_batches
    model.settings.inactive = rr_inactive_batches
    model.settings.particles = rr_particles

    det_cx = float(x_bounds[-1]) + DETECTOR_GEOMETRY.half_extent('x')
    model.settings.random_ray['adjoint'] = True
    model.settings.random_ray['adjoint_source'] = build_detector_adjoint_source(
        _detector_active_cells(model),
        ww_energy_edges,
        detector_center=(det_cx, 0.0, 0.0),
    )

    model.settings.weight_window_generators = openmc.WeightWindowGenerator(
        method='fw_cadis',
        mesh=mesh,
        energy_bounds=ww_energy_edges,
        max_realizations=rr_batches - rr_inactive_batches,
    )

    _run_openmc(model, ww_dir, output=output, random_ray=True)

    ww_path = os.path.abspath(os.path.join(ww_dir, 'weight_windows.h5'))
    if not os.path.isfile(ww_path):
        raise RuntimeError(f"FW-CADIS run finished but '{ww_path}' was not produced.")
    return ww_path, x_grid, y_grid, z_grid


def run_with_weight_windows(
    x_bounds: list[float],
    materials: list[str],
    ww_path: str,
    ce_dir: str,
    *,
    batches: int = 25,
    particles: int = 20_000,
    use_angle_bias: bool = False,
    max_lower_bound_ratio: float = 1.0,
    max_split: int = 10,
    output: bool = True,
    timeout: float | None = None,
    y_dim: float | None = None,
    z_dim: float | None = None,
) -> str:
    """Phase 2: CE production run biased by FW-CADIS weight windows.

    ``y_dim`` / ``z_dim`` must match the generate_weight_windows call that
    produced ``ww_path``.
    """
    model = build_model(
        x_bounds, materials,
        batches=batches, particles=particles,
        use_angle_bias=use_angle_bias,
        y_dim=y_dim,
        z_dim=z_dim,
    )
    if hasattr(openmc, 'WeightWindowsList'):
        weight_windows = openmc.WeightWindowsList.from_hdf5(ww_path)
    else:
        weight_windows = list(openmc.hdf5_to_wws(ww_path))
    for ww in weight_windows:
        ww.max_lower_bound_ratio = max_lower_bound_ratio
        ww.max_split = max_split

    model.settings.weight_windows = weight_windows
    model.settings.weight_windows_on = True
    model.settings.survival_biasing = False
    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.shared_secondary_bank = True

    return _run_openmc(model, ce_dir, output=output, timeout=timeout)


class OpenMCTimeout(RuntimeError):
    """Raised when an OpenMC run exceeds its allowed wall-time budget."""


def _run_openmc(
    model: openmc.Model,
    cwd: str,
    *,
    output: bool = True,
    timeout: float | None = None,
    random_ray: bool = False,
) -> str:
    """Local/HPC: Phase 1 = OpenMP only; Phase 2 = MPI + OpenMP."""
    os.makedirs(cwd, exist_ok=True)

    if HPC:
        if random_ray:
            n_threads, mpi_args = OMP_THREADS_RR_HPC, None
        else:
            n_threads = OMP_THREADS_CE_HPC
            mpi_args = [
                'mpirun', '-np', str(MPI_TASKS_HPC),
                '--bind-to', 'none',
                '--mca', 'btl', 'self,vader',
                '--mca', 'btl_vader_single_copy_mechanism', 'none',
            ]
    elif random_ray:
        n_threads, mpi_args = THREADS, None
    else:
        n_threads = OMP_THREADS_CE_LOCAL
        mpi_args = [
            'mpirun', '-np', str(MPI_TASKS_LOCAL),
            '--bind-to', 'none',
        ]

    with _openmc_mpi_env(singleton=mpi_args is None):
        if timeout is None:
            return model.run(threads=n_threads, mpi_args=mpi_args, cwd=cwd, output=output)

        model.export_to_model_xml(os.path.join(cwd, 'model.xml'))
        openmc_exec = shutil.which('openmc') or 'openmc'
        if mpi_args is not None:
            cmd = [*mpi_args, openmc_exec, '-s', str(n_threads)]
        else:
            cmd = [openmc_exec, '-s', str(n_threads)]

        try:
            subprocess.run(
                cmd, cwd=cwd, timeout=timeout, check=True,
                stdout=None if output else subprocess.DEVNULL,
                stderr=None if output else subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as err:
            raise OpenMCTimeout(
                f"OpenMC exceeded timeout of {timeout:.0f} s in {cwd!r}"
            ) from err

        statepoints = sorted(glob.glob(os.path.join(cwd, 'statepoint.*.h5')))
        if not statepoints:
            raise RuntimeError(f"OpenMC finished but no statepoint was found in {cwd!r}.")
        return statepoints[-1]


# -- post-processing -----------------------------------------------

def _collapse_cell_filter(tally: openmc.Tally) -> openmc.Tally:
    if any(isinstance(f, openmc.CellFilter) for f in tally.filters):
        return tally.summation(filter_type=openmc.CellFilter, remove_filter=True)
    return tally


def _spectrum_from_statepoint(sp_filename: str):
    sp = openmc.StatePoint(sp_filename)
    tally = _collapse_cell_filter(sp.tallies.get(6) or next(iter(sp.tallies.values())))
    mean = tally.get_slice(scores=['flux']).mean.flatten()
    std = tally.get_slice(scores=['flux']).std_dev.flatten()
    return sp, tally, mean, std


def extract_flux(sp_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Integral-normalized flux and relative uncertainty [%] from a statepoint.

    Normalization is over the *whole* grid. dataset_creation.py and
    calibration.py rely on that; the __main__ printout renormalizes over
    non-zero-target bins afterwards.
    """
    _, _, mean, std = _spectrum_from_statepoint(sp_filename)
    norm = mean / (mean.sum() or 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_unc = np.where(mean > 0, std / mean * 100.0, 0.0)
    return norm, rel_unc


def log_rmse(flux: np.ndarray, target: np.ndarray) -> float:
    """Detector-weighted log-RMSE, the same L as optimizer.compute_log_mse.

    Both spectra are integral-normalised over the full grid. Zero-target bins
    are dropped by ``build_bin_weights`` (``WEIGHT_MODE``), then the weights
    are renormalised to sum to 1. The log floor is ``config.LOG_EPS``.
    """
    flux = np.maximum(np.asarray(flux, dtype=float), 0.0)
    target = np.asarray(target, dtype=float)
    active = active_target_mask(target, flux.size)
    flux_n = flux / (flux.sum() or 1.0)
    target_n = target / (target.sum() or 1.0)
    diff = np.log(flux_n + LOG_EPS) - np.log(target_n + LOG_EPS)
    weights = build_bin_weights(mask=active)
    return float(np.sqrt(np.dot(weights, diff * diff)))


def active_target_mask(target: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins with a non-zero target -- the only ones this module reports."""
    target = np.asarray(target, dtype=float)
    if target.size != n_bins:
        raise ValueError(
            f'Target spectrum has {target.size} bins but the tally has {n_bins}.'
        )
    active = target != 0.0
    if not active.any():
        raise ValueError('Target spectrum is zero in every bin.')
    return active


def _active_view(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.where(active, values, np.nan)


def _active_xlim(edges: list[float], active: np.ndarray) -> tuple[float, float]:
    """Energy range [MeV] of non-zero-target bins; left edge starts mid-bin."""
    idx = np.flatnonzero(active)
    lo, hi = int(idx[0]), int(idx[-1])
    return 0.5 * (edges[lo] + edges[lo + 1]), edges[hi + 1]


def postprocess_statepoint(sp_filename: str, target_values: list[float]) -> None:
    """Extract the spectrum, save the log-log plots, and print the log-RMSE."""
    sp, tally, mean, std = _spectrum_from_statepoint(sp_filename)  # sp must stay alive while we read tally filters

    e_bins = tally.find_filter(openmc.EnergyFilter).bins
    edges = [e_bins[0, 0] * 1e-6, *e_bins[:, 1] * 1e-6]

    target = np.asarray(
        convert_if_lethargy(target_values, edges, PHI_TARGET_IN_LETHARGY), dtype=float,
    )
    active = active_target_mask(target, mean.size)

    ref = float(np.nansum(mean[active]))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = _active_view(mean / ref, active)
        rel_unc = np.where(mean > 0, std / mean, 0.0)
        sum_unc = np.sqrt(np.nansum(std[active] ** 2)) / ref
        norm_std = norm * np.sqrt(rel_unc ** 2 + sum_unc ** 2)
    band = (np.maximum(norm - norm_std, 0.0), norm + norm_std)
    t_norm = _active_view(target / (target[active].sum() or 1.0), active)
    xlim = _active_xlim(edges, active)

    _save_spectrum(
        'neutron_spectrum_log_log.png', edges, norm, band, t_norm,
        xscale='log', yscale='log', xlim=xlim,
    )

    volume = DETECTOR_GEOMETRY.active_volume()
    real_mean = _active_view((mean / volume) * SOURCE_STRENGTH, active)
    real_std = _active_view((std / volume) * SOURCE_STRENGTH, active)
    real_band = (np.maximum(real_mean - real_std, 0.0), real_mean + real_std)
    _save_spectrum(
        'neutron_spectrum_real_log_log.png', edges, real_mean, real_band,
        _active_view(target, active),
        xscale='log', yscale='log', xlim=xlim,
        title='Neutron Spectrum (Real Units)',
        ylabel='Flux [n/cm^2/s]',
    )
    print(f'Weighted Log-RMSE = {log_rmse(mean, target):.6f}')


def _save_spectrum(
    filename: str,
    edges: list[float],
    flux: np.ndarray,
    band: tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
    *,
    xscale: str = 'linear',
    yscale: str = 'linear',
    xlim: tuple[float, float] | None = None,
    title: str = 'Neutron Spectrum (Integral Normalized)',
    ylabel: str = 'Normalized flux [-]',
) -> None:
    blue, red = '#1f77b4', '#d62728'

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.stairs(flux, edges, linewidth=2.2, color=blue, label='OpenMC')
    ax.fill_between(
        edges, np.r_[band[0], band[0][-1]], np.r_[band[1], band[1][-1]],
        step='post', color=blue, alpha=0.2, linewidth=0, label='OpenMC +/- 1sigma',
    )
    ax.stairs(target, edges, linewidth=2.2, color=red, linestyle='--', label='Target')

    ax.set_xlabel('Energy [MeV]', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(
        True, which='both' if xscale == 'log' else 'major',
        linestyle='--', linewidth=0.7, alpha=0.6,
    )
    ax.legend(frameon=False)

    if xscale == 'log':
        ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif xscale == 'log':
        ax.set_xlim(left=(TARGET_ENERGY_EDGES[0] * 1e-6 + TARGET_ENERGY_EDGES[1] * 1e-6) / 2)
    if yscale == 'log':
        ax.set_yscale('log')
    elif xscale != 'log':
        ax.ticklabel_format(style='plain', axis='both', useOffset=False)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_mcmc_validation_comparison(
    sp_filename: str,
    band_path: str = 'mcmc_validation_band.npz',
    out_filename: str = 'spectrum_comparison_log.png',
) -> None:
    """If calibration.py wrote ``mcmc_validation_band.npz``, overlay this run on it."""
    if not os.path.isfile(band_path):
        return

    band = np.load(band_path)
    band_edges_ev = band['energy_edges_ev']
    if band_edges_ev.shape != np.asarray(ENERGY_EDGES).shape or not np.allclose(
        band_edges_ev, ENERGY_EDGES, rtol=0.0, atol=0.0
    ):
        print(
            f"[MCMC validation] '{band_path}' was exported with different ENERGY_EDGES "
            "than this run's ENERGY_EDGES -- skipping overlay plot."
        )
        return

    y_target = band['y_target']
    pp_med = band['pp_med']
    pp_lo = band['pp_lo']
    pp_hi = band['pp_hi']
    pp_cred = float(band['pp_cred'])

    energy_edges_mev = np.asarray(ENERGY_EDGES, dtype=float) * 1e-6
    target = convert_if_lethargy(TARGET_VALUES, ENERGY_EDGES, PHI_TARGET_IN_LETHARGY)
    active = active_target_mask(target, len(ENERGY_EDGES) - 1)

    y_openmc_all, rel_unc = extract_flux(sp_filename)
    y_openmc_all = _active_view(
        y_openmc_all / (y_openmc_all[active].sum() or 1.0), active,
    )
    y_openmc_std_all = y_openmc_all * rel_unc / 100.0

    y_target = _active_view(y_target, active)
    pp_med = _active_view(pp_med, active)
    pp_lo = _active_view(pp_lo, active)
    pp_hi = _active_view(pp_hi, active)

    def _plot_step(ax, edges, values, **kwargs):
        ax.step(edges, np.append(values, values[-1]), where='post', **kwargs)

    def _fill_between_step(ax, edges, v_lo, v_hi, **kwargs):
        ax.fill_between(
            edges, np.append(v_lo, v_lo[-1]), np.append(v_hi, v_hi[-1]),
            step='post', **kwargs,
        )

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_step(ax, energy_edges_mev, y_openmc_all, color='#3498db',
               linewidth=2.5, label='OpenMC', zorder=3)
    _fill_between_step(
        ax, energy_edges_mev,
        np.maximum(y_openmc_all - y_openmc_std_all, 1e-12),
        y_openmc_all + y_openmc_std_all,
        alpha=0.2, color='#3498db', label='OpenMC +/- 1sigma', zorder=1,
    )
    _plot_step(ax, energy_edges_mev, y_target, color='#e74c3c',
               linewidth=2.5, linestyle='--', label='Target values', zorder=4)
    _plot_step(ax, energy_edges_mev, pp_med, color='#16a085',
               linewidth=2.5, linestyle='-.', label='Emulator (predictive median)', zorder=3)
    _fill_between_step(
        ax, energy_edges_mev, np.maximum(pp_lo, 1e-10), pp_hi,
        alpha=0.15, color='#16a085',
        label=f'Emulator {pp_cred:.0f}% predictive band', zorder=1,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [MeV]', fontsize=12)
    ax.set_ylabel('Normalized value [-]', fontsize=12)
    ax.set_title(
        'Neutron Spectrum (Integral Normalized) - Log X/Y - Comparison',
        fontsize=13, fontweight='bold',
    )

    all_y = np.concatenate([np.atleast_1d(y) for y in (y_target, pp_med, y_openmc_all)])
    positive = all_y[np.isfinite(all_y) & (all_y > 0)]
    if positive.size:
        ax.set_ylim(np.min(positive) * 0.5, np.max(positive) * 2.0)

    ax.set_xlim(*_active_xlim(energy_edges_mev.tolist(), active))
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best')

    fig.tight_layout()
    fig.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[MCMC validation] Comparison plot saved as '{out_filename}'.")


def _mesh_cells_in_interval(
    grid: list[float] | np.ndarray, lo: float, hi: float,
) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=float)
    overlap = (g[1:] > lo) & (g[:-1] < hi)
    idx = np.flatnonzero(overlap)
    if idx.size == 0:
        raise ValueError(f'No mesh cells overlap [{lo}, {hi}] in grid {g.tolist()}.')
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    return i0, i1, g[i0:i1 + 1]


def _mid_index(grid: np.ndarray, value: float, lo_idx: int, hi_idx: int) -> int:
    idx = int(np.searchsorted(grid, value, side='right')) - 1
    return max(lo_idx, min(hi_idx - 1, idx))


def _masked_plane(bounds_3d: np.ndarray, iz, iy, ix) -> np.ndarray:
    plane = bounds_3d[iz, iy, ix]
    return np.where(plane > 0, plane, np.nan)


def _draw_material_interfaces(
    ax, *, plane: str,
    x_interfaces: list[float] | tuple[float, ...] | np.ndarray | None,
    plate_half_y: float | None = None,
    plate_half_z: float | None = None,
) -> None:
    style = dict(color='white', linewidth=1.4, linestyle='-', zorder=4)
    style_edge = dict(color='black', linewidth=0.8, linestyle=':', zorder=5)
    if x_interfaces is not None:
        for x in x_interfaces:
            ax.axvline(float(x), **style)
            ax.axvline(float(x), **style_edge)
    if plane == 'xz' and plate_half_z is not None:
        for z in (-plate_half_z, plate_half_z):
            ax.axhline(float(z), **style)
            ax.axhline(float(z), **style_edge)
    if plane == 'xy' and plate_half_y is not None:
        for y in (-plate_half_y, plate_half_y):
            ax.axhline(float(y), **style)
            ax.axhline(float(y), **style_edge)


def _draw_detector_outer_outline(
    ax, plane: str, *,
    detector_center: tuple[float, float, float],
    y_cut: float, z_cut: float,
) -> None:
    outer = DETECTOR_GEOMETRY.regions[0]
    cx, cy, cz = detector_center
    kw = dict(fill=False, edgecolor='white', linewidth=2.0, linestyle='-', zorder=5)
    kw2 = dict(fill=False, edgecolor='black', linewidth=1.0, linestyle='--', zorder=6)

    def _circle(center, r):
        ax.add_patch(mpatches.Circle(center, r, **kw))
        ax.add_patch(mpatches.Circle(center, r, **kw2))

    def _rect(xy, w, h):
        ax.add_patch(mpatches.Rectangle(xy, w, h, **kw))
        ax.add_patch(mpatches.Rectangle(xy, w, h, **kw2))

    if outer.shape == 'sphere':
        radius = outer.diameter / 2.0
        if plane == 'xz':
            dy = y_cut - cy
            if abs(dy) < radius:
                _circle((cx, cz), math.sqrt(radius**2 - dy**2))
        else:
            dz = z_cut - cz
            if abs(dz) < radius:
                _circle((cx, cy), math.sqrt(radius**2 - dz**2))
        return

    if outer.shape == 'box':
        if plane == 'xz':
            _rect((cx - outer.dx / 2.0, cz - outer.dz / 2.0), outer.dx, outer.dz)
        else:
            _rect((cx - outer.dx / 2.0, cy - outer.dy / 2.0), outer.dx, outer.dy)
        return

    radius, half, axis = outer.diameter / 2.0, outer.height / 2.0, outer.axis
    if plane == 'xz':
        if axis == 'x':
            _rect((cx - half, cz - radius), outer.height, outer.diameter)
        elif axis == 'z':
            _rect((cx - radius, cz - half), outer.diameter, outer.height)
        elif abs(y_cut - cy) < half:
            _circle((cx, cz), radius)
    else:
        if axis == 'x':
            _rect((cx - half, cy - radius), outer.height, outer.diameter)
        elif axis == 'y':
            _rect((cx - radius, cy - half), outer.diameter, outer.height)
        elif abs(z_cut - cz) < half:
            _circle((cx, cy), radius)


def plot_weight_windows_bounds(
    ww_path: str,
    e_bin: int | list[int] | tuple[int, ...] | np.ndarray,
    *,
    out_dir: str | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    z_lim: tuple[float, float] | None = None,
    detector_center: tuple[float, float, float] | None = None,
    material_x_interfaces: list[float] | tuple[float, ...] | None = None,
    plate_half_y: float | None = None,
    plate_half_z: float | None = None,
) -> None:
    """XY / XZ projections of WW lower and upper bounds (one figure per energy bin)."""
    import h5py
    from matplotlib.colors import LogNorm

    if out_dir is None:
        out_dir = WW_DIR
    os.makedirs(out_dir, exist_ok=True)

    if plate_half_y is None:
        plate_half_y = PLATE_DIM / 2.0
    if plate_half_z is None:
        plate_half_z = PLATE_DIM / 2.0

    bins = np.atleast_1d(np.asarray(e_bin, dtype=int)).ravel()
    if bins.size == 0:
        return

    try:
        with h5py.File(ww_path, 'r') as f:
            ww_group = f['weight_windows/weight_windows_1']
            lb = np.asarray(ww_group['lower_ww_bounds'][:], dtype=float)
            ub = np.asarray(ww_group['upper_ww_bounds'][:], dtype=float)
            e_bounds = np.asarray(ww_group['energy_bounds'][:], dtype=float)
            mesh_grp = f['meshes'][list(f['meshes'].keys())[0]]
            x_grid = np.asarray(mesh_grp['x_grid'][:], dtype=float)
            y_grid = np.asarray(mesh_grp['y_grid'][:], dtype=float)
            z_grid = np.asarray(mesh_grp['z_grid'][:], dtype=float)

        if x_lim is None:
            x_lim = (float(x_grid[0]), float(x_grid[-1]))
        if y_lim is None:
            y_lim = (-PLATE_DIM / 2.0, PLATE_DIM / 2.0)
        if z_lim is None:
            z_lim = (-PLATE_DIM / 2.0, PLATE_DIM / 2.0)

        energy_groups = lb.shape[0]
        n_x = len(x_grid) - 1
        n_y = len(y_grid) - 1
        n_z = len(z_grid) - 1
        n_mesh = n_x * n_y * n_z
        if lb.shape[1] != n_mesh:
            raise ValueError(
                f'WW bounds length {lb.shape[1]} != mesh cells '
                f'{n_x}x{n_y}x{n_z}={n_mesh}.'
            )

        ix0, ix1, x_view = _mesh_cells_in_interval(x_grid, x_lim[0], x_lim[1])
        iy0, iy1, y_view = _mesh_cells_in_interval(y_grid, y_lim[0], y_lim[1])
        iz0, iz1, z_view = _mesh_cells_in_interval(z_grid, z_lim[0], z_lim[1])

        mid_y_idx = _mid_index(y_grid, 0.0, iy0, iy1)
        mid_z_idx = _mid_index(z_grid, 0.0, iz0, iz1)
        y_cut = (y_grid[mid_y_idx] + y_grid[mid_y_idx + 1]) / 2.0
        z_cut = (z_grid[mid_z_idx] + z_grid[mid_z_idx + 1]) / 2.0

        x_slice, y_slice, z_slice = slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1)
        X_xz, Z_xz = np.meshgrid(x_view, z_view)
        X_xy, Y_xy = np.meshgrid(x_view, y_view)

        def _annotate(ax, plane: str) -> None:
            _draw_material_interfaces(
                ax, plane=plane,
                x_interfaces=material_x_interfaces,
                plate_half_y=plate_half_y,
                plate_half_z=plate_half_z,
            )
            if detector_center is not None:
                _draw_detector_outer_outline(
                    ax, plane,
                    detector_center=detector_center,
                    y_cut=y_cut, z_cut=z_cut,
                )

        for bin_i in bins:
            if bin_i < 1 or bin_i > energy_groups:
                continue

            # HDF5 stores bounds flat per energy group in (z, y, x) C order.
            lb_3d = lb[bin_i - 1].reshape((n_z, n_y, n_x))
            ub_3d = ub[bin_i - 1].reshape((n_z, n_y, n_x))

            lb_xz = _masked_plane(lb_3d, z_slice, mid_y_idx, x_slice)
            ub_xz = _masked_plane(ub_3d, z_slice, mid_y_idx, x_slice)
            lb_xy = _masked_plane(lb_3d, mid_z_idx, y_slice, x_slice)
            ub_xy = _masked_plane(ub_3d, mid_z_idx, y_slice, x_slice)

            all_vals = np.concatenate([
                lb_xz.ravel(), ub_xz.ravel(), lb_xy.ravel(), ub_xy.ravel(),
            ])
            all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
            if all_vals.size == 0:
                continue
            norm = LogNorm(vmin=float(all_vals.min()), vmax=float(all_vals.max()))

            e_lo, e_hi = float(e_bounds[bin_i - 1]), float(e_bounds[bin_i])
            e_label = f'Bin {bin_i} [{e_lo:.3g}, {e_hi:.3g}] eV'

            fig, axs = plt.subplots(2, 2, figsize=(15, 12), layout='constrained')
            panels = [
                (axs[0, 0], lb_xz, X_xz, Z_xz, 'xz', x_lim, z_lim, 'Z [cm]',
                 f'Lower Bound (XZ, y~{y_cut:.1f} cm), {e_label}'),
                (axs[0, 1], lb_xy, X_xy, Y_xy, 'xy', x_lim, y_lim, 'Y [cm]',
                 f'Lower Bound (XY, z~{z_cut:.1f} cm), {e_label}'),
                (axs[1, 0], ub_xz, X_xz, Z_xz, 'xz', x_lim, z_lim, 'Z [cm]',
                 f'Upper Bound (XZ, y~{z_cut:.1f} cm), {e_label}'),
                (axs[1, 1], ub_xy, X_xy, Y_xy, 'xy', x_lim, y_lim, 'Y [cm]',
                 f'Upper Bound (XY, z~{z_cut:.1f} cm), {e_label}'),
            ]
            im0 = None
            for ax, data, X, Y, plane, xlim, ylim, ylabel, title in panels:
                im = ax.pcolormesh(X, Y, data, shading='flat', norm=norm)
                if im0 is None:
                    im0 = im
                ax.set_title(title, fontsize=17)
                ax.set_xlabel('X [cm]', fontsize=15)
                ax.set_ylabel(ylabel, fontsize=15)
                ax.tick_params(axis='both', which='both', labelsize=15)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect('equal', adjustable='box')
                _annotate(ax, plane)

            cbar = fig.colorbar(im0, ax=axs)
            cbar.set_label('WW Bound', fontsize=15)
            cbar.ax.tick_params(labelsize=15)
            fig.savefig(os.path.join(out_dir, f'ww_bounds_bin_{bin_i}.png'), dpi=150)
            plt.close(fig)
    except Exception as err:
        print(f'[ww] Could not plot weight-window bounds: {err}')


def _material_plot_colors(model: openmc.Model) -> dict:
    active_mat_names = {f'detector_{n}' for n in DETECTOR_GEOMETRY.active_region_names()}
    cmap = plt.get_cmap('tab20')
    mat_colors = {}
    for i, mat in enumerate(model.materials):
        if mat.name == 'air':
            mat_colors[mat] = (240, 240, 250)
        elif mat.name in active_mat_names:
            mat_colors[mat] = (255, 0, 0)
        elif mat.name.startswith('detector_'):
            mat_colors[mat] = (210, 180, 140)
        elif mat.name == 'CONCRETE_ordinary_NIST':
            mat_colors[mat] = (150, 150, 150)
        else:
            r, g, b, _ = cmap(i % 20)
            mat_colors[mat] = (int(r * 255), int(g * 255), int(b * 255))
    return mat_colors


def plot_model_geometry(
    x_bounds: list[float], materials: list[str], *,
    y_dim: float | None = None,
    z_dim: float | None = None,
    out_path: str = 'geometry_plot.png',
) -> None:
    """XZ / XY / YZ material cuts of the generated geometry."""
    print('Plotting geometry...')
    y_dim = Y_DIM if y_dim is None else float(y_dim)
    z_dim = Z_DIM if z_dim is None else float(z_dim)
    model = build_model(
        x_bounds, materials,
        particles=10, batches=1,
        is_random_ray=False,
        y_dim=y_dim,
        z_dim=z_dim,
    )
    universe = model.geometry.root_universe

    x_min = X_START - 10.0
    x_max = x_bounds[-1] + 2.0 * DETECTOR_GEOMETRY.half_extent('x') + 10.0
    y_width = y_dim + 20.0
    z_width = z_dim + 20.0
    x_width = x_max - x_min
    x_center = (x_min + x_max) / 2.0
    mat_colors = _material_plot_colors(model)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    try:
        universe.plot(
            width=(x_width, z_width), origin=(x_center, 0.0, 0.0), basis='xz',
            color_by='material', colors=mat_colors, axes=axs[0],
        )
        axs[0].set_title('XZ Plane Cut (y=0)', fontsize=17)
        axs[0].set_xlabel('X [cm]', fontsize=15)
        axs[0].set_ylabel('Z [cm]', fontsize=15)
        axs[0].tick_params(axis='both', which='both', labelsize=15)

        universe.plot(
            width=(x_width, y_width), origin=(x_center, 0.0, 0.0), basis='xy',
            color_by='material', colors=mat_colors, axes=axs[1],
        )
        axs[1].set_title('XY Plane Cut (z=0)', fontsize=17)
        axs[1].set_xlabel('X [cm]', fontsize=15)
        axs[1].set_ylabel('Y [cm]', fontsize=15)
        axs[1].tick_params(axis='both', which='both', labelsize=15)

        full_yz = AIR_SPHERE_RADIUS * 2.0
        universe.plot(
            width=(full_yz, full_yz), origin=(x_center, 0.0, 0.0), basis='yz',
            color_by='material', colors=mat_colors, axes=axs[2],
        )
        axs[2].set_title(f'YZ Plane Cut (x={x_center:.2f})', fontsize=17)
        axs[2].set_xlabel('Y [cm]', fontsize=15)
        axs[2].set_ylabel('Z [cm]', fontsize=15)
        axs[2].tick_params(axis='both', which='both', labelsize=15)

        handles = [
            mpatches.Patch(color=[c / 255 for c in color], label=mat.name)
            for mat, color in mat_colors.items()
        ]
        fig.legend(
            handles=handles, loc='upper center',
            bbox_to_anchor=(0.5, 0.98), ncol=min(8, len(mat_colors)),
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_path, dpi=150)
        print(f"Geometry saved to '{out_path}'.")
    except TypeError:
        print('Falling back to openmc.plot_geometry()...')
        plot_xz = openmc.Plot()
        plot_xz.basis = 'xz'
        plot_xz.origin = (x_center, 0.0, 0.0)
        plot_xz.width = (x_width, z_width)
        plot_xz.pixels = (800, 800)
        plot_xz.color_by = 'material'
        plot_xz.colors = mat_colors
        plot_xz.filename = 'geometry_xz'

        plot_xy = openmc.Plot()
        plot_xy.basis = 'xy'
        plot_xy.origin = (x_center, 0.0, 0.0)
        plot_xy.width = (x_width, y_width)
        plot_xy.pixels = (800, 800)
        plot_xy.color_by = 'material'
        plot_xy.colors = mat_colors
        plot_xy.filename = 'geometry_xy'

        plot_yz = openmc.Plot()
        plot_yz.basis = 'yz'
        plot_yz.origin = (x_center, 0.0, 0.0)
        plot_yz.width = (AIR_SPHERE_RADIUS * 2.0, AIR_SPHERE_RADIUS * 2.0)
        plot_yz.pixels = (800, 800)
        plot_yz.color_by = 'material'
        plot_yz.colors = mat_colors
        plot_yz.filename = 'geometry_yz'

        model.plots = openmc.Plots([plot_xz, plot_xy, plot_yz])
        model.export_to_xml()
        openmc.plot_geometry()
        print("Geometry saved to 'geometry_xz.png', 'geometry_xy.png', 'geometry_yz.png'.")
    finally:
        plt.close(fig)


# -- main ----------------------------------------------------------

if __name__ == '__main__':
    if len(THICKNESSES) != len(MATERIALS):
        raise ValueError('THICKNESSES and MATERIALS must have the same length.')

    x_bounds = thicknesses_to_x_bounds(X_START, THICKNESSES)

    if PLOT_GEOMETRY:
        plot_model_geometry(x_bounds, MATERIALS)

    print('Phase 1  weight windows (Random Ray) ...')
    ww_path, *_ = generate_weight_windows(
        x_bounds, MATERIALS, WW_DIR,
        rr_batches=WW_BATCHES,
        rr_inactive_batches=WW_INACTIVE_BATCHES,
        rr_particles=WW_PARTICLES,
        mgxs_particles=MGXS_PARTICLES,
    )

    print('Phase 2  CE production run ...')
    sp_file = run_with_weight_windows(
        x_bounds, MATERIALS, ww_path, CE_DIR,
        batches=CE_BATCHES,
        particles=CE_PARTICLES,
        use_angle_bias=USE_ANGLE_BIAS,
        max_lower_bound_ratio=MAX_LOWER_BOUND_RATIO,
        max_split=MAX_SPLIT,
    )

    print('Processing results...')
    postprocess_statepoint(sp_file, TARGET_VALUES)

    half = PLATE_DIM / 2.0
    x_det_end = float(x_bounds[-1]) + 2.0 * DETECTOR_GEOMETRY.half_extent('x')
    det_cx = float(x_bounds[-1]) + DETECTOR_GEOMETRY.half_extent('x')
    plot_weight_windows_bounds(
        ww_path, PLOT_WW_BIN,
        out_dir=WW_DIR,
        x_lim=(-MESH_X_BEFORE_SOURCE, x_det_end),
        y_lim=(-half, half),
        z_lim=(-half, half),
        detector_center=(det_cx, 0.0, 0.0),
        material_x_interfaces=tuple(float(x) for x in x_bounds),
        plate_half_y=half,
        plate_half_z=half,
    )

    # extract_flux normalizes over the whole grid (dataset_creation.py /
    # calibration.py rely on that); renormalize here over non-zero-target bins.
    flux_norm, rel_unc = extract_flux(sp_file)
    target = convert_if_lethargy(TARGET_VALUES, ENERGY_EDGES, PHI_TARGET_IN_LETHARGY)
    active = active_target_mask(target, flux_norm.size)
    flux_norm = flux_norm / (flux_norm[active].sum() or 1.0)

    n_skipped = int((~active).sum())
    if n_skipped:
        print(f'\nIgnoring {n_skipped} zero-target bin(s)')
    print(f'\n{"bin":>4}  {"flux_norm":>12}  {"rel_unc_%":>10}')
    for i in np.flatnonzero(active):
        print(f'{i:>4}  {flux_norm[i]:>12.4e}  {rel_unc[i]:>10.2f}')
