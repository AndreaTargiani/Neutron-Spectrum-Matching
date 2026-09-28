import openmc
import numpy as np
import os
import shutil
import time
import itertools
import multiprocessing as mp

import config
from config import (
    SOURCE_EDGES, SOURCE_VALUES, TARGET_ENERGY_EDGES,
)
from config_materials import build_material_library

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

_MATERIAL_LIBRARY = build_material_library()
MATERIALS = {
    name: _MATERIAL_LIBRARY[name]
    for name in (
        "PE_BO", "concrete", "HDPE", "Pb", "cast_iron", "SS_304", "water",
        "tungsten", "graphite", "iron", "cadmium", "Al_6061", "Nickel",
        "brass", "Copper", "Zirconium", "B4C", "PTFE", "Mg", "Bi",
    )
}

THICKNESSES = [0.2, 0.8, 3.0, 10.0, 28.0, 75.0, 130.0]   # cm

N_MU = 5      # forward ordinates

SN_BATCHES = int(os.environ.get("SN_BATCHES", 100))
SN_PARTICLES_INITIAL = int(os.environ.get("SN_PARTICLES_INITIAL", 8_000))
SN_PARTICLES_MAX = int(os.environ.get("SN_PARTICLES_MAX", 100_000))
SN_THRESHOLD = float(os.environ.get("SN_THRESHOLD", 8e-6))
SN_REL_ERR = float(os.environ.get("SN_REL_ERR", 0.15))

OVERWRITE = False   # True: regenerate existing sn_tensor_*.npy files

HPC = False
config.configure_openmc_cross_sections(HPC)

if HPC:
    SN_WORKERS = 48
    SN_THREADS = 4
else:
    SN_WORKERS = 1
    SN_THREADS = max(1, (os.cpu_count() or 2) // 2 - 1)

OPT_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_database")

# ==================================================================

_s, _ = np.polynomial.legendre.leggauss(N_MU)
MU_NODES = 0.5 * (_s + 1.0)          # in (0, 1), ascending (grazing -> normal)
MU_EDGES = np.concatenate(
    ([0.0], 0.5 * (MU_NODES[1:] + MU_NODES[:-1]), [1.0])
)

# -- ENERGY GRID & SOURCE SPECTRUM ----------------------------------
# SOURCE_EDGES / SOURCE_VALUES are the source histogram built in config.py.

def build_refined_grid(target_edges, source_edges, tol=1e-6):
    """
    Refine the target energy grid by adding source edges as needed.
    Works for ANY source grid:
      * finer source  -> adds sub-bins inside the relevant target groups
      * coarser source-> adds no internal edges (unchanged target grid)
      * aligned source-> adds nothing (fine grid == TARGET_ENERGY_EDGES)
    Source edges outside the target range are dropped (outside simulated range).
    """
    target = np.asarray(target_edges, float)
    lo, hi = target[0], target[-1]
    add = []
    for e in np.asarray(source_edges, float):
        if e <= lo + tol or e >= hi - tol:
            continue                                   # outside / on boundary
        if np.min(np.abs(target - e)) < tol * max(1.0, e):
            continue                                   # coincides with a target edge
        add.append(e)
    return np.sort(np.concatenate([target, np.array(add)])) if add else target.copy()


def fine_to_target_matrix(fine_edges, target_edges):
    """(N_target x N_fine) 0/1 collapse matrix; each fine bin nests in one target bin."""
    fine_edges = np.asarray(fine_edges, float); target_edges = np.asarray(target_edges, float)
    mid = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    idx = np.clip(np.searchsorted(target_edges, mid, side="right") - 1,
                  0, target_edges.size - 2)
    C = np.zeros((target_edges.size - 1, fine_edges.size - 1))
    C[idx, np.arange(mid.size)] = 1.0
    return C

def get_source_spec():
    return np.asarray(SOURCE_EDGES, float), np.asarray(SOURCE_VALUES, float)

# -- build the actual GENERATION grid (fine) -----------------------
ENERGY_EDGES = build_refined_grid(TARGET_ENERGY_EDGES, SOURCE_EDGES)

N_E = len(ENERGY_EDGES) - 1
N_OUT = N_MU * N_E                 

def process_flux_output(values_mean, values_std, num_bins, threshold, rel_err_threshold):
    """
    Validation + noise-cleaning applied to the CURRENT score.

    Policy (per bin)
    ----------------
    * A bin is KEPT (left untouched) whenever its relative error is acceptable
      (std <= rel_err_threshold * mean), *regardless of whether its mean is above
      or below* `threshold`.  A well-resolved small value is real signal, not
      noise, so it stays in the matrix.
    * A bin BELOW `threshold` with an UNacceptable relative error (or a
      non-positive mean) is treated as noise and zeroed.  Because it is below
      threshold it NEVER triggers particle refinement.
    * A bin at/above `threshold` with an unacceptable relative error fails
      validation: the caller should rerun with more particles.

    Returns:
        (validation_passed: bool, values_mean: array, reason: str)
    """
    for bin_idx in range(num_bins):
        mean = values_mean[bin_idx]
        std = values_std[bin_idx]
        rel_err_ok = (mean > 0.0) and (std <= rel_err_threshold * mean)

        if mean >= threshold:
            if not rel_err_ok:
                relative_error = (std / mean) if mean > 0.0 else float("inf")
                return False, values_mean, (
                    f"Bin {bin_idx}: relative error {relative_error:.2%} "
                    f"> {rel_err_threshold:.0%}. Need more particles."
                )
        else:
            # Below the noise floor: keep it only if it is statistically solid,
            # otherwise drop it as noise.  Either way, no refinement is triggered.
            if not rel_err_ok:
                values_mean[bin_idx] = 0.0

    return True, values_mean, "Validation passed."



def make_sn_source(mu_in_index, source_bin_index):
    """Monodirectional beam at ordinate ``MU_NODES[mu_in_index]``, energy in ``E_in``.
    """
    mu = float(MU_NODES[mu_in_index])
    sin_theta = float(np.sqrt(max(0.0, 1.0 - mu * mu)))
    return openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)),
        angle=openmc.stats.Monodirectional((sin_theta, 0.0, mu)),
        energy=openmc.stats.Uniform(ENERGY_EDGES[source_bin_index],
                                    ENERGY_EDGES[source_bin_index + 1]),
        strength=1.0,
    )


def _build_and_run_sn(material_name, thickness, mu_in_index, source_bin_index,
                      batches, particles, work_dir, threads):
    """Build the slab model for one incident beam and run it; return statepoint path."""
    material = MATERIALS[material_name]

    z0 = openmc.ZPlane(z0=0.0, boundary_type="vacuum")
    z1 = openmc.ZPlane(z0=thickness, boundary_type="transmission")
    z2 = openmc.ZPlane(z0=thickness + 1.0, boundary_type="vacuum")     # void exit region
    material_cell = openmc.Cell(fill=material, region=+z0 & -z1)
    exit_cell = openmc.Cell(fill=None, region=+z1 & -z2)
    geometry = openmc.Geometry(openmc.Universe(cells=[material_cell, exit_cell]))

    settings = openmc.Settings(
        run_mode="fixed source",
        batches=batches,
        particles=particles,
        source=make_sn_source(mu_in_index, source_bin_index),
    )

    tally = openmc.Tally(name="sn_outgoing")
    tally.filters = [
        openmc.SurfaceFilter(z1),
        openmc.MuSurfaceFilter(MU_EDGES),      
        openmc.EnergyFilter(ENERGY_EDGES),
        openmc.ParticleFilter("neutron"),
    ]
    tally.scores = ["current"]              

    model = openmc.Model(
        geometry=geometry,
        materials=openmc.Materials([material]),
        settings=settings,
        tallies=openmc.Tallies([tally]),
    )

    os.makedirs(work_dir, exist_ok=True)
    _isolate_openmc_from_slurm_mpi()
    sp_file = model.run(output=False, cwd=work_dir, openmc_exec="openmc", threads=threads)
    if sp_file is None:
        expected = os.path.join(work_dir, f"statepoint.{batches}.h5")
        if os.path.isfile(expected):
            sp_file = expected
    return sp_file


def _run_sn_column(material_name, thickness, mu_in_index, source_bin_index, *,
                   batches, particles_initial, particles_max, threshold, rel_err,
                   threads, work_dir):
    """Run one (mu_in, E_in) beam with the adaptive-particle / rel-error gate.

    Returns ``(J_mean, J_std, particles, reason)`` flattened over (mu_out, E_out)
    (length N_OUT).  The returned mean is noise-cleaned by ``process_flux_output``:
    every statistically solid bin is kept (including small, below-threshold ones),
    and only below-threshold bins with unacceptable relative error are zeroed.
    The gate decides the particle count solely from bins at/above threshold.
    """
    particles = particles_initial
    while True:
        sp_path = _build_and_run_sn(material_name, thickness, mu_in_index,
                                    source_bin_index, batches, particles,
                                    work_dir, threads)
        if sp_path is None:
            shutil.rmtree(work_dir, ignore_errors=True)
            if particles >= particles_max:
                raise RuntimeError(
                    f"OpenMC produced no statepoint for {material_name} @ {thickness}cm "
                    f"mu_in={mu_in_index}, E_in={source_bin_index} even at the particle cap."
                )
            particles = min(particles * 2, particles_max)
            continue

        with openmc.StatePoint(sp_path) as sp:
            tally = sp.get_tally(name="sn_outgoing")
            J_mean = tally.get_values(scores=["current"], value="mean").flatten()
            J_std = tally.get_values(scores=["current"], value="std_dev").flatten()
        shutil.rmtree(work_dir, ignore_errors=True)

        passed, J_clean, reason = process_flux_output(
            J_mean.copy(), J_std, N_OUT, threshold, rel_err
        )
        if passed:
            return J_clean, J_std, particles, reason
        if particles >= particles_max:
            return J_clean, J_std, particles, f"cap reached, accepting ({reason})"
        particles = min(particles * 2, particles_max)


def _worker_run_column(args):
    """Worker function for multiprocessing."""
    material_name, thickness, mu_in, e_in, batches, particles_initial, particles_max, threshold, rel_err, threads, base_work_dir = args
    work_dir = f"{base_work_dir}_mu{mu_in}_e{e_in}"
    J_mean, J_std, used, reason = _run_sn_column(
        material_name, thickness, mu_in, e_in,
        batches=batches, particles_initial=particles_initial,
        particles_max=particles_max, threshold=threshold, rel_err=rel_err,
        threads=threads, work_dir=work_dir,
    )
    return mu_in, e_in, J_mean, J_std, used, reason

# SLURM/PMI variables that make Open MPI treat a bare `openmc` process as an srun rank.
_SLURM_MPI_LAUNCH_VARS = (
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_ID",
    "SLURM_STEPID",
    "SLURM_NPROCS",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_CORE",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_NTASKS_PER_SOCKET",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_STEP_NODELIST",
    "SLURM_STEP_NUM_NODES",
    "SLURM_GTIDS",
    "SLURM_TASKS_PER_NODE",
    "SLURM_LAUNCH_NODE_IPADDR",
    "SLURM_SRUN_COMM_HOST",
    "SLURM_SRUN_COMM_PORT",
    "SLURM_PTY_PORT",
)

_resolved_workers_threads = None


def _isolate_openmc_from_slurm_mpi():
    """OpenMC here is OpenMP-only. Drop PMI/srun launch vars before MPI_Init."""
    for key in list(os.environ):
        if key.startswith(("PMI_", "PMIX_")):
            os.environ.pop(key, None)
    for key in _SLURM_MPI_LAUNCH_VARS:
        os.environ.pop(key, None)
    os.environ["OMPI_MCA_ess"] = "singleton"


def _workers_and_threads():
    """Pool size and OpenMC threads.

    Precedence (old behaviour): ``SN_WORKERS`` / ``SN_THREADS`` env, then
    ``SLURM_NTASKS`` / ``SLURM_CPUS_PER_TASK``, then the constants at the top.
    SLURM PMI vars are stripped after they are read so OpenMC does not inherit them.
    """
    global _resolved_workers_threads
    if _resolved_workers_threads is None:
        if HPC:
            slurm_ntasks = int(os.environ.get("SLURM_NTASKS", SN_WORKERS))
            slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", SN_THREADS))
            workers = int(os.environ.get("SN_WORKERS", slurm_ntasks))
            threads = int(os.environ.get("SN_THREADS", slurm_cpus_per_task))
        else:
            workers = int(os.environ.get("SN_WORKERS", SN_WORKERS))
            threads = int(os.environ.get("SN_THREADS", SN_THREADS))
        _isolate_openmc_from_slurm_mpi()
        os.environ["OMP_NUM_THREADS"] = str(threads)
        _resolved_workers_threads = (workers, threads)
    return _resolved_workers_threads


def generate_sn_tensor(material_name, thickness, *,
                       batches=SN_BATCHES,
                       particles_initial=SN_PARTICLES_INITIAL,
                       particles_max=SN_PARTICLES_MAX,
                       threshold=SN_THRESHOLD,
                       rel_err=SN_REL_ERR,
                       work_dir=None,
                       verbose=True):
    """Generate the S_N transfer tensor for ONE (material, thickness).

    Fires a monodirectional beam at every (mu_in, E_in) and assembles
    ``T_SN[mu_out, E_out, mu_in, E_in]``.  Returns ``(T_mean, T_std)``, each of
    shape ``(N_MU, N_E, N_MU, N_E)``.
    """
    if material_name not in MATERIALS:
        raise KeyError(f"Unknown material {material_name!r}; choose from {list(MATERIALS)}")
    if work_dir is None:
        work_dir = f"openmc_temp_sn_{material_name}_{thickness}cm"

    T_mean = np.zeros((N_MU, N_E, N_MU, N_E))     # [mu_out, E_out, mu_in, E_in]
    T_std = np.zeros((N_MU, N_E, N_MU, N_E))

    max_workers, mc_threads = _workers_and_threads()

    if verbose:
        print(f"  S_N tensor: {material_name} @ {thickness} cm  "
              f"(N_MU={N_MU}, N_E={N_E}, {N_MU * N_E} columns)", flush=True)
        print(f"  Executing with {max_workers} parallel workers, {mc_threads} OpenMC threads per worker.", flush=True)

    t0 = time.perf_counter()
    col = 0
    total_cols = N_MU * N_E

    args_list = []
    for mu_in in range(N_MU):
        for e_in in range(N_E):
            args_list.append((
                material_name, thickness, mu_in, e_in,
                batches, particles_initial, particles_max, 
                threshold, rel_err, mc_threads, work_dir
            ))

    if max_workers > 1:
        with mp.Pool(max_workers) as pool:
            for mu_in, e_in, J_mean, J_std, used, reason in pool.imap_unordered(_worker_run_column, args_list):
                col += 1
                T_mean[:, :, mu_in, e_in] = J_mean.reshape(N_MU, N_E)
                T_std[:, :, mu_in, e_in] = J_std.reshape(N_MU, N_E)
                if verbose:
                    transmission = float(J_mean.sum())
                    print(f"    [{col:3d}/{total_cols}] mu_in={MU_NODES[mu_in]:.4f} "
                          f"E_in={e_in:2d}  T={transmission:.3e}  "
                          f"particles={used}  ({reason})", flush=True)
    else:
        for args in args_list:
            mu_in, e_in, J_mean, J_std, used, reason = _worker_run_column(args)
            col += 1
            T_mean[:, :, mu_in, e_in] = J_mean.reshape(N_MU, N_E)
            T_std[:, :, mu_in, e_in] = J_std.reshape(N_MU, N_E)
            if verbose:
                transmission = float(J_mean.sum())
                print(f"    [{col:3d}/{total_cols}] mu_in={MU_NODES[mu_in]:.4f} "
                      f"E_in={e_in:2d}  T={transmission:.3e}  "
                      f"particles={used}  ({reason})", flush=True)

    if verbose:
        print(f"  -> done in {time.perf_counter() - t0:.1f} s", flush=True)
    return T_mean, T_std


def _sn_paths(material_name, thickness):
    mean_path = os.path.join(OPT_DB_DIR, f"sn_tensor_{material_name}_{thickness}cm.npy")
    std_path = os.path.join(OPT_DB_DIR, f"sn_tensor_{material_name}_{thickness}cm_std.npy")
    return mean_path, std_path


def main():
    os.makedirs(OPT_DB_DIR, exist_ok=True)

    mats_env = os.environ.get("SN_MATERIALS")
    materials = [m.strip() for m in mats_env.split(",")] if mats_env else list(MATERIALS.keys())
    thk_env = os.environ.get("SN_THICKNESSES")
    thicknesses = [float(t) for t in thk_env.split(",")] if thk_env else list(THICKNESSES)
    print("=" * 70)
    print("  S_N TRANSFER-TENSOR GENERATION  (full grid, no compression)")
    print("=" * 70)
    print(f"  Ordinates N_MU : {N_MU}   (mu in {MU_NODES[0]:.3f} .. {MU_NODES[-1]:.3f})")
    print(f"  Energy groups  : {N_E}")
    print(f"  Materials      : {len(materials)}  {materials}")
    print(f"  Thicknesses    : {thicknesses}")
    print(f"  Gate           : {SN_REL_ERR:.0%} above {SN_THRESHOLD:g}, "
          f"{SN_PARTICLES_INITIAL}->{SN_PARTICLES_MAX} particles (adaptive)")
    print(f"  Overwrite      : {OVERWRITE}")
    print("=" * 70, flush=True)

    grid_t0 = time.perf_counter()
    for mat_name, thickness in itertools.product(materials, thicknesses):
        mean_path, std_path = _sn_paths(mat_name, thickness)
        if not OVERWRITE and os.path.exists(mean_path) and os.path.exists(std_path):
            print(f"  [skip] {mat_name} @ {thickness} cm  (already exists)", flush=True)
            continue

        T_mean, T_std = generate_sn_tensor(mat_name, thickness)
        np.save(mean_path, T_mean)
        np.save(std_path, T_std)
        print(f"  saved {os.path.basename(mean_path)}  (+ _std)", flush=True)

    print("=" * 70)
    print(f"  Grid complete in {time.perf_counter() - grid_t0:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
    


