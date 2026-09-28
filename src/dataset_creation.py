import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from base_plates import (
    N_BINS, HPC, THREADS,
    MPI_TASKS_HPC, OMP_THREADS_RR_HPC, OMP_THREADS_CE_HPC,
    thicknesses_to_x_bounds,
    generate_weight_windows, run_with_weight_windows, extract_flux,
)

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

MATERIALS = ['cast_iron', 'HDPE']

X_START_BOUNDS = (0.1, 5.0)          # cm, source -> first plate
THICKNESS_BOUNDS = [                 # cm, one (min, max) per layer
    (30.0, 95.0),
    (0.1, 20.0),
]
L_BOUNDS = (25.0, 90.0)             # cm, square plate side

N_SAMPLES = 350
SOBOL_SEED = 42

RR_BATCHES = 100
RR_INACTIVE_BATCHES = 50
RR_PARTICLES = 60_000
MGXS_PARTICLES = 60_000

CE_PARTICLES = 200_000
CE_BATCHES = 100
MAX_LOWER_BOUND_RATIO = 1.0
MAX_SPLIT = 1000
CE_TIMEOUT_S = 600.0                 # abort one CE run after this many seconds
USE_ANGLE_BIAS = True

SAMPLE_WORK_DIR = 'sample_workdir'
MGXS_PATH = Path(SAMPLE_WORK_DIR) / 'mgxs.h5'   # one library for the whole sweep
OUTPUT_CSV = Path('examples/demo_hcpb_vv.csv')

# ==================================================================

if len(THICKNESS_BOUNDS) != len(MATERIALS):
    raise ValueError(
        f"THICKNESS_BOUNDS has {len(THICKNESS_BOUNDS)} entries but "
        f"MATERIALS has {len(MATERIALS)}. They must match."
    )

N_PLATES = len(MATERIALS)


# -- Sobol design ---------------------------------------------------

def sobol_design_points(n, skip):
    """n new points. skip is the number of CSV rows already written."""
    engine = qmc.Sobol(d=2 + N_PLATES, scramble=True, seed=SOBOL_SEED)
    if skip > 0:
        engine.fast_forward(skip)
    unit = engine.random(n=n)

    lower = np.array(
        [X_START_BOUNDS[0], L_BOUNDS[0], *[lo for lo, _ in THICKNESS_BOUNDS]]
    )
    upper = np.array(
        [X_START_BOUNDS[1], L_BOUNDS[1], *[hi for _, hi in THICKNESS_BOUNDS]]
    )
    samples = qmc.scale(unit, lower, upper)
    return [
        (thicknesses_to_x_bounds(x_start=row[0], thicknesses=row[2:].tolist()), float(row[1]))
        for row in samples
    ]


# -- One sample -----------------------------------------------------

def run_sample(x_bounds, L, run_dir):
    """FW-CADIS pair. Returns flux, absolute sigma, phase-1 seconds, phase-2 seconds."""
    ww_dir = os.path.join(run_dir, 'ww')
    ce_dir = os.path.join(run_dir, 'ce')

    print('[ww] ', end='', flush=True)
    t0 = time.perf_counter()
    ww_path, *_ = generate_weight_windows(
        x_bounds=x_bounds,
        materials=MATERIALS,
        ww_dir=ww_dir,
        rr_batches=RR_BATCHES,
        rr_inactive_batches=RR_INACTIVE_BATCHES,
        rr_particles=RR_PARTICLES,
        mgxs_particles=MGXS_PARTICLES,
        output=False,
        mgxs_path=str(MGXS_PATH),
        y_dim=L,
        z_dim=L,
    )
    ww_seconds = time.perf_counter() - t0

    print(f'{ww_seconds:.0f}s [ce] ', end='', flush=True)
    t0 = time.perf_counter()
    statepoint = run_with_weight_windows(
        x_bounds=x_bounds,
        materials=MATERIALS,
        ww_path=ww_path,
        ce_dir=ce_dir,
        batches=CE_BATCHES,
        particles=CE_PARTICLES,
        use_angle_bias=USE_ANGLE_BIAS,
        max_lower_bound_ratio=MAX_LOWER_BOUND_RATIO,
        max_split=MAX_SPLIT,
        output=False,
        timeout=CE_TIMEOUT_S,
        y_dim=L,
        z_dim=L,
    )
    ce_seconds = time.perf_counter() - t0

    flux, rel_unc_pct = extract_flux(statepoint)
    sigma = flux * (rel_unc_pct / 100.0)
    return flux, sigma, ww_seconds, ce_seconds


# -- CSV ------------------------------------------------------------

def csv_header():
    material_cols = [f'mat_{i + 1}' for i in range(N_PLATES)]
    x_cols = [f'x{i + 1}' for i in range(N_PLATES + 1)]
    flux_cols = [f'y_{i:02d}' for i in range(N_BINS)]
    sigma_cols = [f'sigma_{i:02d}' for i in range(N_BINS)]
    return (
        ['run_idx']
        + material_cols
        + x_cols
        + ['L']
        + ['use_angle_bias', 'ce_particles', 'ce_batches',
           'wall_time_ww_s', 'wall_time_ce_s', 'wall_time_s']
        + flux_cols
        + sigma_cols
    )


def count_csv_rows(path):
    if not path.exists():
        return 0
    with open(path, newline='') as f:
        return sum(1 for _ in csv.DictReader(f))


def last_run_idx(path):
    if not path.exists():
        return 0
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]['run_idx']) if rows else 0


def print_banner(file_mode, n_existing):
    print('Sobol refinement - N-layer plate FW-CADIS sampling')
    print(f'  layers          : {N_PLATES}')
    print(f'  stack           : {" | ".join(MATERIALS)}')
    print(f'  x_start bounds  : {X_START_BOUNDS} cm')
    for i, (lo, hi) in enumerate(THICKNESS_BOUNDS):
        print(f'  t{i + 1} bounds       : ({lo:.1f}, {hi:.1f}) cm  [{MATERIALS[i]}]')
    print(f'  L bounds        : {L_BOUNDS} cm  (square plate transverse size)')
    print(f'  sobol seed      : {SOBOL_SEED}')
    print(f'  n_samples       : {N_SAMPLES}')
    print(f'  existing rows   : {n_existing}  '
          f'(Sobol sequence fast-forwarded by {n_existing})')
    print(f'  WW gen          : rr_batches={RR_BATCHES}, '
          f'rr_particles={RR_PARTICLES}, mgxs_particles={MGXS_PARTICLES}')
    print(f'  mgxs            : {MGXS_PATH}  '
          f'({"reuse" if MGXS_PATH.is_file() else "generate once, then reuse"})')
    print(f'  CE run          : particles={CE_PARTICLES}, batches={CE_BATCHES}')
    print(f'  angle bias      : {USE_ANGLE_BIAS}')
    if HPC:
        print('  backend         : HPC')
        print(f'    Phase 1 (random ray) : {OMP_THREADS_RR_HPC} OMP threads, no MPI')
        print(f'    Phase 2 (CE + WW)    : {MPI_TASKS_HPC} MPI ranks * '
              f'{OMP_THREADS_CE_HPC} OMP thread(s), shared_secondary_bank=True')
    else:
        print(f'  backend         : local threads ({THREADS})')
    print(f'  output          : {OUTPUT_CSV}  (mode={file_mode!r})')
    print()


# -- Run ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sobol sampling with FW-CADIS for an N-layer plate shielding stack.'
    )
    parser.add_argument(
        '--regen', action='store_true',
        help='Overwrite the CSV and restart the Sobol sequence.',
    )
    parser.add_argument(
        '--refine', action='store_true',
        help='Append N_SAMPLES rows, skipping points already in the CSV.',
    )
    args = parser.parse_args()

    if args.regen or not OUTPUT_CSV.exists():
        file_mode, n_existing, run_idx = 'w', 0, 0
        if args.regen and MGXS_PATH.is_file():
            MGXS_PATH.unlink()
    elif args.refine:
        file_mode = 'a'
        n_existing = count_csv_rows(OUTPUT_CSV)
        run_idx = last_run_idx(OUTPUT_CSV)
    else:
        print(
            f"'{OUTPUT_CSV}' already exists.\n"
            f"  Use --regen to overwrite it, or --refine to append more samples."
        )
        return

    design_points = sobol_design_points(N_SAMPLES, skip=n_existing)
    print_banner(file_mode, n_existing)

    n_attempt = 0
    n_written = 0
    with open(OUTPUT_CSV, file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow(csv_header())
            f.flush()

        for x_bounds, L in design_points:
            run_idx += 1
            n_attempt += 1
            x_tag = '(' + ', '.join(f'{x:6.1f}' for x in x_bounds) + ')'
            print(
                f'[{n_attempt:4d}/{N_SAMPLES}] idx={run_idx:4d}  x={x_tag}  L={L:6.1f} ... ',
                end='', flush=True,
            )

            run_dir = os.path.join(SAMPLE_WORK_DIR, f'run_{run_idx:04d}')
            t0 = time.perf_counter()
            try:
                flux, sigma, ww_seconds, ce_seconds = run_sample(x_bounds, L, run_dir)
            except Exception as err:
                print(f'FAILED ({type(err).__name__}: {err})')
                shutil.rmtree(run_dir, ignore_errors=True)
                continue
            elapsed = time.perf_counter() - t0

            writer.writerow([
                run_idx,
                *MATERIALS,
                *[f'{x:.4f}' for x in x_bounds],
                f'{L:.4f}',
                int(USE_ANGLE_BIAS),
                CE_PARTICLES,
                CE_BATCHES,
                f'{ww_seconds:.2f}',
                f'{ce_seconds:.2f}',
                f'{elapsed:.2f}',
                *[f'{v:.6e}' for v in flux.tolist()],
                *[f'{v:.6e}' for v in sigma.tolist()],
            ])
            f.flush()
            n_written += 1

            shutil.rmtree(run_dir, ignore_errors=True)
            print(f'done ({elapsed:6.1f} s  |  ww {ww_seconds:5.1f} s, ce {ce_seconds:5.1f} s)')

    print(f'\nWrote {n_written} new rows to {OUTPUT_CSV}.')


if __name__ == '__main__':
    main()
