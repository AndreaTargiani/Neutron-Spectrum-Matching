import os
import pickle
import sys

from optimizer import StudyResult, report_optimization_results

# ==================================================================
#  CONFIGURATION  (edit these)
# ==================================================================

PKL_PATH = "study_results_multistart.pkl"
TOP_N_CONFIGURATIONS = 10
SAVE_PLOTS = True

# ==================================================================


def load_all_studies(path=PKL_PATH):
    """Rebuild the `(seed, StudyResult)` list written by `save_all_studies`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    studies = sorted(data["studies"], key=lambda d: d["restart_idx"])
    return [
        (
            d["seed"],
            StudyResult(
                d["best_value"],
                d["trials"],
                d.get("direction") or d.get("directions"),
                directions=d.get("directions"),
                n_pareto=d.get("n_pareto", 0),
                hypervolume=d.get("hypervolume", 0.0),
            ),
        )
        for d in studies
    ]


def main():
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else PKL_PATH
    all_studies = load_all_studies(pkl_path)

    print(f"\n{'─'*62}")
    print(f"  Loaded {sum(len(s.trials) for _, s in all_studies)} total trials "
          f"({len(all_studies)} restarts) ← {pkl_path}")

    report_optimization_results(
        all_studies,
        save_plots=SAVE_PLOTS,
        top_n=TOP_N_CONFIGURATIONS,
    )


if __name__ == "__main__":
    main()
