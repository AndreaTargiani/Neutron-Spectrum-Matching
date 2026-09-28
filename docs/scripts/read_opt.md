# `read_opt.py` — Replay a Saved MOTPE Study

## Purpose

This script is **optional**. After [`optimizer.py`](optimizer.md) has written `study_results_multistart.pkl`, run `read_opt.py` to reprint the MOTPE summary and regenerate the design-choice plots without re-running the search.

Replay is pickle-only. That means you can also compare **different runs**, including runs that targeted different spectra, as long as each run's pickle is still on disk.

The pickle format, the terminal report, and the two PNG files are the same as in [optimizer outputs](optimizer.md#outputs). This page only covers how to replay them.

---

## Configuration

```python
PKL_PATH = "study_results_multistart.pkl"
TOP_N_CONFIGURATIONS = 40
SAVE_PLOTS = True
```

| Constant | Role |
|---|---|
| `PKL_PATH` | Default pickle. Override with a command-line argument |
| `TOP_N_CONFIGURATIONS` | How many unique sequences to list. Independent of the value in `optimizer.py` — raise it here if you want a longer catalogue than the original run printed |
| `SAVE_PLOTS` | Rewrite `cost_error_tradeoff.png` and `pareto_designs.png` |

---

## Usage

```bash
python src/read_opt.py
python src/read_opt.py path/to/study_results_multistart.pkl
```

The working directory is wherever you launch it: plots are written there.

---

## Outputs

Same as [`optimizer.py`](optimizer.md#outputs): the per-restart table, combined Pareto front, design-choice listing, and (if `SAVE_PLOTS`) the two PNG files.
