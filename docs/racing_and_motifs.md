# Racing, duplicate removal and the motif library

How to run the QuantumVQC (Iris, 4 qubits) evolution with the features added on the
`quantum-iris` and `motif-library` branches. Every feature is behind an environment
variable and is off by default.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `LLMGE_RACING` | `0` | Train each child 3 times, then add starts (3 at a time, up to 10) only to circuits whose Pareto rank is uncertain. Code: `src/racing.py`, `race_population()` in `run_improved.py`. |
| `LLMGE_RACING_INITIAL_STARTS`, `_STEP`, `_MAX_STARTS`, `_Z`, `_MAX_RANK` | `3`, `3`, `10`, `2.0`, `1` | Racing's knobs: starts per child, starts per round, cap, how many standard errors count as uncertain, which Pareto ranks are worth resolving. |
| `LLMGE_DEDUPE` | `0` | Keep one gene per distinct program before selection. Two genes are the same program when they build the same circuit (saved QASM) and their readout and training code is the same. Parent slots are filled by picking distinct programs again. |
| `LLMGE_NOOP_RETRIES` | `2` | Re-ask the LLM when its reply changes nothing that runs, or rewrites code outside the block. `0` turns this off. |
| `LLMGE_STOP_AT_STARTS` | `0` | Stop after the generation in which this many training starts have been used. |
| `LLMGE_MOTIFS` | `0` | Motif library (`motif-library` branch only): rebuild it from generation 2, every 2 generations, and offer up to 3 motifs for the block being edited in normal (non-EoT) prompts. |
| `LLMGE_MOTIF_START_GEN`, `_EVERY`, `LLMGE_MOTIFS_PER_PROMPT`, `LLMGE_MOTIF_MAX_RANK` | `2`, `2`, `3`, `1` | The motif library's knobs. |

These are always on:
* The mutation job trims whole-program replies down to the block it asked for
  (`src/block_fit.py`).
* The seed (`sota/QuantumVQC/network.py`) uses the same AVX2 numeric code on every
  node, so the same program scores the same on PACE's Intel and AMD nodes.

## Output per run

| File | Contents |
|---|---|
| `<checkpoints>/progress.jsonl` | One line per generation: training starts used so far, population, Pareto front |
| `<checkpoints>/motif_library.json` | The motif library, when `LLMGE_MOTIFS=1` |
| `sota/QuantumVQC/results/<gene>_metrics.json` | Every training start, merged across racing rounds |
| `sota/QuantumVQC/results/<gene>_mutation.json` | Parent, block, template, outcome, each LLM attempt, motifs offered |

## Running on PACE-ICE

```bash
uv run python tools/racing_smoke.py                 # racing only, no LLM, ~5 min
bash tools/racing_full_run.sh                       # one short run with racing, own server
RUNS=1 CONTROL_GENS=4 bash tools/racing_experiment.sh pilot
bash tools/racing_experiment.sh full                # control vs racing, 3 runs each
ARMS="racing motifs" bash tools/racing_experiment.sh lib   # motif-library branch
```

`tools/racing_experiment.sh` starts one shared LLM server and every run, puts each run
in `experiments/<name>/<arm>_<i>/`, and cancels the server when the runs end. The
racing arms run until they reach the control's maximum compute, so arms can be
compared at equal training starts.

When the runs are done:

```bash
uv run python tools/rescore_front.py experiments/<name>    # 30 fresh starts per front circuit
uv run python tools/racing_analysis.py experiments/<name>  # analysis.json, hypervolume.png
```

The analysis reports, for each run:
* hypervolume against training starts
* hypervolume at the equal-compute budget
* the starts needed to reach the control's hypervolume
* how far reported errors were from the re-scored errors

## Tests

```bash
uv run python -m pytest --noconftest tests/test_racing.py tests/test_block_fit.py tests/test_racing_analysis.py
```

`--noconftest` skips `tests/conftest.py`, which tries to start the LLM server.
