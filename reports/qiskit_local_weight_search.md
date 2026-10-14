# Qiskit Local Weight Search Results

## Status

The local weight-search prototype and three-way ablation are complete on the fixed Breast Cancer Wisconsin split. The results below come from `qiskit_vqc_optimizer_ablation.csv`; per-step validation curves are in `qiskit_vqc_training_curves.csv`.

| Optimizer | Initialization | Validation accuracy | Validation error | Validation cross-entropy | Gates | Training passes used / max | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| COBYLA | Random | 86% | 14% | 0.4773 | 56 | 149 / 178 | 95.0 |
| SPSA | Random | 22% | 78% | 0.6971 | 56 | 88 / 89 | 58.7 |
| SPSA | COBYLA-trained warm start | 85% | 15% | 0.4557 | 56 | 88 / 89 | 58.1 |

## Findings

- Random-start SPSA was substantially worse than COBYLA at this budget: 22% versus 86% validation accuracy.
- Initializing SPSA from COBYLA-trained angles recovered most of that gap (85% validation accuracy) and achieved the lowest validation cross-entropy (0.4557). This supports weight inheritance/warm starts as a useful local-search strategy for this circuit and split.
- All three circuits have 56 gates, so this experiment compares weight training only; it does not show a structural or gate-count improvement.
- Early stopping ended COBYLA after 149 of 178 possible full training passes and both SPSA trials after 88 of 89. The saved curves show 63, 44, and 44 validation checks, respectively. Runtime was about 37 seconds lower for SPSA than COBYLA in this single run, while the warm start retained similar accuracy. These timings are indicative only, not a benchmark.
- A separate smoke run completed SPSA with a 2-pass budget and wrote finite metrics (validation accuracy 22%, validation cross-entropy 0.7003). This confirms the budget enforcement and output path work in the current project virtualenv.

## Relation to the Paper

The cited paper, [Investigating Quantum Circuit Designs Using Neuro-Evolution](https://arxiv.org/html/2602.03840v1), reports 200-epoch Adam training and uses Lamarckian weight inheritance in evolutionary search. Its binary crossover also recombines shared gate parameters with a randomized line search. It does not present SPSA as its local optimizer. This repo's SPSA routine is therefore a lightweight local-search experiment inspired by the broader weight-inheritance idea, not a reproduction of the paper's optimizer or protocol.

The paper reports classification performance and circuit gate counts across benchmark runs; this local ablation reports validation accuracy/error, validation cross-entropy, gate count, training-set passes, and runtime. The dataset name is shared, but the splits, training protocol, and evaluation budget differ, so these numbers should not be compared as a direct paper-versus-repo score. For notebook comparisons, use the same data split and clearly label validation versus test metrics. Test data was not used to select weights here.

## Reproduce

From the repository root, run the completed three-way comparison:

```bash
.venv/bin/python sota/QuantumVQC/run_optimizer_ablation.py
```

To run a small SPSA budget smoke check:

```bash
.venv/bin/python sota/QuantumVQC/seed_vqc.py \
  --gene-id local_weight_smoke \
  --out-dir /tmp/qiskit-local-weight-smoke \
  --repr "" --optimizer spsa --train-budget 2
```

The training curves can be plotted directly from `qiskit_vqc_training_curves.csv`, grouping by `optimizer` and `initialization`, with `train_evals` on the x-axis and `validation_cross_entropy` on the y-axis. The aggregate notebook-ready table is `qiskit_vqc_optimizer_ablation.csv`.

## Caveats / Next Comparison

This is one seeded run per optimizer configuration, not a multi-seed statistical comparison. SPSA's random-start result is weak, and the warm-start result depends on a separately trained COBYLA solution. Before choosing an optimizer for the evolutionary pipeline, run multiple seeds and compare matched actual circuit-simulation counts (not only configured maxima), while keeping the same validation split and reporting held-out test performance only after selection.
