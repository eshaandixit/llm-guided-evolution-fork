"""Report for one or more QuantumVQC evolution islands (e.g. the RAG / no-RAG pair).

    python tools/qvqc_run_report.py --run qvqc_rag=run_job_outputs/islands/<rag log> \
                                    --run qvqc_norag=run_job_outputs/islands/<norag log> \
                                    --out reports/qvqc_runs

Islands share sota/QuantumVQC/results/, so each gene is assigned to an island and a
generation from that island's orchestrator log ("STARTING GENERATION: n" markers;
gen 0 = before the first marker). Fitness comes from <gene>_results.txt; everything
else from <gene>_metrics.json.

Selection objectives (both minimised, exactly what NSGA-II saw):
    obj1 = 1 - validation accuracy averaged over N_STARTS trainings
    obj2 = plain gate count
An individual is on the Pareto front if no other individual of the same island is
<= on both objectives and < on at least one. Hypervolume is measured against the
reference point (obj1 = 0.5 [chance], obj2 = 150 gates).

Every invalid individual is also given a failure cause from its evaluation log
(hallucinated import, wrong attribute, syntax, ...), and every valid one is checked for
being STRUCTURALLY new (gate histogram differs from the seed's) rather than a clone.

Outputs: pareto_fronts.png, hypervolume_validity.png, vs_exaqc.png, pareto_circuits.png,
individuals.csv, summary.md
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "sota" / "QuantumVQC" / "results"
REF = (0.5, 150.0)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})
GENE = re.compile(r"(xXx[A-Za-z0-9]{24})")
# EXAQC (Kar, Krutz, Desell 2026, arXiv:2602.03840) Table 1, Breast Cancer rows:
# (test accuracy, # gates). Single-objective cross-entropy search, Adam 200 epochs.
EXAQC_BC = [(0.892, 10), (0.908, 11), (0.886, 11), (0.910, 12)]
FAILURE_CLASSES = [
    ("hallucinated import/API", ("ImportError", "ModuleNotFoundError")),
    ("wrong attribute/method", ("AttributeError",)),
    ("undefined name", ("NameError",)),
    ("syntax/indentation", ("SyntaxError", "IndentationError")),
    ("circuit contract (qubits/params/values)", ("ValueError", "CircuitError", "QiskitError")),
    ("type/index error", ("TypeError", "IndexError", "KeyError")),
]


def failure_index():
    """gene -> last exception line in its evaluation log (one pass over all logs)."""
    index = {}
    for f in glob.glob(str(ROOT / "run_job_outputs" / "evaluation" / "*.out")):
        text = open(f, errors="replace").read()
        errs = re.findall(r"^(\w*(?:Error|Exception)\b.*)$", text, re.M)
        for g in set(GENE.findall(text)):
            index[g] = errs[-1][:200] if errs else None
    return index


def classify(line):
    if line is None:
        return "no evaluation (LLM stage failed or still running)"
    name = line.split(":")[0].split(".")[-1]
    for label, names in FAILURE_CLASSES:
        if name in names:
            return label
    return "other: " + name


def _lines(log_paths):
    for path in log_paths.split(","):  # a resumed run spans several island logs, oldest first
        yield from open(path, errors="replace")


def genes_by_generation(log_path):
    """{gene: first generation it was evaluated in}. Only genes that reached evaluation
    ("Checking the Results for Gene" or a fitness line) or were created count.
    log_path may be several comma-separated logs of one resumed run, oldest first."""
    first, gen = {}, 0
    for line in _lines(log_path):
        m = re.search(r"STARTING GENERATION: (\d+)", line)
        if m:
            gen = int(m.group(1))
            continue
        if ("Bash script saved to" in line or "Checking the Results for Gene" in line
                or "Running py File for" in line):
            for g in GENE.findall(line):
                first.setdefault(g, gen)
    return first


def load_gene(g):
    rec = {"gene": g, "valid": False}
    rf, mf = RESULTS / f"{g}_results.txt", RESULTS / f"{g}_metrics.json"
    if rf.exists():
        try:
            vals = [float(v) for v in rf.read_text().split(",")]
            if np.isfinite(vals[:2]).all():
                rec.update(obj1=vals[0], obj2=vals[1], valid=True)
        except ValueError:
            pass
    if mf.exists():
        m = json.loads(mf.read_text())
        for k in ("val_accuracy", "val_accuracy_sd", "test_accuracy", "best_start_test_accuracy",
                  "train_accuracy", "two_qubit_gates", "depth", "n_params", "gate_histogram",
                  "weighted_gate_cost", "seconds", "dead_rotations"):
            rec[k] = m.get(k)
    return rec


def pareto(points):
    idx = []
    for i, (a, b) in enumerate(points):
        if not any((c <= a and d <= b) and (c < a or d < b) for j, (c, d) in enumerate(points) if j != i):
            idx.append(i)
    return idx


def hypervolume(points):
    pts = sorted({(min(a, REF[0]), min(b, REF[1])) for a, b in points}, key=lambda p: (p[1], p[0]))
    hv, best = 0.0, REF[0]
    # sweep in increasing gates; area between consecutive gate levels at the best error so far
    levels = [p[1] for p in pts] + [REF[1]]
    for k, (e, gates) in enumerate(pts):
        best = min(best, e)
        hv += (REF[0] - best) * (levels[k + 1] - gates)
    return hv


def seed_metrics():
    """The seed evaluated exactly like an individual (5 starts, same budget)."""
    for name in ("seed_metrics.json", "xXxSMOKEtest_metrics.json"):
        m = RESULTS / name
        if m.exists():
            return json.loads(m.read_text())
    return None


def seed_point():
    d = seed_metrics()
    return (d["obj1_val_error"], d["obj2_gates"], d.get("test_accuracy")) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path/to/orchestrator.log[,resumed.log...]")
    ap.add_argument("--out", default=str(ROOT / "reports" / "qvqc_runs"))
    ap.add_argument("--max-circuits", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    failures = failure_index()
    seed_hist = (seed_metrics() or {}).get("gate_histogram")
    runs = {}
    for spec in args.run:
        name, log = spec.split("=", 1)
        gens = genes_by_generation(log)
        recs = []
        for g, gen in gens.items():
            r = load_gene(g)
            r.update(run=name, generation=gen)
            r["structurally_new"] = r["valid"] and r.get("gate_histogram") != seed_hist
            if not r["valid"]:
                r["failure"] = classify(failures.get(g))
            recs.append(r)
        runs[name] = recs
    seed = seed_point()

    # ---------- Pareto fronts ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    summary = ["# QuantumVQC evolution report\n",
               "Objectives (both minimised): obj1 = 1 - mean validation accuracy over "
               "N_STARTS trainings; obj2 = plain gate count. Front = non-dominated set per island. "
               f"Hypervolume reference point = (error {REF[0]}, {REF[1]:.0f} gates).\n"]
    rows = []
    fronts = {}
    for k, (name, recs) in enumerate(runs.items()):
        valid = [r for r in recs if r["valid"]]
        pts = [(r["obj1"], r["obj2"]) for r in valid]
        front = [valid[i] for i in pareto(pts)] if pts else []
        front.sort(key=lambda r: r["obj2"])
        fronts[name] = front
        for r in recs:
            r["on_front"] = r in front
        ax.scatter([r["obj2"] for r in valid], [1 - r["obj1"] for r in valid], s=18,
                   color=COLORS[k], alpha=0.35, edgecolor="none")
        ax.step([r["obj2"] for r in front], [1 - r["obj1"] for r in front], where="post",
                color=COLORS[k], linewidth=2, label=f"{name}: front ({len(front)})")
        ax.scatter([r["obj2"] for r in front], [1 - r["obj1"] for r in front], s=48,
                   color=COLORS[k], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        n_gen = max([r["generation"] for r in recs], default=0)
        summary.append(f"## {name}\n")
        summary.append(f"- generations evaluated: {n_gen + 1} (0..{n_gen}); individuals: {len(recs)}; "
                       f"valid: {len(valid)} ({len(valid) / max(len(recs), 1):.0%})")
        summary.append(f"- final hypervolume: {hypervolume(pts) if pts else 0:.2f}"
                       f" (seed alone: {hypervolume([seed[:2]]) if seed else 0:.2f})")
        summary.append(f"- structurally new (gate histogram differs from seed): "
                       f"{sum(r['structurally_new'] for r in valid)}/{len(valid)} valid individuals")
        if seed:
            dom = [r for r in valid if r["obj1"] <= seed[0] and r["obj2"] <= seed[1]
                   and (r["obj1"], r["obj2"]) != tuple(seed[:2])]
            summary.append(f"- individuals that dominate the seed: {len(dom)}")
        causes = {}
        for r in recs:
            if not r["valid"]:
                causes[r["failure"]] = causes.get(r["failure"], 0) + 1
        summary.append("- failure causes: " + (", ".join(
            f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) or "none"))
        summary.append("\n| gene | gen | gates | 2q gates | depth | params | val acc (mean±sd) | test acc (mean) | histogram |")
        summary.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in front:
            summary.append(f"| {r['gene']} | {r['generation']} | {r['obj2']:.0f} | {r.get('two_qubit_gates')} | "
                           f"{r.get('depth')} | {r.get('n_params')} | {1 - r['obj1']:.1%} ± {r.get('val_accuracy_sd') or 0:.1%} | "
                           f"{(r.get('test_accuracy') or float('nan')):.1%} | {r.get('gate_histogram')} |")
        summary.append("")
        rows += recs
    if seed:
        ax.scatter([seed[1]], [1 - seed[0]], marker="*", s=220, color=INK, zorder=4, label="seed")
    ax.set_xlabel("Gate count (obj2, minimise)")
    ax.set_ylabel("Mean validation accuracy (1 - obj1)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Pareto fronts: accuracy vs gate count (all evaluated individuals faded)", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out / "pareto_fronts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- hypervolume + validity per generation ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (name, recs) in enumerate(runs.items()):
        n_gen = max([r["generation"] for r in recs], default=0)
        hv, valid_rate, gens = [], [], list(range(n_gen + 1))
        for g in gens:
            upto = [(r["obj1"], r["obj2"]) for r in recs if r["valid"] and r["generation"] <= g]
            hv.append(hypervolume(upto) if upto else 0)
            this = [r for r in recs if r["generation"] == g]
            valid_rate.append(sum(r["valid"] for r in this) / max(len(this), 1))
        axes[0].plot(gens, hv, color=COLORS[k], marker="o", markersize=5, label=name)
        axes[1].plot(gens, valid_rate, color=COLORS[k], marker="o", markersize=5, label=name)
    axes[0].set_title("Hypervolume of the cumulative front", loc="left")
    axes[1].set_title("Share of each generation's new individuals that evaluate", loc="left")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=9)
    fig.savefig(out / "hypervolume_validity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- EXAQC comparison on its own axes: TEST accuracy vs # gates ----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, (name, front) in enumerate(fronts.items()):
        f = [r for r in front if r.get("test_accuracy") is not None]
        ax.scatter([r["obj2"] for r in f], [r["test_accuracy"] for r in f], s=55, color=COLORS[k],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=f"LLM-GE {name} front")
    ax.scatter([g for _, g in EXAQC_BC], [a for a, _ in EXAQC_BC], marker="D", s=50,
               color="#4a3aa7", edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="EXAQC Table 1 (Breast Cancer)")
    if seed and seed[2] is not None:
        ax.scatter([seed[1]], [seed[2]], marker="*", s=220, color=INK, zorder=4, label="seed")
    ax.set_xlabel("# gates")
    ax.set_ylabel("Test accuracy (mean over starts; never used for selection)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Breast Cancer: compared on EXAQC's Table 1 axes", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out / "vs_exaqc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- circuit gallery of front members ----------
    gallery = [(name, r) for name, front in fronts.items() for r in front[: args.max_circuits]
               if (RESULTS / f"{r['gene']}_circuit.png").exists()]
    if gallery:
        fig, axes = plt.subplots(len(gallery), 1, figsize=(12, 2.4 * len(gallery)))
        axes = np.atleast_1d(axes)
        for ax, (name, r) in zip(axes, gallery):
            ax.imshow(mpimg.imread(RESULTS / f"{r['gene']}_circuit.png"))
            ax.set_axis_off()
            ax.set_title(f"{name}  {r['gene']}  gen {r['generation']}:  {r['obj2']:.0f} gates, "
                         f"val {1 - r['obj1']:.1%}, test {(r.get('test_accuracy') or float('nan')):.1%}",
                         loc="left", fontsize=9)
        fig.savefig(out / "pareto_circuits.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    keys = ["run", "gene", "generation", "valid", "failure", "structurally_new", "on_front", "obj1", "obj2", "val_accuracy",
            "val_accuracy_sd", "test_accuracy", "best_start_test_accuracy", "train_accuracy",
            "two_qubit_gates", "depth", "n_params", "weighted_gate_cost", "dead_rotations", "seconds"]
    with open(out / "individuals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print("wrote", out)


if __name__ == "__main__":
    main()
