"""Figures + summary table for the offline QuantumVQC studies.

    python tools/plot_qiskit_studies.py

Reads reports/qiskit_studies/*.json (from sota/QuantumVQC/run_training_studies.py)
and writes reports/figures/*.png plus reports/qiskit_studies/summary.md.
Draws only the studies whose JSON exists.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "qiskit_studies"
FIG = ROOT / "reports" / "figures"
# Reference categorical slots 1-3 (validated all-pairs, light mode) + text tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})
summary = []


def load(name):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def strip(ax, groups, colors, ylabel):
    """Dots per run + a mean bar per group."""
    for i, (label, values) in enumerate(groups):
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
        ax.scatter(i + np.asarray(jitter), values, s=36, color=colors[i % len(colors)],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.hlines(np.mean(values), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.annotate(f"{np.mean(values):.1%}", (i + 0.28, np.mean(values)), va="center",
                    color=INK, fontsize=9)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def falloff():
    runs = load("falloff")
    if not runs:
        return
    grid = np.arange(0, 601, 5)

    def curve(r, key):
        x = np.array([c["train_evals"] for c in r["curve"]])
        y = np.array([c[key] for c in r["curve"]])
        out = np.interp(grid, x, y, left=np.nan)
        out[grid > x.max()] = y[-1]                 # COBYLA converged: hold its last value
        return out

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    stats = {}
    for ax, key, label in ((axes[0], "validation_cross_entropy", "Validation cross-entropy"),
                           (axes[1], "validation_accuracy", "Validation accuracy")):
        M = np.array([curve(r, key) for r in runs])
        mean, lo, hi = np.nanmean(M, 0), np.nanpercentile(M, 10, 0), np.nanpercentile(M, 90, 0)
        ax.fill_between(grid, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(grid, mean, color=BLUE, label="mean of 10 starts (band: 10th-90th pct)")
        for x, txt in ((178, "current budget\n178 passes"),):
            ax.axvline(x, color=INK2, linewidth=1, linestyle="--")
            ax.annotate(txt, (x + 8, np.nanmax(mean) if key.endswith("entropy") else np.nanmin(mean)),
                        color=INK2, fontsize=8, va="top" if key.endswith("entropy") else "bottom")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Training passes over the 341-sample train set")
        stats[key] = (grid, mean)
        if key == "validation_accuracy":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Seed VQC, COBYLA, early stopping off: where does training stop paying?",
                 x=0.01, y=1.05, ha="left", color=INK)
    save(fig, "fig_epoch_falloff.png")

    g, ce = stats["validation_cross_entropy"]
    _, acc = stats["validation_accuracy"]
    valid = ~np.isnan(ce)
    start, end = ce[valid][0], ce[valid][-1]
    knee95 = g[valid][np.argmax(ce[valid] <= start - 0.95 * (start - end))]
    knee99 = g[valid][np.argmax(ce[valid] <= start - 0.99 * (start - end))]
    acc_final = acc[valid][-1]
    acc_knee = g[valid][np.argmax(acc[valid] >= acc_final - 0.01)]
    summary.append("## Epoch fall-off (COBYLA, seed, 10 starts, 600-pass budget, no early stop)\n")
    summary.append(f"- validation CE: {start:.3f} at first check -> {end:.3f} at 600 passes; "
                   f"95% of the improvement by **{knee95} passes**, 99% by {knee99}")
    summary.append(f"- validation accuracy within 1 pt of its 600-pass value ({acc_final:.1%}) "
                   f"by **{acc_knee} passes**")
    i178 = np.argmin(abs(g - 178))
    summary.append(f"- at the current 178-pass budget: CE {ce[i178]:.3f}, accuracy {acc[i178]:.1%}\n")


def seed_multistart():
    m = load("seed_multistart")
    if not m:
        return
    vals = [s["val_accuracy"] for s in m["starts"]]
    tests = [s["test_accuracy"] for s in m["starts"]]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    strip(ax, [("validation (114)", vals), ("test (114)", tests)], [BLUE, ORANGE], "Accuracy")
    ax.set_title("Same circuit, 20 random starting angles", loc="left")
    save(fig, "fig_seed_multistart.png")
    sd = np.std(vals)
    summary.append("## Seed multi-start noise (20 starts, 178-pass COBYLA)\n")
    summary.append(f"- validation accuracy {np.mean(vals):.1%} mean, range {min(vals):.1%}-{max(vals):.1%}, "
                   f"SD {sd:.1%}; test {np.mean(tests):.1%} mean")
    summary.append(f"- standard error of an N-start mean: N=1 {sd:.1%}, N=5 {sd/np.sqrt(5):.1%}, "
                   f"N=10 {sd/np.sqrt(10):.1%}\n")


def optimizers():
    runs = load("optimizers")
    if not runs:
        return
    order = [("cobyla", "COBYLA\n178 passes"), ("spsa", "SPSA\n178 passes"),
             ("adam", "Adam + param-shift\n6400 passes")]
    groups = [(label, [r["val_accuracy"] for r in runs if r["optimizer"] == o]) for o, label in order]
    groups = [g for g in groups if g[1]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    strip(ax, groups, [BLUE, ORANGE, AQUA], "Validation accuracy")
    ax.set_title("Weight optimizers on the seed circuit (5 starts each)", loc="left")
    save(fig, "fig_optimizers.png")
    summary.append("## Local weight optimizers (seed, 5 starts each)\n")
    summary.append("| optimizer | budget (passes) | val acc mean | val acc range | test acc mean | passes used | s/start |")
    summary.append("|---|---:|---:|---:|---:|---:|---:|")
    for o, label in order:
        rs = [r for r in runs if r["optimizer"] == o]
        if rs:
            v = [r["val_accuracy"] for r in rs]
            summary.append(f"| {o} | {rs[0]['budget']} | {np.mean(v):.1%} | {min(v):.1%}-{max(v):.1%} | "
                           f"{np.mean([r['test_accuracy'] for r in rs]):.1%} | "
                           f"{np.mean([r['train_passes_used'] for r in rs]):.0f} | "
                           f"{np.mean([r['seconds'] for r in rs]):.0f} |")
    summary.append("")


def inheritance():
    runs = load("inheritance")
    if not runs:
        return
    variants = [("cz_ring", "CX ring -> CZ ring"), ("linear_cx", "ring -> linear chain"),
                ("three_layers", "2 -> 3 layers (+16 params)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    summary.append("## Lamarckian weight inheritance (structural children of the seed)\n")
    summary.append("| child | budget | random init val acc | inherited val acc | gain |")
    summary.append("|---|---:|---:|---:|---:|")
    for ax, (v, title) in zip(axes, variants):
        for inherited, color, label in ((False, BLUE, "random init"), (True, ORANGE, "inherited from parent")):
            budgets = sorted({r["budget"] for r in runs if r["variant"] == v})
            means = [np.mean([r["val_accuracy"] for r in runs if r["variant"] == v and
                              r["budget"] == b and r["inherited"] == inherited]) for b in budgets]
            ax.plot(budgets, means, color=color, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        for b in budgets:
            a = np.mean([r["val_accuracy"] for r in runs if r["variant"] == v and r["budget"] == b and not r["inherited"]])
            c = np.mean([r["val_accuracy"] for r in runs if r["variant"] == v and r["budget"] == b and r["inherited"]])
            summary.append(f"| {v} | {b} | {a:.1%} | {c:.1%} | {c - a:+.1%} |")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Training budget (passes)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Validation accuracy (mean of 5)")
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "fig_weight_inheritance.png")
    summary.append("")


def readout():
    runs = load("readout")
    if not runs:
        return
    groups = [(f"q{q}", [r["val_accuracy"] for r in runs if r["readout"] == [q]]) for q in range(8)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    strip(ax, groups, [BLUE], "Validation accuracy")
    ax.set_title("Which qubit is measured: seed retrained per readout qubit (5 starts)", loc="left")
    save(fig, "fig_readout_qubit.png")
    summary.append("## Readout-qubit selection (local cost on one qubit; seed, 5 starts each)\n")
    summary.append("| readout | val acc mean | min | max |")
    summary.append("|---|---:|---:|---:|")
    for label, v in groups:
        summary.append(f"| {label} | {np.mean(v):.1%} | {min(v):.1%} | {max(v):.1%} |")
    summary.append("")


if __name__ == "__main__":
    for fn in (falloff, seed_multistart, optimizers, inheritance, readout):
        fn()
    (DATA / "summary.md").write_text("\n".join(summary) + "\n")
    print((DATA / "summary.md").read_text())
