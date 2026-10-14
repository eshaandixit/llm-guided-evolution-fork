"""Re-score and plot the RAG forcing study (sota/QuantumVQC/run_rag_forcing_study.py).

    python tools/plot_rag_forcing.py

The study saves every raw LLM answer. This re-extracts the code from each answer with
the CURRENT src.llm_utils.clean_code_from_llm (fixed on 2026-10-08 to survive Llama's
empty leading fence) and re-runs the same splice/validate/train checks, so both arms
are scored with the extraction evolution now uses. Writes
reports/qiskit_rag_forcing_rescored.json, reports/figures/fig_rag_forcing.png,
reports/figures/fig_qiskit_freeform.png and prints the tables quoted in the notebook.
"""
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sota" / "QuantumVQC"))
import run_rag_forcing_study as study  # noqa: E402
from src.llm_utils import clean_code_from_llm, split_file  # noqa: E402

IN = ROOT / "reports" / "qiskit_rag_forcing.json"
OUT = ROOT / "reports" / "qiskit_rag_forcing_rescored.json"
FIG = ROOT / "reports" / "figures"
STAGES = ("parsed", "valid", "trainable", "changed", "compliant")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlecolor": INK, "legend.frameon": False,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rescore_forcing(rec, parts, seed_hist):
    out = {k: v for k, v in rec.items() if k not in STAGES + ("circuit", "error")}
    out.update({k: False for k in STAGES})
    code = clean_code_from_llm(rec.get("raw") or "")
    if code in ("ERROR", ""):
        return out
    out["parsed"] = True
    candidate = list(parts)
    candidate[rec["block"]] = f"\n{code}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("# --OPTION--".join(candidate))
    try:
        info = study.analyse(f.name, train=False)
        out["valid"], out["circuit"] = True, info
        out["changed"] = info["histogram"] != seed_hist
        out["compliant"] = bool(study.PROPERTIES[rec["property"]][2](info["histogram"], info))
        info.update(study.analyse(f.name, train=True))
        out["trainable"] = True
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    return out


def rescore_freeform(rec):
    out = {k: v for k, v in rec.items() if k not in ("parsed", "executes", "correct", "error", "trace")}
    out.update(parsed=False, executes=False, correct=False)
    code = clean_code_from_llm(rec.get("raw") or "")
    if code in ("ERROR", ""):
        return out
    out["parsed"] = True
    try:
        ns = {}
        exec(compile(code, f"<{rec['task']}>", "exec"), ns)
        fn = ns.get("parity_probability") or ns["make_circuit"]
        out["executes"] = True
        out["correct"] = bool(study.FREEFORM[rec["task"]][1](fn))
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    return out


def _forcing_job(rec):
    parts = split_file(str(study.SEED_PATH))
    seed_hist = study.analyse(study.SEED_PATH, train=False)["histogram"]
    return rescore_forcing(rec, parts, seed_hist)


def main():
    data = json.loads(IN.read_text())
    with ProcessPoolExecutor(4) as pool:
        forcing = list(pool.map(_forcing_job, data["forcing"]))
    freeform = [rescore_freeform(r) for r in data["freeform"]]
    OUT.write_text(json.dumps({**data, "forcing": forcing, "freeform": freeform,
                               "original_extraction": {
                                   "forcing": [{k: r[k] for k in STAGES} for r in data["forcing"]],
                                   "freeform": [{k: r[k] for k in ("parsed", "executes", "correct")}
                                                for r in data["freeform"]]}}, indent=1))

    props = list(study.PROPERTIES)
    print("| property | arm | " + " | ".join(STAGES) + " |")
    print("|---|---|" + "---:|" * len(STAGES))
    for prop in props:
        for arm in ("no_rag", "rag"):
            rs = [r for r in forcing if r["property"] == prop and r["arm"] == arm]
            print(f"| {prop} | {arm} | " + " | ".join(f"{sum(r[k] for r in rs)}/{len(rs)}"
                                                    for k in STAGES) + " |")
    tot = {arm: [r for r in forcing if r["arm"] == arm and r["property"] != "control"]
           for arm in ("no_rag", "rag")}
    print("\nall forced properties pooled:")
    for arm, rs in tot.items():
        print(f"  {arm}: " + ", ".join(f"{k} {sum(r[k] for r in rs)}/{len(rs)}" for k in STAGES))
    orig = data["forcing"]
    print("\noriginal (pre-fix) extraction, parsed: " + ", ".join(
        f"{arm} {sum(r['parsed'] for r in orig if r['arm'] == arm)}/{sum(1 for r in orig if r['arm'] == arm)}"
        for arm in ("no_rag", "rag")))

    # Figure 1: per-property funnel, two arms side by side.
    fig, axes = plt.subplots(1, len(props), figsize=(3.0 * len(props), 3.6), sharey=True)
    x = np.arange(len(STAGES))
    for ax, prop in zip(axes, props):
        for j, (arm, color) in enumerate((("no_rag", BLUE), ("rag", ORANGE))):
            rs = [r for r in forcing if r["property"] == prop and r["arm"] == arm]
            stats = [wilson(sum(r[k] for r in rs), len(rs)) for k in STAGES]
            p = [s[0] for s in stats]
            err = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
            ax.bar(x + (j - 0.5) * 0.38, p, 0.36, color=color, edgecolor=SURFACE, linewidth=2,
                   yerr=err, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2},
                   label="no RAG" if arm == "no_rag" else "with Qiskit RAG")
        ax.set_xticks(x, STAGES, rotation=40, ha="right")
        ax.set_title(prop.replace("_", " "), loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel(f"share of {len(rs)} samples (95% Wilson CI)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Can a one-line goal in the mutation prompt force a circuit property? "
                 "(Llama-3.3-70B, T=0.3)", x=0.01, ha="left", color=INK)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rag_forcing.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: free-form Qiskit generation, correct rate per task.
    tasks = list(study.FREEFORM)
    print("\n| task | arm | parsed | executes | correct |")
    print("|---|---|---:|---:|---:|")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(tasks))
    for j, (arm, color) in enumerate((("no_rag", BLUE), ("rag", ORANGE))):
        ps, errs = [], [[], []]
        for t in tasks:
            rs = [r for r in freeform if r["task"] == t and r["arm"] == arm]
            print(f"| {t} | {arm} | {sum(r['parsed'] for r in rs)}/{len(rs)} | "
                  f"{sum(r['executes'] for r in rs)}/{len(rs)} | {sum(r['correct'] for r in rs)}/{len(rs)} |")
            p, lo, hi = wilson(sum(r["correct"] for r in rs), len(rs))
            ps.append(p)
            errs[0].append(p - lo)
            errs[1].append(hi - p)
        ax.bar(x + (j - 0.5) * 0.38, ps, 0.36, color=color, edgecolor=SURFACE, linewidth=2,
               yerr=errs, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2},
               label="no RAG" if arm == "no_rag" else "with Qiskit RAG")
    ax.set_xticks(x, [t.replace("_", " ") for t in tasks])
    ax.set_ylabel("correct (statevector / unitary oracle)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Can the LLM write working Qiskit from scratch?", loc="left")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig_qiskit_freeform.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    errors = {}
    for r in forcing + freeform:
        if r.get("error"):
            key = (r["arm"], r["error"].split(":")[0])
            errors[key] = errors.get(key, 0) + 1
    print("\nerror types:", sorted(errors.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
