"""Analyse a racing experiment made by tools/racing_experiment.sh.

    uv run python tools/racing_analysis.py experiments/<name> [--ref-gates 56]

Each run writes checkpoints/progress.jsonl, one line per generation with the
training starts used so far and the Pareto front so far. For every run this reports:

  * hypervolume (HV) of the front against training starts used; both objectives are
    minimised, the reference point is (error 1.0, --ref-gates gates)
  * HV at the equal-compute budget: the smallest total the control runs reached;
    each run is cut at the last generation within that budget
  * for racing runs, the starts needed to reach the control runs' mean final HV
  * if tools/rescore_front.py has run: how far each front's reported error was from
    its error over 30 fresh starts, and the HV recomputed with those errors

Writes <exp>/analysis.json and <exp>/hypervolume.png and prints a summary.
"""
import argparse
import glob
import json
import os
import re
import statistics

RUN_RE = re.compile(r"^(control|racing)_(\d+)$")
RESCORE_OFFSET = 1000


def hypervolume(points, ref_error=1.0, ref_gates=56.0):
    """2-D hypervolume for minimised (error, gates) points against a reference point."""
    pts = sorted((e, g) for e, g in points if e < ref_error and g < ref_gates)
    hv, prev_gates = 0.0, ref_gates
    for error, gates in pts:
        if gates < prev_gates:
            hv += (ref_error - error) * (prev_gates - gates)
            prev_gates = gates
    return hv


def load_runs(exp):
    runs = {}
    for path in sorted(glob.glob(os.path.join(exp, "*", "checkpoints", "progress.jsonl"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        match = RUN_RE.match(name)
        if not match:
            continue
        with open(path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if entries:
            runs[name] = {"arm": match.group(1), "entries": entries}
    return runs


def budget(runs):
    """The equal-compute budget: the smallest final total among control runs."""
    totals = [r["entries"][-1]["training_starts"] for r in runs.values() if r["arm"] == "control"]
    return min(totals) if totals else None


def entry_at(entries, limit):
    """The last generation whose starts fit within `limit` (the first one if none do)."""
    within = [e for e in entries if limit is None or e["training_starts"] <= limit]
    return within[-1] if within else entries[0]


def rescored_error(rescore_dir, gene):
    path = os.path.join(rescore_dir, f"{gene}_starts{RESCORE_OFFSET}_metrics.json")
    try:
        with open(path, encoding="utf-8") as f:
            starts = json.load(f)["starts"]
    except (OSError, ValueError, KeyError):
        return None
    return 1.0 - statistics.fmean(s["val_accuracy"] for s in starts)


def analyse(exp, ref_gates=56.0):
    runs = load_runs(exp)
    if not runs:
        raise SystemExit(f"No progress.jsonl files under {exp}/<arm>_<i>/checkpoints/")
    limit = budget(runs)
    hv = lambda front: hypervolume([(p[1], p[2]) for p in front], ref_gates=ref_gates)
    control_final = [hv(r["entries"][-1]["front"]) for r in runs.values() if r["arm"] == "control"]
    target = statistics.fmean(control_final) if control_final else None
    rescore_dir = os.path.join(exp, "rescore")

    report = {"budget_starts": limit, "ref_gates": ref_gates, "control_mean_final_hv": target, "runs": {}}
    for name, run in runs.items():
        entries = run["entries"]
        at = entry_at(entries, limit)
        curve = [(e["training_starts"], hv(e["front"])) for e in entries]
        reached = next((s for s, h in curve if target is not None and h >= target), None)
        row = {
            "arm": run["arm"],
            "generations": entries[-1]["generation"],
            "final_starts": entries[-1]["training_starts"],
            "hv_at_budget": hv(at["front"]),
            "generation_at_budget": at["generation"],
            "starts_to_reach_control_hv": reached if run["arm"] == "racing" else None,
            "curve": curve,
            "front_at_budget": at["front"],
        }
        true = [(p, rescored_error(rescore_dir, p[0])) for p in at["front"]]
        true = [(p, t) for p, t in true if t is not None]
        if true:
            row["rescored"] = len(true)
            row["mean_reported_minus_true_error"] = statistics.fmean(p[1] - t for p, t in true)
            row["mean_abs_error_gap"] = statistics.fmean(abs(p[1] - t) for p, t in true)
            row["hv_at_budget_rescored"] = hypervolume([(t, p[2]) for p, t in true], ref_gates=ref_gates)
        report["runs"][name] = row
    return report


def summarize(report):
    lines = [f"Equal-compute budget: {report['budget_starts']} training starts "
             f"(reference point: error 1.0, {report['ref_gates']:.0f} gates)", ""]
    header = f"{'run':<12} {'gens':>4} {'starts':>7} {'HV@budget':>10} {'HV rescored':>12} {'error gap':>10} {'starts to control HV':>21}"
    lines.append(header)
    for name, r in sorted(report["runs"].items()):
        rescored = f"{r['hv_at_budget_rescored']:.4f}" if "hv_at_budget_rescored" in r else "-"
        gap = f"{r['mean_reported_minus_true_error']:+.4f}" if "mean_reported_minus_true_error" in r else "-"
        reach = r["starts_to_reach_control_hv"] if r["starts_to_reach_control_hv"] is not None else "-"
        lines.append(f"{name:<12} {r['generations']:>4} {r['final_starts']:>7} {r['hv_at_budget']:>10.4f} "
                     f"{rescored:>12} {gap:>10} {str(reach):>21}")
    lines.append("")
    for arm in ("control", "racing"):
        vals = [r["hv_at_budget"] for r in report["runs"].values() if r["arm"] == arm]
        if vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"{arm:<8} HV at budget: mean {statistics.fmean(vals):.4f}, sd {sd:.4f}, n {len(vals)}")
    lines.append("error gap = reported error minus error over 30 fresh starts; positive means the "
                 "front looked better than it is")
    return "\n".join(lines)


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"control": "#5b6b8c", "racing": "#c4572b"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in sorted(report["runs"].items()):
        xs, ys = zip(*r["curve"])
        ax.step(xs, ys, where="post", color=colors[r["arm"]], alpha=0.85,
                label=r["arm"] if name.endswith("_1") else None)
    if report["budget_starts"]:
        ax.axvline(report["budget_starts"], color="#888", linestyle="--", linewidth=1)
        ax.text(report["budget_starts"], ax.get_ylim()[0], " equal budget", color="#666",
                va="bottom", fontsize=8)
    ax.set_xlabel("training starts used")
    ax.set_ylabel(f"hypervolume (ref: error 1.0, {report['ref_gates']:.0f} gates)")
    ax.set_title("Pareto-front hypervolume vs compute")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="experiments/<name>")
    ap.add_argument("--ref-gates", type=float, default=56.0,
                    help="gate count of the reference point (default: twice the seed's 28)")
    args = ap.parse_args()
    report = analyse(args.exp, args.ref_gates)
    with open(os.path.join(args.exp, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    plot(report, os.path.join(args.exp, "hypervolume.png"))
    print(summarize(report))
    print(f"\nWrote {args.exp}/analysis.json and {args.exp}/hypervolume.png")


if __name__ == "__main__":
    main()
