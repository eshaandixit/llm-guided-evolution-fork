"""Racing: extra training starts only for circuits whose selection is in doubt.

A VQC's validation accuracy depends heavily on its random starting angles, so obj1 is
an average over several training starts and still carries noise. Instead of giving
every circuit the same fixed number of starts, racing gives each child a few, then
adds starts only where the ranking is uncertain:

  * obj1 (1 - val accuracy) is noisy; obj2 (gate count) is exact.
  * A circuit is UNCERTAIN when its non-domination rank changes between an optimistic
    (mean - z*SE) and a pessimistic (mean + z*SE) value of its obj1, with every other
    circuit held at its mean, and its better rank is one selection can reach.
  * Uncertain circuits get more starts, up to a cap; the rest stop early.
  * Circuits that behave identically are raced once. The LLM often returns edits that
    read differently but build the same program (second Iris run: 13 children, 2
    distinct behaviors), and racing identical twins against each other only spends
    starts. A group is raced through one member and the result is copied to the rest.

Everything here is pure file and list logic so it can be tested without Slurm.
The orchestrator side (submitting the extra-start jobs) lives in run_improved.py.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics

EXTRA_RE = re.compile(r"_starts(\d+)_metrics\.json$")


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def metric_files(results_dir, gene_id):
    """The gene's first-run metrics file, then its extra-start files in start order."""
    main = os.path.join(results_dir, f"{gene_id}_metrics.json")
    extras = glob.glob(os.path.join(results_dir, f"{gene_id}_starts*_metrics.json"))
    extras.sort(key=lambda p: int(EXTRA_RE.search(p).group(1)) if EXTRA_RE.search(p) else 0)
    return [main] + extras


def collect_starts(results_dir, gene_id):
    """Every training start recorded for the gene, one per seed, in seed order.

    The first-run metrics file is rewritten by merge_starts() to hold all starts, so
    starts are de-duplicated by seed; merging any number of times gives the same result.
    """
    by_seed = {}
    for path in metric_files(results_dir, gene_id):
        data = _load(path)
        if not data:
            continue
        for start in data.get("starts", []):
            if "seed" in start and "val_accuracy" in start:
                by_seed.setdefault(start["seed"], start)
    return [by_seed[seed] for seed in sorted(by_seed)]


def summarize(starts):
    """(mean validation error, sample sd of validation error, number of starts)."""
    errors = [1.0 - s["val_accuracy"] for s in starts]
    n = len(errors)
    if n == 0:
        return None
    sd = statistics.stdev(errors) if n > 1 else 0.0
    return statistics.fmean(errors), sd, n


def nondominated_ranks(points):
    """Pareto rank (0 = front) of each (obj1, obj2) point; both objectives minimized."""
    n = len(points)
    remaining = set(range(n))
    ranks = [None] * n
    rank = 0
    while remaining:
        front = [i for i in remaining
                 if not any(_dominates(points[j], points[i]) for j in remaining if j != i)]
        for i in front:
            ranks[i] = rank
        remaining -= set(front)
        rank += 1
    return ranks


def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _rank_of(i, points, value):
    """Rank of point i if its obj1 were `value`, with every other point unchanged."""
    moved = list(points)
    moved[i] = (value, points[i][1])
    return nondominated_ranks(moved)[i]


def uncertain_genes(stats, z=2.0, max_rank=1, max_starts=10):
    """Genes that need more starts.

    stats: {gene_id: (mean_error, sd, n_starts, gates)}
    z: how many standard errors the optimistic/pessimistic bounds sit from the mean.
    max_rank: only ranks 0..max_rank matter (0 = Pareto front). A gene that stays
        worse than this whether lucky or unlucky is not worth more starts.
    max_starts: genes already at the cap are never returned.

    The sd of a gene with few starts is unreliable (three equal accuracies give sd 0),
    so each gene uses at least the median sd across the population.
    """
    genes = sorted(stats)
    if len(genes) < 2:
        return []
    sds = [stats[g][1] for g in genes if stats[g][2] > 1]
    floor_sd = statistics.median(sds) if sds else 0.0
    points = [(stats[g][0], stats[g][3]) for g in genes]
    out = []
    for i, g in enumerate(genes):
        mean, sd, n, _ = stats[g]
        if n >= max_starts:
            continue
        se = max(sd, floor_sd) / math.sqrt(n)
        if se == 0:
            continue
        best = _rank_of(i, points, mean - z * se)
        worst = _rank_of(i, points, mean + z * se)
        if best != worst and best <= max_rank:
            out.append(g)
    return out


def merge_starts(results_dir, gene_id):
    """Fold every start into <gene>_metrics.json and rewrite <gene>_results.txt.

    Returns the new (obj1, obj2) fitness, or None when the gene has no metrics.
    Writes go through a temporary file and a rename, so a reader never sees half a file.
    """
    main_path = os.path.join(results_dir, f"{gene_id}_metrics.json")
    metrics = _load(main_path)
    if metrics is None:
        return None
    starts = collect_starts(results_dir, gene_id)
    mean_err, sd_err, n = summarize(starts)

    def avg(key):
        values = [s[key] for s in starts if key in s]
        return float(statistics.fmean(values)) if values else metrics.get(key)

    best = max(starts, key=lambda s: s["val_accuracy"])
    metrics.update({
        "starts": starts,
        "n_starts": n,
        "start_offset": 0,
        "val_accuracy": 1.0 - mean_err,
        "val_accuracy_sd": float(statistics.pstdev([s["val_accuracy"] for s in starts])),
        "obj1_val_error": mean_err,
        "val_cross_entropy": avg("val_cross_entropy"),
        "train_accuracy": avg("train_accuracy"),
        "train_cross_entropy": avg("train_cross_entropy"),
        "test_accuracy": avg("test_accuracy"),
        "test_cross_entropy": avg("test_cross_entropy"),
        "best_start_val_accuracy": best["val_accuracy"],
        "best_start_test_accuracy": best.get("test_accuracy"),
        "racing": True,
    })
    if "weights" in best:
        metrics["trained_weights"] = best["weights"]
    obj2 = metrics["obj2_gates"]

    _atomic_write(main_path, json.dumps(metrics, indent=1, sort_keys=True))
    _atomic_write(os.path.join(results_dir, f"{gene_id}_results.txt"),
                  f"{mean_err}, {obj2}, {metrics['val_cross_entropy']}")
    return mean_err, obj2


def _atomic_write(path, text):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def population_stats(results_dir, gene_ids):
    """{gene_id: (mean_error, sd, n_starts, gates)} for genes with usable metrics."""
    stats = {}
    for gene_id in gene_ids:
        metrics = _load(os.path.join(results_dir, f"{gene_id}_metrics.json"))
        if not metrics or "obj2_gates" not in metrics:
            continue
        summary = summarize(collect_starts(results_dir, gene_id))
        if summary is None:
            continue
        stats[gene_id] = (*summary, float(metrics["obj2_gates"]))
    return stats


def fingerprint(results_dir, gene_id, n=3):
    """What the gene did on its first n starts: gate count plus per-seed validation loss.

    Starts use fixed seeds, so two genes with the same fingerprint ran the same
    computation; genuinely different programs essentially never match to 9 decimals.
    Returns None when the metrics don't have what's needed, so the gene stays alone.
    """
    metrics = _load(os.path.join(results_dir, f"{gene_id}_metrics.json"))
    starts = collect_starts(results_dir, gene_id)[:n]
    if not metrics or len(starts) < n or any("val_cross_entropy" not in s for s in starts):
        return None
    return (float(metrics["obj2_gates"]),
            tuple((s["seed"], round(float(s["val_cross_entropy"]), 9)) for s in starts))


def group_duplicates(results_dir, gene_ids, n=3):
    """{representative: [other members]}; the member with the most starts represents a group."""
    groups = {}
    for gene_id in sorted(gene_ids):
        key = fingerprint(results_dir, gene_id, n) or ("alone", gene_id)
        groups.setdefault(key, []).append(gene_id)
    out = {}
    for members in groups.values():
        members.sort(key=lambda g: (-len(collect_starts(results_dir, g)), g))
        out[members[0]] = members[1:]
    return out


def copy_starts(results_dir, src, dst):
    """Give dst every start src has, then rewrite dst's metrics and results like a merge."""
    path = os.path.join(results_dir, f"{dst}_metrics.json")
    metrics = _load(path)
    if metrics is None:
        return None
    metrics["starts"] = collect_starts(results_dir, src)
    metrics["racing_copied_from"] = src
    _atomic_write(path, json.dumps(metrics, indent=1, sort_keys=True))
    return merge_starts(results_dir, dst)
