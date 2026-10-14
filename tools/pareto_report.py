"""Pareto front + Pareto-optimal circuits for a QuantumVQC evolution run.

Reads the fitness files an evolution run leaves behind, works out which
generation each individual first appeared in from the orchestrator log, and
draws the front. Both objectives are MINIMISED: obj1 = validation error
(1 - accuracy, averaged over N_STARTS trainings), obj2 = plain gate count.

  python tools/pareto_report.py --results <dir> --log <file> --out <dir>
"""
import argparse, json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SEED_OBJ = (0.15087719298245617, 56.0)   # seed since 2026-10-08: 8q breast cancer, 5-start mean val error, 56 gates
TEAL, PURPLE, LTEAL, GRAY = "#418faf", "#4c4a86", "#7fc5d4", "#9ba0a5"
NAVY, MUTED, ACC = "#2d2a54", "#6e6f7b", "#b5504a"


def load_fitness(results_dir):
    out = {}
    for f in glob.glob(os.path.join(results_dir, "xXx*_results.txt")):
        gid = os.path.basename(f)[:-len("_results.txt")]
        try:
            # The first two values are the objectives the harness selects on; the
            # seed writes extra diagnostics after them, so take only what we need.
            parts = open(f).read().split(",")
            out[gid] = (float(parts[0]), float(parts[1]))
        except Exception:
            pass                                 # unparsable = invalid individual
    return out


def load_metric_pair(results_dir, xk, yk):
    """Any two measures from <gene>_metrics.json. BOTH are treated as minimised."""
    out = {}
    for f in glob.glob(os.path.join(results_dir, "xXx*_metrics.json")):
        gid = os.path.basename(f)[:-len("_metrics.json")]
        try:
            m = json.load(open(f))
            out[gid] = (float(m[xk]), float(m[yk]))
        except Exception:
            pass
    return out


def load_generations(log_path):
    """Generation each gene id is first mentioned in, and whether this run resumed.

    The resume flag matters: on a resumed run the orchestrator restores a
    population from a checkpoint and the fresh log names ONLY those survivors
    until print_ancestry() dumps the full history, which does not happen until
    partway through the first resumed generation. Everything evaluated in earlier
    generations of the same search is therefore missing from gen_of for that
    whole window, and is indistinguishable from a genuinely abandoned run.
    """
    gen_of, gen, resumed = {}, 0, False
    if not log_path or not os.path.exists(log_path):
        return gen_of, resumed
    for line in open(log_path, errors="replace"):
        if "Loaded checkpoint from" in line:
            resumed = True
        m = re.search(r"STARTING GENERATION:\s*(\d+)", line)
        if m:
            gen = int(m.group(1)) + 1            # pop created before gen 0 => 0
            continue
        for gid in re.findall(r"xXx[A-Za-z0-9]+", line):
            gen_of.setdefault(gid, gen)
    return gen_of, resumed


def pareto(points):
    """Indices of non-dominated points; both objectives minimised."""
    keep = []
    for i, (a1, a2) in enumerate(points):
        if not any((b1 <= a1 and b2 <= a2) and (b1 < a1 or b2 < a2)
                   for j, (b1, b2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="pareto_out")
    ap.add_argument("--strict", action="store_true",
                    help="on a resumed run, drop genes missing from this log "
                         "instead of including them")
    ap.add_argument("--x", default=None,
                    help="redraw on any measure from <gene>_metrics.json, e.g. "
                         "val_cross_entropy, weighted_gate_cost, depth, two_qubit_gates. "
                         "Both axes are treated as lower-is-better, so do not pass an "
                         "accuracy field - use val_error.")
    ap.add_argument("--y", default=None)
    ap.add_argument("--max-circuits", type=int, default=40,
                    help="cap rows in the circuit grid (front members first)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    global SEED_OBJ
    if bool(a.x) != bool(a.y):
        print("--x and --y must be given together"); return
    if a.x:
        fit = load_metric_pair(a.results, a.x, a.y)
        SEED_OBJ = None            # the recorded seed point only applies to the objectives
        XLAB, YLAB = a.x, a.y
    else:
        fit = load_fitness(a.results)
        XLAB, YLAB = "obj1  —  validation error (1 - accuracy)", "obj2  —  gate count"
    if not fit:
        print("no parsable results yet"); return
    gen_of, resumed = load_generations(a.log)
    # results/ is never cleared between runs, so it accumulates genes from every
    # previous (often crashed) run. Those are absent from THIS run's log, and
    # gen_of.get(g, 0) would silently relabel them as generation 0 and let them
    # into the front. Keep only genes this run actually produced. If the log is
    # missing entirely gen_of is empty - fall back to reporting everything rather
    # than emitting an empty front.
    if gen_of:
        unknown = [g for g in fit if g not in gen_of]
        if unknown and resumed and not a.strict:
            # On a resume these are AMBIGUOUS, not stale: they are equally
            # consistent with earlier generations of this same search, whose gene
            # ids do not reach the log until the ancestry dump. Dropping them
            # would silently delete real front members. Keep them and say so -
            # an overstated front the operator can see beats an understated one
            # they cannot.
            print(f"  !! RESUMED RUN - {len(unknown)} genes are in results/ but not yet in")
            print(f"     this log. They may be earlier generations of THIS search, so they")
            print(f"     are INCLUDED and their generation numbers are unreliable.")
            print(f"     Pass --strict to drop them instead.")
        else:
            for g in unknown:
                del fit[g]
            if unknown:
                why = "forced by --strict" if a.strict else "from earlier runs"
                print(f"  ignored stale results   : {len(unknown)} ({why})")
        if not fit:
            print("no results from this run yet"); return
    gids = sorted(fit, key=lambda g: (gen_of.get(g, 0), fit[g][0]))
    pts = [fit[g] for g in gids]
    gens = [gen_of.get(g, 0) for g in gids]
    front = pareto(pts)
    fseq = sorted((pts[i] for i in front), key=lambda p: p[0])

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eceef0", lw=0.9); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)

    gmax = max(gens) if gens else 0
    sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                    c=gens, cmap="viridis", vmin=0, vmax=max(gmax, 1),
                    s=95, edgecolor="white", lw=1.2, zorder=3)
    if gmax > 0:
        plt.colorbar(sc, ax=ax, label="generation first seen", pad=0.02)

    # Distinct front coordinates. Several individuals routinely land on the SAME
    # point - an LLM edit that only adds comments or uncalled helpers leaves the
    # circuit, and so the fitness, bit-identical - and plotting that point five
    # times stacks five labels on top of each other.
    fpts = sorted(set(fseq))

    # The frontier proper: the staircase of what is actually achievable. Anything
    # inside a step is dominated, so the step is the honest boundary.
    ax.step([p[0] for p in fpts], [p[1] for p in fpts], where="post",
            color=ACC, lw=1.8, ls=(0, (5, 3)), zorder=2, label="Pareto frontier")

    # The trade-off curve through the same points. This one is a reading aid, not
    # a claim: no individual is known to exist between two front points, so the
    # curve is interpolation. PCHIP keeps it monotone - a spline would overshoot
    # and imply circuits better than anything actually evaluated.
    fx = [p[0] for p in fpts]
    fy = [p[1] for p in fpts]
    if len(fpts) >= 2 and all(b > a for a, b in zip(fx, fx[1:])):
        if len(fpts) >= 3:
            try:
                from scipy.interpolate import PchipInterpolator
                xs = np.linspace(fx[0], fx[-1], 250)
                ys = PchipInterpolator(fx, fy)(xs)
            except Exception:
                xs, ys = fx, fy
        else:
            xs, ys = fx, fy
        ax.plot(xs, ys, color=ACC, lw=1.5, alpha=0.55, zorder=2,
                label="trade-off curve (interpolated)")

    ax.scatter(fx, fy, s=210, facecolor="none", edgecolor=ACC, lw=2.0, zorder=4)
    if SEED_OBJ:
        ax.scatter([SEED_OBJ[0]], [SEED_OBJ[1]], marker="*", s=420, color=NAVY,
                   zorder=5, label="seed (individual zero)")

    # Label EVERY individual, not just the non-dominated ones - the dominated
    # points are the record of what the search actually tried. Genes sharing a
    # coordinate are collapsed into one stacked label.
    at_point = {}
    for i, g in enumerate(gids):
        at_point.setdefault(pts[i], []).append(g[3:11])
    # Headroom first, so a tall stacked label has somewhere to go.
    ax.margins(x=0.10, y=0.14)
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    for (o1, o2), gl in at_point.items():
        on_front = (o1, o2) in set(fpts)
        shown = gl[:4] + ([f"+{len(gl) - 4} more"] if len(gl) > 4 else [])
        # A label hangs UPWARD by default, which walks straight into the title for
        # anything near the top - and the seed's tied cluster is both the tallest
        # label and the highest point. Drop those below the marker instead. Same
        # for the right edge: labels there would run off the axes.
        high = o2 > ylo + 0.78 * (yhi - ylo)
        right = o1 > xlo + 0.72 * (xhi - xlo)
        ax.annotate("\n".join(shown), (o1, o2), textcoords="offset points",
                    xytext=(-10 if right else 10, -8 if high else 6),
                    ha="right" if right else "left",
                    va="top" if high else "bottom",
                    fontsize=6.8, linespacing=1.25,
                    color=ACC if on_front else MUTED,
                    weight="bold" if on_front else "normal", zorder=6)

    ax.set_xlabel(f"{XLAB}  (lower is better)", fontsize=11, color="#3f4149")
    ax.set_ylabel(f"{YLAB}  (lower is better)", fontsize=11, color="#3f4149")
    ax.set_title(f"Pareto front — {len(pts)} evaluated individuals, "
                 f"{len(front)} non-dominated, {gmax + 1} generation(s)",
                 fontsize=13.5, weight="bold", color=NAVY, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "pareto_front.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # the Pareto-optimal circuits themselves
    # Every individual, front first, then the rest by loss - the dominated
    # circuits are how you see WHAT the LLM edited, which is the whole question.
    order = [i for i in front] + [i for i in range(len(gids)) if i not in set(front)]
    imgs = [(gids[i], pts[i], os.path.join(a.results, f"{gids[i]}_circuit.png"))
            for i in order]
    imgs = [t for t in imgs if os.path.exists(t[2])]
    # The grid is one row per individual at 2.5in / 170dpi, so its height grows
    # without bound - measured ~425 px per individual. Past ~135 individuals the
    # PNG trips PIL's decompression-bomb limit and mpimg.imread, Jupyter and most
    # viewers refuse to open it; at the 500-genome budget it is ~212,000 px tall
    # and needs multiple GB of RSS to render. `order` puts the front first, so
    # capping keeps every non-dominated circuit.
    if len(imgs) > a.max_circuits:
        dropped = len(imgs) - a.max_circuits
        imgs = imgs[:a.max_circuits]
        print(f"  circuit grid capped     : {a.max_circuits} shown, {dropped} omitted "
              f"(front kept; raise with --max-circuits)")
    if imgs:
        n = len(imgs)
        fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.5 * n), dpi=170)
        if n == 1:
            axes = [axes]
        for ax2, (gid, (o1, o2), path) in zip(axes, imgs):
            ax2.imshow(mpimg.imread(path)); ax2.axis("off")
            mark = "  [FRONT]" if (o1, o2) in set(fseq) else ""
            ax2.set_title(f"{gid[3:11]}   obj1 {o1:.4f}   obj2 {o2:.0f}   gen {gen_of.get(gid, 0)}{mark}",
                          fontsize=10, color=NAVY if mark else MUTED, loc="left")
        fig.suptitle(f"All {len(imgs)} evaluated circuits — front marked [FRONT]",
                     fontsize=13.5, weight="bold", color=NAVY, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "pareto_circuits.png"), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"  evaluated individuals : {len(pts)}")
    print(f"  generations covered   : 0..{gmax}")
    print(f"  non-dominated         : {len(front)}")
    print(f"  circuits rendered     : {len(imgs)}")
    print()
    print(f"  {'gene':<12} {'obj1':>10} {'obj2':>7}  gen   vs seed")
    # Dominance is strict: at least as good on BOTH objectives AND strictly better
    # on at least one. An individual that merely TIES the seed has not beaten it -
    # and ties are common here, because an LLM edit that only adds comments or
    # uncalled helper functions leaves the circuit, and therefore the fitness,
    # bit-identical. Labelling those "DOMINATES seed" overstates the whole run.
    seen = {}
    for i in front:
        o1, o2 = pts[i]
        if SEED_OBJ and o1 <= SEED_OBJ[0] and o2 <= SEED_OBJ[1]:
            tag = "ties seed" if (o1, o2) == SEED_OBJ else "DOMINATES seed"
        else:
            tag = ""
        dup = seen.setdefault((o1, o2), [])
        dup.append(gids[i][3:11])
        if len(dup) > 1:
            continue                              # same point, already printed
        print(f"  {gids[i][3:11]:<12} {o1:>10.5f} {o2:>7.0f}  {gen_of.get(gids[i],0):<4} {tag}")
    extra = {k: v for k, v in seen.items() if len(v) > 1}
    if extra:
        print()
        for (o1, o2), gl in extra.items():
            print(f"  note: {len(gl)} individuals share ({o1:.5f}, {o2:.0f}): {', '.join(gl)}")


if __name__ == "__main__":
    main()
