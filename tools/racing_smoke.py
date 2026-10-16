"""Racing smoke test on the real cluster, with no LLM involved.

Writes a handful of hand-edited variants of the QuantumVQC seed, evaluates them as
ordinary Slurm eval jobs (RACING_INITIAL_STARTS starts each), then runs the same
race_population() the orchestrator runs, so extra-start jobs are submitted, waited
on and merged exactly as in a real evolution run.

Run it from the repo root on the login node. It only submits and polls jobs; all
training happens inside Slurm jobs. Expect roughly 10-20 minutes.

  python tools/racing_smoke.py              # submit with sbatch
  python tools/racing_smoke.py --local      # run each job with bash, no Slurm
  python tools/racing_smoke.py --clean      # delete files from earlier smoke runs

Look for "Round N: k of 6 distinct genes uncertain" and "<gene>: obj1 ... after N starts" in
the output, and for <gene>_starts3_metrics.json files in sota/QuantumVQC/results/.
"""
import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                        # run_improved uses paths relative to the repo root
sys.path.insert(0, ROOT)
os.environ["LLMGE_RACING"] = "1"      # read by the config when run_improved imports it

import run_improved as ri             # noqa: E402
from deap import creator              # noqa: E402
from src import racing                # noqa: E402

PREFIX = "smoke"

# (name, [(old, new), ...]) edits applied to the seed. Several land close together on
# accuracy at similar gate counts, so racing has real close calls to resolve.
VARIANTS = [
    ("seed", []),
    ("one_layer", [("N_LAYERS = 2", "N_LAYERS = 1")]),
    ("three_layers", [("N_LAYERS = 2", "N_LAYERS = 3")]),
    ("cz_ring", [("qc.cx(i, (i + 1) % N_QUBITS)", "qc.cz(i, (i + 1) % N_QUBITS)")]),
    ("linear_chain", [("    for i in range(N_QUBITS):\n        qc.cx(i, (i + 1) % N_QUBITS)",
                       "    for i in range(N_QUBITS - 1):\n        qc.cx(i, i + 1)")]),
    ("one_layer_cz", [("N_LAYERS = 2", "N_LAYERS = 1"),
                      ("qc.cx(i, (i + 1) % N_QUBITS)", "qc.cz(i, (i + 1) % N_QUBITS)")]),
]


def smoke_files():
    results = os.path.join(ri.SOTA_ROOT, "results")
    return (glob.glob(os.path.join(ri.VARIANT_DIR, f"{ri.MODEL}_{PREFIX}_*.py"))
            + glob.glob(os.path.join(results, f"{PREFIX}_*")))


def write_variants():
    with open(ri.SEED_NETWORK, "r", encoding="utf-8") as f:
        seed = f.read()
    os.makedirs(ri.VARIANT_DIR, exist_ok=True)
    tag = ri.generate_random_string(6)
    genes = {}
    for name, edits in VARIANTS:
        code = seed
        for old, new in edits:
            if code.count(old) != 1:
                sys.exit(f"Variant {name}: expected one '{old[:40]}' in the seed, found {code.count(old)}")
            code = code.replace(old, new)
        gene_id = f"{PREFIX}_{tag}_{name}"
        with open(os.path.join(ri.VARIANT_DIR, f"{ri.MODEL}_{gene_id}.py"), "w", encoding="utf-8") as f:
            f.write(code)
        genes[gene_id] = name
    return genes


def evaluate(genes, loop_delay):
    """Submit one eval job per variant and wait for all of them."""
    for gene_id in genes:
        ri.GLOBAL_DATA[gene_id] = {"status": "subbed file", "start_time": time.time()}
        ri.submit_run(gene_id)
    started = time.time()
    while True:
        waiting = [g for g in genes if ri.GLOBAL_DATA[g]["status"] == "running eval"]
        for gene_id in waiting:
            ri.check4results(gene_id)
        waiting = [g for g in genes if ri.GLOBAL_DATA[g]["status"] == "running eval"]
        if not waiting:
            return
        if time.time() - started > ri.EVAL_NO_PROGRESS_TIMEOUT_SECONDS:
            sys.exit(f"Timed out waiting for eval jobs: {waiting}")
        print(f"  waiting on {len(waiting)} eval job(s)...", flush=True)
        time.sleep(loop_delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="run each job with bash instead of sbatch")
    ap.add_argument("--loop-delay", type=int, default=30, help="seconds between job checks")
    ap.add_argument("--clean", action="store_true", help="delete files from earlier smoke runs and exit")
    args = ap.parse_args()

    if args.clean:
        files = smoke_files()
        for path in files:
            os.remove(path)
        print(f"Removed {len(files)} smoke-test file(s).")
        return

    if args.local:
        ri.LOCAL, ri.RUN_COMMAND = True, "bash"
    os.makedirs(os.path.join(ri.SLURM_OUTPUT_PATH, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(ri.OUTPUT_DIR, str(ri.GENERATION)), exist_ok=True)

    print(f"Racing settings: {ri.RACING_INITIAL_STARTS} initial starts, +{ri.RACING_STEP} per round, "
          f"cap {ri.RACING_MAX_STARTS}, z={ri.RACING_Z}, ranks 0..{ri.RACING_MAX_RANK}")
    genes = write_variants()
    print(f"Evaluating {len(genes)} variants ({'bash' if args.local else 'sbatch'})...", flush=True)
    evaluate(genes, args.loop_delay)

    results_dir = os.path.join(ri.SOTA_ROOT, "results")
    population, before = [], {}
    for gene_id in genes:
        fitness = ri.GLOBAL_DATA[gene_id]["fitness"]
        if fitness == ri.INVALID_FITNESS_MAX:
            print(f"  ☠ {gene_id} failed its first evaluation; see run_job_outputs/evaluation/")
            continue
        ind = creator.Individual([gene_id])
        ind.fitness.values = fitness
        population.append(ind)
        before[gene_id] = fitness
    if len(population) < 2:
        sys.exit("Fewer than two variants evaluated; nothing to race.")

    t0 = time.time()
    ri.race_population(population, loop_delay=args.loop_delay)

    print(f"\nRacing took {time.time() - t0:.0f}s.\n")
    print(f"{'variant':<14} {'gates':>5} {'starts':>6} {'error before':>13} {'error after':>12}")
    for ind in sorted(population, key=lambda i: i.fitness.values):
        gene_id = ind[0]
        starts = len(racing.collect_starts(results_dir, gene_id))
        print(f"{genes[gene_id]:<14} {ind.fitness.values[1]:>5.0f} {starts:>6} "
              f"{before[gene_id][0]:>13.4f} {ind.fitness.values[0]:>12.4f}")
    raced = [ind[0] for ind in population
             if len(racing.collect_starts(results_dir, ind[0])) > ri.RACING_INITIAL_STARTS]
    print(f"\n{len(raced)} of {len(population)} variants got extra starts.")
    print(f"Files are kept for inspection; remove them with: python tools/racing_smoke.py --clean")


if __name__ == "__main__":
    main()
