"""Re-score each run's front at the equal-compute budget with fresh training starts.

    uv run python tools/rescore_front.py experiments/<name> [--starts 30]

A front's reported errors come from the starts that selected it, so they are biased
towards lucky circuits. This trains every front circuit --starts more times with seeds
that evolution never used (starting at SEED + 1000) and saves the results in
<exp>/rescore/, where tools/racing_analysis.py picks them up. Each circuit is one
small Slurm job; the script waits for them and skips circuits already re-scored.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from racing_analysis import RESCORE_OFFSET, budget, entry_at, load_runs  # noqa: E402

VARIANTS = os.path.join(ROOT, "sota", "QuantumVQC", "models")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="experiments/<name>")
    ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--poll", type=int, default=30, help="seconds between checks")
    args = ap.parse_args()

    runs = load_runs(args.exp)
    limit = budget(runs)
    genes = sorted({p[0] for r in runs.values() for p in entry_at(r["entries"], limit)["front"]})
    out_dir = os.path.abspath(os.path.join(args.exp, "rescore"))
    logs = os.path.join(ROOT, "run_job_outputs", "rescore")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    python = os.path.join(ROOT, ".venv", "bin", "python")

    jobs = {}
    for gene in genes:
        if os.path.exists(os.path.join(out_dir, f"{gene}_starts{RESCORE_OFFSET}_metrics.json")):
            continue
        variant = os.path.join(VARIANTS, f"network_{gene}.py")
        command = (f"{python} {variant} --gene-id {gene} --out-dir {out_dir} --repr '' "
                   f"--n-starts {args.starts} --start-offset {RESCORE_OFFSET}")
        result = subprocess.run(
            ["sbatch", "--parsable", "-t", "1:00:00", "--mem", "8G", "-c", "2", "-J", "rescore",
             "--output", os.path.join(logs, "slurm-%j.out"), "--wrap", command],
            capture_output=True, text=True)
        if result.returncode == 0:
            jobs[gene] = result.stdout.strip()
        else:
            print(f"Could not submit {gene}: {result.stderr.strip()}")
    print(f"{len(genes)} front circuits at the {limit}-start budget; {len(jobs)} submitted, "
          f"{len(genes) - len(jobs)} already done or failed to submit")

    while jobs:
        queued = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(jobs.values())],
                                capture_output=True, text=True).stdout.split()
        jobs = {g: j for g, j in jobs.items() if j in queued}
        if jobs:
            print(f"  waiting on {len(jobs)} job(s)...", flush=True)
            time.sleep(args.poll)
    done = sum(os.path.exists(os.path.join(out_dir, f"{g}_starts{RESCORE_OFFSET}_metrics.json")) for g in genes)
    print(f"Re-scored {done} of {len(genes)}. Next: uv run python tools/racing_analysis.py {args.exp}")


if __name__ == "__main__":
    main()
