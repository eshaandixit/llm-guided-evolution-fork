"""
Adapter so the quantum seed satisfies LLM-GE's existing eval contract.

run_improved.py invokes every individual as:

    python <train_file> -bs 216 -network "models.network_<gene_id>" \
           -data <DATA_PATH> -end_lr 0.001 -seed 21 -val_r 0.2 -amp

That signature comes from ExquisiteNetV2, where train.py imports a mutated
network module and trains it. Our seed is self-contained instead - the mutated
file builds its circuit, trains the angles, scores on validation and writes its
own results file.

Rather than change run_improved.py to suit the seed, this shim makes the seed
speak the contract LLM-GE already has. The only edit to the evolutionary loop is
the default train_file path.

The ExquisiteNetV2 flags (-bs, -data, -end_lr, -val_r, -amp, -epoch) do not apply
to a variational circuit and are accepted and ignored on purpose, so the call
signature keeps working untouched.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-network", required=True, help='e.g. "models.network_abc123"')
    # Accepted for signature compatibility, unused for a quantum circuit.
    ap.add_argument("-bs", type=int, default=None)
    ap.add_argument("-data", default=None)
    ap.add_argument("-end_lr", type=float, default=None)
    ap.add_argument("-seed", type=int, default=None)
    ap.add_argument("-val_r", type=float, default=None)
    ap.add_argument("-epoch", type=int, default=None)
    ap.add_argument("-amp", action="store_true")
    ap.add_argument("-parent", default=None,
                    help="parent gene id; its trained angles warm-start the first training start")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[train.py] ignoring unused args: {unknown}")

    # "models.network_<gene_id>" -> "<gene_id>"
    gene_id = args.network.split("network_")[-1]

    sys.path.insert(0, HERE)
    module = importlib.import_module(args.network)

    # The variant's own CLI does the work; hand it the gene id and output dir.
    out_dir = os.path.join(HERE, "results")
    sys.argv = ["network", "--gene-id", gene_id, "--out-dir", out_dir]
    if args.parent:
        parent_metrics = os.path.join(out_dir, f"{args.parent}_metrics.json")
        if os.path.exists(parent_metrics):
            sys.argv += ["--initial-weights", parent_metrics]
    module.main()

    # check_contents_for_error() in run_improved.py treats a job as finished only
    # when this exact string appears in its slurm log; sota/ExquisiteNetV2/train.py
    # prints it on line 304. Without it check4error() returns None forever - not
    # done, not errored - and the orchestrator waits out its whole walltime on a
    # job that finished in 20 seconds.
    print("=" * 120); print("job done"); print("=" * 120)


if __name__ == "__main__":
    main()
