"""Compare matched-budget COBYLA, SPSA, and warm-started SPSA runs."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEED = Path(__file__).with_name("seed_vqc.py")
DEFAULT_OUTPUT = ROOT / "reports" / "qiskit_vqc_optimizer_ablation.csv"


def run_trial(name, optimizer, output_dir, train_budget, initial_weights=None):
    env = os.environ.copy()
    env["QISKIT_EARLY_STOP_PATIENCE"] = "8"
    command = [
        sys.executable, str(SEED), "--gene-id", name,
        "--out-dir", str(output_dir), "--repr", "", "--optimizer", optimizer,
        "--train-budget", str(train_budget),
    ]
    if initial_weights:
        command.extend(("--initial-weights", str(initial_weights)))
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:] or completed.stdout[-3000:])
    metrics_path = output_dir / f"{name}_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8")), completed.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    curve_rows = []
    with tempfile.TemporaryDirectory(prefix="qiskit-weight-ablation-") as tmp:
        tmp_path = Path(tmp)
        cobyla, _ = run_trial("cobyla_random", "cobyla", tmp_path / "cobyla", 178)
        spsa, _ = run_trial("spsa_random", "spsa", tmp_path / "spsa", 89)
        warm, _ = run_trial(
            "spsa_warm", "spsa", tmp_path / "warm", 89,
            tmp_path / "cobyla" / "cobyla_random_metrics.json",
        )
        trials = [
            ("COBYLA", "random initialization", cobyla),
            ("SPSA", "random initialization", spsa),
            ("SPSA", "COBYLA-trained warm start", warm),
        ]
        for optimizer, initialization, metrics in trials:
            rows.append({
                "optimizer": optimizer,
                "initialization": initialization,
                "validation_accuracy": metrics["val_accuracy"],
                "validation_error": metrics["obj1_val_error"],
                "validation_cross_entropy": metrics["val_cross_entropy"],
                "gate_count": metrics["obj2_gates"],
                "training_evaluations": metrics["train_evals"],
                "validation_curve_steps": len(metrics["training_curve"]),
                "seconds": metrics["seconds"],
            })
            curve_rows.extend({
                "optimizer": optimizer,
                "initialization": initialization,
                **point,
            } for point in metrics["training_curve"])

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curves_path = args.output.with_name("qiskit_vqc_training_curves.csv")
    if curve_rows:
        with curves_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {curves_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
