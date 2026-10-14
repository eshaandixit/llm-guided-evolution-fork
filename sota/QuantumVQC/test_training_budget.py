"""Every individual gets the same training budget, however it trains.

    python test_training_budget.py

Training is an evolvable block, so a variant can rewrite the optimizer or raise
MAX_ITER - run 5872577 had both. The budget is therefore enforced in the protected
header, which counts circuit simulations while training runs. This builds variants
that try to train longer and checks the budget holds.
"""
import json, os, pathlib, re, shutil, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).parent
SEED = (HERE / "seed_vqc.py").read_text()
PY = sys.executable
BUDGET = 10

BLOCK = SEED[SEED.rindex("# --OPTION--"):]          # the training block

VARIANTS = {
    "stock": BLOCK,
    "greedy_maxiter": BLOCK.replace("MAX_ITER = 178", "MAX_ITER = 5000"),
    "differential_evolution": '''# --OPTION--
MAX_ITER = 5000
from scipy.optimize import differential_evolution

def train_angles(qc, x_params, w_params, X, y, *args, **kwargs):
    rng = np.random.default_rng(SEED)
    w0 = rng.uniform(0, 2 * np.pi, len(w_params))

    def objective(w):
        return cross_entropy(qc, x_params, w_params, w, X, y)

    res = differential_evolution(objective, [(0, 2 * np.pi)] * len(w_params),
                                 x0=w0, maxiter=MAX_ITER)
    return res.x


if __name__ == "__main__":
    main()
''',
}
assert VARIANTS["greedy_maxiter"] != BLOCK, "MAX_ITER anchor missed"

failures = []
with tempfile.TemporaryDirectory() as tmp:
    for name, block in VARIANTS.items():
        src = SEED[:SEED.rindex("# --OPTION--")] + block
        f = pathlib.Path(tmp) / f"v_{name}.py"
        f.write_text(src)
        out = pathlib.Path(tmp) / name
        r = subprocess.run([PY, str(f), "--gene-id", name, "--out-dir", str(out), "--repr", "",
                    "--train-budget", str(BUDGET)],
                           capture_output=True, text=True, timeout=1800)
        mfile = out / f"{name}_metrics.json"
        if r.returncode != 0 or not mfile.exists():
            failures.append(f"{name}: did not finish (exit {r.returncode}) {r.stderr.strip()[-200:]}")
            continue
        m = json.loads(mfile.read_text())
        used = m["train_evals"]
        print(f"  {name:24s} used {used:6.0f} of {BUDGET} evals   obj1 {m['obj1_val_error']:.4f}  "
              f"obj2 {m['obj2_gates']:.0f}  ({m['seconds']:.0f}s)")
        if used > BUDGET + 1:
            failures.append(f"{name}: trained for {used:.0f} evals, budget is {BUDGET}")

print()
if failures:
    print("FAILED:")
    for x in failures:
        print("  -", x)
    sys.exit(1)
print(f"all variants trained within the {BUDGET}-evaluation budget")
