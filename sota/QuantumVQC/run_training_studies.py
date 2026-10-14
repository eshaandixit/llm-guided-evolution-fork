"""Offline training studies on the QuantumVQC seed (no LLM involved).

    python sota/QuantumVQC/run_training_studies.py [falloff optimizers inheritance readout]

Writes reports/qiskit_studies/<study>.json; tools/plot_qiskit_studies.py draws them.

  falloff      COBYLA with early stopping OFF and a 3.4x larger budget, 10 starts:
               where does validation loss/accuracy stop improving? (sets TRAIN_BUDGET_EVALS)
  optimizers   COBYLA vs SPSA at the evolution budget, and Adam on parameter-shift
               gradients (EXAQC trains with Adam) at its own, much larger budget.
  inheritance  Lamarckian weight inheritance (EXAQC/EXAMM): structural children of the
               seed trained from random angles vs from their parent's trained angles,
               at several budgets.
  readout      Which qubit is measured (the "qubit selection for local cost functions"
               idea): seed retrained with each of the 8 qubits as the readout.
"""

import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "reports" / "qiskit_studies"
N_STARTS = 5


def load_seed():
    spec = importlib.util.spec_from_file_location("seed_study", HERE / "seed_vqc.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def variant_circuit(m, variant):
    """Seed circuit or one of its structural children (same parameter ordering)."""
    if variant == "seed":
        return m.build_circuit()
    if variant == "cz_ring":
        def ent(qc):
            for i in range(m.N_QUBITS):
                qc.cz(i, (i + 1) % m.N_QUBITS)
        m.build_entanglement = ent
    elif variant == "linear_cx":
        def ent(qc):
            for i in range(m.N_QUBITS - 1):
                qc.cx(i, i + 1)
        m.build_entanglement = ent
    elif variant == "three_layers":
        m.N_LAYERS = 3
    else:
        raise ValueError(variant)
    return m.build_circuit()


def train_once(m, qc, xp, wp, data, optimizer, budget, seed, init=None, patience=18,
               lr=None):
    X_tr, X_va, X_te, y_tr, y_va, y_te = data
    m.TRAIN_BUDGET_EVALS = budget
    m.MAX_ITER = budget
    m.EARLY_STOP_PATIENCE = patience
    m.SEED = seed
    m._PASS_LIMIT = budget * len(X_tr)
    m._PASSES = 0
    m._BEST = [float("inf"), None]
    m._TRAINING_CURVE.clear()
    if lr is not None:
        m.adam_shift.__defaults__ = (None, lr, 0.9, 0.999, 1e-8)
    t0 = time.perf_counter()
    m._COUNTING = True
    try:
        w = m.train_angles(qc, xp, wp, X_tr, y_tr, X_va, y_va,
                           initial_weights=init, optimizer=optimizer)
    except m.BudgetSpent:
        w = m._BEST[1]
    finally:
        m._COUNTING = False
    w = np.asarray(w, dtype=float)
    return {
        "optimizer": optimizer, "budget": budget, "seed": seed, "patience": patience,
        "inherited": init is not None, "lr": lr,
        "val_accuracy": m.accuracy(qc, xp, wp, w, X_va, y_va),
        "val_cross_entropy": m.cross_entropy(qc, xp, wp, w, X_va, y_va),
        "test_accuracy": m.accuracy(qc, xp, wp, w, X_te, y_te),
        "train_accuracy": m.accuracy(qc, xp, wp, w, X_tr, y_tr),
        "train_passes_used": m._PASSES / len(X_tr),
        "gates": m.gate_count(qc), "n_params": len(wp),
        "seconds": time.perf_counter() - t0,
        "curve": list(m._TRAINING_CURVE),
        "weights": w.tolist(),
    }


def task(spec):
    m = load_seed()
    if spec.get("readout") is not None:
        m.READOUT = spec["readout"]
    qc, xp, wp = variant_circuit(m, spec.get("variant", "seed"))
    data = m.load_data()
    init = spec.get("init")
    if init is not None and len(init) < len(wp):            # new layer: random angles
        rng = np.random.default_rng(spec["seed"] + 1000)
        init = list(init) + rng.uniform(0, 2 * np.pi, len(wp) - len(init)).tolist()
    r = train_once(m, qc, xp, wp, data, spec["optimizer"], spec["budget"], spec["seed"],
                   init=init, patience=spec.get("patience", 18), lr=spec.get("lr"))
    r.update({k: v for k, v in spec.items() if k not in ("init",)})
    return r


def run(study, specs, processes=4):
    t0 = time.perf_counter()
    with Pool(processes) as pool:
        results = pool.map(task, specs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{study}.json").write_text(json.dumps(results, indent=1))
    print(f"{study}: {len(results)} trainings in {time.perf_counter() - t0:.0f}s -> {OUT / study}.json")
    return results


def main(studies):
    seeds = [42 + k for k in range(N_STARTS)]
    if "falloff" in studies:
        run("falloff", [{"optimizer": "cobyla", "budget": 600, "seed": 42 + k, "patience": 0}
                        for k in range(10)])
    if "optimizers" in studies:
        specs = [{"optimizer": o, "budget": 178, "seed": s} for o in ("cobyla", "spsa") for s in seeds]
        # Adam on parameter-shift gradients: 2*32 passes per step. 100 steps = 6400 passes.
        specs += [{"optimizer": "adam", "budget": 6400, "seed": s, "patience": 0, "lr": 0.05}
                  for s in seeds]
        run("optimizers", specs)
    if "inheritance" in studies:
        parents = run("inheritance_parents",
                      [{"optimizer": "cobyla", "budget": 178, "seed": s, "variant": "seed"}
                       for s in seeds])
        specs = []
        for variant in ("cz_ring", "linear_cx", "three_layers"):
            for budget in (20, 45, 90, 178):
                for p in parents:
                    specs.append({"optimizer": "cobyla", "budget": budget, "seed": p["seed"],
                                  "variant": variant, "init": None, "patience": 0})
                    specs.append({"optimizer": "cobyla", "budget": budget, "seed": p["seed"],
                                  "variant": variant, "init": p["weights"], "patience": 0})
        run("inheritance", specs)
    if "readout" in studies:
        run("readout", [{"optimizer": "cobyla", "budget": 178, "seed": s, "readout": [q]}
                        for q in range(8) for s in seeds])


if __name__ == "__main__":
    main(sys.argv[1:] or ["falloff", "optimizers", "inheritance", "readout"])
