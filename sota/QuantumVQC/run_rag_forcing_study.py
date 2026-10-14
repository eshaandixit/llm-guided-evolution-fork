"""Can the prompt force a circuit property, and does Qiskit RAG help? (2026-10-08)

Runs against the live LLM server (hostname.log), using the same client call as
evolution (src.llm_utils.submit_mixtral_local, Llama-3.3-70B):

    python sota/QuantumVQC/run_rag_forcing_study.py [--samples 8]

Part A - property forcing. For each target property the evolvable block that owns it
is sent with the stock structural-mutation template + ConstantRules + one "Goal:"
sentence, with and without the BM25-retrieved Qiskit notes appended (the ONLY
difference between arms). Each answer is spliced into the seed exactly as
llm_mutation.py does, executed, and scored:
    parsed      a code block came back
    valid       the spliced module builds an 8-qubit circuit with trainable params
    trainable   it trains for 45 passes and returns finite validation accuracy
    changed     the gate histogram differs from the seed (a real structural edit)
    compliant   the target property holds in the built circuit
Part B - free-form Qiskit generation: short coding tasks whose answers are executed
and checked against a statevector oracle.

Output: reports/qiskit_rag_forcing.json (+ raw LLM text for every sample).
"""

import argparse
import importlib.util
import json
import random
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.llm_utils import clean_code_from_llm, split_file, submit_mixtral_local  # noqa: E402
from src.qiskit_rag import retrieve, retrieve_context  # noqa: E402

SEED_PATH = ROOT / "sota" / "QuantumVQC" / "network.py"
TEMPLATE = (ROOT / "templates/QuantumVQC/Normal/structural_mutation.txt").read_text()
RULES = (ROOT / "templates/QuantumVQC/ConstantRules.txt").read_text()
OUT = ROOT / "reports" / "qiskit_rag_forcing.json"

# block index in split_file(): 1 feature map, 2 variational layer, 3 entanglement,
# 4 circuit assembly, 5 readout, 6 training
PROPERTIES = {
    "control": (3, None, lambda h, c: True),
    "use_cz": (3, "Replace the CNOT (cx) entanglement with controlled-Z (cz) gates.",
               lambda h, c: h.get("cz", 0) > 0 and h.get("cx", 0) == 0),
    "use_rzz": (3, "Use parameterized Ising ZZ entanglers (rzz) with trainable angles "
                   "instead of fixed CNOTs.",
                lambda h, c: h.get("rzz", 0) > 0),
    "data_reuploading": (4, "Add data re-uploading: apply the feature map again inside "
                            "every layer, before each variational layer.",
                         lambda h, c: c["encoding_gates"] > 8),
    "fewer_gates": (2, "Reduce the total gate count of the circuit below 56 gates while "
                       "keeping it trainable.",
                    lambda h, c: c["gates"] < 56),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def analyse(module_path, train=True):
    m = load_module(module_path, f"variant_{random.random():.9f}".replace(".", "_"))
    qc, xp, wp = m.build_circuit()
    if qc.num_qubits != m.N_QUBITS or len(wp) == 0:
        raise ValueError(f"qubits={qc.num_qubits} params={len(wp)}")
    hist = Counter(i.operation.name for i in qc.data if i.operation.name != "barrier")
    xset = set(xp)
    enc = sum(1 for i in qc.data
              if any(getattr(p, "parameters", set()) & xset for p in i.operation.params))
    info = {"gates": int(m.gate_count(qc)), "histogram": dict(hist), "encoding_gates": enc,
            "n_params": len(wp), "depth": qc.depth()}
    if train:
        X_tr, X_va, X_te, y_tr, y_va, y_te = m.load_data()
        m.TRAIN_BUDGET_EVALS, m._PASS_LIMIT = 45, 45 * len(X_tr)
        m._PASSES, m._BEST, m._COUNTING = 0, [float("inf"), None], True
        try:
            w = m.train_angles(qc, xp, wp, X_tr, y_tr, X_va, y_va)
        except m.BudgetSpent:
            w = m._BEST[1]
        finally:
            m._COUNTING = False
        acc = m.accuracy(qc, xp, wp, w, X_va, y_va)
        if not np.isfinite(acc):
            raise ValueError("non-finite accuracy")
        info["val_accuracy_45pass"] = acc
    return info


def ask(prompt, temperature):
    t0 = time.time()
    raw = submit_mixtral_local(prompt, temperature=temperature, top_p=0.1)
    return raw, time.time() - t0


def part_a(samples, workers):
    parts = split_file(str(SEED_PATH))
    seed_info = analyse(SEED_PATH, train=False)
    jobs = []
    for prop, (idx, goal, _) in PROPERTIES.items():
        task = TEMPLATE.replace("{}", parts[idx].strip(), 1) + "\n" + RULES
        if goal:
            task += f"\nGoal: {goal}\n"
        query = (goal or "structural mutation entanglement topology") + " " + parts[idx]
        context = retrieve_context(query, top_k=3)
        for arm, prompt in (("no_rag", task),
                            ("rag", task + "\n\nRetrieved Qiskit API guidance (use as "
                                           "references, not as code to copy):\n" + context)):
            for k in range(samples):
                jobs.append({"property": prop, "arm": arm, "sample": k, "block": idx,
                             "prompt": prompt, "temperature": 0.3,
                             "retrieved": [d.source for d in retrieve(query, 3)] if arm == "rag" else []})

    def run(job):
        raw, secs = ask(job["prompt"], job["temperature"])
        rec = {k: v for k, v in job.items() if k != "prompt"}
        rec.update({"raw": raw, "llm_seconds": secs, "parsed": False, "valid": False,
                    "trainable": False, "changed": False, "compliant": False})
        code = clean_code_from_llm(raw) if raw else "ERROR"
        if code in ("ERROR", ""):
            return rec
        rec["parsed"] = True
        candidate = list(parts)
        candidate[job["block"]] = f"\n{code}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("# --OPTION--".join(candidate))
        try:
            info = analyse(f.name, train=False)
            rec["valid"], rec["circuit"] = True, info
            rec["changed"] = info["histogram"] != seed_info["histogram"]
            rec["compliant"] = bool(PROPERTIES[job["property"]][2](info["histogram"], info))
            info.update(analyse(f.name, train=True))
            rec["trainable"] = True
        except Exception as error:
            rec["error"] = f"{type(error).__name__}: {error}"[:300]
        return rec

    with ThreadPoolExecutor(workers) as pool:
        return seed_info, list(pool.map(run, jobs))


FREEFORM = {
    "ghz": ("Write a Python function `make_circuit(n)` using Qiskit that returns a "
            "QuantumCircuit preparing the n-qubit GHZ state (|0...0> + |1...1>)/sqrt(2). "
            "No measurements. Return one Python code block.",
            lambda f: _sv_close(f(4), np.eye(1, 16, 0)[0] / np.sqrt(2) + np.eye(1, 16, 15)[0] / np.sqrt(2))),
    "bell_phi_minus": ("Write a Python function `make_circuit()` using Qiskit that returns "
                       "a 2-qubit QuantumCircuit preparing (|00> - |11>)/sqrt(2). No "
                       "measurements. Return one Python code block.",
                       lambda f: _sv_close(f(), np.array([1, 0, 0, -1]) / np.sqrt(2))),
    "hea_params": ("Write a Python function `make_circuit(n, layers)` using Qiskit that "
                   "returns a hardware-efficient ansatz: each layer applies a trainable RY "
                   "on every qubit followed by a linear chain of CX gates. Use a single "
                   "ParameterVector named 'theta' of length n*layers. Return one Python code block.",
                   lambda f: (lambda qc: qc.num_parameters == 12 and
                              sum(i.operation.name == "cx" for i in qc.data) == 9)(f(4, 3))),
    "angle_encoding": ("Write a Python function `make_circuit(x)` using Qiskit that encodes "
                       "a list of real features x into a circuit with one qubit per feature, "
                       "applying RY(x[i]) to qubit i. Return one Python code block.",
                       lambda f: _sv_close(f([np.pi, 0.0]), np.array([0, 1, 0, 0]))),
    "qft3": ("Write a Python function `make_circuit()` using Qiskit that returns the 3-qubit "
             "quantum Fourier transform circuit (including the final swaps) built from H, "
             "controlled-phase (cp) and swap gates only. Return one Python code block.",
             lambda f: _unitary_close(f(), _qft(3))),
    "parity_readout": ("Write a Python function `parity_probability(qc)` that, given a "
                       "Qiskit QuantumCircuit without measurements, uses "
                       "qiskit.quantum_info.Statevector to return the probability that the "
                       "measured bitstring has even parity. Return one Python code block.",
                       lambda f: abs(f(_h_on_first()) - 0.5) < 1e-9),
}


def _sv_close(qc, target):
    from qiskit.quantum_info import Statevector
    sv = Statevector(qc).data
    return abs(abs(np.vdot(sv, target)) - 1) < 1e-6


def _unitary_close(qc, target):
    from qiskit.quantum_info import Operator
    u = Operator(qc).data
    return abs(abs(np.trace(u.conj().T @ target)) / len(target) - 1) < 1e-6


def _qft(n):
    d = 2 ** n
    w = np.exp(2j * np.pi / d)
    return np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)


def _h_on_first():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    return qc


def part_b(samples, workers):
    jobs = []
    for name, (prompt, _) in FREEFORM.items():
        context = retrieve_context(prompt, top_k=3)
        for arm, p in (("no_rag", prompt),
                       ("rag", prompt + "\n\nRetrieved Qiskit API guidance:\n" + context)):
            for k in range(samples):
                jobs.append({"task": name, "arm": arm, "sample": k, "prompt": p})

    def run(job):
        raw, secs = ask(job["prompt"], 0.3)
        rec = {k: v for k, v in job.items() if k != "prompt"}
        rec.update({"raw": raw, "llm_seconds": secs, "parsed": False, "executes": False,
                    "correct": False})
        code = clean_code_from_llm(raw) if raw else "ERROR"
        if code in ("ERROR", ""):
            return rec
        rec["parsed"] = True
        try:
            namespace = {}
            exec(compile(code, f"<{job['task']}>", "exec"), namespace)
            fn = namespace.get("parity_probability") or namespace["make_circuit"]
            rec["executes"] = True
            rec["correct"] = bool(FREEFORM[job["task"]][1](fn))
        except Exception as error:
            rec["error"] = f"{type(error).__name__}: {error}"[:300]
            rec["trace"] = traceback.format_exc()[-600:]
        return rec

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(run, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--freeform-samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    seed_info, forcing = part_a(args.samples, args.workers)
    freeform = part_b(args.freeform_samples, args.workers)
    OUT.write_text(json.dumps({"seed": seed_info, "forcing": forcing, "freeform": freeform,
                               "model": "Llama-3.3-70B-Instruct", "temperature": 0.3,
                               "top_p": 0.1, "seconds": time.time() - t0}, indent=1))
    for prop in PROPERTIES:
        for arm in ("no_rag", "rag"):
            rs = [r for r in forcing if r["property"] == prop and r["arm"] == arm]
            print(f"{prop:17s} {arm:7s} " + "  ".join(
                f"{k} {sum(r[k] for r in rs)}/{len(rs)}"
                for k in ("parsed", "valid", "trainable", "changed", "compliant")))
    for name in FREEFORM:
        for arm in ("no_rag", "rag"):
            rs = [r for r in freeform if r["task"] == name and r["arm"] == arm]
            print(f"{name:17s} {arm:7s} executes {sum(r['executes'] for r in rs)}/{len(rs)}  "
                  f"correct {sum(r['correct'] for r in rs)}/{len(rs)}")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
