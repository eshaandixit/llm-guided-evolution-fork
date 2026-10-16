import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from block_fit import fit_to_block

SEED = ROOT / "sota" / "QuantumVQC" / "network.py"
PARTS = SEED.read_text().split("# --OPTION--")
ENT = next(i for i, b in enumerate(PARTS) if "def build_entanglement" in b)
ASSEMBLY = next(i for i, b in enumerate(PARTS) if "def build_circuit" in b)
TRAINING = next(i for i, b in enumerate(PARTS) if "def train_angles" in b)


def others(i):
    return PARTS[:i] + PARTS[i + 1:]


# The shape of the replies that got rejected: a whole program around the edited block.
WHOLE_PROGRAM = '''
import numpy as np
from qiskit import QuantumCircuit

N_QUBITS = 4

def load_data():
    return None

def build_entanglement(qc):
    for i in range(N_QUBITS - 1):
        qc.cz(i, i + 1)

def build_circuit():
    qc = QuantumCircuit(N_QUBITS)
    build_entanglement(qc)
    return qc, [], []

if __name__ == "__main__":
    main()
'''


def test_whole_program_is_cut_down_to_the_block():
    code, changes = fit_to_block(WHOLE_PROGRAM, PARTS[ENT], others(ENT))
    assert "def build_entanglement" in code and "qc.cz(i, i + 1)" in code
    for gone in ("def load_data", "N_QUBITS = 4", "def build_circuit", "__main__"):
        assert gone not in code
    assert "dropped N_QUBITS" in changes and "dropped load_data" in changes


def test_helper_the_llm_added_is_kept():
    reply = "def pairs():\n    return [(0, 1), (1, 2)]\n\ndef build_entanglement(qc):\n    for a, b in pairs():\n        qc.cx(a, b)\n"
    code, changes = fit_to_block(reply, PARTS[ENT], others(ENT))
    assert code == reply and changes == []


def test_dropped_setting_is_restored():
    # The crash in the first Iris run: build_circuit() kept, N_LAYERS = 2 dropped.
    block = PARTS[ASSEMBLY]
    reply = block.split("N_LAYERS = 2", 1)[1]
    code, changes = fit_to_block(reply, block, others(ASSEMBLY))
    assert code.startswith("N_LAYERS = 2") and "def build_circuit" in code
    assert changes == ["restored N_LAYERS"]


def test_main_guard_is_kept_in_the_block_that_owns_it():
    block = PARTS[TRAINING]
    code, changes = fit_to_block(block, block, others(TRAINING))
    assert changes == [] and code == block


def test_unparseable_reply_is_left_for_the_usual_checks():
    reply = "def build_entanglement(qc)\n    pass"
    assert fit_to_block(reply, PARTS[ENT], others(ENT)) == (reply, [])


def test_forbidden_pattern_ignores_comparisons():
    import llm_mutation
    assert llm_mutation.validate_generated_chunk("assert qc.num_qubits == N_QUBITS")[0]
    assert llm_mutation.validate_generated_chunk("if N_QUBITS == 4:\n    pass")[0]
    assert not llm_mutation.validate_generated_chunk("N_QUBITS = 5")[0]


def test_whole_program_reply_becomes_a_working_child(tmp_path, monkeypatch):
    import llm_mutation
    template = tmp_path / "template.txt"
    template.write_text("Improve this block:\n{}")
    monkeypatch.setattr(llm_mutation, "generate_augmented_code", lambda *a, **k: WHOLE_PROGRAM.strip())
    monkeypatch.setattr(llm_mutation.np.random, "randint", lambda lo, hi: ENT)
    monkeypatch.setattr(llm_mutation, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_mutation, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(llm_mutation, "QISKIT_CRITIC", False)
    child = tmp_path / "network_child.py"

    llm_mutation.augment_network(str(SEED), str(child), "template.txt", mutation_type="structural_mutation")

    record = json.loads((tmp_path / "results" / "child_mutation.json").read_text())
    assert record["outcome"] == "ok" and record["fit_changes"]
    namespace = {"__name__": "child"}
    exec(compile(child.read_text(), str(child), "exec"), namespace)
    qc, _, weights = namespace["build_circuit"]()
    assert qc.num_qubits == 4 and weights
    assert qc.count_ops().get("cz") == 3 * namespace["N_LAYERS"]


# A reply cut off by the length limit, with demo code and a Qiskit 0.x import:
# the shape of most replies still rejected after the first fix (second Iris run).
CUT_OFF = '''
from qiskit import QuantumCircuit
from qiskit import execute

N_QUBITS = 4

def build_entanglement(qc):
    for i in range(N_QUBITS - 1):
        qc.cz(i, i + 1)

qc, xp, wp = build_circuit()
print(qc)

def main():
    X, y = load_data()
    qc = QuantumCircuit(N_QUBITS
'''


def test_cut_off_reply_is_salvaged_without_demo_code_or_broken_imports():
    code, changes = fit_to_block(CUT_OFF, PARTS[ENT], others(ENT))
    assert "def build_entanglement" in code and "from qiskit import QuantumCircuit" in code
    for gone in ("execute", "build_circuit()", "print(qc)", "def main", "N_QUBITS = 4"):
        assert gone not in code
    assert changes[0] == "salvaged a cut-off reply"
    assert any(c.startswith("dropped missing import") for c in changes)
    assert any(c.startswith("dropped top-level code") for c in changes)


def test_reply_that_never_wrote_the_block_is_left_to_be_rejected():
    reply = "def load_data():\n    return None\n\ndef something_else():\n    pass\n"
    assert fit_to_block(reply, PARTS[ENT], others(ENT)) == (reply, [])


def test_block_settings_that_call_functions_are_kept():
    block = "WEIGHTS = list(range(3))\n\ndef build_entanglement(qc):\n    pass\n"
    code, changes = fit_to_block(block + "\nprint('demo')\n", block, others(ENT))
    assert "WEIGHTS = list(range(3))" in code and "print" not in code
