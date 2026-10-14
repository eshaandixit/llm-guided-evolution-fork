"""Offline RAG and structural-edit sanity probe; does not call an LLM."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.qiskit_rag import retrieve


SEED_PATH = Path(__file__).with_name("seed_vqc.py")
OUTPUT = ROOT / "reports" / "qiskit_rag_structural_probe.json"


def load_seed():
    spec = importlib.util.spec_from_file_location("qiskit_seed_probe", SEED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    seed = load_seed()
    circuit, _, _ = seed.build_circuit()
    baseline = Counter(instruction.operation.name for instruction in circuit.data)
    structural_variant = circuit.copy()
    structural_variant.cz(0, seed.N_QUBITS - 1)
    variant = Counter(
        instruction.operation.name for instruction in structural_variant.data
    )
    query = (
        "Make a structural mutation to the variational circuit by changing its "
        "CNOT/CX or CZ entanglement topology; preserve every qubit index."
    )
    retrieved = retrieve(query, top_k=3)
    result = {
        "dataset": "Breast Cancer Wisconsin (scikit-learn)",
        "qubits": seed.N_QUBITS,
        "baseline_gate_count": int(sum(baseline.values())),
        "structural_variant_gate_count": int(sum(variant.values())),
        "baseline_gate_histogram": dict(sorted(baseline.items())),
        "structural_variant_gate_histogram": dict(sorted(variant.items())),
        "added_gate": "cz",
        "retrieved_documents": [
            {"source": doc.source, "score": round(doc.score, 4)}
            for doc in retrieved
        ],
        "retrieval_result_count": len(retrieved),
        "llm_inference_performed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
