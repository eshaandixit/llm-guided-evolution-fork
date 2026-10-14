import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qiskit_rag import retrieve, retrieve_context


def test_structural_query_retrieves_gate_topology_guidance():
    results = retrieve("change CNOT entanglement topology and add a CZ gate", top_k=2)
    assert results
    # two_qubit_gates.md (added 2026-10-08) is the more specific match for a CZ query;
    # either gate-topology note is a correct top hit.
    assert results[0].source in ("structural_gates.md", "two_qubit_gates.md")
    assert "cx" in results[0].text.lower()


def test_weight_query_distinguishes_parameter_binding_from_topology():
    results = retrieve("ParameterVector assign_parameters optimize rotation weights", top_k=2)
    assert results
    assert results[0].source == "parameters_and_binding.md"


def test_formatted_context_has_provenance():
    context = retrieve_context("Qiskit Statevector readout probabilities", top_k=1)
    assert "statevector_readout.md" in context
    assert "Reference:" in context


def test_structural_prompt_includes_retrieved_qiskit_guidance(monkeypatch):
    import run_improved

    monkeypatch.setattr(
        run_improved,
        "PROMPT_GLOB",
        "templates/QuantumVQC/Normal/structural_mutation.txt",
    )
    prompt, mutation_type = run_improved.generate_template(
        0, 0, [], run_improved.SOTA_ROOT, run_improved.SEED_NETWORK,
        run_improved.ROOT_DIR, "llama3",
    )
    assert mutation_type == "structural_mutation"
    assert "Do not change only feature-encoding constants" in prompt
    assert "Retrieved Qiskit API guidance" in prompt
    assert "structural_gates.md" in prompt
    assert "Mujoco RL hard rules" not in prompt
