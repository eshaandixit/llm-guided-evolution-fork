"""Compare one structural Qiskit mutation with and without retrieved context.

Run only while the PACE LLM server is ready:
    python sota/QuantumVQC/run_live_rag_ablation.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm_utils import clean_code_from_llm, split_file, submit_mixtral_local
from src.qiskit_rag import retrieve, retrieve_context


SEED_PATH = ROOT / "sota" / "QuantumVQC" / "network.py"
TEMPLATE_PATH = ROOT / "templates" / "QuantumVQC" / "Normal" / "structural_mutation.txt"
RULES_PATH = ROOT / "templates" / "QuantumVQC" / "ConstantRules.txt"
OUTPUT_PATH = ROOT / "reports" / "live_rag_prompt_ablation.json"
TARGET_BLOCK_INDEX = 3  # protected header, encoding, variational layer, entanglement


def compile_variant(parts, code):
    candidate = list(parts)
    candidate[TARGET_BLOCK_INDEX] = code
    source = "# --OPTION--".join(candidate)
    compile(source, "<llm-qiskit-mutation>", "exec")
    return source


def main():
    parts = split_file(str(SEED_PATH))
    target_block = parts[TARGET_BLOCK_INDEX]
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    rules = RULES_PATH.read_text(encoding="utf-8")
    task = template.replace("{}", target_block.strip(), 1) + "\n" + rules
    query = (
        "Qiskit structural quantum circuit mutation, variational RY RZ rotations, "
        "CX CZ entanglement topology, valid qubit indices, QuantumCircuit API."
    )
    context = retrieve_context(query, top_k=3)
    prompts = {
        "without_rag": task,
        "with_rag": task + "\n\nRetrieved Qiskit API guidance:\n" + context,
    }

    results = {}
    for label, prompt in prompts.items():
        raw = submit_mixtral_local(
            prompt, max_new_tokens=1200, temperature=0.1, top_p=0.15
        )
        code = clean_code_from_llm(raw)
        record = {"raw_output": raw, "code": code, "valid_python": False,
                  "structural_change": False, "added_cz": False}
        if code and code != "ERROR":
            try:
                compile_variant(parts, code)
                record["valid_python"] = True
            except SyntaxError as error:
                record["syntax_error"] = str(error)
            record["structural_change"] = code.strip() != target_block.strip()
            record["added_cz"] = ".cz(" in code or ".cz(" in raw
        results[label] = record

    output = {
        "task": "Force a structural entanglement mutation in the build_entanglement block",
        "dataset": "Breast Cancer Wisconsin, 8-qubit PCA seed",
        "retrieved_sources": [item.source for item in retrieve(query, top_k=3)],
        "changed_variable": "retrieved Qiskit documentation context only",
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
