"""
Regression test for the seed's OPTION-block structure.

Run before handing the seed to the harness:
    python test_seed_blocks.py

Checks the three things that silently break an LLM-GE seed:

 1. The marker appears ONLY as a standalone separator. A `# --OPTION--` inside
    a docstring or comment splits the file there, which silently dumps the
    protected scaffolding (load_data, cross_entropy, main) into a mutable block
    where the LLM can rewrite the scoring function. This has already happened
    once.
 2. Splitting and rejoining round-trips byte-identically, matching
    src/llm_utils.py split_file() and llm_crossover.py's join.
 3. The protected header holds every function that must not be evolved.
"""
import re
import sys
import pathlib

SEED_FILE = pathlib.Path(__file__).parent / "network.py"
MUST_BE_PROTECTED = ["load_data", "gate_count", "weighted_gate_cost", "forward", "forward_batch", "cross_entropy",
                     "accuracy", "write_results", "emit_representations",
                     "count_dead_gates", "main"]
MUST_BE_EVOLVABLE = ["build_feature_map", "build_variational_layer", "build_entanglement",
                     "build_circuit", "readout_probabilities", "train_angles"]

src = SEED_FILE.read_text()
parts = re.split(r"# --OPTION--", src)
failures = []

stray = [(i + 1, l) for i, l in enumerate(src.splitlines())
         if "--OPTION--" in l and l.strip() != "# --OPTION--"]
if stray:
    for n, l in stray:
        failures.append(f"stray marker inside text at line {n}: {l.strip()[:60]!r}")

if "# --OPTION--".join(parts) != src:
    failures.append("split/join does not round-trip byte-identically")

header = parts[0]
for fn in MUST_BE_PROTECTED:
    if f"def {fn}(" not in header:
        failures.append(f"{fn}() is NOT in the protected header - the LLM could rewrite it")

body = "# --OPTION--".join(parts[1:])
for fn in MUST_BE_EVOLVABLE:
    if f"def {fn}(" not in body:
        failures.append(f"{fn}() is not in any evolvable block")

print(f"{len(parts) - 1} evolvable blocks, {len(header.splitlines())}-line protected header")
if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
