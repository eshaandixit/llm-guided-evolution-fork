import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import motif_library as ml

SEED = ROOT / "sota" / "QuantumVQC" / "network.py"
SEED_TEXT = SEED.read_text()
SEED_BLOCKS = ml.split_blocks(SEED_TEXT)
ENT = next(i for i, b in enumerate(SEED_BLOCKS) if "def build_entanglement" in b)
ASSEMBLY = next(i for i, b in enumerate(SEED_BLOCKS) if "N_LAYERS = 2" in b)

CZ_RING = "qc.cx(i, (i + 1) % N_QUBITS)", "qc.cz(i, (i + 1) % N_QUBITS)"
LINEAR = ("    for i in range(N_QUBITS):\n        qc.cx(i, (i + 1) % N_QUBITS)",
          "    for i in range(N_QUBITS - 1):\n        qc.cx(i, i + 1)")
ONE_LAYER = "N_LAYERS = 2", "N_LAYERS = 1"


def make_run(tmp_path, genes):
    """genes: {gene_id: (edits, accuracies, gates, parent, edited block index)}"""
    variants, results = tmp_path / "models", tmp_path / "results"
    variants.mkdir()
    results.mkdir()
    for gene_id, (edits, accs, gates, parent, index) in genes.items():
        code = SEED_TEXT
        for old, new in edits:
            assert code.count(old) == 1
            code = code.replace(old, new)
        (variants / f"network_{gene_id}.py").write_text(code)
        starts = [{"seed": 42 + i, "val_accuracy": a} for i, a in enumerate(accs)]
        (results / f"{gene_id}_metrics.json").write_text(
            json.dumps({"obj2_gates": gates, "starts": starts}))
        if parent:
            (results / f"{gene_id}_mutation.json").write_text(json.dumps(
                {"child": gene_id, "parent": parent, "block_index": index, "outcome": "ok"}))
    return variants, results


def build(tmp_path, genes, **kwargs):
    variants, results = make_run(tmp_path, genes)
    out = tmp_path / "lib.json"
    lib = ml.build_library(list(genes), str(results), str(variants), "network", str(SEED),
                           str(out), **kwargs)
    return lib, variants, results, out


def test_normalize_ignores_comments_and_spacing():
    a = "def f(qc):\n    # ring\n    qc.cx(0, 1)\n"
    b = "\n\ndef f(qc):\n\n    qc.cx(0,1)  # comment\n"
    assert ml.normalize(a) == ml.normalize(b)


def test_front_block_that_differs_from_seed_becomes_a_motif(tmp_path):
    lib, *_ = build(tmp_path, {
        "base": ([], [0.7, 0.7, 0.7], 28, None, None),
        "cz": ([CZ_RING], [0.8, 0.8, 0.8], 28, None, None),     # dominates base
    })
    motifs = lib["motifs"]
    assert len(motifs) == 1
    assert motifs[0]["block_index"] == ENT and motifs[0]["block"] == "build_entanglement"
    assert "qc.cz" in motifs[0]["code"] and motifs[0]["front_genes"] == ["cz"]


def test_edit_evidence_is_weighted_by_starts(tmp_path):
    lib, *_ = build(tmp_path, {
        "parent": ([], [0.6] * 10, 28, None, None),
        "few": ([LINEAR], [0.9, 0.9, 0.9], 26, "parent", ENT),
        "many": ([LINEAR], [0.65] * 10, 26, "parent", ENT),
    }, max_rank=0)
    (motif,) = lib["motifs"]
    w_few, w_many = 10 * 3 / 13, 10 * 10 / 20
    expected = (w_few * -0.3 + w_many * -0.05) / (w_few + w_many)
    assert abs(motif["avg_delta_error"] - expected) < 1e-9
    assert motif["avg_delta_gates"] == -2


def test_edit_that_made_things_worse_is_not_a_motif(tmp_path):
    lib, *_ = build(tmp_path, {
        "parent": ([], [0.8] * 5, 28, None, None),
        "worse": ([CZ_RING], [0.5] * 5, 28, "parent", ENT),
    }, max_rank=0)
    assert lib["motifs"] == []


def test_motif_that_breaks_the_circuit_is_rejected():
    bad_qubits = {"block_index": ASSEMBLY,
                  "code": SEED_BLOCKS[ASSEMBLY].replace("QuantumCircuit(N_QUBITS)", "QuantumCircuit(N_QUBITS + 1)")}
    syntax = {"block_index": ENT, "code": "def build_entanglement(qc)\n    pass"}
    assert ml.validate(bad_qubits, SEED_BLOCKS)[0] is False
    assert ml.validate(syntax, SEED_BLOCKS)[0] is False
    good = {"block_index": ENT, "code": SEED_BLOCKS[ENT].replace(*CZ_RING)}
    assert ml.validate(good, SEED_BLOCKS) == (True, "")


def test_naming_uses_llm_json_even_when_the_prompt_is_echoed(tmp_path):
    def generate(prompt):
        ids = [line.split()[1] for line in prompt.splitlines() if line.startswith("id: ")]
        return prompt + json.dumps([{"id": i, "name": "CZ ring", "description": "Uses CZ."} for i in ids])

    lib, *_ = build(tmp_path, {
        "base": ([], [0.7] * 3, 28, None, None),
        "cz": ([CZ_RING], [0.8] * 3, 28, None, None),
    }, generate=generate)
    assert lib["motifs"][0]["name"] == "CZ ring"
    assert lib["motifs"][0]["description"] == "Uses CZ."


def test_naming_falls_back_when_llm_fails(tmp_path):
    def generate(prompt):
        raise ConnectionError("server down")

    lib, *_ = build(tmp_path, {
        "base": ([], [0.7] * 3, 28, None, None),
        "cz": ([CZ_RING], [0.8] * 3, 28, None, None),
    }, generate=generate)
    assert lib["motifs"][0]["name"].startswith("build_entanglement variant")


def test_credit_counts_shown_and_used_and_prunes_harmful_motifs(tmp_path):
    genes = {
        "base": ([], [0.7] * 4, 28, None, None),
        "cz": ([CZ_RING], [0.8] * 4, 28, None, None),
    }
    for i in range(3):   # three children were offered the CZ motif, used it, and got worse
        genes[f"user{i}"] = ([CZ_RING], [0.4] * 4, 28, "base", ENT)
    genes["ignorer"] = ([LINEAR], [0.4] * 4, 26, "base", ENT)
    variants, results = make_run(tmp_path, genes)
    cz_id = ml.motif_id(ENT, SEED_BLOCKS[ENT].replace(*CZ_RING))
    for g in ("user0", "user1", "user2", "ignorer"):
        path = results / f"{g}_mutation.json"
        record = json.loads(path.read_text())
        record["motifs_shown"] = [cz_id]
        path.write_text(json.dumps(record))

    candidates, _ = ml.gather_evidence(list(genes), str(results), str(variants), "network", str(SEED))
    motifs = ml.credit(ml.score_candidates(candidates), list(genes), str(results), str(variants),
                       "network", str(SEED))
    cz = next(m for m in motifs if m["id"] == cz_id)
    assert cz["shown"] == 4 and cz["used"] == 3
    assert abs(cz["used_delta_error"] - 0.3) < 1e-9     # 0.6 error vs the parent's 0.3
    assert cz_id not in [m["id"] for m in ml.prune(motifs)]


def test_select_skips_motifs_identical_to_current_block():
    cz_code = SEED_BLOCKS[ENT].replace(*CZ_RING)
    motifs = [{"block_index": ENT, "code": ml.clean_code(cz_code), "id": "a"},
              {"block_index": ENT, "code": ml.clean_code(SEED_BLOCKS[ENT].replace(*LINEAR)), "id": "b"},
              {"block_index": ASSEMBLY, "code": "N_LAYERS = 1", "id": "c"}]
    assert [m["id"] for m in ml.select_for_block(motifs, ENT, cz_code)] == ["b"]


def test_mutation_prompt_gets_motifs_except_for_eot(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "src"))
    import llm_mutation

    library = tmp_path / "lib.json"
    motif = {"id": "m1", "block_index": ENT, "block": "build_entanglement", "name": "CZ ring",
             "description": "Uses CZ.", "code": ml.clean_code(SEED_BLOCKS[ENT].replace(*CZ_RING)),
             "front_genes": ["g"], "avg_delta_error": -0.05, "avg_delta_gates": 0.0}
    library.write_text(json.dumps({"motifs": [motif]}))
    template = tmp_path / "template.txt"
    template.write_text("Improve this block:\n{}")
    prompts = []
    monkeypatch.setattr(llm_mutation, "generate_augmented_code",
                        lambda txt, *a, **k: prompts.append(txt) or SEED_BLOCKS[ENT].replace(*LINEAR))
    monkeypatch.setattr(llm_mutation.np.random, "randint", lambda lo, hi: ENT)
    monkeypatch.setattr(llm_mutation, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_mutation, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(llm_mutation, "QISKIT_CRITIC", False)

    for mutation_type, child in (("structural_mutation", "a"), ("EoT", "b")):
        llm_mutation.augment_network(str(SEED), str(tmp_path / f"network_{child}.py"), "template.txt",
                                     mutation_type=mutation_type, motif_library_path=str(library))
    normal, eot = prompts
    assert "CZ ring: Uses CZ." in normal and "Patterns that have worked" in normal
    assert "Patterns that have worked" not in eot
    assert json.loads((tmp_path / "results" / "a_mutation.json").read_text())["motifs_shown"] == ["m1"]
    assert json.loads((tmp_path / "results" / "b_mutation.json").read_text())["motifs_shown"] == []
