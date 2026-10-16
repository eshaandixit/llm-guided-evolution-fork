"""Motif library: reusable circuit patterns learned from the current run.

Every few generations the orchestrator rebuilds the library from what the run has
already scored, and mutation prompts offer the motifs that fit the block being edited.

A motif is one version of one evolvable block (the code between two "# --OPTION--"
markers) that differs from the seed's version and has evidence behind it:

  * front evidence: Pareto-front circuits (rank <= max_rank) that contain the block
  * edit evidence: mutations that wrote the block, with the change in error and gates
    from parent to child. Each change is weighted by n_parent*n_child/(n_parent+n_child),
    which is proportional to 1/variance of the change in mean error, so a change
    measured on 3 starts counts for less than one measured on 10.

Rebuilding is stateless: the library is recomputed from the run's files each time, so
a motif's id (a hash of its block index and code) and its credit are stable.

Credit: each mutation records which motifs its prompt offered. A child "used" a motif
when its edited block matches the motif's code closely. Motifs whose users did worse
than their parents, over enough uses, are dropped.

Nothing here talks to Slurm; the LLM is only used to name motifs, through a callable
passed in, and naming falls back to plain labels if that call fails.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import textwrap

from src import racing

OPTION_MARKER = "# --OPTION--"
DEF_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(", re.M)


# ---------------------------------------------------------------------------
# Blocks

def split_blocks(text):
    """Same split as llm_utils.split_file; index 0 is the protected header."""
    return text.split(OPTION_MARKER)


def read_blocks(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return split_blocks(f.read())
    except OSError:
        return None


def normalize(block):
    """Code with comments, blank lines and formatting removed, for comparing blocks."""
    code = textwrap.dedent(block).strip()
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        lines = [ln.split("#", 1)[0].rstrip() for ln in code.splitlines()]
        return "\n".join(ln for ln in lines if ln.strip())


def clean_code(block):
    """The block as shown to the LLM: no NOTE markers, comments kept, trimmed."""
    if "# -- NOTE --" in block:
        parts = block.split("# -- NOTE --")
        block = parts[0] + (parts[2] if len(parts) > 2 else "")
    return textwrap.dedent(block).strip()


def block_name(block):
    match = DEF_RE.search(block)
    if match:
        return match.group(1)
    for line in block.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            return line.split("=")[0].strip()
    return "block"


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def motif_id(block_index, code):
    digest = hashlib.sha1(f"{block_index}\n{normalize(code)}".encode()).hexdigest()
    return digest[:8]


# ---------------------------------------------------------------------------
# Evidence

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def gather_evidence(gene_ids, results_dir, variant_dir, model, seed_path, max_rank=1):
    """Collect candidate blocks from the current run's genes.

    Returns (candidates, seed_blocks) where candidates maps (block_index, normalized
    code) to a dict with the code, the genes on the front that contain it, and the
    weighted edits that produced it.
    """
    seed_blocks = read_blocks(seed_path)
    stats = racing.population_stats(results_dir, gene_ids)
    genes = sorted(stats)
    ranks = dict(zip(genes, racing.nondominated_ranks([(stats[g][0], stats[g][3]) for g in genes])))
    blocks = {g: read_blocks(os.path.join(variant_dir, f"{model}_{g}.py")) for g in genes}
    seed_norm = [normalize(b) for b in seed_blocks]

    candidates = {}

    def entry(index, code):
        key = (index, normalize(code))
        if key not in candidates:
            candidates[key] = {"block_index": index, "code": clean_code(code),
                               "front_genes": [], "edits": []}
        return candidates[key]

    for g in genes:
        parts = blocks[g]
        if not parts or len(parts) != len(seed_blocks) or ranks[g] > max_rank:
            continue
        for index in range(1, len(parts)):
            if normalize(parts[index]) != seed_norm[index]:
                entry(index, parts[index])["front_genes"].append(g)

    for g in genes:
        record = _load_json(os.path.join(results_dir, f"{g}_mutation.json"))
        if not record or record.get("outcome") != "ok":
            continue
        parent, index = record.get("parent"), record.get("block_index")
        parts = blocks[g]
        if parent not in stats or not parts or index is None or not 0 < index < len(parts):
            continue
        if len(parts) != len(seed_blocks) or normalize(parts[index]) == seed_norm[index]:
            continue
        n_p, n_c = stats[parent][2], stats[g][2]
        entry(index, parts[index])["edits"].append({
            "child": g, "parent": parent,
            "delta_error": stats[g][0] - stats[parent][0],
            "delta_gates": stats[g][3] - stats[parent][3],
            "weight": n_p * n_c / (n_p + n_c),
        })
    return candidates, seed_blocks


def weighted_mean(edits, key):
    total = sum(e["weight"] for e in edits)
    return sum(e[key] * e["weight"] for e in edits) / total if total else None


def score_candidates(candidates, min_front=1, max_error_cost=0.02):
    """Keep candidates with evidence, ranked best first.

    A candidate qualifies when it appears in at least `min_front` front circuits, or
    when the edits that produced it lowered error on average, or lowered gates on
    average without raising error by more than `max_error_cost`.
    """
    kept = []
    for cand in candidates.values():
        d_err = weighted_mean(cand["edits"], "delta_error")
        d_gates = weighted_mean(cand["edits"], "delta_gates")
        improved = d_err is not None and (d_err < 0 or (d_gates < 0 and d_err <= max_error_cost))
        if len(cand["front_genes"]) < min_front and not improved:
            continue
        cand = dict(cand, avg_delta_error=d_err, avg_delta_gates=d_gates,
                    id=motif_id(cand["block_index"], cand["code"]),
                    block=block_name(cand["code"]))
        kept.append(cand)
    kept.sort(key=lambda c: (-len(c["front_genes"]),
                             c["avg_delta_error"] if c["avg_delta_error"] is not None else 1.0,
                             c["avg_delta_gates"] if c["avg_delta_gates"] is not None else 0.0))
    return kept


# ---------------------------------------------------------------------------
# Credit from earlier prompts

def used_motif(child_block, motif_code, parent_block=None, threshold=0.9):
    """Whether a child's edited block took up the motif.

    Blocks are short, so two different edits of the same block can still look alike.
    The child must be close to the motif and closer to it than to the block it started
    from, so an unrelated small edit doesn't count as a use.
    """
    to_motif = similarity(child_block, motif_code)
    if to_motif < threshold:
        return False
    return parent_block is None or to_motif > similarity(child_block, parent_block)


def credit(motifs, gene_ids, results_dir, variant_dir, model, seed_path=None):
    """Add shown/used counts and the used children's average change to each motif."""
    by_id = {m["id"]: m for m in motifs}
    for m in motifs:
        m.update(shown=0, used=0, used_delta_error=None, used_delta_gates=None)
    used_edits = {m["id"]: [] for m in motifs}
    stats = racing.population_stats(results_dir, gene_ids)
    for g in gene_ids:
        record = _load_json(os.path.join(results_dir, f"{g}_mutation.json"))
        if not record or not record.get("motifs_shown"):
            continue
        parts = read_blocks(os.path.join(variant_dir, f"{model}_{g}.py"))
        index, parent = record.get("block_index"), record.get("parent")
        parent_path = seed_path if parent == "seed" else os.path.join(variant_dir, f"{model}_{parent}.py")
        parent_parts = read_blocks(parent_path) if parent_path else None
        for mid in record["motifs_shown"]:
            m = by_id.get(mid)
            if m is None:
                continue
            m["shown"] += 1
            if not parts or index != m["block_index"] or index >= len(parts):
                continue
            parent_block = parent_parts[index] if parent_parts and index < len(parent_parts) else None
            if not used_motif(parts[index], m["code"], parent_block):
                continue
            m["used"] += 1
            if g in stats and parent in stats:
                n_p, n_c = stats[parent][2], stats[g][2]
                used_edits[mid].append({"delta_error": stats[g][0] - stats[parent][0],
                                        "delta_gates": stats[g][3] - stats[parent][3],
                                        "weight": n_p * n_c / (n_p + n_c)})
    for m in motifs:
        m["used_delta_error"] = weighted_mean(used_edits[m["id"]], "delta_error")
        m["used_delta_gates"] = weighted_mean(used_edits[m["id"]], "delta_gates")
    return motifs


def prune(motifs, min_uses=3, max_size=12):
    """Drop motifs that made their users worse on both objectives; keep the best max_size."""
    kept = []
    for m in motifs:
        if (m["used"] >= min_uses and m["used_delta_error"] is not None
                and m["used_delta_error"] > 0 and (m["used_delta_gates"] or 0) >= 0):
            continue
        kept.append(m)
    return kept[:max_size]


# ---------------------------------------------------------------------------
# Validation and naming

def validate(motif, seed_blocks, n_qubits=None):
    """Splice the motif into the seed and build the circuit. Returns (ok, reason)."""
    parts = list(seed_blocks)
    parts[motif["block_index"]] = "\n" + motif["code"] + "\n"
    source = OPTION_MARKER.join(parts)
    namespace = {"__name__": "motif_check"}
    try:
        exec(compile(source, "<motif>", "exec"), namespace)
        qc, x_params, w_params = namespace["build_circuit"]()
        expected = n_qubits if n_qubits is not None else namespace["N_QUBITS"]
        if qc.num_qubits != expected:
            return False, f"builds {qc.num_qubits} qubits, expected {expected}"
        if not w_params:
            return False, "no trainable parameters"
        import numpy as np
        from qiskit.quantum_info import Statevector
        values = dict(zip(x_params, np.full(len(x_params), 0.3)))
        values.update(zip(w_params, np.full(len(w_params), 0.7)))
        Statevector(qc.assign_parameters(values, strict=False))
    except Exception as error:      # any failure means the motif can't be offered
        return False, f"{type(error).__name__}: {error}"
    return True, ""


def naming_prompt(motifs):
    lines = ["You are cataloguing circuit patterns found in a run that evolves Qiskit "
             "variational quantum classifiers. For each numbered code block below, give a "
             "short name (at most 6 words) and one sentence saying what the pattern does "
             "and why it might help. Reply with JSON only: a list of objects with keys "
             "\"id\", \"name\" and \"description\".", ""]
    for m in motifs:
        lines += [f"id: {m['id']}  (block {m['block']}, in {len(m['front_genes'])} Pareto circuits)",
                  "```python", m["code"], "```", ""]
    return "\n".join(lines)


def parse_names(text):
    match = re.search(r"\[\s*\{.*\}\s*\]", text or "", re.S)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return {}
    return {str(i.get("id")): i for i in items if isinstance(i, dict) and i.get("id")}


def name_motifs(motifs, generate=None):
    """Fill in name and description, via `generate(prompt) -> text` if given."""
    names = {}
    if generate and motifs:
        try:
            names = parse_names(generate(naming_prompt(motifs)))
        except Exception as error:
            print(f"\t☠ Motif naming failed ({error}); using plain labels", flush=True)
    for m in motifs:
        named = names.get(m["id"], {})
        m["name"] = str(named.get("name") or f"{m['block']} variant {m['id'][:4]}").strip()
        m["description"] = str(named.get("description") or "").strip()
    return motifs


# ---------------------------------------------------------------------------
# Build and use

def build_library(gene_ids, results_dir, variant_dir, model, seed_path, out_path,
                  generate=None, generation=None, max_rank=1, max_size=12):
    """Rebuild the library for the current run and write it to out_path atomically."""
    candidates, seed_blocks = gather_evidence(gene_ids, results_dir, variant_dir, model,
                                              seed_path, max_rank=max_rank)
    motifs = score_candidates(candidates)
    valid, rejected = [], 0
    for m in motifs:
        ok, reason = validate(m, seed_blocks)
        if ok:
            valid.append(m)
        else:
            rejected += 1
            print(f"\t‣ Motif {m['id']} rejected: {reason}", flush=True)
    motifs = prune(credit(valid, gene_ids, results_dir, variant_dir, model, seed_path), max_size=max_size)
    motifs = name_motifs(motifs, generate)
    library = {"generation": generation, "n_genes": len(gene_ids), "rejected": rejected,
               "motifs": motifs}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=1)
    os.replace(tmp, out_path)
    return library


def load_library(path):
    data = _load_json(path) if path else None
    return data.get("motifs", []) if isinstance(data, dict) else []


def select_for_block(motifs, block_index, current_block, k=3):
    """Up to k motifs for this block, skipping one that is the current code already."""
    current = normalize(current_block)
    return [m for m in motifs
            if m["block_index"] == block_index and normalize(m["code"]) != current][:k]


def prompt_section(motifs):
    """The text appended to a mutation prompt; empty when there is nothing to offer."""
    if not motifs:
        return ""
    lines = ["", "",
             "Patterns that have worked in this run for this block. They are optional: "
             "use one if it fits, adapt it, or ignore them. Keep following every constraint above."]
    for m in motifs:
        evidence = [f"in {len(m['front_genes'])} Pareto circuits"]
        if m.get("avg_delta_error") is not None:
            evidence.append(f"edits that wrote it changed error by {m['avg_delta_error']:+.3f}"
                            f" and gate count by {m['avg_delta_gates']:+.0f}")
        title = m["name"] + (f": {m['description']}" if m.get("description") else "")
        lines += ["", f"# {title}", f"# ({'; '.join(evidence)})", "```python", m["code"], "```"]
    return "\n".join(lines)
