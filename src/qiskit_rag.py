"""Small local BM25 retriever for Qiskit mutation guidance.

Retrieval is per mutation (2026-10-13). The Oct 8 RAG run built one query from the
prompt template before the code block was chosen, so every one of its 405 mutations
got one of two fixed sets of notes, and the import/API notes never made the top 3.
build_rag_section() is now called by the LLM job AFTER it has picked the block, with:
  * the block's own code and the mutation goal as the BM25 query,
  * the errors that recent children of the same block died with (failure log),
  * a pinned note with the imports and gate calls that exist in Qiskit 2.x,
  * the block's contract (names/signatures other blocks call), parsed from its code.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "qiskit_rag"
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
PINNED_NOTE = "qiskit_imports_api.md"
ERROR_RE = re.compile(r"^(\w*(?:Error|Exception)\b.*)$", re.M)


@dataclass(frozen=True)
class RetrievedDocument:
    source: str
    text: str
    score: float


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def retrieve(query: str, top_k: int = 3, exclude: tuple[str, ...] = ()) -> list[RetrievedDocument]:
    """Rank local Qiskit notes against a per-mutation query using BM25."""
    if top_k < 1:
        return []
    paths = sorted(p for p in DOCS_DIR.glob("*.md") if p.name not in exclude)
    documents = [(path.name, path.read_text(encoding="utf-8")) for path in paths]
    if not documents:
        return []

    query_terms = _tokens(query)
    tokenized = [_tokens(text) for _, text in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized)
        for term in set(query_terms)
    }
    query_counts = {term: query_terms.count(term) for term in set(query_terms)}
    ranked = []
    for (source, text), tokens, length in zip(documents, tokenized, lengths):
        score = 0.0
        for term, query_count in query_counts.items():
            frequency = tokens.count(term)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log1p(
                (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * length / max(average_length, 1.0)
            )
            # Repeated query terms (e.g. an error seen three times) weigh more, capped.
            score += min(query_count, 3) * inverse_frequency * frequency * 2.5 / denominator
        if score > 0:
            ranked.append(RetrievedDocument(source, text.strip(), score))

    return sorted(ranked, key=lambda item: (-item.score, item.source))[:top_k]


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Format retrieved notes with source labels for inclusion in an LLM prompt."""
    results = retrieve(query, top_k=top_k)
    return "\n\n".join(
        f"[{item.source}; BM25={item.score:.3f}]\n{item.text}"
        for item in results
    )


def block_contract(code: str) -> str:
    """Names and signatures defined in a block, which the rest of the file relies on.

    Renamed functions and changed signatures (NameError: name 'build_circuit' is not
    defined, TypeError: build_circuit() missing 1 required positional argument) were
    the second-largest failure group in the Oct 8 runs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    lines = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            lines.append(f"- def {node.name}({ast.unparse(node.args)})")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    lines.append(f"- module constant {target.id}")
    if not lines:
        return ""
    return ("Block contract - other code calls these, so keep every name and "
            "signature exactly as written (you may add helpers):\n" + "\n".join(lines))


def recent_failures(failure_log: str | Path | None, block: int | None, limit: int = 3) -> list[str]:
    """Distinct error lines from the most recent failed children of this block."""
    if not failure_log or not Path(failure_log).exists():
        return []
    errors = []
    for line in reversed(Path(failure_log).read_text(encoding="utf-8").splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if block is not None and record.get("block") not in (block, None):
            continue
        error = (record.get("error") or "").split(" (/")[0][:160]   # drop site-packages paths
        if error and error not in errors:
            errors.append(error)
        if len(errors) >= limit:
            break
    return errors


def build_rag_section(goal: str, code: str, failures: list[str] | None = None,
                      top_k: int = 2) -> tuple[str, list[str]]:
    """Prompt text appended to one mutation/crossover prompt, and the notes it used."""
    failures = failures or []
    query = " ".join([goal, code] + failures * 2)
    results = retrieve(query, top_k=top_k, exclude=(PINNED_NOTE,))
    pinned = (DOCS_DIR / PINNED_NOTE).read_text(encoding="utf-8").strip()
    parts = [f"[{PINNED_NOTE}; pinned]\n{pinned}"]
    parts += [f"[{item.source}; BM25={item.score:.3f}]\n{item.text}" for item in results]
    contract = block_contract(code)
    text = ("\n\nRetrieved Qiskit API guidance for THIS block (use as references, "
            "not as code to copy):\n" + "\n\n".join(parts))
    if contract:
        text += "\n\n" + contract
    if failures:
        text += ("\n\nRecent mutations of this block crashed with these errors; "
                 "do not repeat them:\n" + "\n".join(f"- {e}" for e in failures))
    return text, [PINNED_NOTE] + [item.source for item in results]


def last_error(log_text: str) -> str | None:
    errors = ERROR_RE.findall(log_text)
    return errors[-1][:300] if errors else None
