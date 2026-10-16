"""Trim an LLM reply down to the block it was asked to edit.

A mutation prompt hands the LLM one evolvable block (the code between two
"# --OPTION--" markers). Llama often answers with a whole program instead: it
rewrites load_data(), redefines N_QUBITS, adds a main(). The mutation job then
rejected the reply outright (first Iris run, 2026-10-15: 31 of 48 mutations
rejected this way, 3 accepted), so almost no children were ever made.

fit_to_block() keeps the reply's own version of the block and drops everything
that redefines code living elsewhere in the file. It also puts back module-level
settings the block defines (N_LAYERS = 2, MAX_ITER = 178) when the reply dropped
them, since other code still reads them.

It only removes or restores whole top-level statements and never edits inside
one, so anything it can't parse is returned unchanged for the usual checks.
"""

from __future__ import annotations

import ast
import textwrap

OPTION_MARKER = "# --OPTION--"


def _parse(code):
    try:
        return ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None


def _defined_names(node):
    """Names a top-level statement binds."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    names = set()
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
    return names


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def _top_level_names(code):
    tree = _parse(code)
    names = set()
    if tree:
        for node in tree.body:
            names |= _defined_names(node)
    return names


def fit_to_block(reply, block, other_blocks):
    """Return (code, changes): the reply cut down to `block`, and what was changed.

    reply: the code the LLM returned
    block: the original block it was asked to edit
    other_blocks: every other part of the file (protected header and other blocks)
    """
    reply_code = textwrap.dedent(reply).strip()
    tree, block_tree = _parse(reply_code), _parse(block)
    if tree is None or block_tree is None:
        return reply, []

    block_names = set()
    for node in block_tree.body:
        block_names |= _defined_names(node)
    block_has_main = any(_is_main_guard(n) for n in block_tree.body)
    elsewhere = set()
    for part in other_blocks:
        elsewhere |= _top_level_names(part)
    elsewhere -= block_names

    kept, dropped = [], []
    for node in tree.body:
        names = _defined_names(node)
        if names & elsewhere or (_is_main_guard(node) and not block_has_main):
            dropped.append(", ".join(sorted(names & elsewhere)) or "__main__ guard")
        else:
            kept.append(node)

    reply_names = set()
    for node in kept:
        reply_names |= _defined_names(node)
    restored = []
    for node in block_tree.body:
        names = _defined_names(node)
        if (isinstance(node, (ast.Assign, ast.AnnAssign)) and names
                and not names & reply_names and not names & elsewhere):
            restored.append(node)

    if not dropped and not restored:
        return reply, []
    if not kept:
        return reply, []          # nothing of the block survives; let the usual checks reject it

    block_src = textwrap.dedent(block)
    pieces = [ast.get_source_segment(block_src, n) for n in restored]
    pieces += [ast.get_source_segment(reply_code, n) for n in kept]
    changes = [f"dropped {d}" for d in dropped] + [
        f"restored {', '.join(sorted(_defined_names(n)))}" for n in restored]
    return "\n\n".join(p for p in pieces if p), changes
