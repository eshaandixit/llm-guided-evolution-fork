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
one. Besides redefinitions it drops two kinds of statement that crash a child when
its file is imported: demo code at the top level (bare calls, loops, assignments
that call functions, like "qc = build_circuit()"), and imports that don't exist in
the installed packages (Llama often writes Qiskit 0.x imports such as
"from qiskit import execute"). A reply cut off by the LLM's length limit doesn't parse as a whole, so it is
split into top-level statements and the ones that parse on their own are kept;
the cut-off statement at the end is dropped.
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


def _top_level_segments(code):
    """Split code into top-level statements by indentation, for code that won't parse whole."""
    segments, current = [], []
    continuation = ("else", "elif", "except", "finally", ")", "]", "}")
    for line in code.splitlines():
        starts_new = (line and not line[0].isspace() and not line.lstrip().startswith("#")
                      and not line.startswith(continuation) and not line.startswith("@"))
        if starts_new and current and not current[-1].startswith("@"):
            segments.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        segments.append("\n".join(current))
    return segments


def _statements(code):
    """[(node, source)] for each top-level statement; unparseable statements are left out."""
    tree = _parse(code)
    if tree is not None:
        return [(n, ast.get_source_segment(code, n)) for n in tree.body], False
    kept = []
    for segment in _top_level_segments(code):
        seg_tree = _parse(segment)
        if seg_tree is not None:
            kept += [(n, ast.get_source_segment(segment, n)) for n in seg_tree.body]
    return kept, True


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


def _calls_something(node):
    return any(isinstance(sub, ast.Call) for sub in ast.walk(node))


def _is_demo_code(node, block_names):
    """Top-level statements that run code on import instead of defining something."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                         ast.Import, ast.ImportFrom)):
        return False
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return _calls_something(node) and not _defined_names(node) & block_names
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return False                                  # a docstring
    return not _is_main_guard(node)                   # bare calls, loops, with, try, if


def _import_fails(node):
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return False
    try:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<import>", "exec"), {})
    except Exception:
        return True
    return False


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
    block_tree = _parse(block)
    statements, salvaged = _statements(reply_code)
    if block_tree is None or not statements:
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
    for node, src in statements:
        names = _defined_names(node)
        if names & elsewhere or (_is_main_guard(node) and not block_has_main):
            dropped.append(", ".join(sorted(names & elsewhere)) or "__main__ guard")
        elif _is_demo_code(node, block_names):
            dropped.append(f"top-level code ({src.splitlines()[0].strip()[:40]})")
        elif _import_fails(node):
            dropped.append(f"missing import ({src.strip()[:50]})")
        else:
            kept.append((node, src))

    reply_names = set()
    for node, _ in kept:
        reply_names |= _defined_names(node)
    if not reply_names & block_names:
        return reply, []          # the reply never wrote this block; let the usual checks reject it
    restored = []
    for node in block_tree.body:
        names = _defined_names(node)
        if (isinstance(node, (ast.Assign, ast.AnnAssign)) and names
                and not names & reply_names and not names & elsewhere):
            restored.append(node)

    if not dropped and not restored and not salvaged:
        return reply, []

    block_src = textwrap.dedent(block)
    pieces = [ast.get_source_segment(block_src, n) for n in restored]
    pieces += [src for _, src in kept]
    changes = (["salvaged a cut-off reply"] if salvaged else []) + [f"dropped {d}" for d in dropped] + [
        f"restored {', '.join(sorted(_defined_names(n)))}" for n in restored]
    return "\n\n".join(p for p in pieces if p), changes
