"""Checks for the two failure modes an adversarial review found in pareto_report.

Run:  .venv/bin/python tools/test_pareto_report.py
"""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORT = os.path.join(HERE, "pareto_report.py")
SEED_PNG = os.path.join(ROOT, "sota/QuantumVQC/results")


def _find_a_circuit_png():
    for f in sorted(os.listdir(SEED_PNG)):
        if f.endswith("_circuit.png"):
            return os.path.join(SEED_PNG, f)
    return None


def build(tmp, genes, logged, resumed, with_png=False):
    """genes -> results files; logged -> gene ids the fake log mentions."""
    res = os.path.join(tmp, "results"); os.makedirs(res, exist_ok=True)
    png = _find_a_circuit_png()
    for i, g in enumerate(genes):
        open(os.path.join(res, f"{g}_results.txt"), "w").write(f"0.{400 + i}, {60 - i}.0")
        if with_png and png:
            shutil.copy(png, os.path.join(res, f"{g}_circuit.png"))
    log = os.path.join(tmp, "run.out")
    with open(log, "w") as fh:
        if resumed:
            fh.write("Loaded checkpoint from first_test/checkpoint_gen_6.pkl\n")
        fh.write("STARTING GENERATION: 0\n")
        for g in logged:
            fh.write(f"Gene: {g}\n")
    return res, log


def run(res, log, out, *extra):
    p = subprocess.run([sys.executable, REPORT, "--results", res, "--log", log,
                        "--out", out, *extra], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


G = [f"xXx{c}{'a' * 23}" for c in "ABCDEFGHIJKLMNOPQRSTUVWX"]

with tempfile.TemporaryDirectory() as tmp:
    # 1. fresh run: genes absent from the log are genuinely stale -> dropped
    res, log = build(tmp, G[:6], G[:4], resumed=False)
    o = run(res, log, os.path.join(tmp, "o1"))
    assert "ignored stale results   : 2 (from earlier runs)" in o, o
    assert "evaluated individuals : 4" in o, o

    # 2. RESUMED run: the same shape must NOT silently drop - those genes may be
    #    earlier generations of this search, which was the reviewed defect.
    res, log = build(tmp, G[:6], G[:4], resumed=True)
    o = run(res, log, os.path.join(tmp, "o2"))
    assert "RESUMED RUN" in o, o
    assert "evaluated individuals : 6" in o, o
    assert "ignored stale" not in o, o

    # 3. --strict restores dropping on a resumed run
    o = run(res, log, os.path.join(tmp, "o3"), "--strict")
    assert "forced by --strict" in o, o
    assert "evaluated individuals : 4" in o, o

with tempfile.TemporaryDirectory() as tmp:
    # 4. circuit grid is capped, and the cap is announced rather than silent
    res, log = build(tmp, G, G, resumed=False, with_png=True)
    o = run(res, log, os.path.join(tmp, "o4"), "--max-circuits", "5")
    if "_circuit.png" in "".join(os.listdir(res)):
        assert "circuit grid capped" in o, o
        assert "5 shown" in o, o
        grid = os.path.join(tmp, "o4", "pareto_circuits.png")
        if os.path.exists(grid):
            from PIL import Image
            h = Image.open(grid).size[1]
            assert h < 6000, f"capped grid still {h}px tall"
    else:
        print("  (skipped cap render check - no circuit PNG available to copy)")

print("all pareto_report checks passed")
