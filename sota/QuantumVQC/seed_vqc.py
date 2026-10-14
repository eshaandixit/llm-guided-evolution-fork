# --PROMPT LOG--

"""
Base seed algorithm: a variational quantum classifier in Qiskit.

This is individual zero. Evolution mutates the blocks below the first
OPTION marker (see the separators further down); everything above it is fixed
scaffolding and is never handed to the LLM.

Contract with the LLM-GE harness:
  python seed_vqc.py --gene-id <id> --out-dir <dir>
writes "<obj1>, <obj2>" to <out_dir>/<gene_id>_results.txt, matching
run_improved.py check4results(). Both objectives are MINIMIZED:
    obj1 = 1 - validation accuracy, averaged over N_STARTS trainings
    obj2 = plain gate count                  (circuit complexity, EXAQC's "# Gates")
A crash, an invalid circuit, or a missing file leaves the harness to assign
INVALID_FITNESS_MAX, which is the intended failure path - do not catch broadly
and report a fake score.

Fitness is computed on VALIDATION. Every circuit is also scored on test right after
training (like EXAQC's Table 1 and ExquisiteNetV2), but test only goes into
<gene>_metrics.json; selection never reads it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter, ParameterExpression
from qiskit.quantum_info import Statevector, Operator
# Pre-imported so mutated blocks can reach for them; the import region is not evolvable.
import math
import random
import itertools

SEED = 42
N_QUBITS = 8
N_CLASSES = 2
READOUT = [0]             # one readout qubit represents the two classes
TWO_QUBIT_COST = 5.0      # two-qubit error rates are ~an order of magnitude worse
MAX_GATE_COST = 500.0     # cost ceiling; runaway circuits are not interesting


def load_data():
    """Fixed 60/20/20 split, shared by every individual. Scalers fit on TRAIN ONLY.

    Breast Cancer Wisconsin (569 samples, 30 features) -> 341 train / 114 val / 114 test.
    Changed from 70/30-then-75/25 (298/100/171) on 2026-10-08 to match the quantum-seed
    branch's 60/20/20 protocol; runs before that date are not comparable with runs after.
    Features: standardize -> PCA to N_QUBITS components -> MinMax to [0, pi] (angle encoding).
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=SEED,
        stratify=y_tr_full)

    standardizer = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (standardizer.transform(values)
                        for values in (X_tr, X_va, X_te))
    reducer = PCA(n_components=N_QUBITS, random_state=SEED).fit(X_tr)
    X_tr, X_va, X_te = (reducer.transform(values)
                        for values in (X_tr, X_va, X_te))
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X_tr)
    return (scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te),
            y_tr, y_va, y_te)


def gate_count(qc):
    """Objective 2. Plain gate count, the "# Gates" column EXAQC reports."""
    return float(sum(1 for inst in qc.data
                     if inst.operation.name not in ("barrier", "measure")))


def weighted_gate_cost(qc):
    """Hardware-weighted count. Diagnostic only; obj2 uses gate_count()."""
    cost = 0.0
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        cost += TWO_QUBIT_COST if inst.operation.num_qubits >= 2 else 1.0
    return cost


# ---------------------------------------------------------------------------
# Batched statevector simulator (2026-10-08).
#
# The original forward() bound one sample at a time and built a fresh Statevector:
# ~2 ms per sample, 177 s per individual for ONE training start. This compiles the
# circuit once into a list of (qubits, matrix-or-angle-function) steps and pushes the
# whole batch of samples through numpy at once, which is ~100x faster and makes
# multi-start averaging affordable. It is checked against Statevector on every
# compile (one random sample); any mismatch or any operation it does not understand
# (measure, reset, initialize, classically-controlled ops, ...) silently falls back
# to the exact per-sample Statevector path, so it can never change a score.

def _rotation_matrices(name, angles):
    """Batched matrices for common parameterized gates; angles: list of (N,) arrays."""
    if name == "rx":
        c, s = np.cos(angles[0] / 2), np.sin(angles[0] / 2)
        z = np.zeros_like(c)
        return np.stack([np.stack([c + 0j, -1j * s], -1), np.stack([-1j * s, c + 0j], -1)], -2)
    if name == "ry":
        c, s = np.cos(angles[0] / 2), np.sin(angles[0] / 2)
        return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)
    if name == "rz":
        e = np.exp(-0.5j * angles[0])
        z = np.zeros_like(e)
        return np.stack([np.stack([e, z], -1), np.stack([z, np.conj(e)], -1)], -2)
    if name in ("p", "u1"):
        e = np.exp(1j * angles[0])
        o, z = np.ones_like(e), np.zeros_like(e)
        return np.stack([np.stack([o, z], -1), np.stack([z, e], -1)], -2)
    if name in ("u", "u3"):
        t, p, l = angles
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.stack([np.stack([c + 0j, -np.exp(1j * l) * s], -1),
                         np.stack([np.exp(1j * p) * s, np.exp(1j * (p + l)) * c], -1)], -2)
    return None


def _angle_function(expr):
    """Return f(x_row_dict, w_dict) -> value, vectorized over the sample axis."""
    if not isinstance(expr, ParameterExpression):
        value = float(expr)
        return lambda env: value
    params = list(expr.parameters)
    if not params:
        value = float(expr)
        return lambda env: value
    if len(params) == 1 and isinstance(expr, Parameter):
        p = params[0]
        return lambda env: env[p]
    import sympy
    sym = expr.sympify()
    symbols = [sympy.Symbol(p.name) for p in params]
    fn = sympy.lambdify(symbols, sym, modules="numpy")
    return lambda env: np.real_if_close(fn(*(env[p] for p in params)))


class _CompiledCircuit:
    """A circuit compiled for batched simulation over many input samples."""

    def __init__(self, qc):
        self.n = qc.num_qubits
        self.steps = []
        for inst in qc.data:
            op = inst.operation
            if op.name == "barrier":
                continue
            if op.name in ("measure", "reset", "initialize", "delay") or \
                    getattr(op, "condition", None) is not None:
                raise NotImplementedError(op.name)
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            if op.params and any(isinstance(p, ParameterExpression) and p.parameters
                                 for p in op.params):
                fns = [_angle_function(p) for p in op.params]
                self.steps.append((qubits, "param", op, fns))
            else:
                self.steps.append((qubits, "fixed", Operator(op).data, None))

    def _apply(self, state, qubits, matrix):
        n, k, batch = self.n, len(qubits), state.shape[0]
        axes = [1 + (n - 1 - q) for q in reversed(qubits)]       # q_{k-1} ... q_0
        rest = [a for a in range(1, n + 1) if a not in axes]
        perm = [0] + rest + axes
        moved = state.transpose(perm).reshape(batch, -1, 2 ** k)
        if matrix.ndim == 2:
            moved = moved @ matrix.T
        else:
            moved = np.matmul(moved, matrix.transpose(0, 2, 1))
        moved = moved.reshape([batch] + [2] * n)
        return moved.transpose(np.argsort(perm))

    def statevectors(self, env, batch):
        state = np.zeros((batch, 2 ** self.n), dtype=complex)
        state[:, 0] = 1.0
        state = state.reshape([batch] + [2] * self.n)
        for qubits, kind, payload, fns in self.steps:
            if kind == "fixed":
                matrix = payload
            else:
                angles = [np.broadcast_to(np.asarray(f(env), dtype=float), (batch,))
                          for f in fns]
                matrix = _rotation_matrices(payload.name, angles)
                if matrix is None:                       # uncommon gate: build per sample
                    gate = payload.copy()
                    mats = []
                    for b in range(batch):
                        gate.params = [float(a[b]) for a in angles]
                        mats.append(Operator(gate).data)
                    matrix = np.stack(mats)
            state = self._apply(state, qubits, matrix)
        return state

    def marginal(self, state, readout):
        probs = np.abs(state) ** 2
        n = self.n
        keep = [1 + (n - 1 - q) for q in reversed(readout)]
        drop = tuple(a for a in range(1, n + 1) if a not in keep)
        out = probs.sum(axis=drop) if drop else probs
        # remaining axes are in ascending axis order; reorder to [r_{m-1} ... r_0]
        remaining = sorted(keep)
        out = out.transpose([0] + [1 + remaining.index(a) for a in keep])
        return out.reshape(state.shape[0], -1)


_COMPILED = {}


def _compiled(qc, x_params, w_params):
    key = id(qc)
    entry = _COMPILED.get(key)
    if entry is not None and entry[0] is qc and entry[1] == len(qc.data):
        return entry[2]
    sim = None
    try:
        sim = _CompiledCircuit(qc)
        rng = np.random.default_rng(0)
        xv = rng.uniform(0, np.pi, len(x_params))
        wv = rng.uniform(0, 2 * np.pi, len(w_params))
        env = {**{p: np.array([v]) for p, v in zip(x_params, xv)},
               **{p: np.array([v]) for p, v in zip(w_params, wv)}}
        fast = sim.marginal(sim.statevectors(env, 1), READOUT)[0]
        bound = qc.assign_parameters({**dict(zip(x_params, xv)), **dict(zip(w_params, wv))},
                                      strict=False)
        exact = Statevector(bound).probabilities(READOUT)
        if not np.allclose(fast, exact, atol=1e-8):
            sim = None
    except Exception:
        sim = None
    _COMPILED[key] = (qc, len(qc.data), sim)
    return sim


def readout_batch(qc, x_params, w_params, X, w_vals):
    """Marginal READOUT probabilities for every row of X: shape (len(X), 2**len(READOUT))."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    sim = _compiled(qc, x_params, w_params)
    if sim is not None:
        env = {p: X[:, i] for i, p in enumerate(x_params)}
        env.update({p: float(v) for p, v in zip(w_params, w_vals)})
        return sim.marginal(sim.statevectors(env, len(X)), READOUT)
    rows = []
    for xi in X:
        bound = qc.assign_parameters(
            {**dict(zip(x_params, xi)), **dict(zip(w_params, w_vals))}, strict=False)
        rows.append(Statevector(bound).probabilities(READOUT))
    return np.asarray(rows)


# ---------------------------------------------------------------------------
# Training budget. Every individual gets the same amount of compute, and the
# count is kept HERE rather than in the training block, which stays evolvable.
# A variant may rewrite the optimizer, the initial angles, the batching, any of
# it - but every training pass runs the circuit through forward()/forward_batch(),
# so the budget is enforced from a place the LLM cannot edit. Counting simulations
# rather than optimizer steps keeps mini-batch training honest: fewer samples per
# step buys more steps, not more compute.
TRAIN_BUDGET_EVALS = 178        # full passes over the training set (see reports/qiskit_epoch_falloff.md)
_PASSES = 0                     # sample simulations made while training
_PASS_LIMIT = None              # set in main() once the training set size is known
_COUNTING = False               # only training counts, scoring afterwards does not
_BEST = [float("inf"), None]    # best (training loss, weights) seen while training
N_STARTS = int(os.getenv("QISKIT_N_STARTS", "5"))  # trainings per individual; obj1 averages them
EARLY_STOP_PATIENCE = int(os.getenv("QISKIT_EARLY_STOP_PATIENCE", "18"))
EARLY_STOP_MIN_DELTA = 1e-4
_TRAINING_CURVE = []


class BudgetSpent(Exception):
    """Raised inside forward() when an individual has used its training budget."""


class ValidationPlateau(Exception):
    """Raised when validation loss stops improving for the configured patience."""


def _charge(n_samples):
    global _PASSES
    if _COUNTING:
        _PASSES += n_samples
        if _PASS_LIMIT is not None and _PASSES > _PASS_LIMIT:
            raise BudgetSpent(f"training budget of {TRAIN_BUDGET_EVALS} passes is spent")


def forward_batch(qc, x_params, w_params, X, w_vals):
    """Class probabilities for every row of X, shape (len(X), N_CLASSES)."""
    _charge(len(X))
    marg = readout_batch(qc, x_params, w_params, X, w_vals)
    return np.stack([readout_probabilities(row) for row in marg])


def forward(qc, x_params, w_params, x_vals, w_vals):
    """Bind one sample plus the weights and return class probabilities."""
    return forward_batch(qc, x_params, w_params, [x_vals], w_vals)[0]


def cross_entropy(qc, xp, wp, w_vals, X, y):
    eps = 1e-10
    p = forward_batch(qc, xp, wp, X, w_vals)
    loss = float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y)] + eps)))
    # Remember the best angles seen while training, so an individual that runs out
    # of budget mid-search still gets scored on its best work rather than dying.
    if _COUNTING and loss < _BEST[0]:
        _BEST[0], _BEST[1] = loss, np.array(w_vals, dtype=float)
    return loss


def accuracy(qc, xp, wp, w_vals, X, y):
    p = forward_batch(qc, xp, wp, X, w_vals)
    return float(np.mean(np.argmax(p, axis=1) == np.asarray(y)))


def write_results(out_dir, gene_id, obj1, obj2, extra=None):
    """run_improved.py reads the first two values; anything after is diagnostic."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{gene_id}_results.txt")
    with open(path, "w") as f:
        f.write(f"{obj1}, {obj2}" + ("" if extra is None else f", {extra}"))
    return path


def count_dead_gates(qc, x_params, w_params, w_vals, tol=1e-2):
    """EXAQC's champion circuits carried dead gates - R(0) no-ops. Report trainable
    rotations whose trained angle is ~0 mod 2pi alongside accuracy. Input-encoding
    rotations are skipped (their angle changes per sample). RZ directly after |0>
    is also inert but not detected here - see the notebook for that caveat."""
    bound = qc.assign_parameters(dict(zip(w_params, w_vals)), strict=False)
    dead = rotations = 0
    for inst in bound.data:
        if not inst.operation.params:
            continue
        try:
            angle = float(inst.operation.params[0])
        except (TypeError, ValueError):
            continue
        rotations += 1
        r = abs(angle) % (2 * np.pi)
        if min(r, 2 * np.pi - r) < tol:
            dead += 1
    return dead, rotations


def emit_representations(qc, out_dir, gene_id, arms=("qasm", "ascii", "image")):
    """Three views of the SAME circuit object, rendered UNBOUND so parameter names
    stay visible. Never fatal: a drawing failure must not cost a valid individual
    its fitness (2026-10-08: a missing pylatexenc would have failed every gene)."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for arm in arms:
        try:
            if arm == "qasm":
                from qiskit.qasm3 import dumps
                path = os.path.join(out_dir, f"{gene_id}_circuit.qasm")
                with open(path, "w") as f:
                    f.write(dumps(qc))
            elif arm == "ascii":
                path = os.path.join(out_dir, f"{gene_id}_circuit.txt")
                with open(path, "w") as f:
                    f.write(str(qc.draw("text", fold=-1)))
            elif arm == "image":
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                path = os.path.join(out_dir, f"{gene_id}_circuit.png")
                fig = qc.draw("mpl", fold=-1)
                fig.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig)
            else:
                continue
            written[arm] = path
        except Exception as error:      # diagnostic output only
            print(f"  repr [{arm}] skipped: {type(error).__name__}: {error}")
    return written


def main(n_starts=N_STARTS):
    # n_starts is bound here, when the protected header runs, so a block that
    # rebinds N_STARTS cannot change how many times an individual is trained.
    global TRAIN_BUDGET_EVALS, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-id", default="seed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--repr", default="qasm,ascii,image",
                    help="circuit views to emit; empty string to skip")
    ap.add_argument("--optimizer", choices=("cobyla", "spsa", "adam"),
                    default=os.getenv("QISKIT_WEIGHT_OPTIMIZER", "cobyla"))
    ap.add_argument("--initial-weights", default=None,
                    help="Optional JSON metrics file or list of inherited angles "
                         "(Lamarckian warm start; used for the first start only)")
    ap.add_argument("--train-budget", type=int, default=TRAIN_BUDGET_EVALS,
                    help="Maximum training-set passes per start")
    ap.add_argument("--n-starts", type=int, default=n_starts)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.train_budget < 1 or args.n_starts < 1:
        ap.error("--train-budget and --n-starts must be positive")
    TRAIN_BUDGET_EVALS = args.train_budget
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_data()

    qc, xp, wp = build_circuit()
    if len(wp) == 0:
        raise ValueError("circuit has no trainable parameters")
    if qc.num_qubits != N_QUBITS:
        raise ValueError(f"circuit has {qc.num_qubits} qubits, expected {N_QUBITS}")

    initial_weights = None
    if args.initial_weights and os.path.exists(args.initial_weights):
        with open(args.initial_weights, encoding="utf-8") as file:
            initial_weights = json.load(file)
        if isinstance(initial_weights, dict):
            initial_weights = initial_weights["trained_weights"]
        if len(initial_weights) != len(wp):
            print(f"  inherited weights ignored: {len(initial_weights)} != {len(wp)} params")
            initial_weights = None

    # Multi-start (from the quantum-seed branch, 2026-10-03): one training run is a
    # lottery - the seed alone spans >15 points of validation accuracy across starting
    # angles - so a single-start "win" means nothing. Every circuit is trained
    # n_starts times from different starting angles (SEED, SEED+1, ...), each with the
    # full budget, and obj1 is the AVERAGE validation error over the starts.
    global _COUNTING, _PASS_LIMIT, _PASSES, _BEST
    _PASS_LIMIT = TRAIN_BUDGET_EVALS * len(X_tr)
    base_seed, starts, curves = SEED, [], []
    for k in range(args.n_starts):
        SEED = base_seed + k               # train_angles reads SEED for its starting angles
        _PASSES = 0
        _BEST = [float("inf"), None]
        _TRAINING_CURVE.clear()
        _COUNTING = True
        try:
            w = train_angles(qc, xp, wp, X_tr, y_tr, X_va, y_va,
                             initial_weights=initial_weights if k == 0 else None,
                             optimizer=args.optimizer)
        except BudgetSpent:
            if _BEST[1] is None:
                raise                      # trained without using cross_entropy: no angles to keep
            w = _BEST[1]
        finally:
            _COUNTING = False
        w = np.asarray(w, dtype=float)
        starts.append({
            "seed": SEED, "train_passes": _PASSES,
            "val_accuracy": accuracy(qc, xp, wp, w, X_va, y_va),
            "val_cross_entropy": cross_entropy(qc, xp, wp, w, X_va, y_va),
            "test_accuracy": accuracy(qc, xp, wp, w, X_te, y_te),
            "test_cross_entropy": cross_entropy(qc, xp, wp, w, X_te, y_te),
            "train_accuracy": accuracy(qc, xp, wp, w, X_tr, y_tr),
            "train_cross_entropy": cross_entropy(qc, xp, wp, w, X_tr, y_tr),
            "weights": w.tolist(),
        })
        curves.append(list(_TRAINING_CURVE))
    SEED = base_seed
    keys = [key for key in starts[0] if key not in ("seed", "weights")]
    avg = {key: float(np.mean([s[key] for s in starts])) for key in keys}
    best_start = max(range(len(starts)), key=lambda i: starts[i]["val_accuracy"])
    w_best = np.asarray(starts[best_start]["weights"])

    val_acc, val_ce = avg["val_accuracy"], avg["val_cross_entropy"]
    obj1 = 1.0 - val_acc
    obj2 = min(gate_count(qc), MAX_GATE_COST)
    if not np.isfinite(val_ce):
        raise ValueError(f"non-finite validation loss: {val_ce}")

    path = write_results(args.out_dir, args.gene_id, obj1, obj2, extra=val_ce)
    print(f"gene {args.gene_id}")
    print(f"  qubits {qc.num_qubits}  depth {qc.depth()}  params {len(wp)}  "
          f"fast-sim {'on' if _compiled(qc, xp, wp) is not None else 'OFF (Statevector fallback)'}")
    print(f"  obj1 val error {obj1:.4f} ({val_acc:.1%} mean acc over {args.n_starts} starts)"
          f"   obj2 gates {obj2:.0f}   [val CE {val_ce:.4f}, weighted cost {weighted_gate_cost(qc):.0f}]")
    print(f"  mean train acc {avg['train_accuracy']:.1%}   val acc {val_acc:.1%}"
          f"   test acc {avg['test_accuracy']:.1%}")
    print("  val acc per start: " + " ".join(f"{s['val_accuracy']:.1%}" for s in starts))

    dead, rotations = count_dead_gates(qc, xp, wp, w_best)
    two_qubit = sum(1 for inst in qc.data
                    if inst.operation.num_qubits >= 2
                    and inst.operation.name not in ("barrier", "measure"))
    ops = {}
    for inst in qc.data:
        ops[inst.operation.name] = ops.get(inst.operation.name, 0) + 1
    # Every metric we might want to plot, so a Pareto front can be redrawn on any
    # pair afterwards without re-running anything. The harness never reads this
    # file; obj1 and obj2 in the results file are what selection uses.
    metrics = {
        "gene_id": args.gene_id,
        "obj1_val_error": obj1, "obj2_gates": obj2,
        "val_accuracy": val_acc, "val_cross_entropy": val_ce,
        "val_accuracy_sd": float(np.std([s["val_accuracy"] for s in starts])),
        "test_accuracy": avg["test_accuracy"], "test_cross_entropy": avg["test_cross_entropy"],
        "best_start_val_accuracy": starts[best_start]["val_accuracy"],
        "best_start_test_accuracy": starts[best_start]["test_accuracy"],
        "train_accuracy": avg["train_accuracy"], "train_cross_entropy": avg["train_cross_entropy"],
        "gates": gate_count(qc), "weighted_gate_cost": weighted_gate_cost(qc),
        "two_qubit_gates": two_qubit, "depth": qc.depth(), "gate_histogram": ops,
        "n_qubits": qc.num_qubits, "n_params": len(wp),
        "n_starts": args.n_starts, "starts": starts,
        "train_passes": max(s["train_passes"] for s in starts),
        "train_evals": max(s["train_passes"] for s in starts) / max(len(X_tr), 1),
        "budget_evals": TRAIN_BUDGET_EVALS,
        "budget_spent": any(s["train_passes"] >= _PASS_LIMIT for s in starts),
        "optimizer": args.optimizer,
        "inherited_weights": initial_weights is not None,
        "trained_weights": w_best.tolist(),
        "training_curves": curves,
        "dead_rotations": dead, "rotations": rotations,
        "fast_sim": _compiled(qc, xp, wp) is not None,
        "seconds": time.perf_counter() - t0,
    }
    mpath = os.path.join(args.out_dir, f"{args.gene_id}_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=1, sort_keys=True)
    print(f"  metrics -> {mpath}   (used at most {metrics['train_evals']:.0f} of "
          f"{TRAIN_BUDGET_EVALS} training evals per start)")

    arms = tuple(a for a in args.repr.split(",") if a)
    if arms:
        for arm, rpath in emit_representations(qc, args.out_dir, args.gene_id, arms).items():
            print(f"  repr [{arm}] -> {rpath}")

    print(f"  wrote {path}   ({time.perf_counter() - t0:.1f}s)")


# --OPTION--
# -- NOTE --
# Feature encoding: how the 8 PCA components of each sample become rotation angles.
# Data is already scaled to [0, pi]. A single rotation per feature is the
# baseline; data re-uploading (repeating this map between variational layers)
# is a known way to raise expressivity and is deliberately NOT used here.
# -- NOTE --
def build_feature_map(qc, x_params):
    for i in range(N_QUBITS):
        qc.ry(x_params[i], i)


# --OPTION--
# Variational layer: the trainable rotations applied to every qubit.
def build_variational_layer(qc, w_params, offset):
    for i in range(N_QUBITS):
        qc.ry(w_params[offset + 2 * i], i)
        qc.rz(w_params[offset + 2 * i + 1], i)
    return offset + 2 * N_QUBITS


# --OPTION--
# Entanglement topology. A CNOT ring is the baseline; linear, all-to-all and
# hardware-native couplings are all reasonable alternatives.
def build_entanglement(qc):
    for i in range(N_QUBITS):
        qc.cx(i, (i + 1) % N_QUBITS)


# --OPTION--
# Circuit assembly. Depth lives here.
N_LAYERS = 2

def build_circuit():
    n_weights = N_LAYERS * 2 * N_QUBITS
    x_params = ParameterVector("x", N_QUBITS)
    w_params = ParameterVector("w", n_weights)

    qc = QuantumCircuit(N_QUBITS)
    build_feature_map(qc, x_params)
    offset = 0
    for _ in range(N_LAYERS):
        offset = build_variational_layer(qc, w_params, offset)
        build_entanglement(qc)
    return qc, list(x_params), list(w_params)


# --OPTION--
# Readout: marginal over READOUT qubits -> class probabilities.
# EXAQC's scheme throws away the out-of-range mass and renormalizes the rest.
def readout_probabilities(probs):
    p = np.asarray(probs[:N_CLASSES], dtype=float)
    total = p.sum()
    if total <= 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    return p / total


# --OPTION--
# Training the angles. COBYLA is gradient-free and cheap. "spsa" is a warm-startable
# local search; "adam" uses parameter-shift-style central differences (EXAQC trains
# with Adam). The protected sample-evaluation budget is authoritative.
MAX_ITER = 178

def local_weight_search(objective, initial_weights, max_steps, callback=None,
                        learning_rate=6.0, perturbation=0.2):
    """Warm-started SPSA: two directional probes per gradient estimate.
    Gains tuned 2026-10-08 on the seed (5 starts, 178 passes): lr 0.12 left the loss
    flat (0.701 -> 0.697, val acc 46%); lr 6.0 / c 0.2 reaches 85.3%, on par with COBYLA."""
    rng = np.random.default_rng(SEED)
    weights = np.asarray(initial_weights, dtype=float).copy()
    for step in range(max_steps):
        direction = rng.choice((-1.0, 1.0), size=weights.size)
        radius = perturbation / (step + 1) ** 0.101
        plus = objective(weights + radius * direction)
        minus = objective(weights - radius * direction)
        gradient = ((plus - minus) / (2.0 * radius)) * direction
        rate = learning_rate / (step + 1) ** 0.602
        weights = np.mod(weights - rate * gradient, 2.0 * np.pi)
        if callback is not None:
            callback(weights)
    return weights


def adam_shift(objective, initial_weights, max_steps, callback=None,
               learning_rate=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam on parameter-shift (pi/2 central difference) gradients. For gates of the
    form R(w) = exp(-i w P / 2) this is the exact analytic gradient."""
    weights = np.asarray(initial_weights, dtype=float).copy()
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    shift = np.pi / 2
    for step in range(1, max_steps + 1):
        grad = np.zeros_like(weights)
        for j in range(weights.size):
            e = np.zeros_like(weights)
            e[j] = shift
            grad[j] = 0.5 * (objective(weights + e) - objective(weights - e))
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        weights = weights - learning_rate * (m / (1 - beta1 ** step)) / (
            np.sqrt(v / (1 - beta2 ** step)) + eps)
        if callback is not None:
            callback(weights)
    return weights


def train_angles(qc, x_params, w_params, X, y, X_val=None, y_val=None,
                 initial_weights=None, optimizer="cobyla"):
    rng = np.random.default_rng(SEED)
    weights = (rng.uniform(0, 2 * np.pi, len(w_params))
               if initial_weights is None else np.asarray(initial_weights, dtype=float))
    if weights.shape != (len(w_params),):
        raise ValueError(f"expected {len(w_params)} initial angles, got {weights.shape}")

    def objective(values):
        return cross_entropy(qc, x_params, w_params, values, X, y)

    best_validation = [float("inf"), None]
    stale_steps = [0]

    def monitor(values):
        global _COUNTING
        if X_val is None or y_val is None:
            return
        was_counting = _COUNTING
        _COUNTING = False
        try:
            validation_loss = cross_entropy(qc, x_params, w_params, values, X_val, y_val)
            validation_acc = accuracy(qc, x_params, w_params, values, X_val, y_val)
        finally:
            _COUNTING = was_counting
        _TRAINING_CURVE.append({
            "step": len(_TRAINING_CURVE) + 1,
            "train_evals": _PASSES / max(len(X), 1),
            "best_train_loss": _BEST[0],
            "validation_cross_entropy": float(validation_loss),
            "validation_accuracy": float(validation_acc),
        })
        if validation_loss < best_validation[0] - EARLY_STOP_MIN_DELTA:
            best_validation[:] = [float(validation_loss), np.asarray(values).copy()]
            stale_steps[0] = 0
        else:
            stale_steps[0] += 1
            if EARLY_STOP_PATIENCE > 0 and stale_steps[0] >= EARLY_STOP_PATIENCE:
                raise ValidationPlateau

    try:
        if optimizer == "spsa":
            local_weight_search(objective, weights,
                                max_steps=max(1, TRAIN_BUDGET_EVALS // 2), callback=monitor)
        elif optimizer == "adam":
            adam_shift(objective, weights,
                       max_steps=max(1, TRAIN_BUDGET_EVALS // (2 * len(w_params))),
                       callback=monitor)
        elif optimizer == "cobyla":
            minimize(objective, weights, method="COBYLA", callback=monitor,
                     options={"maxiter": min(MAX_ITER, TRAIN_BUDGET_EVALS), "disp": False})
        else:
            raise ValueError(f"unsupported weight optimizer: {optimizer}")
    except (BudgetSpent, ValidationPlateau):
        pass

    if best_validation[1] is not None:
        return best_validation[1]
    if _BEST[1] is not None:
        return _BEST[1]
    return weights


# -- NOTE --
# Entry point. Mutating this block breaks the results contract and the
# individual will score INVALID_FITNESS_MAX.
# -- NOTE --
if __name__ == "__main__":
    main()
