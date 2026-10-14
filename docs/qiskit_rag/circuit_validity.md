# Common Qiskit validity errors in evolved circuit blocks

- Qubit index out of range: every index must be in `range(qc.num_qubits)`; use
  `% N_QUBITS` for wrap-around rings.
- `ParameterVector` index errors: `w_params[offset + k]` must stay below
  `len(w_params)`; the assembly function sizes the vector, the layer function returns
  the next offset.
- Do not add `qc.measure`, `qc.measure_all`, `qc.reset`, or `qc.initialize` to a
  statevector classifier block; probabilities are read from the statevector.
- `qc.cx(i, i)` (same control and target) raises a CircuitError.
- Return values: the assembly function must return `(qc, list(x_params), list(w_params))`.
- `qc.qasm()` was removed in Qiskit 1.x/2.x; use `qiskit.qasm3.dumps(qc)`.
- Gate methods live on the circuit: `qc.ry(theta, q)`, not `RYGate(theta, q)`
  (use `qc.append(RYGate(theta), [q])` for gate objects).
