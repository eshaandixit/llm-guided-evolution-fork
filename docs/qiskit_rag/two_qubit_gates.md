# Qiskit two-qubit gates and entanglers

`QuantumCircuit` methods for two-qubit interactions (control first, target second):
- `qc.cx(control, target)` - CNOT. `qc.cz(a, b)` - controlled-Z, symmetric in its qubits.
- `qc.swap(a, b)`, `qc.iswap(a, b)`, `qc.ecr(a, b)` - fixed (non-parameterized) gates.
- `qc.rzz(theta, a, b)`, `qc.rxx(theta, a, b)`, `qc.ryy(theta, a, b)` - Ising-type
  parameterized entanglers exp(-i theta/2 Z⊗Z) etc.; `theta` may be a `Parameter`,
  so they add a trainable weight AND entanglement in one gate.
- `qc.crx(theta, c, t)`, `qc.cry(theta, c, t)`, `qc.crz(theta, c, t)`, `qc.cp(theta, c, t)` -
  controlled rotations; also trainable.
Each of these counts as ONE gate in a plain gate count. A ring on n qubits uses n
two-qubit gates (`i -> (i + 1) % n`), a linear chain uses n - 1 (`i -> i + 1` for
i in range(n - 1)), all-to-all uses n(n - 1)/2.

Reference: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit
