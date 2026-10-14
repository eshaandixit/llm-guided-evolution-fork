# Qiskit circuit structure

Qiskit `QuantumCircuit` exposes operations such as `ry(theta, qubit)`,
`rz(theta, qubit)`, `cx(control, target)`, `cz(control, target)`, and `swap(a, b)`.
Changing the gate operation or its qubit operands changes circuit structure;
changing only a `Parameter` value changes weights, not topology. Keep every
qubit index in `range(circuit.num_qubits)` and preserve a path from data encoding
to measured readout when modifying entanglement.

Reference: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit
