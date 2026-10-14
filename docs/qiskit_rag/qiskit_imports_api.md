# Qiskit 2.x imports and gate calls that exist in this environment

The block runs inside a file that ALREADY imports everything it needs:
`np`, `math`, `random`, `itertools`, `QuantumCircuit`, `ParameterVector`, `Parameter`,
`ParameterExpression`, `Statevector`, `Operator`, `minimize` (scipy). Do not add imports.
If you must import, these are the only correct paths:
- `from qiskit import QuantumCircuit`
- `from qiskit.circuit import ParameterVector, Parameter`   (NOT `from qiskit import ParameterVector`)
- `from qiskit.quantum_info import Statevector, Operator`
- `from qiskit.circuit.library import RYGate, RZGate, CXGate, CZGate`

These do NOT exist in Qiskit 2.x and crash the circuit:
- `qiskit.providers.aer`, `qiskit_aer`, `from qiskit import Aer`, `from qiskit import execute`
- `qc.cnot(...)` (use `qc.cx(control, target)`), `qc.toffoli` (use `qc.ccx`), `qc.sd`
- `ZZFeatureMap(...).build()`; library circuits are not needed here.

Gate calls on a circuit `qc`: `qc.rx/ry/rz(theta, q)`, `qc.h(q)`, `qc.x(q)`,
`qc.cx(c, t)`, `qc.cz(a, b)`, `qc.swap(a, b)`, `qc.rzz(theta, a, b)`,
`qc.crx/cry/crz(theta, c, t)`. `theta` may be a `ParameterVector` element.
Never call `qc.measure`, `transpile`, a backend, or a simulator in a block.
