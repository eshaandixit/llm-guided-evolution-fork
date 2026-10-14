# Statevector readout

Qiskit's `Statevector` can simulate a bound circuit and return measurement
probabilities. The chosen qubit subset defines the marginal distribution; keep
the readout dimension consistent with the classification labels. For a binary
classifier, one readout qubit has two outcomes. This simulation is a local
statevector metric, not a hardware-noise estimate.

Reference: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Statevector
