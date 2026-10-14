# Qiskit parameters and binding

Use `ParameterVector` to create named symbolic rotation parameters. A circuit
can bind values with `assign_parameters({parameter: value})`; binding values
does not add or remove gates. A structural mutation should edit circuit-building
operations, while weight optimization should change only the bound values.

Reference: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.ParameterVector
