# Reducing gate cost without breaking a Qiskit circuit

The size objective is a WEIGHTED gate count: every single-qubit gate costs 1 and
every two-qubit gate (cx, cz, swap, rzz, crx, ...) costs 5, because two-qubit gates
are about ten times noisier on hardware. Barriers and measurements cost 0.
Ways to lower it while keeping the circuit valid:
- Use a linear chain instead of a ring (saves one two-qubit gate per layer).
- Entangle only the pairs that feed the readout qubit; gates outside its light cone
  cannot change the prediction.
- An RZ applied directly to |0> only adds a phase - remove it.
- Fewer layers (`N_LAYERS`) remove a full rotation + entangling block each.
- Merge rotations: RY(a) followed by RY(b) on the same qubit equals RY(a + b).
If you remove gates that consumed trainable parameters, also shrink the
`ParameterVector` size in the circuit-assembly function or keep offsets consistent,
so every weight in the returned list is actually used by the circuit.
