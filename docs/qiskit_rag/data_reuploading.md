# Data re-uploading in a variational classifier

Data re-uploading applies the feature-encoding map more than once, interleaved with
trainable layers: encode -> variational -> entangle -> encode -> variational -> ...
In code this means calling the feature-map function inside the layer loop of the
circuit-assembly function, reusing the SAME input `ParameterVector` (the inputs are
bound per sample; no new input parameters are needed):

    for layer in range(N_LAYERS):
        build_feature_map(qc, x_params)
        offset = build_variational_layer(qc, w_params, offset)
        build_entanglement(qc)

It increases expressivity (a single-qubit classifier with re-uploading is a universal
approximator; Perez-Salinas et al., Quantum 4, 226, 2020) at the cost of N_QUBITS extra
encoding gates per extra upload. Trainable weights must still be allocated for every
layer: `ParameterVector("w", N_LAYERS * weights_per_layer)`.
