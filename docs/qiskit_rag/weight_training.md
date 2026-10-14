# Training block: optimizer and train_angles contract

`main()` calls `train_angles(qc, x_params, w_params, X, y, X_val, y_val,
initial_weights=..., optimizer=...)` and expects a 1-D numpy array of
`len(w_params)` angles back. Keep that name and signature exactly, and keep every
helper it calls (`local_weight_search`, `adam_shift`) defined in the same block.
Inside, compute the loss with `cross_entropy(qc, x_params, w_params, values, X, y)`;
the protected training budget stops the search by raising `BudgetSpent`, which
`train_angles` catches - do not swallow it inside your own loop.
Valid scipy optimizers (`minimize` is already imported): `minimize(objective, w0,
method="COBYLA" | "Powell" | "Nelder-Mead" | "L-BFGS-B", options={"maxiter": MAX_ITER})`.
Angles are periodic, so wrap with `np.mod(w, 2 * np.pi)` rather than clipping.
