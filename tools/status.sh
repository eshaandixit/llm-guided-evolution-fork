#!/bin/bash
# one-line machine-readable status of the current long run
cd ~/scratch/Quantam/llm-guided-evolution-fork || exit 9
J=$(cat .long_run_jobid 2>/dev/null); L="llm_opt_${J}.out"
STATE=$(squeue -h -j "$J" -o %T 2>/dev/null)
ELAP=$(squeue -h -j "$J" -o %M 2>/dev/null)
GENS=$(grep -c 'STARTING GENERATION' "$L" 2>/dev/null); GENS=${GENS:-0}
RES=$(ls sota/QuantumVQC/results/xXx*_results.txt 2>/dev/null | wc -l)
VAR=$(ls sota/QuantumVQC/models/network_xXx*.py 2>/dev/null | wc -l)
printf 'job=%s state=%s elapsed=%s gens=%s results=%s variants=%s\n' \
  "$J" "${STATE:-DONE}" "${ELAP:-na}" "$GENS" "$RES" "$VAR"
