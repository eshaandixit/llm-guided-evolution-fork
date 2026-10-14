#!/bin/bash
#SBATCH --job-name=pareto_rpt
#SBATCH -t 00:20:00
#SBATCH -A coc
#SBATCH -q coe-ice
#SBATCH -p pace-cpu
#SBATCH -N 1 -n 2
#SBATCH --mem 8G
# Runs when the orchestrator ends, however it ends, so the figures exist even if
# nobody is connected. Reads the job id at RUNTIME - do not bake it in.
cd "$HOME/scratch/Quantam/llm-guided-evolution-fork" || exit 9
J=$(cat .long_run_jobid)
OUT="runs/final_${J}"
mkdir -p "$OUT"
cp -r sota/QuantumVQC/results "$OUT/results" 2>/dev/null
cp -r sota/QuantumVQC/models  "$OUT/models"  2>/dev/null
cp "llm_opt_${J}.out" "$OUT/" 2>/dev/null
.venv/bin/python tools/pareto_report.py \
  --results "$OUT/results" --log "$OUT/llm_opt_${J}.out" --out "$OUT/pareto"
echo "PARETO REPORT DONE -> $OUT/pareto"
