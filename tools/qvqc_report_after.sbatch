#!/bin/bash
#SBATCH --job-name=qvqc_report
#SBATCH -p pace-cpu
#SBATCH -t 1:00:00
#SBATCH --mem 8G
#SBATCH -c 2
#SBATCH --output=run_job_outputs/islands/qvqc_report-%j.out
# Renders the RAG vs no-RAG comparison as soon as the island finishes, however it ends:
#     sbatch --dependency=afterany:<island job> \
#            --export=ALL,RAG_LOGS=<log>,NORAG_LOGS=<log>[,<resumed log>],CANCEL_JOBS="<spare server ids>" \
#            tools/qvqc_report_after.sbatch
# Output: reports/qvqc_rag_vs_norag/ (summary.md + figures).
cd "$SLURM_SUBMIT_DIR"
for j in $CANCEL_JOBS; do scancel "$j" && echo "cancelled spare job $j"; done
.venv/bin/python tools/qvqc_run_report.py \
    --run "rag=$RAG_LOGS" --run "no_rag=$NORAG_LOGS" \
    --out reports/qvqc_rag_vs_norag
echo "report done: reports/qvqc_rag_vs_norag/summary.md"
