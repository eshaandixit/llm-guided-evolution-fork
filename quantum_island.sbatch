#!/bin/bash
#SBATCH --job-name=qvqc_island
#SBATCH -p pace-cpu
#SBATCH -t 16:00:00
#SBATCH --mem 16G
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/qvqc_island-%x-%j.out
# One QuantumVQC evolution island (the orchestrator only; it needs no GPU).
# The LLM server must be started separately:
#     SUBMIT_ISLAND_CONTROLLER=0 sbatch server.sh
# then, e.g. the RAG / no-RAG pair used for the 2026-10-08 ablation:
#     sbatch -J rag   --export=ALL,RUN=qvqc_rag,LLMGE_QISKIT_RAG=1 quantum_island.sbatch
#     sbatch -J norag --export=ALL,RUN=qvqc_norag,LLMGE_QISKIT_RAG=0 quantum_island.sbatch
# Resubmitting with the same RUN resumes from that run's last checkpoint.
cd "$SLURM_SUBMIT_DIR"
module load uv
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

RUN=${RUN:-qvqc_rag}
export LLMGE_MIGRATION_GEN=${LLMGE_MIGRATION_GEN:-0}       # no 5-generation island stop
export LLMGE_NUM_GENERATIONS=${LLMGE_NUM_GENERATIONS:-40}
export LLMGE_START_POPULATION=${LLMGE_START_POPULATION:-20}
export LLMGE_POPULATION=${LLMGE_POPULATION:-16}
export LLMGE_QISKIT_RAG=${LLMGE_QISKIT_RAG:-1}
echo "run=$RUN rag=$LLMGE_QISKIT_RAG gens=$LLMGE_NUM_GENERATIONS pop=$LLMGE_START_POPULATION/$LLMGE_POPULATION host=$(hostname)"

.venv/bin/python -u run_improved.py \
    --checkpoints "quantum_runs/$RUN/checkpoints" \
    --global_path "quantum_runs/$RUN/global_data" \
    --llm_model llama3 \
    --prompt_group QuantumVQC/Normal
