#!/bin/bash
# Short evolution run with racing on, to check racing inside the real loop.
#
# Submits two Slurm jobs:
#   1. the LLM server (server.py on 2 H200s), which writes hostname.log
#   2. run_improved.py with racing on and a small population, which starts once the
#      server job starts, waits for the server to answer, and cancels it when done
#
# Run from anywhere on the login node:  bash tools/racing_full_run.sh
# Settings can be overridden, e.g.:      LLMGE_POPULATION=24 bash tools/racing_full_run.sh
#
# This does not touch run.sh, server.sh or island_controller.sbatch, and uses fresh
# checkpoint folders, so it never resumes an earlier run.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="racing_test_$(date +%m%d_%H%M)"
# num_elites is 8, and a generation whose population drops below 8 is regenerated from
# scratch, so keep the population well above 8. Generations run 1..N-1, so 3 gives two.
POP="${LLMGE_POPULATION:-16}"
GENS="${LLMGE_NUM_GENERATIONS:-3}"
PORT=$(grep -E '^PORT *=' src/cfg/constants_Mujoco.py | grep -oE '[0-9]+')

mkdir -p run_job_outputs/server run_job_outputs/islands run_job_outputs/evaluation run_job_outputs/evolution
rm -f hostname.log    # a stale file would point the run at a server that no longer exists

SERVER_JOB=$(sbatch --parsable --export=ALL,PORT="$PORT" <<'EOF'
#!/bin/bash
#SBATCH --job-name=LLMGE_racing_server
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
module load cuda
module load uv
export USE_TF=0 USE_TORCH=1
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
export SERVER_HOSTNAME=$(hostname)
echo "$SERVER_HOSTNAME" > hostname.log
uv run python -m uvicorn server:app --host "$SERVER_HOSTNAME" --port "$PORT" --workers 1
EOF
)

RUN_JOB=$(sbatch --parsable --dependency=after:"$SERVER_JOB" \
  --export=ALL,SERVER_JOB="$SERVER_JOB",RUN_NAME="$RUN_NAME",LLMGE_RACING=1,LLMGE_START_POPULATION="$POP",LLMGE_POPULATION="$POP",LLMGE_NUM_GENERATIONS="$GENS",LLMGE_MIGRATION_GEN=0 <<'EOF'
#!/bin/bash
#SBATCH --job-name=LLMGE_racing_run
#SBATCH -t 4:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
module load uv
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
export SERVER_HOSTNAME=$(hostname)
trap 'scancel "$SERVER_JOB"' EXIT    # stop the GPU server when the run ends, however it ends
uv run python run_improved.py \
  --checkpoints "$RUN_NAME/checkpoints" \
  --global_path "$RUN_NAME/global_data" \
  --llm_model llama3 \
  --prompt_group QuantumVQC/Normal
EOF
)

cat <<EOF
Submitted:
  LLM server   job $SERVER_JOB   log: run_job_outputs/server/slurm-$SERVER_JOB.out
  Evolution    job $RUN_JOB   log: run_job_outputs/islands/slurm-$RUN_JOB.out
Run name: $RUN_NAME  (population $POP, $((GENS - 1)) generations, racing on)

Check the queue:     squeue -u \$USER
Follow the run:      tail -f run_job_outputs/islands/slurm-$RUN_JOB.out
See racing rounds:   grep -E "RACING|Round|after [0-9]+ starts|Extra starts failed" run_job_outputs/islands/slurm-$RUN_JOB.out
Stop everything:     scancel $RUN_JOB $SERVER_JOB
EOF
