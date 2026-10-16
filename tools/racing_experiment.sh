#!/bin/bash
# Racing experiment: control (fixed starts) vs racing, several runs each, sharing one LLM server.
#
#   bash tools/racing_experiment.sh <name>                 # full: 3 runs per arm
#   RUNS=1 CONTROL_GENS=4 bash tools/racing_experiment.sh pilot
#
# Both arms drop duplicate programs before selection (LLMGE_DEDUPE=1). The control arm
# trains every circuit QISKIT_N_STARTS (10) times and runs CONTROL_GENS generations.
# The racing arm (LLMGE_RACING=1) runs until it has used as many training starts as
# the control could have, so the two are compared at equal compute
# (tools/racing_analysis.py cuts both at the control's actual total).
#
# Everything goes under experiments/<name>/<arm>_<i>/ (checkpoints, global data and
# progress.jsonl). One server job serves every run; a last job cancels it once all
# runs have ended.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: bash tools/racing_experiment.sh <name>}"
RUNS="${RUNS:-3}"
CONTROL_GENS="${CONTROL_GENS:-10}"
POP="${LLMGE_POPULATION:-16}"
CONTROL_STARTS="${QISKIT_N_STARTS:-10}"
# Upper bound on what the control can use: the first population plus every generation's
# children, each trained CONTROL_STARTS times. Racing stops once it reaches this.
BUDGET=$(( (CONTROL_GENS + 1) * POP * CONTROL_STARTS ))
RACING_GENS="${RACING_GENS:-$(( CONTROL_GENS * 3 ))}"
HOURS="${HOURS:-8}"
EXP="experiments/$NAME"
PORT=$(grep -E '^PORT *=' src/cfg/constants_Mujoco.py | grep -oE '[0-9]+')

if [ -e "$EXP" ]; then echo "$EXP already exists; pick another name" >&2; exit 1; fi
mkdir -p "$EXP" run_job_outputs/server run_job_outputs/islands run_job_outputs/evaluation run_job_outputs/evolution
rm -f hostname.log

SERVER_JOB=$(sbatch --parsable -t "$HOURS:00:00" --export=ALL,PORT="$PORT" <<'SH'
#!/bin/bash
#SBATCH --job-name=LLMGE_exp_server
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
SH
)

submit_run() {   # arm, index, extra env
  local arm=$1 i=$2 env=$3 run="$EXP/${1}_${2}"
  mkdir -p "$run"
  sbatch --parsable -t "$HOURS:00:00" --dependency=after:"$SERVER_JOB" \
    --export=ALL,RUN="$run",LLMGE_DEDUPE=1,LLMGE_MIGRATION_GEN=0,LLMGE_START_POPULATION="$POP",LLMGE_POPULATION="$POP",$env <<'SH'
#!/bin/bash
#SBATCH --job-name=LLMGE_exp_run
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
module load uv
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
export SERVER_HOSTNAME=$(hostname)
echo "$RUN started on $(hostname) at $(date)"
uv run python run_improved.py --checkpoints "$RUN/checkpoints" --global_path "$RUN/global_data" \
  --llm_model llama3 --prompt_group QuantumVQC/Normal
SH
}

RUN_JOBS=()
for i in $(seq 1 "$RUNS"); do
  RUN_JOBS+=("$(submit_run control "$i" "LLMGE_RACING=0,LLMGE_NUM_GENERATIONS=$((CONTROL_GENS + 1))")")
  RUN_JOBS+=("$(submit_run racing "$i" "LLMGE_RACING=1,LLMGE_NUM_GENERATIONS=$((RACING_GENS + 1)),LLMGE_STOP_AT_STARTS=$BUDGET")")
done
DEPS=$(IFS=:; echo "${RUN_JOBS[*]}")
STOP_JOB=$(sbatch --parsable -t 0:05:00 --mem 1G -c 1 --dependency=afterany:"$DEPS" \
  --job-name=LLMGE_exp_stop --output=run_job_outputs/server/stop-%j.out --wrap "scancel $SERVER_JOB")

cat > "$EXP/config.txt" <<CFG
name=$NAME
started=$(date)
commit=$(git rev-parse --short HEAD)
runs_per_arm=$RUNS
population=$POP
control_generations=$CONTROL_GENS
control_starts_per_circuit=$CONTROL_STARTS
racing_max_generations=$RACING_GENS
racing_budget_starts=$BUDGET
server_job=$SERVER_JOB
run_jobs=${RUN_JOBS[*]}
stop_job=$STOP_JOB
CFG

cat <<MSG
Experiment $NAME: $RUNS run(s) per arm, population $POP
  control: $CONTROL_GENS generations, $CONTROL_STARTS starts per circuit
  racing:  until $BUDGET training starts (at most $RACING_GENS generations)
  server job $SERVER_JOB, run jobs ${RUN_JOBS[*]}, server stops via job $STOP_JOB
Settings saved to $EXP/config.txt

Progress:  tail -n 1 $EXP/*/checkpoints/progress.jsonl | cut -c1-120
Queue:     squeue -u \$USER
Stop all:  scancel $SERVER_JOB ${RUN_JOBS[*]} $STOP_JOB
When done: uv run python tools/rescore_front.py $EXP   then   uv run python tools/racing_analysis.py $EXP
MSG
