#!/bin/bash
#SBATCH -A math
#SBATCH -q coe-ice
#SBATCH --job-name=wine_seed
#SBATCH -t 1:00:00
#SBATCH --mem 8G
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --output=run_job_outputs/evaluation/seed-%j.out
unset VIRTUAL_ENV
.venv/bin/python sota/QuantumVQC/network.py --gene-id seed --out-dir sota/QuantumVQC/results