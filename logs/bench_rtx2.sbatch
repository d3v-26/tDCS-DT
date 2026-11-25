#!/bin/bash
#SBATCH --job-name=tdcs_bench_rtx
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
cd "$SLURM_SUBMIT_DIR"
PY=/blue/ruogu.fang/pated32.vanderbilt/IXI_project/envs/crown/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
run() { $PY logs/bench_gpu.py "$@" 2>&1 | grep RESULT; }
run "256 fp32 bs1"                               256 1  0   0    0     0
run "256 bf16 bs1"                               256 1  1   0    0     0
run "256 bf16 flash bs1"                         256 1  1   0    1     0
run "256 bf16 flash ckpt bs1"                    256 1  1   1    1     0
run "192 bf16 flash bs1"                         192 1  1   0    1     0
run "192 bf16 flash bs2"                         192 2  1   0    1     0
run "128 bf16 bs2 (no flash)"                    128 2  1   0    0     0
run "128 bf16 flash ckpt bs2"                    128 2  1   1    1     0
