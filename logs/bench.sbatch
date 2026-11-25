#!/bin/bash
#SBATCH --job-name=tdcs_bench
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
cd "$SLURM_SUBMIT_DIR"
PY=/blue/ruogu.fang/pated32.vanderbilt/IXI_project/envs/crown/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
run() { $PY logs/bench_gpu.py "$@" 2>&1 | grep RESULT; }
#   name                                   roi bs amp ckpt flash empty
run "256 fp32 bs1 (original, minus bs2)"   256 1  0   0    0     1
run "256 bf16 flash bs1"                   256 1  1   0    1     0
run "256 bf16 flash ckpt bs1"              256 1  1   1    1     0
run "256 bf16 flash ckpt bs2"              256 1  1   1    1     0
run "192 bf16 flash bs1"                   192 1  1   0    1     0
run "160 bf16 flash bs2"                   160 2  1   0    1     0
run "128 fp32 bs2 empty_cache (orig-style)" 128 2 0   0    0     1
run "128 fp32 bs2"                         128 2  0   0    0     0
run "128 bf16 bs2"                         128 2  1   0    0     0
run "128 bf16 flash bs2"                   128 2  1   0    1     0
run "128 bf16 flash bs4"                   128 4  1   0    1     0
