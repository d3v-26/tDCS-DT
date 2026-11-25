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
#   name                                   roi bs amp ckpt flash empty
#   name                                         roi bs amp ckpt flash empty
run "256 fp32 bs2 empty_cache (original setup)"  256 2  0   0    0     1
run "256 fp32 bs2"                               256 2  0   0    0     0
run "256 bf16 bs2"                               256 2  1   0    0     0
run "256 bf16 flash bs2"                         256 2  1   0    1     0
run "256 bf16 flash ckpt bs2"                    256 2  1   1    1     0
run "128 fp32 bs2"                               128 2  0   0    0     0
run "128 bf16 flash bs2"                         128 2  1   0    1     0
run "128 bf16 flash bs8"                         128 8  1   0    1     0
