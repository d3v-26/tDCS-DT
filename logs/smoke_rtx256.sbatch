#!/bin/bash
#SBATCH --job-name=tdcs_smoke_rtx256
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --gpus=rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
cd "$SLURM_SUBMIT_DIR"
PY=$SLURM_SUBMIT_DIR/envs/tdcs/bin/python
$PY -m torch.distributed.run --standalone --nproc_per_node=2 train.py -root_dir cache -x 256 -y 256 -z 256 \
    -batch_size 1 -num_epochs 1 -val_every 1 -limit_batches 30 -log_every 10 -num_workers 7 \
    -save_path logs/smoke/rtx256.pth 2>&1 | grep -v -i warn
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
