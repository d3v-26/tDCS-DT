#!/bin/bash
#SBATCH --job-name=tdcs_train
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --gpus=rtx_pro_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
# Full-volume 256^3, bf16, batch 1 per GPU (bs2 per GPU OOMs even on 96 GB).
# 2 GPUs -> global batch 2 = original setting. 1 GPU also works: sbatch --gpus=rtx_pro_6000:1 slurm/train.sbatch
# Resume after a timeout:  sbatch slurm/train.sbatch -resume Models/best_model_last.pth
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$SLURM_SUBMIT_DIR/envs/tdcs/bin/python}
NGPU=${SLURM_GPUS_ON_NODE:-1}
nvidia-smi
$PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU train.py \
    -root_dir cache -x 256 -y 256 -z 256 -batch_size 1 -num_epochs 120 \
    -num_workers $(( (SLURM_CPUS_PER_TASK - 2) / NGPU )) -save_path Models/best_model.pth "$@"
