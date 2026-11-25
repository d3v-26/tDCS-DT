#!/bin/bash
#SBATCH --job-name=tdcs_smoke_ddp
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --gpus=l4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
cd "$SLURM_SUBMIT_DIR"
PY=/blue/ruogu.fang/pated32.vanderbilt/IXI_project/envs/crown/bin/python
$PY -m torch.distributed.run --standalone --nproc_per_node=2 train.py -root_dir logs/smoke/cache -x 128 -y 128 -z 128 \
    -batch_size 1 -num_epochs 2 -val_every 1 -num_workers 3 -log_every 2 -save_path logs/smoke/ddp.pth 2>&1 | grep -v -i warn
$PY -m torch.distributed.run --standalone --nproc_per_node=2 train.py -root_dir logs/smoke/cache -x 128 -y 128 -z 128 \
    -batch_size 1 -num_epochs 3 -val_every 1 -num_workers 3 -log_every 2 -save_path logs/smoke/ddp.pth -resume logs/smoke/ddp_last.pth 2>&1 | grep -v -i warn
$PY test.py -root_dir logs/smoke/cache -x 128 -y 128 -z 128 -save_path logs/smoke/ddp.pth -no_save 2>&1 | grep -v -i warn | tail -3
