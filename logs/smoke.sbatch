#!/bin/bash
#SBATCH --job-name=tdcs_smoke
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
cd "$SLURM_SUBMIT_DIR"
PY=/blue/ruogu.fang/pated32.vanderbilt/IXI_project/envs/crown/bin/python
C=logs/smoke/cache
# 4 subjects/split incl. a padded-label subject in Test (108121)
$PY preprocess.py -root_dir tDCS_DT_data -out_dir $C -workers 6 -limit 4 2>&1 | grep -v -i warn | tail -5
$PY train.py -root_dir $C -x 128 -y 128 -z 128 -batch_size 1 -num_epochs 3 -val_every 1 -num_workers 4 \
    -log_every 2 -flash_attention -save_path logs/smoke/m.pth 2>&1 | grep -v -i warn
$PY train.py -root_dir $C -x 128 -y 128 -z 128 -batch_size 1 -num_epochs 4 -val_every 1 -num_workers 4 \
    -log_every 4 -flash_attention -save_path logs/smoke/m.pth -resume logs/smoke/m_last.pth 2>&1 | grep -v -i warn
$PY test.py -root_dir $C -x 128 -y 128 -z 128 -flash_attention -save_path logs/smoke/m.pth \
    -results_path ../logs/smoke/results 2>&1 | grep -v -i warn
ls logs/smoke/results | head
