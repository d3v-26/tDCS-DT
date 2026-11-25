#!/bin/bash
#SBATCH --job-name=tdcs_preprocess
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#SBATCH --partition=hpg-default,hpg-milan,hpg-turin
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# One-time conversion of tDCS_DT_data (NIfTI) -> cache/ (.npy). Re-running skips finished subjects.
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$SLURM_SUBMIT_DIR/envs/tdcs/bin/python}
$PY preprocess.py -root_dir tDCS_DT_data -out_dir cache -workers $SLURM_CPUS_PER_TASK
