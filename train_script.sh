#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=PShift
#SBATCH --nodes=1
#SBATCH --tasks=20
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

cd $SLURM_SUBMIT_DIR

module load miniconda3
module load cuda
source activate torch_linux

python train_ddpm.py --config config/hpDIRC_config_test.json