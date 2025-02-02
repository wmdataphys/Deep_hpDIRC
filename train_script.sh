#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=PDDPM_hpDIRC
#SBATCH --nodes=1
#SBATCH --tasks=20
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

cd $SLURM_SUBMIT_DIR

module load miniconda3
module load cuda
source activate torch_linux

python /sciclone/home/mcmartinez/Deep_hpDIRC/train_ddpm.py --config /sciclone/home/mcmartinez/Deep_hpDIRC/config/hpDIRC_config_gulf.json --overwrite