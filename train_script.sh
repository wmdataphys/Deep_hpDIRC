#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=PScore_hpDIRC
#SBATCH --nodes=1
#SBATCH --tasks=20
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH -t 61:00:00


cd $SLURM_SUBMIT_DIR

module load miniforge3/24.9.2-0
module load cuda/12.4
source activate torch_linux

python /sciclone/home/mcmartinez/Deep_hpDIRC/train_score.py --config /sciclone/home/mcmartinez/Deep_hpDIRC/config/hpDIRC_config_gulf.json --overwrite
