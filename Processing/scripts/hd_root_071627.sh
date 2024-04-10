#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=hd_root_071627
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --constraint=gust 


module load anaconda3/2021.11
source activate ptorch

python /sciclone/home/jgiroux/Cherenkov_FastSim/Processing/make_GlueX_data.py --file /sciclone/data10/jgiroux/Cherenkov/Real_Data/json/hd_root_071627.json
