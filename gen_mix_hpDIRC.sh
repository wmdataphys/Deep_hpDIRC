#!/bin/bash

#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=gen_spec_thetas
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --mem-per-cpu=2000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

n_tracks=100
n_dump=25
theta=50-70
momentum=0-10.1
model_type="DDPM"

config_file="config/hpDIRC_config_test.json"

python generate_mix_hpDIRC.py --config "$config_file" --n_tracks $n_tracks --n_dump $n_dump --method "MixPK" --momentum $momentum --theta $theta --model_type $model_type  --fine_grained_prior

