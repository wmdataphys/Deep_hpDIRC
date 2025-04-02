#!/bin/bash

#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=gen_fullphase
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --mem-per-cpu=2000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

python generate_phasespace_hpDIRC.py --config config/hpDIRC_config_clean.json --n_particles 50000 --method Pion --model_type GSGM --fine_grained_prior