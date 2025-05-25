#!/bin/bash

#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=gen_mix_thetas
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --mem-per-cpu=2000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

momentum=3
model_type="DDPM"
config_file="config/hpDIRC_config_local.json"

if [ -n "$SLURM_JOB_ID" ]; then
    temp_config_file="config/temp/config_temp_${SLURM_JOB_ID}.json"
else
    # Fallback to a timestamp if not running under SLURM
    temp_config_file="config/temp/config_temp_$(date +%s).json"
fi

# cp "$config_file" "$temp_config_file"

output_dir=$(python -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)
print(config['Inference']['fixed_point_dir'])
")

if [ -z "$output_dir" ]; then
    echo "Error: Unable to extract output directory from config file."
    exit 1
fi

output_dir="Generations/$output_dir"

for theta in 30; do
    for timestep in 0 5 10 20 30 40 50 60 70; do
        if ls "${output_dir}"/*Kaon*theta_${theta}* 1> /dev/null 2>&1; then
            echo "Kaon file for theta $theta already exists. Skipping..."
        else
            python generate_fixedpoint_hpDIRC.py --config "$config_file" --momentum $momentum --theta $theta --method "Kaon" --model_type $model_type --timesteps $timestep --fine_grained_prior 
        fi

    done
done

python make_plots.py --config "$config_file" --momentum $momentum