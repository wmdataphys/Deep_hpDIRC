#!/bin/bash

#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=gen_thetas
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --mem-per-cpu=2000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

theta=25
momentum=6
model_type="Score"
config_file="config/hpDIRC_config_gulf.json"
momentum=3

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

while [ $theta -le 155 ]
do
    if ls "${output_dir}"/*Pion*theta_${theta}* 1> /dev/null 2>&1; then
        echo "Pion file for theta $theta already exists. Skipping..."
    else
        python generate_fixedpoint_hpDIRC.py --config "$config_file" --momentum $momentum --theta $theta --method "Pion" --model_type $model_type --fine_grained_prior
    fi

    if ls "${output_dir}"/*Kaon*theta_${theta}* 1> /dev/null 2>&1; then
       echo "Kaon file for theta $theta already exists. Skipping..."
    else
       python generate_fixedpoint_hpDIRC.py --config "$config_file" --momentum $momentum --theta $theta --method "Kaon" --model_type $model_type --fine_grained_prior
    fi

    theta=$((theta + 5))
done

python make_plots.py --config "$config_file" --momentum $momentum
