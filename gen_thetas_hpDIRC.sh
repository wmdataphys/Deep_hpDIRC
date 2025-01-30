#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=gen_thetas_hpDIRC
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --mem-per-cpu=2000
#SBATCH --gpus=1
#SBATCH -t 72:00:00

# Initialize theta value
theta=30
momentum=9
model_type="Score"

# Loop over theta values from 30 to 150 in steps of 5
while [ $theta -le 150 ]
do
    # Run the Python script for "Pion"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config_gulf.json --momentum $momentum --theta $theta --method "Pion" --model_type $model_type

    # Run the Python script for "Kaon"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config_gulf.json --momentum $momentum --theta $theta --method "Kaon" --model_type $model_type

    # Increment theta by 5
    theta=$((theta + 5))
done

# Make 2D plots
python make_plots.py --config config/hpDIRC_config_gulf.json --momentum $momentum
