#!/bin/bash

# Initialize theta value
theta=30
momentum=6
model_type="FlowMatching"

# Loop over theta values from 30 to 150 in steps of 5
while [ $theta -le 150 ]
do
    # Run the Python script for "Pion"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config_clean.json --momentum $momentum --theta $theta --method "Pion" --model_type $model_type

    # Run the Python script for "Kaon"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config_clean.json --momentum $momentum --theta $theta --method "Kaon" --model_type $model_type

    # Increment theta by 5
    theta=$((theta + 5))
done

# Make plots
python make_plots.py --config config/hpDIRC_config_clean.json --momentum $momentum
