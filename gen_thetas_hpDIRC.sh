#!/bin/bash

# Initialize theta value
theta=30
momentum=6

# Loop over theta values from 30 to 150 in steps of 5
while [ $theta -le 150 ]
do
    # Run the Python script for "Pion"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config.json --momentum $momentum --theta $theta --method "Pion"

    # Run the Python script for "Kaon"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config.json --momentum $momentum --theta $theta --method "Kaon"

    # Increment theta by 5
    theta=$((theta + 5))
done

# Make plots
python make_plots.py --config config/hpDIRC_config.json --momentum $momentum
