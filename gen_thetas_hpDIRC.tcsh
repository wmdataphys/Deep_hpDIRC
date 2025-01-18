#!/bin/tcsh

# Initialize theta value
set theta = 30
set momentum = 6
model_type = "NF"

# Loop over theta values from 30 to 150 in steps of 5
while ($theta <= 150)
    # Run the Python script for "Pion"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config_clean.json --momentum $momentum --theta $theta --method "Pion" --model_type $model_type

    # Run the Python script for "Kaon"
    python generate_fixedpoint_hpDIRC.py --config config/hpDIRC_config_clean.json --momentum $momentum --theta $theta --method "Kaon" --model_type $model_type

    # Increment theta by 5
    @ theta = $theta + 5
end

# Make plots
python make_plots.py --config config/hpDIRC_config_clean.json --momentum $momentum
