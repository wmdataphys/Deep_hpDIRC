#!/bin/tcsh

set theta = 30
set momentum = 6
set model_type = "FlowMatching"
set config_file = "config/hpDIRC_config_clean.json"

set output_dir = `python -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)
print(config['Inference']['fixed_point_dir'])
"`

if ("$output_dir" == "") then
    echo "Error: Unable to extract output directory from config file."
    exit 1
endif

set output_dir = "Generations/$output_dir"

while ($theta <= 150)
    if (`ls ${output_dir}/*Pion*theta_${theta}* 1> /dev/null 2>&1; echo $?` == 0) then
        echo "Pion file for theta $theta already exists. Skipping..."
    else
        python generate_fixedpoint_hpDIRC.py --config "$config_file" --momentum $momentum --theta $theta --method "Pion" --model_type $model_type
    endif

    if (`ls ${output_dir}/*Kaon*theta_${theta}* 1> /dev/null 2>&1; echo $?` == 0) then
        echo "Kaon file for theta $theta already exists. Skipping..."
    else
        python generate_fixedpoint_hpDIRC.py --config "$config_file" --momentum $momentum --theta $theta --method "Kaon" --model_type $model_type
    endif

    @ theta = $theta + 5
end

python make_plots.py --config "$config_file" --momentum $momentum
