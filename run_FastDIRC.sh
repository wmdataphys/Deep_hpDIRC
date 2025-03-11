#!/bin/bash

# Define the support values for Geant PDF with k suffix
geant_support_values=("100k" "200k" "400k" "500k" "800k")

# JSON config file path
CONFIG_FILE="config/hpDIRC_config_clean.json"
momentum=6.0

if [ ! -d "KDE_Fits" ]; then
    echo "Making KDE_Fits directory."
    mkdir "KDE_Fits"
fi

out_dir="KDE_Fits/$momentum"
mkdir -p "$out_dir"


cp "$CONFIG_FILE" "$CONFIG_FILE.bak"

for geant_support in "${geant_support_values[@]}"; do
    geant_support_number="${geant_support}"
    geant_support_number_=$(echo "$geant_support" | sed 's/k//') 
    geant_support_full="${geant_support_number_}000"

    KDE_DIR=$(python -c "
import json
config_file = '$CONFIG_FILE'
geant_support = '$geant_support_number'
kde_dir=f'KDE_Fits/${momentum}/Test_FastDIRC_800k_NF_${geant_support}G'
with open(config_file, 'r') as f:
    config = json.load(f)
config['Inference']['KDE_dir'] = kde_dir
with open(config_file, 'w') as f:
    json.dump(config, f, indent=4)
print(kde_dir)
")

    echo "Updated KDE_dir to: $KDE_DIR"

    python KDE_Fits.py \
        --config "$CONFIG_FILE" \
        --momentum $momentum \
        --fine_grained_prior \
        --fs_support 800000 \
        --geant_support ${geant_support_full}
done

rm "$CONFIG_FILE.bak"