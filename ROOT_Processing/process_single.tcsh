#!/bin/bash

# Create the json folder if it doesn't exist
mkdir -p /sciclone/data10/jgiroux/Cherenkov/Real_Data/InvMassData/071952/json

# Iterate over all .root files in the current directory
for root_file in /sciclone/data10/jgiroux/Cherenkov/Real_Data/InvMassData/071952/*.root; do
    # Extract the filename without extension
    filename=$(basename "$root_file")
    filename_noext="${filename%.*}"

    # Construct the output JSON filename in the json folder
    json_file="/sciclone/data10/jgiroux/Cherenkov/Real_Data/InvMassData/071952/json/${filename_noext}.json"

    # Execute the root command with the appropriate parameters
    root -q DrcHit.cc+ DrcEvent.cc+ 'MakeDictionaries.C("'"${root_file}"'", "'"${json_file}"'")'
    
done

