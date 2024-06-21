#!/bin/tcsh

# Check if the folder argument is provided
if ($#argv < 1) then
    echo "Usage: $0 <folder>"
    exit 1
endif

# Source the ROOT environment script
source /sciclone/home/jgiroux/root/bin/thisroot.csh

# Base directory containing the numbered folders
set folder = $argv[1]

# Create a single json directory in the base directory
#set json_dir = "/sciclone/data10/jgiroux/Cherenkov/Real_Data/InvMassData/json"
set json_dir = "/sciclone/data10/jgiroux/Cherenkov/Real_Data/June5_2024/json"

mkdir -p $json_dir

# Get the folder name
set folder_name = `basename "$folder"`

# Iterate over all .root files in the current numbered folder
foreach root_file (${folder}/*.root)
    # Extract the filename without extension
    set filename = `basename "$root_file"`
    set filename_noext = "${filename:r}"

    # Construct the output JSON filename in the centralized json folder
    set json_file = "${json_dir}/${folder_name}_${filename_noext}.json"

    # Execute the root command with the appropriate parameters
    root -q DrcHit.cc+ DrcEvent.cc+ 'MakeDictionaries.C("'"${root_file}"'", "'"${json_file}"'")'
end