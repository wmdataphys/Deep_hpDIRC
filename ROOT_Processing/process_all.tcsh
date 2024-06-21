#!/bin/tcsh
# Going to stop at folder 071502 for now. - June 6 2024
# Source the ROOT environment script
source /sciclone/home/jgiroux/root/bin/thisroot.csh

# Base directory containing the numbered folders
set base_dir = /sciclone/data10/jgiroux/Cherenkov/Real_Data/June5_2024

# Create a single json directory in the base directory
set json_dir = "${base_dir}/json"
mkdir -p $json_dir

# Iterate over all numbered folders within the base directory
foreach folder (${base_dir}/*)
    if (-d $folder) then
        # Get the folder name
        set folder_name = `basename "$folder"`

        # Iterate over all .root files in the current numbered folder
        foreach root_file (${folder}/*.root)
            # Extract the filename without extension
            set filename = `basename "$root_file"`
            set filename_noext = "${filename:r}"

            # Construct the output JSON filename in the centralized json folder
            set json_file = "${json_dir}/${filename_noext}.json"
            # Execute the root command with the appropriate parameters
            root -q DrcHit.cc+ DrcEvent.cc+ 'MakeDictionaries.C("'"${root_file}"'", "'"${json_file}"'")'
        end
    endif
end