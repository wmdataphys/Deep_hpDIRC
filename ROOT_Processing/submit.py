import os
import subprocess

# Base directory containing the numbered folders
base_dir = '/sciclone/data10/jgiroux/Cherenkov/Real_Data/June5_2024'
#base_dir = '/sciclone/data10/jgiroux/Cherenkov/Real_Data/InvMassData'
json_base_dir = os.path.join(base_dir, 'json')
os.makedirs(json_base_dir, exist_ok=True)

# Function to create a SLURM batch script
def create_batch_script(folder_name, folder_path):
    script_content = f"""#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=process_{folder_name}
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu=2000

# Execute the tcsh script with the folder as an argument
tcsh process_batch.tcsh {folder_path}
"""
    script_path = f'sbatch_{folder_name}.sh'
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)
    return script_path

# Iterate over all numbered folders within the base directory
i = 0
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        # Create and submit the batch script
        batch_script_path = create_batch_script(folder_name, folder_path)
        
        # Print debug information
        print(f"Created batch script: {batch_script_path}")
        with open(batch_script_path, 'r') as file:
            print(file.read())
        
        # Submit the batch script
        result = subprocess.run(['sbatch', batch_script_path])
        
        i += 1 

    # if i > 3:  # You can adjust this limit as needed
    #     break