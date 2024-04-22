import os
import subprocess

# Directory containing the files
files_directory = "/sciclone/data10/jgiroux/Cherenkov/Real_Data/json"

# Path to the Python script to be executed
python_script = "/sciclone/home/jgiroux/Cherenkov_FastSim/Processing/make_GlueX_data.py"

# Directory to store submission scripts
submission_scripts_directory = "/sciclone/home/jgiroux/Cherenkov_FastSim/Processing/scripts"

# Create the submission scripts directory if it doesn't exist
os.makedirs(submission_scripts_directory, exist_ok=True)

# Iterate over the files in the directory
for filename in os.listdir(files_directory):
    # Check if the item is a file
    if os.path.isfile(os.path.join(files_directory, filename)):
        # Construct the full file path
        file_path = os.path.join(files_directory, filename)
        #file_path = os.path.join(files_directory,"hd_root_071725.json")
        # Calculate the RAM requirement based on the size of the file (in megabytes)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to megabytes
        ram_per_job = int(file_size_mb * 2)  # Set RAM to be 2 times the size of the file
        
        # Create a submission script for each file
        submission_script = f"""\
#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name={os.path.splitext(filename)[0]}
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --constraint=gust 


module load anaconda3/2021.11
source activate ptorch

python {python_script} --file {file_path}
"""

        # Write the submission script to a file in the submission scripts directory
        submission_script_path = os.path.join(submission_scripts_directory, f"{os.path.splitext(filename)[0]}.sh")
        with open(submission_script_path, 'w') as f:
            f.write(submission_script)
            
        # Execute the submission script
        subprocess.run(["sbatch", submission_script_path])
      
    #break  # For testing purposes, remove this line to process all files

