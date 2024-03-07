# If you're submitting this on the new cluster you will need to do the following:

1. module load anaconda3/2023.09
2. conda activate ptorch (note you will need to install normflows into this original ptorch env I created)
3. sbatch slurm_sub (make whatever changes inside here to the file names, path to config, etc.)

You need to have your conda env activated before submitting the job. The submission script will export all current system states (i.e., python packages) to the node.
This was just the first way I could get slurm to not complain. Probably a better way for this.
