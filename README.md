# If you're submitting this on the new cluster you will need to do the following:

1. module load anaconda3/2023.09
2. conda activate ptorch (note you will need to install normflows into this original ptorch env I created)
3. sbatch slurm_sub (make whatever changes inside here to the file names, path to config, etc.)

You need to have your conda env activated before submitting the job. The submission script will export all current system states (i.e., python packages) to the node.
This was just the first way I could get slurm to not complain. Probably a better way for this.

# Import Notes:

To run an interactive job: salloc -N -n 8 --gpus=1 --time=01:00:00

To submit a job with the submit_slurm script: sbatch slurm_sub
This is similar to torque

Also, something very important I forgot. We are working on linux, we can use multiple workers. I have modified the dataloader scripts to use 8 workers, so request 8 cores. This seems to be about perfect before we degrade performance due to bottlenecking.
This will give you training times on the order of 15 minutes / epoch with a batch size of 2048 (30 minutes if using both pions and kaons)

I have made the dataloader so you can utilize both classes more easily. There are new fields in the config file (i.e., different data paths for the new files) Update these as you need.
You can control which dataset is loaded within the config file through the "method" field, which can be Pion, Kaon or combined.

All scaling is also handled in the dataset now as well. Take a look over these files to see the changes and how things tie together. But in general this is a more sound method.

Updated the generation (run_inference.py) and DLL files (run_DLL_v2.py). These should work with new data pipeline. Also changed the statistics we are using slightly, specifically the min and max values for the x,y position. These should increased/decreased by 3 from what we had prior.

# TODO:

Need to rewrite DIRC DLL file.
