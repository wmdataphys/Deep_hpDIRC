# Fast Simulation for the hpDIRC

This repository contains all the necessary files for running the hpDIRC fast simulation. I've organized them in a way that should be easy to modify and generate new simulations based on your needs.

Note: The generations will not be performed bar-by-bar, but at fixed regions of momentum and theta. We are using two conditionals for the simulation, as it is set up with $\phi = 0$. This setup should be fine for now, and we can make use of symmetry arguments..

### **What You Will Need:**

1. **The Config File**  
   The config file contains crucial information such as scalings, data paths, and other settings. I will provide access to this file on the cluster.

2. **The Models (Freia)**  
   The models contain variables such as `allowed_x`, `allowed_y`, etc., that are used for masking and resampling operations.

3. **The `gen_thetas_hpDIRC` Shell Command**  
   I have provided three versions of the `gen_thetas_hpDIRC` shell command:
   - On the cluster, use the `.csh` or `.sh` file depending on whether you're using `tcsh` or `bash`.
   - On Windows, use the `.bat` file.

---

### **Using the `.sh` / `.tcsh` Files for Generation**

An important thing to note is that if you make a change in the .tcsh or .sh files, you need to recompile them as executables. Theta will loop in these automatically, but you will need to go in and change the momentum values. These will index specific files and generate the full theta range at that momentum:

```bash
chmod +x gen_thetas_hpDIRC.sh
```

Then you can simply do:

```bash
./gen_thetas_hpDIRC.sh 
```

In the config file (also please feel free to remove all the model paths I have in there) you will see the Inference/fixed_point_dir field. You will need to change this to whatever you want. I recommend having something that gives information of the model, and also the momentum range, e.g., SDE_V1_6GeV. This will create a folder called Generations (only once), along with another folder with the name you have given in that fixed_point_dir field. Here it will dump the generations and ground truth into .pkl files, loop over these .pkl files and create individual .png images, and then combine these into a single .pdf. These images will be inside a folder called 2DPlots.

### **Creating your own simulation**
This is more complicated. I don't think you should need to do it so I am going to save myself from detailing this but we can talk about it if you wish. Probably easier for me to run it for you.


