# [A Fast Simulation Suite for Cherenkov Detectors at the future Electron Ion Collider]()

# Abstract

The integration of Deep Learning (DL) into experimental nuclear and particle physics has driven significant progress in simulation and reconstruction workflows. However, traditional simulation frameworks such as GEANT4 remain computationally intensive, especially for Cherenkov detectors, where simulating optical photon transport through complex geometries and reflective surfaces introduces a major bottleneck.
To address this, we present an open, standalone fast simulation tool for Detection of Internally Reflected Cherenkov Light (DIRC) detectors, with a focus on the High-Performance DIRC (hpDIRC) at the future Electron-Ion Collider (EIC). Our framework incorporates a suite of generative models tailored to accelerate particle identification (PID) tasks by offering a scalable, GPU-accelerated alternative to full \geant-based simulations. Designed with accessibility in mind, our simulation package enables both DL researchers and physicists to efficiently generate high-fidelity large-scale datasets on demand, without relying on complex traditional simulation stacks. This flexibility supports the development and benchmarking of novel DL-driven PID methods. Moreover, this fast simulation pipeline represents a critical step toward enabling EIC-wide PID strategies that depend on virtually unlimited simulated samples, spanning the full acceptance of the hpDIRC.

![Example Hit Patterns](assets/Overlayed_hits.png)

# Contents
- [Data used for Training](#Section-1)
- [Environment](#Section-1)
- [Available Architectures](#Section-3)
- [Usage](#Section-4)
- [Reference Packages](#Section-5)


# Data used for Training

The data used for training our generative models was created using a standalone simulation. More details can be found at [eicdirc](https://github.com/rdom/eicdirc).
Our dataset was constructed over a singular bar (no azimuthal angle variance), and without magnetic field. In future updates, we will provide generative models trained under these scenarios.

# Environment 

Noteable requirements: 

- Python:     3.12.8
- Pytorch:    2.5.1
- CUDA:       12.4
- [Framework for Easily Invertible Architectures](https://github.com/vislearn/FrEIA)
- [Flow Matching](https://github.com/facebookresearch/flow_matching)
- [NFlows](https://github.com/bayesiains/nflows)

The dependencies for the networks can be installed with the following command:

```bash
$ conda env create -f env.yml
```

In the case that some packages do not install through the provided conda command, you can install them using pip once your conda environment is activated:

```bash
$ python3 -m pip install <package>
```


# Available Architectures

Our package currently consists of 5 state-of-the-art generative models, each providing fast simulation for both Pions and Kaons in the hpDIRC:

* Discrete Normalizing Flows - Default simulation method
* Continuous Normalizing Flows
* Flow Matching
* Score-Based Generative Models
* Denoising Diffusion Probabalistic Models (DDPM)

The default simulation method provides both the fastest, and highest quality generations.

# Usage 

Note we have provided all code required to reproduce the results found within the paper, although these require large amounts of simulation from GEANT4. For those interested in training their own models, or reproducing our work, please open an issue. If their exists a high demand, we will update our documentation to and provide instructions for dataset creation, model training and model evaluation.

## Running Fast Simulation

We have provided an example script to allow simulation at both fixed kinematics, and continuously over the phase space. An example command and argument details is provided below:

```
python run_simulation.py --config config/config.json --n_tracks {} --n_dump {} --method {} --momentum {} --theta {} --model_type {} --fine_grained_prior
```

| Argument               | Type    | Default       | Description                                                              |
|------------------------|---------|---------------|--------------------------------------------------------------------------|
| `--config`             | `str`   | `config.json` | Path to the config file                                                  |
| `--n_tracks`           | `int`   | `1e5`         | Number of particles to generate                                          |
| `--n_dump`             | `int`   | `None`        | Number of particles to dump per `.pkl` file                              |
| `--method`             | `str`   | `"MixPK"`     | Generated particle type (`Kaon`, `Pion`, or `MixPK`)                     |
| `--momentum`           | `str`   | `"3"`         | Momentum to generate     (e.g., "6", "1-10")                             |
| `--theta`              | `str`   | `"30"`        | Theta angle to generate  (e.g., "30", "25-155")                          |
| `--model_type`         | `str`   | `"NF"`        | Which model to use                                                       |
| `--fine_grained_prior` | `flag`  | `False`       | Enable fine-grained prior (just include the flag to activate the option) |

## Hit Pattern Creation

We have provided an example script to generate hit patterns at fixed kinematics for viewing purposes, in which the arguments inherit from the script above for simplicity.

```
python hit_patterns.py --config config/config.json --n_tracks {} --method {} --momentum {} --theta {} --model_type {} --fine_grained_prior
```

To further automate the process for those that wish to generate various hit patterns as a function of angle we have provided *hit_patterns.sh*. Which will produce hit patterns in increments of 5 degrees, at a fixed momentum value (see script for setting of momentum.) 

# Reference Packages

The following is a list of references used in the creation of this repository.

[OT-Flow](https://github.com/EmoryMLIP/OT-Flow)
[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8)
[FastDIRC](https://github.com/jmhardin/FastDIRC)