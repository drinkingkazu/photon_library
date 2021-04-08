# Learning an optical model (look-up table)

This repository contains utility tools to attack a challenge of learning an "_optical model_" for Liquid Argon Time Projection Chambers.

If you would like to jump onto a code example, refer to [this page](https://web.stanford.edu/~kterao/HowTo.html) or try running `HowTo.ipynb` with `jupyter` (not in the `requirements.txt`).
## Background
Due to complex geometry and a time-consuming ray tracing procedure, many experiments do not simulate every single step of photon propagation process in a detector for every physics simulation. 
Instead, in many cases, experiments use a "look up table", which contains a probability for a photon produced at any position to be detected by an optical detector.
This look-up table is produced by running a huge campaign of simulation once, in which a large number of photons are produced at a fixed location inside a detector to measure how much of them may be detected by each optical detector.
The location of producing photons is sampled uniformly in the rectangular grids ("voxels") defined along the axis of the cartesian coordinate (x,y,z).
The result is a look-up table, which records a 1D array of probability (length = number of optical detectors) per voxelized position in the detector.

In a specific instance of ICARUS experiment, the number of voxels are (74,77,394) along (x,y,z), or 2245012 voxels total.

## What's in here?
`photon_library.py` contains a Python class API to interact with the look up table. Please see [this page](https://web.stanford.edu/~kterao/HowTo.html) for a quick instructions or try running `HowTo.ipynb` with `jupyter` (not in the `requirements.txt`).


## Challenge

**Goal** of the challenge is to come up with a learnable model that can accurately reproduce the knowledge (i.e. visibility per optical detector) of a PhotonLibrary. 
