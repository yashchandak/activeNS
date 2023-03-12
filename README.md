# Off-Policy Evaluation for Action-Dependent Non-stationary Environments


This repository contains the code for the following paper.

> Chandak Y, Shankar S, Bastian ND, da Silva BC, Brunskil E, Thomas PS. Off-Policy Evaluation for Action-Dependent Non-Stationary Environments. In Advances in Neural Information Processing Systems, 2022. 


## Requirements

The code for environments and agents which are used for collecting trajectories, as well as the code for doing off-policy evlauation are written in Python (+ Pytorch). 


## Data Collection

The following files  contain python code to collect data:
- CollectData/run_Diabetes.py
- CollectData/run_Maze.py
- CollectData/run_Reco.py
- CollectData/run_Reco.py

Note that data was collected on the cluster with additional wrapper code for SLURM.
Code for that is available in /Swarm.

## OPEN Estimator + Baselines

Code for estimating the future performance:
- OPE/hybrid.py 
- OPE/passive.py 
- OPE/stationary.py 


## Plots

To produce the plots in the paper, use:
- Plots/plotter.ipynb


## Bibliography

```bash
@article{chandak2023off,
  title={Off-Policy Evaluation for Action-Dependent Non-Stationary Environments},
  author={Chandak, Yash and Shankar, Shiv and Bastian, Nathaniel D and da Silva, Bruno Castro and Brunskil, Emma and Thomas, Philip S},
  journal={arXiv preprint arXiv:2301.10330},
  year={2023}
}
```
