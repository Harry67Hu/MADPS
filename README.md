# Multi-agent Dynamic Parameter Sharing

Official implementation of the AAMAS 2024 paper Measruing Policy Distance for MARL.

## Requirements

To run this repository you need:

i) Install the code's requirements. A virtual environment based on Conda is recommended (We will update the docker approach soon). Install with 
```setup
conda create --name madps --file requirements.txt
```
ii) Install the supported MARL-environments, for example:
- Multi-agent Spread of [Petting-Zoo Version](https://github.com/semitable/PettingZoo)
(Careful! In our research, we have upgraded the task's difficulty level. Replace the file directory at `madps/large_spread_example.py` with `pettingzoo/mpe/scenarios/large_spread.py` to access the updated task.



## Running

You can then simply run ac_NF.py using:
```python
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v1' time_limit=50
```
This command runs our multi-agent actor-critic training framework, with `pettingzoo:pz-mpe-large-spread-v1` as the training scenario and the maximum episode step length as 50. The scenario versions, from large-spread-`v1` to `v6`, correspond to the `15a_3c`, \ `30a_3c`, \  `30a_5c`, \ `30a_5c_super`, \ `15a_3c_shuffle`, and  `30a_3c_shuffle` scenarios in our paper.




## Structure of MADPS
The MADPS code is structured as follows:
### 1. Actor-Critic Training and Execution Framework (ac_NF.py)

ac_NF.py includes:
- Multi-agent Environment Construction and Maintenance.
- Environment Sampling and Sample Pool Construction.
- Neural Network Evaluating and Training.

More details can be found in the comments of main function in ac_NF.py.

### 2. Multi-agent Policy Distance Computing and Multi-agent Dynamic Parameter Sharing (MADPS_NF.py)

MADPS_NF.py includes:

- `compute_fusions` (including MAPD and MADPS):
    1. Train Conditional VAE (Learning the conditional representations of agents' decisions).
    2. Use VAE and agents' models to calculate dij（Compute the multi-agent policy distance matrix using the learned conditional representations).
    3. Use dij to automatically adjusting parameter sharing（Optional. Using the the multi-agent policy distance matrix to adjust the parameter sharing scheme of the agents).
- `calculate_N_Gaussians_BD`: This function is used for parallel calculation of the Bhattacharyya distance between multiple Gaussian distributions using PyTorch.
- `calculate_N_Gaussians_Hellinger_through_BD`: This function is used for parallel calculation of the Hellinger distance between multiple Gaussian distributions using PyTorch. It requires results from the `calculate_N_Gaussians_BD` function.
- `calculate_N_Gaussians_WD`: This function is used for the parallel calculation of the Wasserstein distance between multiple Gaussian distributions using PyTorch.

### 3. Multi-agent Nerual Network Models (model_NF.py)

model_NF.py includes:
- `MADPSNet`: Multi-agent neural network models that support dynamic adjustment and hierarchical adjustment of parameter sharing.
- `MultiAgentFCNetwork`: Multi-agent neural network models that support adjustment of parameter sharing (Replication of [SePS](https://proceedings.mlr.press/v139/christianos21a/christianos21a.pdf) algorithm).
- `Policy`: Multi-agent policy models.
- `ConditionalVAE`: Conditional VAE model.



<!-- ### Hyperparameters 
ac_NF.py: This is the main file for training and evaluating the agent. It contains the following functions:



This repository is an implementation of SePS and not necessarily the identical code that generated the experiments (i.e. small bug-fixes, features, or improved hyperparameters _will_ be contained in this repository). Any major bug fixes (if necessary) will be documented below. 

Right now, the proposed hyperparameters can be found in the config function of `ops_utils.py` (for the SePS procedure) and for `ac.py` (A2C hyperparameters).
For SePS (`ops_utils.py`):
- `pretraining_steps`: always returns stabler results when increased. Usually works with as low as 2,000, but should be increased to 10,000 if time/memory allows. Simpler environments (like BPS) can handle much smaller values.
- `clusters`: the number of clusters (K in the paper). Please see respective section in the paper. When # of clusters is unknown Davies–Bouldin index is recommended (scikit-learn function [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) works well). DB-Index will be used automatically if `clusters=None`.
- `z_features`: Typical values of ~5-10 work well with the tested number of types. If this value is set to 2 then a visualisation will be saved at "clusters.png"
- `kl_weight`: Values near zero work well since it tends to overwhelm the reconstruction loss. Try setting to zero when debugging.
- `reconstruct`: Could be changed from `["next_obs", "rew"]` to only `["next_obs"]` or `["rew"]` if it known that the environment only changes the observation or reward function respectively (and not both).
- `delay/delay_training/pretraining_times`: can be used in situation when differences between types are shown later in the training.

In `ac.py`:
- "algorithm_mode": Can be set to "ops", "iac", "snac-a", "snac-b". These values correspond to "SePS", "NoPS", "FuPS", "FuPS+id" respectively. -->



# Cite:

The correct version is coming soon.
```
@inproceedings{tianyihu2024MAPD,
   title={Measruing Policy Distance for Multi-agent Reinforcement Learning},
   author={Tianyi Hu et.al},
   booktitle={International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
   year={2024}
}
```
