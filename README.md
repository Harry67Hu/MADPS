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

Note: More details are coming soon.

# Cite:

The paper can be quickly accessed via the [arxiv link](https://arxiv.org/pdf/2401.11257.pdf).
After the AAMAS 2024 conference, please cite as follows: 
```
@inproceedings{hu2024MAPD,
  title={Measuring Policy Distance for Multi-Agent Reinforcement Learning},
  author={Hu, Tianyi and Pu, Zhiqiang and Ai, Xiaolin and Qiu, Tenghai and Yi, Jianqiang},
  booktitle={Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
  pages={834--842},
  year={2024}
}
```
Note: Since there is no appendix on the AAMAS official link, I put the appendix of the paper in this repository
