import numpy as np
import torch
import logging
import os
import time
from collections import deque
from functools import partial
from os import path
from collections import defaultdict
import json
from cpprb import ReplayBuffer
import gym
from sacred import Experiment
from sacred.observers import (
    FileStorageObserver,
    MongoObserver,  
    QueuedMongoObserver,
    QueueObserver,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

from model_NF import Policy
from MADPS_NF import compute_fusions, madps_ingredient
from wrappers import *

ex = Experiment(ingredients=[madps_ingredient])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver('test/'))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


import subprocess

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')

    memory_used = [int(x) for x in result.strip().split('\n')]
    return memory_used

def select_least_used_gpu():
    """Select the GPU with the least memory usage."""
    memory_map = get_gpu_memory_map()
    return f'cuda:{memory_map.index(min(memory_map))}'

@ex.config
def config(madps):
    name = "MADPS"

    env_name = None
    time_limit = None
    env_args = {}

    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
        SMACCompatible,
    )
    dummy_vecenv = True

    seed = 9999 # environment seed

    # everything below is update steps (not env steps!)
    total_steps = int(10e6)
    log_interval = int(500)
    save_interval = int(1e6)
    eval_interval = int(1e4)

    # 
    architecture = {
        "actor": [64, 64],
        "critic": [64, 64],
    }
    lr = 3e-4
    optim_eps = 0.00001

    # 
    parallel_envs = 8
    n_steps = 5
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    use_proper_termination = True
    central_v = False

    # 
    algorithm_mode = "madps" # "madps","seps", "iac", "sac", or "sac-id", 
    if_measure = True  # 


    #
    device = select_least_used_gpu() # auto-gpu or cpu
    # device = "cpu"
    # device = "cuda:0" # selected gpu



class Torcherize(VecEnvWrapper):
    @ex.capture
    def __init__(self, venv, algorithm_mode):
        super().__init__(venv)
        self.observe_agent_id = algorithm_mode == "sac-id"
        if self.observe_agent_id:
            agent_count = len(self.observation_space)
            self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=((x.shape[0] + agent_count),), dtype=x.dtype) for x in self.observation_space]))

    @ex.capture
    def reset(self, device, parallel_envs):
        obs = self.venv.reset()
        obs = [torch.from_numpy(o).to(device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(parallel_envs, 0).view(len(obs), -1, len(obs)).to(device)
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]
        return obs

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    @ex.capture
    def step_wait(self, device, parallel_envs):
        obs, rew, done, info = self.venv.step_wait()
        obs = [torch.from_numpy(o).float().to(device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(parallel_envs, 0).view(len(obs), -1, len(obs)).to(device)
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]

        return (
            obs,
            torch.from_numpy(rew).float().to(device),
            torch.from_numpy(done).float().to(device),
            info,
        )


class SMACWrapper(VecEnvWrapper):
    def _make_action_mask(self, n_agents):
        action_mask = self.venv.env_method("get_avail_actions")
        action_mask = [
            torch.tensor([avail[i] for avail in action_mask]) for i in range(n_agents)
        ]
        return action_mask

    def _make_state(self, n_agents):
        state = self.venv.env_method("get_state")
        state = torch.from_numpy(np.stack(state))
        return n_agents * [state]

    def reset(self):
        obs = self.venv.reset()
        state = self._make_state(len(obs))
        action_mask = self._make_action_mask(len(obs))
        return obs, state, action_mask

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        state = self._make_state(len(obs))
        action_mask = self._make_action_mask(len(obs))

        return (
            (obs, state, action_mask),
            rew,
            done,
            info,
        )


@ex.capture
def _compute_returns(storage, next_value, gamma):
    returns = [next_value]
    for rew, done in zip(reversed(storage["rewards"]), reversed(storage["done"])):
        ret = returns[0] * gamma + rew * (1 - done.unsqueeze(1))
        returns.insert(0, ret)

    return returns


@ex.capture
def _make_envs(env_name, env_args, parallel_envs, dummy_vecenv, wrappers, time_limit, seed):
    def _env_thunk(seed):

        env = gym.make(env_name, **env_args)
        if time_limit:
            env = TimeLimit(env, time_limit)
        for wrapper in wrappers:
            env = wrapper(env)
        env.seed(seed)
        return env

    env_thunks = [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    if dummy_vecenv:
        envs = DummyVecEnv(env_thunks)
        envs.buf_rews = np.zeros(
            (parallel_envs, len(envs.observation_space)), dtype=np.float32
        )
    else:
        envs = SubprocVecEnv(env_thunks, start_method="fork")
    envs = Torcherize(envs)
    envs = SMACWrapper(envs)
    return envs


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


@ex.capture
def _log_progress(
    infos,
    prev_time,
    step,
    parallel_envs,
    n_steps,
    total_steps,
    log_interval,
    _log,
    _run,
):

    elapsed = time.time() - prev_time
    ups = log_interval / elapsed
    fps = ups * parallel_envs * n_steps
    mean_reward = sum(sum([ep["episode_reward"] for ep in infos]) / len(infos))
    battles_won = 100 * sum([ep.get("battle_won", 0) for ep in infos]) / len(infos)

    _log.info(f"Updates {step}, Environment timesteps {parallel_envs* n_steps * step}")
    _log.info(
        f"UPS: {ups:.1f}, FPS: {fps:.1f}, ({100*step/total_steps:.2f}% completed)"
    )

    _log.info(f"Last {len(infos)} episodes with mean reward: {mean_reward:.3f}")
    _log.info(f"Battles won: {battles_won:.1f}%")
    _log.info("-------------------------------------------")

    squashed_info = _squash_info(infos)
    for k, v in squashed_info.items():
        _run.log_scalar(k, v, step)


@ex.capture
def _compute_loss(model, storage, value_loss_coef, entropy_coef, central_v):
    with torch.no_grad():
        next_value = model.get_value(storage["state" if central_v else "obs"][-1])
    returns = _compute_returns(storage, next_value)

    input_obs = zip(*storage["obs"])
    input_obs = [torch.stack(o)[:-1] for o in input_obs]

    if central_v:
        input_state = zip(*storage["state"])
        input_state = [torch.stack(s)[:-1] for s in input_state]
    else:
        input_state = None

    input_action_mask = zip(*storage["action_mask"])
    input_action_mask = [torch.stack(a)[:-1] for a in input_action_mask]

    input_actions = zip(*storage["actions"])
    input_actions = [torch.stack(a) for a in input_actions]

    values, action_log_probs, entropy, act_probs = model.evaluate_actions(
        input_obs, input_actions, input_action_mask, input_state,
    )

    returns = torch.stack(returns)[:-1]
    advantage = returns - values

    actor_loss = (
        -(action_log_probs * advantage.detach()).sum(dim=2).mean()
        - entropy_coef * entropy
    )
    value_loss = (returns - values).pow(2).sum(dim=2).mean()

    loss = actor_loss + value_loss_coef * value_loss
    return loss




@ex.automain
def main(
    _run,
    seed,
    total_steps,
    log_interval,
    save_interval,
    eval_interval,
    architecture,
    lr,
    optim_eps,
    parallel_envs,
    n_steps,
    use_proper_termination,
    central_v,

    madps,
    algorithm_mode,
    env_name,

    if_measure,

    device,
    _log,
):
    # make environments
    torch.set_num_threads(1)
    envs = _make_envs(seed=seed)

    customized_representation_learning = madps['customized_representation_learning']
    customized_feature_size = madps['customized_feature_size'] if customized_representation_learning else None

    agent_count = len(envs.observation_space)
    state_size = envs.get_attr("state_size")[0] if central_v else None
    obs_size = envs.observation_space[0].shape
    obs_size = obs_size[0] - customized_feature_size if customized_representation_learning else obs_size
    act_size = envs.action_space[0].n

    # make sample buffer
    env_dict = {
        "obs": {"shape": obs_size, "dtype": np.float32},
        "hidden": {"shape": obs_size, "dtype": np.float32},
        "rew": {"shape": 1, "dtype": np.float32},
        "next_obs": {"shape": obs_size, "dtype": np.float32},
        "done": {"shape": 1, "dtype": np.float32},
        "act": {"shape": act_size, "dtype": np.float32},
        "act_probs": {"shape": act_size, "dtype": np.float32},
        "act_probs_with_mask": {"shape": act_size, "dtype": np.float32},
        "act_mask": {"shape": act_size, "dtype": np.float32},
        "agent": {"shape": agent_count, "dtype": np.float32},
        # "customized_feature": {"shape": customized_feature_size, "dtype": np.float32}, 
        # The above line needs to be commented when running without customized_feature
    }
    rb = ReplayBuffer(int(agent_count * madps['max_rb_steps'] * parallel_envs * n_steps), env_dict) 

    # set model count
    if algorithm_mode.startswith("sac"):
        model_count = 1
    elif algorithm_mode == "iac":
        assert madps["model_count"]
        model_count = madps["model_count"]
    elif algorithm_mode == "seps":
        if madps["clusters"] is not None:
            model_count = madps["clusters"]
        else:
            model_count = madps["model_count"]
    elif algorithm_mode == "madps":
        assert madps["model_count"]
        model_count = madps["model_count"]
    
    # make actor-critic model
    if customized_representation_learning:
        obs_shape = [obs_size for i in range(agent_count)]
        act_shape = [act_size for i in range(agent_count)]
        model = Policy(None, None, architecture, model_count, state_size, initial_as_the_same=madps['initial_as_the_same'],
                   obs_shape=obs_shape, act_shape=act_shape)
    else:
        model = Policy(envs.observation_space, envs.action_space, architecture, model_count, state_size, initial_as_the_same=madps['initial_as_the_same'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, eps=optim_eps)

    # creates and initialises storage
    obs, state, action_mask = envs.reset()

    if customized_representation_learning:
        obs = [tensor[:, :-customized_feature_size] for tensor in obs]
        customized_feature = [tensor[:, -customized_feature_size:] for tensor in obs]

    storage = defaultdict(lambda: deque(maxlen=n_steps))
    storage["obs"] = deque(maxlen=n_steps + 1)
    storage["done"] = deque(maxlen=n_steps + 1)
    storage["obs"].append(obs)
    storage["done"].append(torch.zeros(parallel_envs))
    storage["info"] = deque(maxlen=10)

    # for SMAC:
    storage["state"] = deque(maxlen=n_steps + 1)
    storage["action_mask"] = deque(maxlen=n_steps + 1)
    if central_v:
        storage["state"].append(state)
    storage["action_mask"].append(action_mask)

    model.sample_laac(parallel_envs)
    if algorithm_mode == "iac":  # independent actor-critic
        model.laac_shallow = torch.arange(len(envs.observation_space)).repeat(parallel_envs, 1)
        model.laac_deep = torch.arange(len(envs.observation_space)).repeat(parallel_envs, 1)

    if algorithm_mode.startswith("sac"): # fully-shared actor-critic (with/out id)
        model.laac_shallow = torch.zeros(parallel_envs, agent_count).long() 
        model.laac_deep = torch.zeros(parallel_envs, agent_count).long() 

    if algorithm_mode == "madps": # automatic-shared actor-critic
        model.laac_shallow = torch.zeros(parallel_envs, agent_count).long()
        model.laac_deep = torch.arange(len(envs.observation_space)).repeat(parallel_envs, 1)
    
    if algorithm_mode == "seps":
        model.laac_shallow = torch.zeros(parallel_envs, agent_count).long()
        model.laac_deep = torch.zeros(parallel_envs, agent_count).long()

    # print the parameter sharing scheme
    print(model.laac_shallow)
    print(model.laac_deep)

    start_time = time.time()
    # train and record
    for step in range(total_steps):

        if (algorithm_mode == "madps" or if_measure) and step in [madps["delay"] + madps["reparameter_steps"]*(i+1) for i in range(madps["reparameter_times"])]:
            print(f"Reparameter at step: {step}")
            only_measure = False if algorithm_mode == "madps" else True

            BD, Hellinger,WD, laac_s, laac_d = compute_fusions(rb.get_all_transitions(), agent_count, model, only_measure=only_measure)
            
            # print the distances (in MAPD and MADPS, we only recommend to use WD)
            (BD, Hellinger, WD) = (BD.cpu().numpy(),Hellinger.cpu().numpy(),WD.cpu().numpy())
            (avg_BD, avg_Hellinger, avg_WD) = (BD.mean().item(), Hellinger.mean().item(), WD.mean().item())

            _log.info(model.laac_shallow)
            _log.info(model.laac_deep)

        if step % log_interval == 0 and len(storage["info"]):
            _log_progress(storage["info"], start_time, step)
            start_time = time.time()
            storage["info"].clear()

        for n_step in range(n_steps):
            with torch.no_grad():
                actions, act_probs_with_mask = model.act(storage["obs"][-1], storage["action_mask"][-1])
                act_probs = model.get_act_probs(storage["obs"][-1])
            (obs, state, action_mask), reward, done, info = envs.step(actions)

            if customized_representation_learning:
                obs = [tensor[:, :-customized_feature_size] for tensor in obs]
                customized_feature = [tensor[:, -customized_feature_size:] for tensor in obs]

            if use_proper_termination:
                bad_done = torch.FloatTensor(
                    [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
                ).to(device)
                done = done - bad_done

            storage["obs"].append(obs)
            storage["actions"].append(actions)
            storage["rewards"].append(reward)
            storage["done"].append(done)
            storage["info"].extend([i for i in info if "episode_reward" in i])
            storage["laac_rewards"] += reward

            if (algorithm_mode == "madps" or algorithm_mode == "seps" or if_measure) and step < (madps["delay"] + madps["reparameter_times"] * madps["reparameter_steps"]):
                for agent in range(len(obs)):

                    one_hot_action = torch.nn.functional.one_hot(actions[agent], act_size).squeeze().cpu().numpy()
                    one_hot_agent = torch.nn.functional.one_hot(torch.tensor(agent), agent_count).repeat(parallel_envs, 1).numpy()

                    if bad_done[0]:
                        nobs = info[0]["terminal_observation"]
                        nobs = [torch.tensor(o).unsqueeze(0) for o in nobs]
                        if customized_representation_learning:
                            nobs = [tensor[:, :-customized_feature_size] for tensor in nobs]
                    else:
                        nobs = obs
                        
                    data = {
                        "obs": storage["obs"][-2][agent].cpu().numpy(),
                        "hidden": storage["obs"][-2][agent].cpu().numpy(), 
                        "act_mask": action_mask[agent].cpu().numpy(),  
                        "act": one_hot_action,
                        "act_probs": act_probs[agent].cpu().numpy(),  
                        "act_probs_with_mask": act_probs_with_mask[agent].cpu().numpy(),  
                        "next_obs": nobs[agent].cpu().numpy(),
                        "rew":  reward[:, agent].unsqueeze(-1).cpu().numpy(),
                        "done": done[:].unsqueeze(-1).cpu().numpy(),
                        # "policy": np.array([model.laac_sample[0, agent].float().item()]),
                        "agent": one_hot_agent,
                        # "customized_feature" : customized_feature[agent].cpu().numpy(), # this line need to be 注释掉当跑常规的时候
                    }
                    rb.add(**data)
                    assert True

            # for smac:
            if central_v:
                storage["state"].append(state)

            storage["action_mask"].append(action_mask)
            # ---------

        if algorithm_mode == "seps" and step < madps["reparameter_steps"] and madps["delay_training_seps"]:
            continue

        loss = _compute_loss(model, storage)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


    envs.close()
