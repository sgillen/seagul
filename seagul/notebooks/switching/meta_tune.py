import seagul.envs
import gym

import ray
from ray import tune
import torch
import ray.rllib.agents.ppo as ppo
import numpy as np
from numpy import pi

m1 = 1; m2 = 1
l1 = 1; l2 = 1
lc1 = .5; lc2 = .5
I1 = .2; I2 = 1.0
g = 9.8
max_torque = 5
max_t = 10.0

trial_num = input("trial name / number please:\n")

def control(q):
    k = np.array([[-1649.86567367, -460.15780461, -716.07110032, -278.15312267]])
    gs = np.array([pi / 2, 0, 0, 0])
    return -k.dot(q - gs)

def reward_fn(s, a):
    reward = -.1*((s[0] - pi/2)**2 + .25*s[1]**2)
    #reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

def reward_fn_sin(s,a):
    reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

env_name = "su_acroswitch-v0"
config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 1
config["num_envs_per_worker"] = 1
config["lambda"] = 0.2
config["gamma"] = 0.95
config["num_gpus"] = 0
config["eager"] = False
config["model"]["free_log_std"] = True
config["lr"] = 0.0001
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 10
config["batch_mode"] = "truncate_episodes"
config["observation_filter"] = "MeanStdFilter"
config["sgd_minibatch_size"] = 512
# config["train_batch_size"] = tune.sample_from(lambda spec: spec.config.sgd_minibatch_size*32)
config["train_batch_size"] = 2048
config["vf_clip_param"] = 10
config["env"] = env_name
config["model"]["fcnet_hiddens"] = [32]
config["no_done_at_end"] = True


config["env_config"] = {
    "init_state": [-pi/2, 0, 0, 0],
    "max_torque": max_torque,
    "init_state_weights": [0, 0, 0, 0],
    "dt": .01,
    "reward_fn" : reward_fn_sin,
    "max_t" : max_t,
    "m2": m2,
    "m1": m1,
    "l1": l1,
    "lc1": lc1,
    "lc2": lc2,
    "i1": I1,
    "i2": I2,
    "act_hold" : 20,
    "gate_fn" : torch.load("warm/lqr_gate_better"),
    "controller" : control
}

ray.init(
    num_cpus=12,
    memory=int(8e9),
    object_store_memory=int(4e9),
    driver_object_store_memory= int(2e9)
)

analysis = tune.run(
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 5e6},
    num_samples=4,
    local_dir="./data6/tune/switch/trial" + trial_num + "/" ,
    checkpoint_at_end=True,
)
