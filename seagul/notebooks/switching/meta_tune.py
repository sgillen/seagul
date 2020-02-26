import seagul.envs
import gym

env_name = "su_acrobot-v0"

import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import numpy as np

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
config["model"]["fcnet_hiddens"] = [16, 16]
config["no_done_at_end"] = True


def reward_fn(ns, act):
    # return -1e-4*(5*(ns[0] - np.pi)**2 + ns[1]**2 + .5*ns[2]**2 + .5*ns[3]**2)
    return -1e-2 * (np.cos(ns[0]) + np.cos(ns[0] + ns[1]))
    # return -.1 * np.exp(np.sqrt(.1 * (ns[0] - np.pi) ** 2 + .1 * ns[1] ** 2 + .01 * ns[2] ** 2 + .01 * ns[3] ** 2))

env_name = "su_acrobot-v0"


config["env_config"] = {
    "max_torque" : 5,
    "init_state" : [0.0, 0.0, 0.0, 0.0],
    "init_state_weights" : np.array([0, 0, 0, 0]),
    "dt" : .01,
    "max_t" : 10,
    "act_hold" : 1,
    "reward_fn" : reward_fn
}


analysis = tune.run(
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 5e5},
    num_samples=8,
    local_dir="./data/sg_acro2/tune/trial1/",
    checkpoint_at_end=True,
)
