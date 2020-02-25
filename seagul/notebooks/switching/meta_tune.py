import seagul.envs
import gym

env_name = "su_acro_drake-v0"
env = gym.make(env_name)

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
config["model"]["fcnet_hiddens"] = [24, 24]
config["no_done_at_end"] = True

def reward_fn(ns, act):
    reward = (np.cos(ns[0]) + np.cos(ns[0] + ns[1]))
    #reward -= (abs(ns[2]) > 5)
    #reward -= (abs(ns[3]) > 10)
    #reward -= (abs(act) > 10)
    return 1e-2*reward.item()


config["env_config"] = {
    "max_torque" : 10,
    "init_state" : [0.0, 0.0, 0.0, 0.0],
    "init_state_weights" : np.array([1, 1, 5, 5]),
    "dt" : .01,
    "max_t" : 2,
    "act_hold" : 1,
    "fixed_step" : True,
    "int_accuracy" : .01,
    "reward_fn" : reward_fn 
}


analysis = tune.run(
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 5e5},
    num_samples=4,
    local_dir="./data/tune/swingdown/trial0",
    checkpoint_at_end=True,
)
