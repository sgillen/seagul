# %%

import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG
from seagul.envs.mujoco.five_link import FiveLinkWalkerEnv
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
from ray.tune.registry import register_env
import pybullet_envs
from tqdm import trange


def five_link_creator(env_config):
    return FiveLinkWalkerEnv()  # return an env instance


def bullet_walker_creator(env_config):
    return Walker2DBulletEnv()  # return an env instance


register_env("five_link-v3", five_link_creator)
register_env("Walker2DBulletEnv-v0", bullet_walker_creator)


def train_ppo(config, env_name):
    trainer = ppo.PPOTrainer(config=config, env=env_name)
    for i in trange(500):
        result = trainer.train()

    checkpoint = trainer.save()


# %%

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 15
config["num_gpus"] = tune.grid_search([0, 1])

config["eager"] = False
config["model"]["fcnet_hiddens"] = [64, 64]
config["lr"] = 5e-5
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 20
config["batch_mode"] = "complete_episodes"
config["vf_clip_param"] = 100.0
config["observation_filter"] = "MeanStdFilter"
config["sgd_minibatch_size"] = 2048
config["train_batch_size"] = 20480

ray.init()

# config["env"] = "five_link-v3"
config["env"] = "Walker2DBulletEnv-v0"

analysis = tune.run(
    "PPO", config=config, stop={"timesteps_total": 32e6}, local_dir="./data/batch_grid2/", checkpoint_at_end=True
)
