import ray.rllib.agents.ppo as ppo
from tqdm import trange
import math
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG
from seagul.envs.mujoco.five_link import FiveLinkWalkerEnv
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
from ray.tune.registry import register_env

import pybullet_envs
import seagul.envs


from sym_ppo_policy import PPOTFSymPolicy

def five_link_creator(env_config):
    return FiveLinkWalkerEnv()  # return an env instance


def bullet_walker_creator(env_config):
    return Walker2DBulletEnv()  # return an env instance

register_env("five_link-v3", five_link_creator)
register_env("Walker2DBulletEnv-v0", bullet_walker_creator)



config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 8
config["num_gpus"] = 0

config["eager"] = False
config["model"]["fcnet_hiddens"] = [64, 64]
config["lr"] = 5e-5
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 20
config["batch_mode"] = "complete_episodes"
config['vf_clip_param'] = 100.0
config['observation_filter'] = 'MeanStdFilter'
config["sgd_minibatch_size"] = 2048
config["train_batch_size"] = 20480

env_name = "Walker2DBulletEnv-v0"

config["env"] = env_name



import pprint

pprint.pprint(config)



ray.init()
PPOSymTrainer = ppo.PPOTrainer.with_updates(name="SymPPO", default_policy = PPOTFSymPolicy)
trainer = PPOSymTrainer(config = config, env=env_name)

#trainer = ppo.PPOTrainer(config=config, env=env_name)
# Can optionally call trainer.restore(path) to load a checkpoint.

num_steps = 16e6
for i in trange(math.ceil(num_steps/config["train_batch_size"])):
   # Perform one iteration of training the policy with ARS
    result = trainer.train()

    print()
    print("Reward Mean: ", result['policy_reward_mean'])
    print()
    

print(trainer.collect_metrics())

import ipdb; ipdb.set_trace()
