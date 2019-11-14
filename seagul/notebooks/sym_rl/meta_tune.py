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

from seagul.rllib.sym_ppo_policy import PPOTFSymPolicy
from seagul.rllib.register_envs import register_all_envs

register_all_envs()

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 15
config["num_gpus"] = 0

config["eager"] = False
config["model"]["fcnet_hiddens"] = [64, 64]
config["lr"] =  .0001
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 20
config["batch_mode"] = "complete_episodes"
config['vf_clip_param'] = 50.0
config['observation_filter'] = 'MeanStdFilter'
config["sgd_minibatch_size"] = 8192
config["train_batch_size"] = 80000

env_name =  "Walker2DBulletEnv-v0"
#env_name =   "HumanoidBulletEnv-v0"
config["env"] = env_name  

#import pprint
#pprint.pprint(config)

PPOSymTrainer = ppo.PPOTrainer.with_updates(name="SymPPO", default_policy = PPOTFSymPolicy)

analysis = tune.run(
    PPOSymTrainer,
#    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 32e6},
    local_dir="./data/mirror_walker/",
    checkpoint_at_end=True,
)
