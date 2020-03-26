import ray
from ray import tune
import ray.rllib.agents.ppo as ppo

from seagul.rllib.register_envs import register_all_envs

register_all_envs()

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 8
config["num_gpus"] = 0
config["eager"] = False
config["model"]["fcnet_hiddens"] = [64, 64]
config["lr"] = 0.0001
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 20
config["batch_mode"] = "complete_episodes"
config["vf_clip_param"] = 10.0
config["observation_filter"] = "MeanStdFilter"
config["sgd_minibatch_size"] = tune.grid_search([1024, 2048, 4096, 8192])
config["train_batch_size"] = tune.sample_from(lambda spec: spec.config.sgd_minibatch_size * 15)

# env_name = "Walker2d-v3"
env_name = "Walker2DBulletEnv-v0"
# env_name =   "HumanoidBulletEnv-v0"

config["env"] = env_name


# import pprint
# pprint.pprint(config)

from seagul.rllib.sym_ppo_policy import PPOTFSymPolicy

# PPOSymTrainer = ppo.PPOTrainer.with_updates(name="SymPPO", default_policy = PPOTFSymPolicy)

analysis = tune.run(
    # PPOSymTrainer,
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 8e6},
    local_dir="./data/walker_14_tune/",
    checkpoint_at_end=True,
)
