import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import seagul.envs

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 10
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
config["train_batch_size"] = 1024*10
config["vf_clip_param"] = 10
config["seed"] = tune.grid_search([2, 3, 4, 5])  #
env_name = "linear_z-v0"
config["env"] = env_name
config["env_config"]["xyz_max"] = float("inf")
config["model"]["fcnet_hiddens"] = [64, 64]


# import pprint
# pprint.pprint(config)

analysis = tune.run(
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 5e4},
    local_dir="./data/tune/",
    checkpoint_at_end=True,
)
