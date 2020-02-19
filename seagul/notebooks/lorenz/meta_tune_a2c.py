import ray
from ray import tune
import ray.rllib.agents.a3c.a3c as a3c
import seagul.envs

config = a3c.DEFAULT_CONFIG.copy()
env_name = "linear_z-v0"


config["env"] = env_name
config["model"]["fcnet_hiddens"] = [64, 64]

# config["log_level"] = "DEBUG"

analysis = tune.run(
    a3c.A3CTrainer, config=config, stop={"timesteps_total": 5e5}, local_dir="./data/a2c/", checkpoint_at_end=True
)
