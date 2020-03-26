import ray
from ray import tune
import ray.rllib.agents.sac as sac
import seagul.envs

config = sac.DEFAULT_CONFIG.copy()
env_name = "Humanoid-v2"
# env_name = "sym_pendulum-v0"#
# env_name = "dt_pendulum-v0"
# env_name  = "Pendulum-v0"
# env_name =  "Walker2DBulletEnv-v0"


config["env"] = env_name
# config["eager"] = True
# config["num_cpus_per_worker"] = 15
# config["num_envs_per_worker"]  = 15
# config["train_batch_size"] = 256
config["learning_starts"] = 4096
# config["learning_starts"] =  10000
# config["pure_exploration_steps"] =  10000
# config["evaluation_interval"] = 5
# config["evaluation_num_episode"] = 10
config["use_state_preprocessor"] = True
config["seed"] = tune.grid_search([0, 1, 2, 3])
config["evaluation_interval"] = 1
config["soft_horizon"] = True
config["metrics_smoothing_episodes"] = 10

# config["log_level"] = "DEBUG"

analysis = tune.run(
    sac.SACTrainer, config=config, stop={"timesteps_total": 2e7}, local_dir="./data/sac_hum/", checkpoint_at_end=True
)
