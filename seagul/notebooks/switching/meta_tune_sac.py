import ray
from ray import tune
import ray.rllib.agents.sac as sac
import seagul.envs
import numpy as np

config = sac.DEFAULT_CONFIG.copy()
env_name = "su_acrobot-v0"


def reward_fn(ns, act):
    reward = -1e-2 * (np.cos(ns[0]) + np.cos(ns[0] + ns[1]))

    done = False
    if abs(ns[0] - np.pi) < 1 and abs(ns[1]) < 1 and abs(ns[2]) < 3 and abs(ns[3]) < 3:
        reward += 10
        print("go zone")
        done = True

    return reward, done



config["env"] = env_name
# config["eager"] = True
# config["num_cpus_per_worker"] = 15
# config["num_envs_per_worker"]  = 15
# config["train_batch_size"] = 256
config["learning_starts"] = 10000
# config["learning_starts"] =  10000
# config["pure_exploration_steps"] =  10000
# config["evaluation_interval"] = 5
# config["evaluation_num_episode"] = 10
config["use_state_preprocessor"] = True
config["evaluation_interval"] = 1
config["soft_horizon"] = True
config["metrics_smoothing_episodes"] = 10
config["env_config"] = {
    "max_torque" : 5,
    "init_state" : [0.0, 0.0, 0.0, 0.0],
    "init_state_weights" : np.array([0, 0, 0, 0]),
    "dt" : .01,
    "max_t" : 10,
    "act_hold" : 1,
    "reward_fn" : reward_fn
}

trial_num = input("which trial is this?\n")

analysis = tune.run(
    sac.SACTrainer,
    config=config,
    stop={"timesteps_total": 5e5},
    num_samples=4,
    local_dir="./data/sg_acro2/tune_sac/trial1/",
    checkpoint_at_end=True,
)



