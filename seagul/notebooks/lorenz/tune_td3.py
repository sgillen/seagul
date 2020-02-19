import ray
from ray import tune
import ray.rllib.agents.ddpg.td3 as td3
import seagul.envs

config = td3.DDPG_CONFIG.copy()
config["num_workers"] = 1
config["batch_mode"] = "truncate_episodes"
config["observation_filter"] = "MeanStdFilter"
env_name = "linear_z-v0"
config["env"] = env_name
config["model"]["fcnet_hiddens"] = [32, 32]
config["actor_hiddens"] =  [32, 32]
config["critic_hiddens"] = [32, 32]
config["no_done_at_end"] =  True

def reward_fn(s):
    if s[3] > 0:
        if s[0] > 2 and s[2] > 3:
            reward = 5.0
            s[3] = -10
        else:
            reward = 0.0

    elif s[3] < 0:
        if s[0] < -2 and s[2] < -3:
            reward = 5.0
            s[3] = 10
        else:
            reward = 0.0

    return reward, s

config["env_config"]["reward_fn"] = reward_fn
config["env_config"]["num_steps"] = 100
config["env_config"]["act_hold"] = 10
config["env_config"]["xyz_max"] = float('inf')



# import pprint
# pprint.pprint(config)

analysis = tune.run(
    td3.TD3Trainer,
    config=config,
    stop={"timesteps_total": 5e5, "time_total_s": 1800}, #900s == 15m
    num_samples=4,
    local_dir="./data/tune/box_reward",
    checkpoint_at_end=True,
)
