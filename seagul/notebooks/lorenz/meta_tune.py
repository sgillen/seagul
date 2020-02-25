import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import seagul.envs
from seagul.integration import euler,rk4

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
env_name = "linear_z-v0"
config["env"] = env_name
config["env_config"]["xyz_max"] = float("inf")
config["model"]["fcnet_hiddens"] = [16, 16]
config["no_done_at_end"] = True

def reward_fn(s):
    if s[3] > 0:
        if 12 > s[0] > 2 and 13 > s[2] > 3:
            reward = 5.0
            s[3] = -10
        else:
            reward = 0.0

    elif s[3] < 0:
        if -12 < s[0] < -2 and -13 < s[2] < -3:
            reward = 5.0
            s[3] = 10
        else:
            reward = 0.0

    return reward, s

config["env_config"]["reward_fn"] = reward_fn
config["env_config"]["num_steps"] = 300
config["env_config"]["xyz_max"] = float('inf')
config["env_config"]["integrator"] = euler
config["env_config"]["act_hold"] = 1
config["env_config"]["dt"] = .01

# import pprint
# pprint.pprint(config)

analysis = tune.run(
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 8e6},
    num_samples=4,
    local_dir="./data/tune/euler_250_long",
    checkpoint_at_end=True,
)
