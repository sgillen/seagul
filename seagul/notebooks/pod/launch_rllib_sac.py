import ray
from ray import tune
import ray.rllib.agents.sac as sac
import seagul.envs

config = sac.DEFAULT_CONFIG.copy()
config["num_workers"] = 0
config["horizon"] = .1000
config["soft_horizon"] = False
config["num_gpus"] = 0
config["Q_model"]["hidden_activation"] = "relu"
config["Q_model"]["hiden_layer_sizes"] = [256, 256]
config["policy_model"]["hidden_activaton"] = "relu"
config["policy_model"]["hidden_layer_sizes"] = [256, 256]
config["tau"] = 0.005
config["target_entropy"] = "auto"
config["no_done_at_end"] = True
config["n_step"] = 1
config["sample_batch_size"] = 1
config["prioritized_replay"] = False
config["eager"] = False
config["train_batch_size"] = 256
config["target_network_update_freq"] = 1 
config["timesteps_per_iteration"] = 1000
config["learning_starts"] = 10000
config["exploration_enabled"] = True
config["optimization"]["actor_learning_rate"] = 0.0003
config["optimization"]["critic_learning_rate"] = 0.0003
config["optimization"]["entropy_learning_rate"] = 0.0003
config["clip_actions"] =  False
#config["normalize_actions"] = True
config["evaluation_interval"] = 1
config["metrics_smoothing_episodes"] = 5

#env_name = "Walker2d-v3"
#env_name =  "Walker2DBulletEnv-v0"
env_name = "HalfCheetahBulletEnv-v0"
#env_name =   "HumanoidBulletEnv-v0"
#env_name = "Humanoid-v2"
#env_name = "humanoid_long-v1"
#env_name  = "Pendulum-v0"
#env_name = "sym_pendulum-v0"#
#env_name = "dt_pendulum-v0"
#env_name = "sg_cartpole-v0"
config["env"] = env_name  
#import pprint
#pprint.pprint(config)

analysis = tune.run(
    sac.SACTrainer,
    config=config,
    stop={"episode_reward_mean" : 3000},
    local_dir="./data/bullet_walker/",
    checkpoint_at_end=True,
)

