import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import seagul.envs

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 39
config["lambda"] = .2
config["gamma"] = .95
config["num_gpus"] = 0
config["eager"] = False
config["model"]["free_log_std"] = True
config["lr"] =  .0001
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 20
config["batch_mode"] = "truncate_episodes"
config['observation_filter'] = 'MeanStdFilter'
config["sgd_minibatch_size"] = 512
#config["train_batch_size"] = tune.sample_from(lambda spec: spec.config.sgd_minibatch_size*32)
config["train_batch_size"] = 512*32
config["vf_clip_param"] = 30
config["seed"] = tune.grid_search([0,1,2,3]) #
#env_name = "Walker2d-v3"
env_name =  "Walker2DBulletEnv-v0"
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
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 6e7},
    local_dir="./data/bullet_walker2/",
    checkpoint_at_end=True,
)

