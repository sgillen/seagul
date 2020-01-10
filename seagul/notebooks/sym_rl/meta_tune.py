import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import seagul.envs

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 0
config["num_envs_per_worker"] = 10
config["lambda"] = 0.1
config["gamma"] = 0.95
config["num_gpus"] = 0
config["eager"] = True
config["model"]["fcnet_hiddens"] = [64, 64]
config["lr"] = 0.0003
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 10
config["batch_mode"] = "complete_episodes"
config["vf_clip_param"] = 10.0
# config['vf_loss_coeff'] = 0.0
config["observation_filter"] = "MeanStdFilter"
config["sgd_minibatch_size"] = 64
# config["train_batch_size"] = tune.sample_from(lambda spec: spec.config.sgd_minibatch_size*32)
config["train_batch_size"] = 2048
config["seed"] = tune.grid_search([0, 1, 2, 3, 5, 6, 7])
# env_name = "Walker2d-v3"
# env_name =  "Walker2DBulletEnv-v0"
# env_name =   "HumanoidBulletEnv-v0"
env_name = "Pendulum-v0"
# env_name = "sym_pendulum-v0"#
# env_name = "dt_pendulum-v0"

# env_name = "sg_cartpole-v0"

config["env"] = env_name

# import pprint
# pprint.pprint(config)

from seagul.rllib.sym_ppo_policy import PPOTFSymPolicy
from seagul.rllib.symrew_ppo_policy import PPOTFSymRewPolicy

PPOSymTrainer = ppo.PPOTrainer.with_updates(name="SymPPO", default_policy=PPOTFSymPolicy)
PPOSymRewTrainer = ppo.PPOTrainer.with_updates(name="SymRewPPO", default_policy=PPOTFSymRewPolicy)

analysis = tune.run(
    #    PPOSymRewTrainer,
    PPOSymTrainer,
    # ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 6e5},
    local_dir="./data/bench_model/",
    checkpoint_at_end=True,
)
