import ray
from ray import tune
import ray.rllib.agents.sac as sac
import seagul.envs
import numpy as np
from numpy import pi
import torch

config = sac.DEFAULT_CONFIG.copy()
env_name = "su_acroswitchsin-v0"

trial_num = input("trial name / number please:\n")

m1 = 1; m2 = 1
l1 = 1; l2 = 1
lc1 = .5; lc2 = .5
I1 = .2; I2 = 1.0
g = 9.8
max_torque = 5.0; lqr_max_torque = 5.0
max_t = 10.0

def lqr(q):
    k = np.array([[-1649.86567367, -460.15780461, -716.07110032, -278.15312267]])
    gs = np.array([pi / 2, 0, 0, 0])
    return -k.dot(q - gs)

def reward_fn_sin(s,a):
    reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False


# config["env"] = env_name
# # config["eager"] = True
# #config["num_cpus_per_worker"] = 2
# config["num_workers"] = 2
# # config["num_envs_per_worker"]  = 15
# # config["train_batch_size"] = 256
# config["learning_starts"] = 50000
# # config["learning_starts"] =  10000
# # config["pure_exploration_steps"] =  10000
# # config["evaluation_interval"] = 5
# # config["evaluation_num_episode"] = 10
# config["use_state_preprocessor"] = True
# config["evaluation_interval"] = 1
# config["soft_horizon"] = False
# config["metrics_smoothing_episodes"] = 10
# config["no_done_at_end"] = True
# #config["prioritized_replay"] = True
# config["Q_model"]["hidden_layer_sizes"] = (32, 32)
# config["policy_model"]["hidden_layer_sizes"] = (32, 32)
        
        

# config["env_config"] = {
#     "init_state": [-pi/2, 0, 0, 0],
#     "max_torque": max_torque,
#     "lqr_max_torque" : lqr_max_torque,
#     "init_state_weights": [0, 0, 0, 0],
#     "dt": .01,
#     "reward_fn" : reward_fn_sin,
#     "max_t" : max_t,
#     "m2": m2,
#     "m1": m1,
#     "l1": l1,
#     "lc1": lc1,
#     "lc2": lc2,
#     "i1": I1,
#     "i2": I2,
#     "act_hold" : 20,
#     "gate_fn" : torch.load("warm/lqr_gate_better"),
#     "controller" : lqr
# }


config = {'num_workers': 4,
          'num_envs_per_worker': 1,
          'sample_batch_size': 1,
          'batch_mode': 'truncate_episodes',
          'num_gpus': 0,
          'train_batch_size': 256,
          'model': {'conv_filters': None,
                    'conv_activation': 'relu',
                    'fcnet_activation': 'tanh',
                    'fcnet_hiddens': [32, 32],
                    'free_log_std': False,
                    'no_final_linear': False,
                    'vf_share_layers': True,
                    'use_lstm': False,
                    'max_seq_len': 20,
                    'lstm_cell_size': 256,
                    'lstm_use_prev_action_reward': False,
                    'state_shape': None,
                    'framestack': True,
                    'dim': 84,
                    'grayscale': False,
                    'zero_mean': True,
                    'custom_preprocessor': None,
                    'custom_model': None,
                    'custom_action_dist': None,
                    'custom_options': {}},
          'optimizer': {},
          'gamma': 0.99,
          'horizon': None,
          'soft_horizon': True,
          'no_done_at_end': True,
          'env_config': {'init_state': [-1.5707963267948966, 0, 0, 0],
                         'max_torque': 5,
                         'init_state_weights': [0, 0, 0, 0],
                         'dt': 0.01,
                         'reward_fn': reward_fn_sin,
                         'max_t': 10.0,
                         'm2': 1,
                         'm1': 1,
                         'l1': 1,
                         'lc1': 0.5,
                         'lc2': 0.5,
                         'i1': 0.2,
                         'i2': 1.0,
                         'act_hold': 20,
                         'gate_fn': torch.load('warm/lqr_gate_better'),
                         'controller': lqr,
          },
          'env': 'su_acroswitch-v0',
          'normalize_actions': True,
          'clip_rewards': None,
          'clip_actions': True,
          'preprocessor_pref': 'deepmind',
          'lr': 0.0001,
          'monitor': False,
          'log_level': 'WARN',
          'callbacks': {'on_episode_start': None,
                        'on_episode_step': None,
                        'on_episode_end': None,
                        'on_sample_end': None,
                        'on_train_result': None,
                        'on_postprocess_traj': None},
          'ignore_worker_failures': False,
          'log_sys_usage': True,
          'eager': False,
          'eager_tracing': False,
          'no_eager_on_workers': False,
          'evaluation_interval': 1,
          'evaluation_num_episodes': 1,
          'evaluation_config': {'exploration_enabled': False},
          'sample_async': False,
          'observation_filter': 'NoFilter',
          'synchronize_filters': True,
          'tf_session_args': {'intra_op_parallelism_threads': 2,
                              'inter_op_parallelism_threads': 2,
                              'gpu_options': {'allow_growth': True},
                              'log_device_placement': False,
                              'device_count': {'CPU': 1},
                              'allow_soft_placement': True},
          'local_tf_session_args': {'intra_op_parallelism_threads': 8,
                                    'inter_op_parallelism_threads': 8},
          'compress_observations': False,
          'collect_metrics_timeout': 180,
          'metrics_smoothing_episodes': 10,
          'remote_worker_envs': False,
          'remote_env_batch_wait_ms': 0,
          'min_iter_time_s': 1,
          'timesteps_per_iteration': 4000,
          'seed': None,
          'num_cpus_per_worker': 1,
          'num_gpus_per_worker': 0,
          'custom_resources_per_worker': {},
          'num_cpus_for_driver': 1,
          'memory': 0,
          'object_store_memory': 0,
          'memory_per_worker': 0,
          'object_store_memory_per_worker': 0,
          'input': 'sampler',
          'input_evaluation': ['is', 'wis'],
          'postprocess_inputs': False,
          'shuffle_buffer_size': 0,
          'output': None,
          'output_compress_columns': ['obs', 'new_obs'],
          'output_max_file_size': 67108864,
          'multiagent': {'policies': {},
                         'policy_mapping_fn': None,
                         'policies_to_train': None},
          'twin_q': True,
          'use_state_preprocessor': True,
          'policy': 'GaussianLatentSpacePolicy',
          'Q_model': {'hidden_activation': 'relu', 'hidden_layer_sizes': (32, 32)},
          'policy_model': {'hidden_activation': 'relu',
                           'hidden_layer_sizes': (32, 32)},
          'tau': 0.005,
          'target_entropy': 'auto',
          'n_step': 1,
          'exploration_enabled': True,
          'buffer_size': 1000000,
          'prioritized_replay': False,
          'prioritized_replay_alpha': 0.6,
          'prioritized_replay_beta': 0.4,
          'prioritized_replay_eps': 1e-06,
          'beta_annealing_fraction': 0.2,
          'final_prioritized_replay_beta': 0.4,
          'optimization': {'actor_learning_rate': 0.0003,
                           'critic_learning_rate': 0.0003,
                           'entropy_learning_rate': 0.0003},
          'grad_norm_clipping': None,
          'learning_starts': 500,
          'target_network_update_freq': 0,
          'worker_side_prioritization': False,
          'per_worker_exploration': False,
          'exploration_fraction': 0.1,
          'schedule_max_timesteps': 100000,
          'exploration_final_eps': 0.02
}

ray.init(
    num_cpus=24,
    memory=int(16e9),
    object_store_memory=int(8e9),
    driver_object_store_memory= int(4e9)
)

analysis = tune.run(
    sac.SACTrainer,
    config=config,
    stop={"timesteps_total": 2.5e5},
    num_samples=4,
    local_dir="./data6/tune/sac/" + trial_num + "/" ,
    checkpoint_at_end=True,
)
