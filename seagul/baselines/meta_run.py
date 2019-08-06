from seagul.baselines.run import run_and_save

# algs = ['a2c', 'acer', 'acktr', 'ddpg', 'deepq', 'her', 'ppo2', 'trpo_mpi']
#
arg_dict = {
    #'env': 'bullet_car_ast-v0',
    #'env': 'Walker2d-v2',
    #'env': 'Hopper-v2', # Continous
    "env": "CartPole-v0",
    "alg": "ppo2",
    "network": "mlp",
    "num_timesteps": "1e5",
    "num_env": "1",
    "num_layers": "3",
    "num_hidden": "24",
}

# run_name ='fps_test_noball'

run_name = "bullet debug"
run_desc = "asdasd"
run_and_save(
    arg_dict, run_name=run_name, description=run_desc, base_path="/data/bullet_car_ast/"
)
