from multiprocessing import Pool
import torch.nn as nn
import numpy as np
import gym
from seagul.rl.sac import SACAgent, SACModel, SACMMultiLinModel
from seagul.nn import MLP
import matplotlib.pyplot as plt
import torch
import dill
import seagul.envs

env_name = "hurdle_hopper-v0"
#env = gym.make(env_name)
#torch.set_default_dtype(torch.float64)

def run_and_test(seed, verbose=False):
    # init policy, value fn
    input_size = 12
    output_size = 2
    layer_size = 32
    num_layers = 2
    activation = nn.ReLU

    # model = SACModel(
    #      policy = MLP(input_size, output_size*2, num_layers, layer_size, activation),
    #      value_fn= MLP(input_size, 1, num_layers, layer_size, activation),
    #      q1_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
    #      q2_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
    #      act_limit=1
    # )

    import pickle
    low_level_agent_list = pickle.load(open("./mdim_304045_gap_back", 'rb'))[:2]

    model = SACMMultiLinModel(
         policy = MLP(input_size, output_size*2, num_layers, layer_size, activation),
         value_fn= MLP(input_size, 1, num_layers, layer_size, activation),
         q1_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
         q2_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
         act_limit=1,
         model_list=[a.model for a in low_level_agent_list]
    )

    print(input_size+output_size)

    agent = SACAgent(env_name=env_name, model=model, seed=int(seed), exploration_steps=5000,
                     min_steps_per_update=500, reward_stop=3000, gamma=1, sgd_batch_size=64,
                     replay_batch_size=256, iters_per_update=1000, env_max_steps=2000,
                     polyak=.995, env_config={"gap_set":[30, 45], "hurdle_height":.52})


    model, rewards, var_dict,  = agent.learn(train_steps=1e6)
    torch.save(var_dict, open("./tmp/" + str(seed), 'wb'), pickle_module=dill)

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    return rewards, var_dict["early_stop"]


if __name__ == "__main__":
    import seagul.envs
    seeds = np.random.randint(0, 2**32, 8)
    pool = Pool(processes=8)

    #results = run_and_test(run_and_test(seeds[0]))
    results = pool.map(run_and_test, seeds)

    rewards = []
    finished = []
    for result in results:
        rewards.append(result[0])
        finished.append(result[1])

    #for reward in rewards:
    #    plt.plot(reward, alpha=.8)

    print(finished)

    #plt.show()

    #ws = torch.load(open(f'/home/sgillen/work/seagul/seagul/tests/ppo/todo/tmp/{seeds[0]}', 'rb'))

