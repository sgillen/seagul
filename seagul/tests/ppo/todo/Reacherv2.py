import dill
import torch.nn as nn
from seagul.rl.ppo.ppo2 import PPOAgent
from seagul.nn import MLP
import torch
from seagul.rl.ppo.models import PPOModel
from multiprocessing import Process, Manager, Pool
import numpy as np
from seagul.plot import smooth_bounded_curve, chop_returns
import matplotlib.pyplot as plt
import torch
import dill

def run_and_test(seed, verbose=False):
    input_size = 11
    output_size = 2
    layer_size = 64
    num_layers = 2
    activation = nn.ReLU

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PPOModel(policy, value_fn, init_logstd=-.5, learn_std=True)
    agent = PPOAgent(env_name="Reacher-v2", model=model, epoch_batch_size=2048, gamma=.99, seed=int(seed),
                     entropy_coef=0.0, sgd_batch_size=64, lr_schedule=(1e-3,), sgd_epochs=30, target_kl=float('inf'),
                     clip_val=True, env_no_term_steps=50, reward_stop=-2.5, normalize_return=True, normalize_obs=True,
                     normalize_adv=True)

    t_model, rewards, var_dict = agent.learn(total_steps=2.5e5)

    if verbose:
        if var_dict["early_stop"]:
            print("seed", seed, "achieved 1000 reward in ", len(rewards), "steps")
        else:
            print("Error: seed:", seed, "failed")

    torch.save(var_dict, open("./tmp/" + str(seed), 'wb'), pickle_module=dill)

    return rewards, var_dict["early_stop"]


if __name__ == "__main__":
    seeds = np.random.randint(0, 2**32, 8)
    pool = Pool(processes=8)

    #results = run_and_test(run_and_test(seeds[0]))
    results = pool.map(run_and_test, seeds)

    rewards = []
    finished = []
    for result in results:
        rewards.append(result[0])
        finished.append(result[1])

    for reward in rewards:
        plt.plot(reward, alpha=.8)

    print(finished)

    plt.show()

    ws = torch.load(open(f'/home/sgillen/work/seagul/seagul/tests/ppo/todo/tmp/{seeds[0]}', 'rb'))
    locals().update(ws)

#     import gym
#     import time

#     env = gym.make("Reacher-v2")


#     def discount_cumsum(rewards, discount):
#         future_cumulative_reward = 0
#         cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
#         for i in range(len(rewards) - 1, -1, -1):
#             cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
#             future_cumulative_reward = cumulative_rewards[i]
#         return cumulative_rewards

    import time
    import gym

    def do_rollout(env: object, model: object, n_steps_complete: object) -> object:
        torch.autograd.set_grad_enabled(False)

        act_list = []
        obs_list = []
        rew_list = []
        logp_list = []

        dtype = torch.float32
        obs = env.reset()
        done = False
        cur_step = 0

        while not done:
            obs = torch.as_tensor(obs, dtype=dtype).detach()
            obs_list.append(obs.clone())

            act, _ = model.select_action(obs)
            obs, rew, done, _ = env.step(np.clip(act.numpy(), -1, 1))
            env.render()
            time.sleep(.02)

            act_list.append(torch.as_tensor(act.clone()))
            rew_list.append(rew)

            cur_step += 1

        if cur_step < n_steps_complete:
            ep_term = True
        else:
            ep_term = False

        ep_length = len(rew_list)
        ep_obs = torch.stack(obs_list)
        ep_act = torch.stack(act_list)
        ep_rew = torch.tensor(rew_list, dtype=dtype)
        ep_rew = ep_rew.reshape(-1, 1)

        torch.autograd.set_grad_enabled(True)
        return ep_obs, ep_act, ep_rew, ep_length, None

ep_obs, ep_act, ep_rew, ep_steps, ep_logp = do_rollout(gym.make("Reacher-v2"),self.model,1000)
# ep_rew = torch.cat((ep_rew, self.model.value_fn(ep_obs[-1]).detach().reshape(1, 1).clone()))
# ep_discrew = discount_cumsum(ep_rew, self.gamma)
# plt.plot(ep_discrew)
# plt.show()
