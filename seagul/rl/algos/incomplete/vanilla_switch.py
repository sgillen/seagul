import seagul.envs

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from tqdm import trange

# This will need to be a float 32 if trained
torch.set_default_dtype(torch.double)

# ============================================================================================


# Switching master controller
switching_policy = nn.Sequential(
    nn.Linear(17, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 7),
)

# Swingup sub controller
swingup_policy = nn.Sequential(
    nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 12), nn.Tanh(), nn.Linear(12, 2), nn.Softmax(dim=-1)
)

# Value function approximator used for the baseline
switching_policy = nn.Sequential(
    nn.Linear(17, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)


def select_action(switching_policy, swingup_policy, state):

    t_state = torch.as_tensor(state)
    logprob_swingup = None

    switch_dist = Categorical(switching_policy(t_state))
    controller = switch_dist.sample()
    logprob_switch = switch_dist.log_prob(controller)

    # if controller.item() == 0:
    # print("hello")
    torque_limit = 100.0
    swingup_dist = Categorical(swingup_policy(t_state))
    dir = swingup_dist.sample()
    if dir == 0.0:
        action = np.array([torque_limit])
    elif dir == 1.0:
        action = np.array([-torque_limit])

    logprob_swingup = swingup_dist.log_prob(dir)

    # else:
    # print("goodbye")
    # balancing
    # LQR: K values from MATLAB
    #  K = 1 / 50
    #  k1 = 3
    #  k2 = 140
    #  k3 = 8
    #  k4 = 42
    #  action = K * (k1 * state[0] + k2 * state[1] + k3 * state[2] + k4 * state[3])

    return action, logprob_switch, logprob_swingup


# ============================================================================================


switching_optimizer = optim.Adam(switching_policy.parameters(), lr=1e-2)
swingup_optimizer = optim.Adam(swingup_policy.parameters(), lr=1e-2)
value_optimizer = optim.Adam(value_fn.parameters(), lr=1e-2)

num_epochs = 2000  # number of batches to train on
batch_size = 500  # how many steps we want to use before we update our gradients
num_steps = 200  # number of steps in an episode (unless we terminate early)


env_name = "Pendulum-v0"

# def vanilla_policy_grad(env, policy, policy_optimizer):

env = gym.make(env_name)
env.num_steps = 3000
env.state_noise_max = 0.01
env.torque_limit = 200.0
env.dt = 0.01
env.X_MAX = 20.0

avg_reward_hist = []


for epoch in trange(num_epochs):

    # Probably just want to preallocate these with zeros, as either a tensor or an array
    episode_reward_sum = []
    total_steps = 0
    traj_count = 1

    while True:

        state = env.reset()
        switching_logprob_list = []
        swingup_logprob_list = []
        reward_list = []
        state_list = []

        for t in range(num_steps):

            action, logprob_switch, logprob_swingup = select_action(switching_policy, swingup_policy, state)
            state, reward, done, _ = env.step(action)

            swingup_logprob_list.append(-logprob_switch)
            if logprob_swingup:
                switching_logprob_list.append(-logprob_swingup)

            reward_list.append(reward)
            state_list.append(state)

            total_steps += 1

            if done:
                traj_count += 1
                break

        try:

            # Now Calculate cumulative rewards for each action
            action_rewards = torch.tensor([sum(reward_list[i:]) for i in range(len(reward_list))])
            logprob_sw = torch.stack(switching_logprob_list)

            value_preds = value_fn(torch.tensor(state_list)).squeeze()
            policy_rewards = action_rewards - value_preds

            sw_policy_loss = torch.sum(logprob_sw * policy_rewards) / (traj_count)
            sw_policy_loss.backward(retain_graph=True)

            logprob_su = torch.stack(swingup_logprob_list)
            su_policy_loss = torch.sum(logprob_su * policy_rewards) / (traj_count)
            su_policy_loss.backward(retain_graph=True)

            value_loss = torch.sum(value_preds - action_rewards) / (traj_count * num_steps)
            value_loss.backward(retain_graph=True)

        except RuntimeError:
            pass

        episode_reward_sum.append(sum(reward_list))

        if total_steps > batch_size:
            switching_optimizer.step()
            switching_optimizer.zero_grad()

            swingup_optimizer.step()
            swingup_optimizer.zero_grad()

            value_optimizer.step()
            value_optimizer.zero_grad()

            avg_reward_hist.append(sum(episode_reward_sum) / len(episode_reward_sum))
            break


# ============================================================================================

plt.plot(avg_reward_hist)
plt.title("new")
plt.show()


def policy_render_loop(sw_policy, su_policy, env, select_action):

    """
        Will render the policy passed in in an infinite loop. You can send a keyboard interrupt (Ctrl-C) and it will
        end the render loop without ending your interactive session.

        Tries to close the window when you send the interrupt, doesn't actually work for Mujoco environments though.

        Attributes:
            policy: your (presumably neural network) function that maps states->actions
            env: the environment you want to render actions in
            select_action: function for actually picking an action from the policy, should be the same one you trained with

        Returns:
            Nothing

        Example:
            import torch
            import torch.nn as nn
            from torch.distributions import Normal
            import gym
            from utils.nn_utils import policy_render_loop

            import os
            print(os.getcwd())

            policy = nn.Sequential(
                nn.Linear(4, 12),
                nn.LeakyReLU(),
                nn.Linear(12, 12),
                nn.LeakyReLU(),
                nn.Linear(12, 1),
            )

            load_path = '/Users/sgillen/work_dir/ucsb/notebooks/rl/cont_working'
            policy.load_state_dict(torch.load(load_path))


            def select_action(policy, state):
                # loc is the mean, scale is the variance
                m = Normal(loc = policy(torch.tensor(state))[0], scale = torch.tensor(.7))
                action = m.sample()
                logprob = m.log_prob(action)
                return action.detach().numpy(), logprob


            env_name = 'InvertedPendulum-v2'
            env = gym.make(env_name)


            policy_render_loop(policy,env,select_action)

            # Blocks until you give a Ctrl-C, then drops you back into the shell


    """

    try:
        state = env.reset()
        while True:
            action, _, _ = select_action(sw_policy, su_policy, state)
            state, reward, done, _ = env.step(action)
            env.render()

            if done:
                state = env.reset()

    except KeyboardInterrupt:
        env.close()
