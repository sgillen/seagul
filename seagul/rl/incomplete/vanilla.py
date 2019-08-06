import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import trange


torch.set_default_dtype(torch.double)

# ============================================================================================

env_name = "CartPole-v0"

# Hard coded policy for the cartpole problem
# Will eventually want to build up infrastructure to develop a policy depending on:
# env.action_space
# env.observation_space

policy = nn.Sequential(
    nn.Linear(4, 12), nn.LeakyReLU(), nn.Linear(12, 12), nn.LeakyReLU(), nn.Linear(12, 2), nn.Softmax(dim=-1)
)

value_fn = nn.Sequential(nn.Linear(4, 12), nn.LeakyReLU(), nn.Linear(12, 12), nn.LeakyReLU(), nn.Linear(12, 1))
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
value_optimizer = optim.Adam(value_fn.parameters(), lr=1e-2)

num_epochs = 200
batch_size = 500  # how many steps we want to use before we update our gradients
num_steps = 200  # number of steps in an episode (unless we terminate early)

# ============================================================================================


def select_action(policy, state):
    m = Categorical(policy(torch.Tensor(state)))
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach().numpy(), logprob


# def vanilla_policy_grad(env, policy, policy_optimizer):

env = gym.make(env_name)
avg_reward_hist = []
value_preds_hist = []
value_loss_hist = []
policy_loss_hist = []

max_reward = 200

for epoch in trange(num_epochs):

    # Probably just want to preallocate these with zeros, as either a tensor or an array
    episode_reward_sum = []
    total_steps = 0
    traj_count = 1

    while True:

        state = env.reset()
        logprob_list = []
        reward_list = []
        state_list = []

        # Check if we have completed the task early
        try:
            if avg_reward_hist[-1] == avg_reward_hist[-2] == max_reward:
                break
        except IndexError:
            pass

        for t in range(num_steps):

            action, logprob = select_action(policy, state)
            state, reward, done, _ = env.step(action.item())

            logprob_list.append(-logprob)
            reward_list.append(reward)
            state_list.append(state)

            total_steps += 1

            if done:
                traj_count += 1
                break

        # Now Calculate cumulative rewards for each action
        action_rewards = torch.tensor([sum(reward_list[i:]) for i in range(len(reward_list))])
        logprob_t = torch.stack(logprob_list)

        value_list = value_fn(torch.tensor(state_list)).squeeze()
        value_preds = torch.stack([torch.sum(value_list[i:]) for i in range(len(value_list))])
        policy_rewards = action_rewards - value_preds
        policy_rewards = action_rewards

        policy_loss = torch.sum(logprob_t * policy_rewards) / traj_count
        policy_loss.backward(retain_graph=True)

        value_loss = torch.sum(torch.pow(action_rewards - value_preds, 2)) / (traj_count * num_steps)
        value_loss.backward(retain_graph=True)

        episode_reward_sum.append(sum(reward_list))

        value_preds_hist.append(value_preds)
        value_loss_hist.append(value_loss)
        policy_loss_hist.append(policy_loss)

        if total_steps > batch_size:
            policy_optimizer.step()
            policy_optimizer.zero_grad()

            value_optimizer.step()
            value_optimizer.zero_grad()

            avg_reward_hist.append(sum(episode_reward_sum) / len(episode_reward_sum))
            traj_count = 1

            break


# ============================================================================================

plt.plot(avg_reward_hist)
plt.title("new")
plt.show()
