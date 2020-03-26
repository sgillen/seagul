import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
from tqdm import trange

torch.set_default_dtype(torch.double)

# ============================================================================================

# env_name = 'CartPole-v0' # Discrete
env_name = "InvertedPendulum-v2"  # Continous
# env_name = 'su_cartpole-v0'

# env_name = 'Walker2d-v2'
# env_name = 'lorenz-v0'
# Hard coded policy for the cartpole problem
# Will eventually want to build up infrastructure to develop a policy depending on:
# env.action_space
# env.observation_space

# policy = nn.Sequential(
#     nn.Linear(17, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 6)
# )

# value_fn = nn.Sequential(
#     nn.Linear(17, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 1)
# )


policy = nn.Sequential(
    nn.Linear(4, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 1),
)


value_fn = nn.Sequential(
    nn.Linear(4, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 1),
)

policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
value_optimizer = optim.Adam(value_fn.parameters(), lr=1e-2)

num_epochs = 100
batch_size = 500  # how many steps we want to use before we update our gradients
num_steps = 500  # number of steps in an episode (unless we terminate early)
max_reward = 101

# ============================================================================================

variance = 0.1


def select_action(policy, state):

    # loc is the mean, scale is the variance
    # m = Normal(loc = policy(torch.tensor(state))[0], scale = abs(policy(torch.tensor(state))[1]))

    means = policy(torch.as_tensor(state))
    m = Normal(loc=means, scale=torch.ones_like(means) * variance)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach().numpy(), logprob


# def vanilla_policy_grad(env, policy, policy_optimizer):

env = gym.make(env_name)

# env.num_steps = 3000
# env.state_noise_max = .01
# env.torque_limit = 200.0
# env.dt = .01
# env.X_MAX = 20.0

avg_reward_hist = []
value_preds_hist = []
value_loss_hist = []
policy_loss_hist = []

for epoch in trange(num_epochs):

    # Probably just want to preallocate these with zeros, as either a tensor or an array
    episode_reward_sum = []
    total_steps = 0
    traj_count = 1

    policy_loss = torch.tensor([0], dtype=torch.double)
    value_loss = torch.tensor([0], dtype=torch.double)

    # Check if we have completed the task early
    try:
        if avg_reward_hist[-1] == avg_reward_hist[-2] == max_reward:
            break
    except IndexError:
        pass

    while True:

        state = env.reset()
        logprob_list = []
        reward_list = []
        state_list = []
        action_list = []

        for t in range(num_steps):

            action, logprob = select_action(policy, state)
            state, reward, done, _ = env.step(action)

            logprob_list.append(-logprob)
            reward_list.append(reward)
            state_list.append(state)
            action_list.append(action)

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

        policy_loss += torch.sum(logprob_t.transpose(-1, 0) * policy_rewards) / traj_count
        value_loss += torch.sum(torch.pow(action_rewards - value_preds, 2)) / (traj_count * num_steps)

        episode_reward_sum.append(sum(reward_list))

        value_preds_hist.append(value_preds.detach())
        value_loss_hist.append(value_loss.detach())
        policy_loss_hist.append(policy_loss.detach())

        if total_steps > batch_size:

            policy_loss.backward()
            value_loss.backward()

            policy_optimizer.step()
            policy_optimizer.zero_grad()

            value_optimizer.step()
            value_optimizer.zero_grad()

            policy_loss *= 0
            value_loss += 0

            avg_reward_hist.append(sum(episode_reward_sum) / len(episode_reward_sum))
            traj_count = 1

            break

# ============================================================================================

save_path = "./data/lorenz1"
torch.save(policy.state_dict(), save_path)

plt.plot(avg_reward_hist)
plt.title("Average Reward")
plt.xlabel("Epoch")
plt.ylabel("Avg Reward")
plt.show()
