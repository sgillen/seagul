import gym
import pickle
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
from tqdm import trange
import copy

from seagul.nn import fit_model
import scipy.signal


seed = 1336
torch.set_default_dtype(torch.double)
torch.manual_seed(seed)
np.random.seed(seed)

use_gpu = False
use_cuda = torch.cuda.is_available() and use_gpu
device = torch.device("cuda:0" if use_cuda else "cpu")


# ============================================================================================

# choose the environment

#env_name = 'CartPole-v0' # Discrete
#env_name = 'InvertedPendulum-v2' # Continuous
#env_name = 'su_cartpole-v0'
env_name = 'Walker2d-v2'
#env_name = 'lorenz-v0'

#env_name = 'CartPole-v0'


# create policy and value networks

policy = nn.Sequential(
    nn.Linear(17, 48),
    nn.ReLU(),
    nn.Linear(48, 48),
    nn.ReLU(),
    nn.Linear(48, 48),
    nn.ReLU(),
    nn.Linear(48, 6),
)

# policy = nn.Sequential(
#     nn.Linear(4, 12),
#     nn.LeakyReLU(),
#     nn.Linear(12, 12),
#     nn.LeakyReLU(),
#     nn.Linear(12, 2),
#     nn.Softmax(dim=-1)
# )

old_policy = pickle.loads(pickle.dumps(policy))
value_fn = nn.Sequential(
    nn.Linear(17, 48),
    nn.ReLU(),
    nn.Linear(48, 48),
    nn.ReLU(),
    nn.Linear(48, 48),
    nn.ReLU(),
    nn.Linear(48, 1),
)


# Define our hyper parameters
num_epochs = 100
batch_size = 2048  # how many steps we want to use before we update our gradients
num_steps = 1000 # number of steps in an episode (unless we terminate early)
max_reward = num_steps
p_batch_size = 1024
v_epochs = 1
p_epochs = 10
p_lr = 1e-2
v_lr = 1e-2

gamma = .99
lam = .99
eps = .2

variance = 0.2 # feel like there should be a better way to do this...
optimizer = torch.optim.Adam(policy.parameters(), lr=p_lr)
v_optimizer = torch.optim.Adam(value_fn.parameters(), lr=p_lr)


# ============================================================================================


# takes a policy and the states and sample an action from it... (can we make this faster?)
def select_action(policy, state):
    means = policy(torch.as_tensor(state)).squeeze()
    m = Normal(loc = means, scale = torch.ones_like(means)*variance)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach(), logprob


# given a policy plus a state/action pair, what is the log liklihood of having taken that action?
def get_logp(policy, states, actions):
    means = policy(torch.as_tensor(states)).squeeze()
    m = Normal(loc = means, scale = torch.ones_like(means)*variance)
    logprob = m.log_prob(actions)
    return logprob



# def select_action(policy, state):
#     m = Categorical(policy(torch.as_tensor(state)))
#     action = m.sample()
#     logprob = m.log_prob(action)
#     return action.detach(), logprob
#
# def get_logp(policy, state, action):
#     m = Categorical(policy(torch.as_tensor(state)))
#     logprob = m.log_prob(action)
#     return logprob


def discount_cumsum(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards


# ============================================================================================

# def vanilla_policy_grad(env, policy, policy_optimizer):

env = gym.make(env_name)
env.seed(seed)

# env.num_steps = 3000
# env.state_noise_max = .01
# env.torque_limit = 200.0
# env.dt = .01
# env.X_MAX = 20.0

avg_reward_hist = []
value_preds_hist = []
value_loss_hist = []
policy_loss_hist = []
adv_list = []

for epoch in trange(num_epochs):

    # Probably just want to preallocate these with zeros, as either a tensor or an array
    episode_reward_sum = []
    total_steps = 0
    traj_steps = 0
    traj_count = 1

    adv = torch.empty(0)
    disc_rewards = torch.empty(0)
    state_tensor = torch.empty(0)
    logp_t = torch.empty(0)
    action_tensor = torch.empty(0)

    loss = torch.zeros(1)
    v_loss = torch.zeros(1)


    # Check if we have completed the task early
    try:
        if avg_reward_hist[-1] == avg_reward_hist[-2] == max_reward:
            break
    except IndexError:
        pass

    while True:

        state = torch.as_tensor(env.reset())

        reward_list = []
        action_list = []
        state_list = []
        log_prob_list = []
        traj_steps = 0

        for t in range(num_steps):

            state_list.append(state)

            action, logprob = select_action(policy, state)
            state_np, reward, done, _ = env.step(action.numpy())
            state = torch.as_tensor(state_np)

            log_prob_list.append(logprob)
            reward_list.append(reward)
            action_list.append(action)

            total_steps += 1
            traj_steps += 1

            if done:
                traj_count += 1
                break


        if traj_steps <= 1:
            traj_steps = 0
            break


        # Now Calculate cumulative rewards for each action
        ep_state_tensor = torch.stack(state_list).reshape(-1,env.observation_space.shape[0])
        ep_action_tensor = torch.stack(action_list).reshape(-1,env.action_space.shape[0])
        ep_disc_rewards = torch.as_tensor(discount_cumsum(reward_list, gamma)).reshape(-1, 1)

        value_preds = value_fn(ep_state_tensor)
        value_preds_hist.append(value_preds.detach())

        deltas = torch.as_tensor(reward_list[:-1]) + gamma*value_preds[1:].squeeze() - value_preds[:-1].squeeze()
        ep_adv = discount_cumsum(deltas.detach(), gamma*lam).reshape(-1,1)

        state_tensor  = torch.cat((state_tensor , ep_state_tensor[:-1]))
        action_tensor = torch.cat((action_tensor, ep_action_tensor[:-1]))
        disc_rewards  = torch.cat((disc_rewards , ep_disc_rewards[:-1]))
        #adv = torch.cat((adv,torch.tensor(ep_disc_rewards[:-1])))

        logp_t = torch.cat((logp_t, torch.stack(log_prob_list[:-1])))
        #old_logp = get_logp(old_policy, ep_state_tensor, torch.tensor(action_list))
        adv = torch.cat((adv, ep_adv))

        episode_reward_sum.append(sum(reward_list))

        #loss += -torch.sum(torch.stack(log_prob_list) * ep_adv) / (total_steps)
        #r = torch.exp(torch.tensor(log_prob_list) - old_logp)
        #loss = -torch.sum(torch.min(r * ep_adv, ep_adv * torch.clamp(r, (1 - eps), (1 + eps)))) / (traj_steps)



        if total_steps > batch_size:
            avg_reward_hist.append(sum(episode_reward_sum) / len(episode_reward_sum))

            v_hist = fit_model(value_fn, state_tensor, disc_rewards, v_epochs, v_lr, shuffle=True)

            training_data = data.TensorDataset(state_tensor, action_tensor, adv, logp_t)
            training_generator = data.DataLoader(training_data, batch_size=p_batch_size, shuffle=True)
            p_loss = []
            for epoch in range(p_epochs):
                for local_states, local_actions, local_adv, local_logp_t in training_generator:
                    # Transfer to GPU (if GPU is enabled, else this does nothing)
                    #local_states, local_actions, local_adv = local_states.to(device), local_actions.to(device), local_adv.to(device)

                    # predict and calculate loss for the batch
                    logp = get_logp(policy, local_states, local_actions.squeeze()).reshape(-1,env.action_space.shape[0])
                    old_logp = get_logp(old_policy, local_states, local_actions.squeeze()).reshape(-1,env.action_space.shape[0])
                    r = torch.exp(logp - old_logp)

                    #loss =  -torch.sum(local_logp_t.squeeze() * local_adv.squeeze())
                    #loss =  -torch.sum(logp * local_adv)
                    #loss = -torch.sum(local_logp_t.reshape(-1,1) * local_adv)

                    loss = -torch.sum(torch.min(r*local_adv, local_adv*torch.clamp(r, (1 - eps), (1 + eps))))/r.shape[0]
                    #loss = -torch.sum(r * local_adv)

                    #loss = torch.sum(logp*local_adv)/logp.shape[0]
                    p_loss.append(loss.detach())
                    # do the normal pytorch update
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()


#            policy_loss_hist.append(sum(p_loss)/len(p_loss))
#            value_loss_hist.append(v_hist)
            loss.backward()
            old_policy = pickle.loads(pickle.dumps(policy))

            #old_policy = copy.deepcopy(policy)

            break

# ============================================================================================

#save_path = "./data/lorenz1"
#torch.save(policy.state_dict(), save_path )

plt.plot(avg_reward_hist, 'b')
print(avg_reward_hist)

plt.title('ppo2')
plt.show()