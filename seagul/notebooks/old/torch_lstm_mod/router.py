
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax


# for the training only
from itertools import count
import gym
import torch.optim as optim
import numpy as np



eps = np.finfo(np.float32).eps.item()


class Router(nn.Module):
    def __init__(self, input_size, hidden_size, router_size, output_size):
        super().__init__()
        
        # Routing layer gates
        self.r_linear1 = nn.Linear(input_size, router_size)
        self.r_linear2 = nn.Linear(router_size, 2)
        
        # Swingup layer gates
        self.s_linear1 = nn.Linear(input_size, hidden_size)
        self.s_linear2 = nn.Linear(hidden_size, output_size)
        
        # This is basically our static gain matrix (maybe I should make this a matrix rather than a linear layer...)
        self.k = nn.Linear(input_size, output_size, bias=False) 
        
        # Required for the training
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        # Gating
        g = torch.sigmoid(self.r_linear1(x))
        g = torch.sigmoid(self.r_linear2(g))
        d = softmax(g, dim=-1)

        
        # Swingup
        s = torch.relu(self.s_linear1(x))
        ys = self.s_linear2(s)
        
        # Balance
        yb = self.k(x)
    
        return ys, yb, d

def select_action(x, policy):
    x = torch.from_numpy(x).float().unsqueeze(0)

    ys, yb, d = policy(x)
    m = torch.distributions.Categorical(d)
    path = m.sample()

    policy.saved_log_probs.append(m.log_prob(path))

    #if path.item() == 0:
    #    return ys.item()
    #else:
    #    return yb.item()
    return yb.item()

    # Calculates the time weighted rewards, policy losses, and optimizers

# This is a confusing name...
def finish_episode(policy):
    R = 0
    policy_loss = []
    rewards = []

    gamma = .5
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.append(R)

    rewards = rewards[::-1]
    rewards = torch.tensor(rewards)

    std = rewards.std()
    if torch.isnan(std):
        std = 1

    rewards = (rewards - rewards.mean()) / (std + eps)

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]


if __name__ == '__main__':
        


    
    # Just Naive REINFORCE for pendulum env. 
    env = gym.make('InvertedPendulum-v2')

    #env.step = fixed_step.__get__(env, gym.Env)
    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    policy = Router(input_size = 4, hidden_size = 8, router_size = 8, output_size = 1)

    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    num_episodes = int(2e5)

    running_reward = 10
    for i_episode in range(num_episodes):
        state = env.reset()
        for t in range(10000):
            action = select_action(state, policy)
            state,reward, done, _ = env.step(action)

            policy.rewards.append(reward)
            if done:
                break

            running_reward = running_reward * 0.99 + t * 0.01
            finish_episode(policy) 


        log_interval = 100
        if i_episode % log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))




    state = env.reset()
    while(True):
        action = select_action(state, policy)
        state,reward, done, _ = env.step(action)
        env.render()
        if done:
            state = env.reset()

