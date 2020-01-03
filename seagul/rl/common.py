import torch
import numpy as np

# can make this faster I think?
def discount_cumsum(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer (stolen from https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
    """

    # TODO torchify?
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        self.obs2_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        self.acts_buf = torch.zeros([size, act_dim], dtype=torch.float32)
        self.rews_buf = torch.zeros(size, dtype=torch.float32)
        self.done_buf = torch.zeros(size, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.obs1_buf[idxs],self.obs2_buf[idxs], self.acts_buf[idxs], self.rews_buf[idxs], self.done_buf[idxs]
