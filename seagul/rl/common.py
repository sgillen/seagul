import torch
import numpy as np
from torch.distributions import Normal, Categorical


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer (modifed from from https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs1_buf = torch.zeros([max_size, obs_dim], dtype=torch.float32)
        self.obs2_buf = torch.zeros([max_size, obs_dim], dtype=torch.float32)
        self.acts_buf = torch.zeros([max_size, act_dim], dtype=torch.float32)
        self.rews_buf = torch.zeros([max_size, 1], dtype=torch.float32)
        self.done_buf = torch.zeros([max_size, 1], dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(self, obs, next_obs, act, rew, done):
        
        insert_size = obs.shape[0]
        space_left = self.max_size - (self.ptr + insert_size)
        
        if(space_left >= 0):
            self.obs1_buf[self.ptr:self.ptr + insert_size,:] = obs
            self.obs2_buf[self.ptr:self.ptr + insert_size,:] = next_obs
            self.acts_buf[self.ptr:self.ptr + insert_size,:] = act
            self.rews_buf[self.ptr:self.ptr + insert_size,:] = rew
            self.done_buf[self.ptr:self.ptr + insert_size,:] = done
            
            self.ptr = self.ptr + insert_size
            self.size = max(self.size, self.ptr)
            
        else:
            
            space_at_end = self.max_size - self.ptr
            space_at_beg = (self.ptr + insert_size) - self.max_size
            
            self.obs1_buf[self.ptr:,:] = obs[:space_at_end,:]
            self.obs2_buf[self.ptr:,:] = next_obs[:space_at_end,:]
            self.acts_buf[self.ptr:,:] = act[:space_at_end,:]
            self.rews_buf[self.ptr:,:] = rew[:space_at_end,:]
            self.done_buf[self.ptr:,:] = done[:space_at_end,:]
            
            self.obs1_buf[:space_at_beg,:] = obs[space_at_end:,:]
            self.obs2_buf[:space_at_beg,:] = next_obs[space_at_end:,:]
            self.acts_buf[:space_at_beg,:] = act[space_at_end:,:]
            self.rews_buf[:space_at_beg,:] = rew[space_at_end:,:]
            self.done_buf[:space_at_beg,:] = done[space_at_end:,:]
            
            self.ptr = (self.ptr + insert_size) % self.max_size
            self.size = self.max_size 
            

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs1_buf[idxs], self.obs2_buf[idxs], self.acts_buf[idxs], self.rews_buf[idxs], self.done_buf[idxs])


def update_mean(data, cur_mean, cur_steps):
    new_steps = data.shape[0]
    return (torch.mean(data, 0) * new_steps + cur_mean * cur_steps) / (cur_steps + new_steps)


def update_std(data, cur_std, cur_steps):
    new_steps = data.shape[0]
    batch_var = torch.var(data, 0)

    if torch.isnan(batch_var).any():
        return cur_std
    else:
        cur_var = cur_std ** 2
        new_var = torch.std(data, 0) ** 2
        new_var[new_var < 1e-6] = cur_var[new_var < 1e-6]
        return torch.sqrt((new_var * new_steps + cur_var * cur_steps) / (cur_steps + new_steps))


# can make this faster I think?
def discount_cumsum(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards


class RandModel:
    """
    class that just takes actions from a uniform random distribution
    """

    def __init__(self, act_limit, act_size):
        self.act_limit = act_limit
        self.act_size = act_size

    def select_action(self, state, noise):
        return (torch.rand(self.act_size) * 2 * self.act_limit - self.act_limit, 1 / (self.act_limit * 2))


def make_schedule(std_schedule, num_steps):
    std_schedule = np.asarray(std_schedule)
    sched_length = std_schedule.shape[0]
    x_vals = np.linspace(0, num_steps, sched_length)
    std_lookup = lambda epoch: np.interp(epoch, x_vals, std_schedule)
    return std_lookup


def update_target_fn(fn, target_fn, polyak):

    fn_sd = fn.state_dict()
    target_sd = target_fn.state_dict()
    for layer in target_sd:
        target_sd[layer] = polyak * target_sd[layer] + (1 - polyak) * fn_sd[layer]

    target_fn.load_state_dict(target_sd)

    return target_fn