import gym
import torch
from seagul.rl.common import update_std, update_mean
from torch.multiprocessing import Process,Pipe
import copy
import cProfile
import os


class ARSAgent:
    def __init__(self, seed, env_config=None, n_workers=8, zero_policy=True, env_steps=1000,step_size=.02, n_delta=32, n_top=16, exp_noise=0.03, reward_stop=None):
        self.step_size = step_size
        self.n_delta = n_delta
        self.n_top = n_top
        self.env_steps = env_steps
        self.exp_noise = exp_noise
        self.seed = seed
        self.reward_stop = reward_stop
        self.r_hist = []
        self.lr_hist = []
        self.total_epochs = 0
        self.total_steps = 0

        if env_config is None:
            env_config = {}
        self.env_config = env_config

        self.W = None

        self.state_means = None
        self.state_std = None

        
    def do_vec_rollout(self, env, W):
        obs_list = []
        act_list = []
        rew_list = []
        dne_list = []
        
        env.reset()
        obs = env.observe()
        
        cur_step = 0
        
        while cur_step < self.env_steps:
            obs = torch.as_tensor(obs)
            obs_list.append(obs.detach().clone())

            obs_n  = (obs - self.state_means) / self.state_std

            
            act = torch.bmm(obs_n.reshape(W.shape[0], 1, W.shape[1]), W).reshape(W.shape[0], W.shape[2])

            rew, dne = env.step(act.numpy())
            obs = env.observe()
            
            act_list.append(torch.as_tensor(act))
            rew_list.append(torch.as_tensor(rew))
            dne_list.append(torch.as_tensor(dne))
        
            cur_step+=1


        obs_tens = torch.stack(obs_list)
        act_tens = torch.stack(act_list)
        rew_tens = torch.stack(rew_list)
        dne_tens = torch.stack(dne_list)
        
        return obs_tens, act_tens, rew_tens, dne_tens

        
        
    def learn(self, n_epochs, env):
        torch.autograd.set_grad_enabled(False)

        learn_start_idx = copy.copy(self.total_epochs)
        
        # shortcuts
        ob_dim = env.num_obs
        act_dim = env.num_acts

        if self.W is None:
            self.W = torch.zeros((ob_dim, act_dim))
            self.W = self.W.flatten()

        if self.state_means is None:
            self.state_means = torch.zeros(ob_dim)
            self.state_std = torch.ones(ob_dim)
            
        n_param = self.W.shape[0]
        
        torch.manual_seed(self.seed)
        exp_dist = torch.distributions.Normal(torch.zeros(self.n_delta, n_param), torch.ones(self.n_delta, n_param))

        for epoch in range(n_epochs):

            if len(self.lr_hist) > 2 and self.reward_stop:
                if self.lr_hist[-1] >= self.reward_stop and self.lr_hist[-2] >= self.reward_stop:
                    early_stop = True
                    break

            deltas = exp_dist.sample()
            W_deltas = torch.cat((self.W+(deltas*self.exp_noise), self.W-(deltas*self.exp_noise)),dim=1)
            
            import time
            start = time.time()
            

            obs, acts, rew, _ = self.do_vec_rollout(env, W_deltas.reshape(self.n_delta*2, ob_dim, act_dim))

            
            end = time.time()
            t = (end - start)
            
            rew_sums = rew.sum(dim=0)

            plus_rets = rew_sums[:self.n_delta]
            minus_rets = rew_sums[self.n_delta:]
                                              

            top_rets , top_idx = torch.sort(torch.max(plus_rets, minus_rets), descending=True)
            top_rets = top_rets[:self.n_top]
            top_idx = top_idx[:self.n_top]

            self.r_hist.append(top_rets.mean())

            print(epoch, self.r_hist[-1], rew.numel()/t, acts.max())

            ep_steps = obs.shape[1]
            self.state_means = update_mean(obs.reshape(-1,ob_dim), self.state_means, self.total_steps)
            self.state_std = update_std(obs.reshape(-1, ob_dim), self.state_std, self.total_steps)

            self.total_steps += ep_steps
            self.total_epochs += 1

            #            import ipdb; ipdb.set_trace()

            self.W = self.W + (self.step_size / (self.n_top * (top_rets.std() + 1e-6))) * torch.sum((plus_rets[top_idx] - minus_rets[top_idx])*deltas[top_idx].T, dim=1)

        return self.W, self.lr_hist[learn_start_idx:], locals()

