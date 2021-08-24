import gym
from torch.multiprocessing import Process,Queue
import copy
from numpy.random import default_rng
import numpy as np
import torch
import time
from seagul.rl.common import make_schedule


def update_mean(data, cur_mean, cur_steps):
    new_steps = data.shape[0]
    return (np.mean(data, 0) * new_steps + cur_mean * cur_steps) / (cur_steps + new_steps)


def update_std(data, cur_std, cur_steps):
    new_steps = data.shape[0]

    if np.isnan(data).any():
        return cur_std
    else:
        cur_var = cur_std ** 2
        new_var = np.std(data, 0) ** 2
        new_var[new_var < 1e-6] = cur_var[new_var < 1e-6]
        return np.sqrt((new_var * new_steps + cur_var * cur_steps) / (cur_steps + new_steps))


def worker_fn(worker_q, master_q, model, env_name, env_config, postprocess, seed):
    torch.set_grad_enabled(False)
    env = gym.make(env_name, **env_config)
    env.seed(int(seed))
    while True:
        data = master_q.get()
        if data == "STOP":
            env.close()
            return
        else:
            W_flat,state_mean,state_std = data
            torch.nn.utils.vector_to_parameters(torch.tensor(W_flat,requires_grad=False), model.policy.parameters())
            model.policy.state_means = torch.from_numpy(state_mean)
            model.policy.state_std = torch.from_numpy(state_std)
            states, returns, log_returns = do_rollout_train(env, model, postprocess)
            worker_q.put((states, returns, log_returns))


def do_rollout_train(env, model, postprocess):
    state_list = []
    act_list = []
    reward_list = []

    obs = env.reset()
    done = False
    while not done:
        state_list.append(np.copy(obs))

        actions,_,_,_ = model.step(obs.reshape(1,-1))
        actions = actions.detach().numpy()
        obs, reward, done, _ = env.step(actions)

        act_list.append(np.array(actions))
        reward_list.append(reward)

    state_arr = np.stack(state_list)
    act_arr = np.stack(act_list)
    preprocess_sum = np.array(sum(reward_list))

    state_arr_n = (state_arr - np.asarray(model.policy.state_means))/np.asarray(model.policy.state_std)
    reward_list = postprocess(state_arr_n, act_arr, np.array(reward_list))
    reward_sum = (np.sum(reward_list).item())

    return state_arr, reward_sum, preprocess_sum


def postprocess_default(obs,acts,rews):
    return rews


class ARSTorchModel:
    """
    Just here to be compatible with the rest of seagul, ARS has such a simple policy that it doesn't really matter
    """
    def __init__(self, policy):
        self.policy = policy

    def step(self, obs):
        with torch.no_grad():
            action = self.policy(obs)
            return action, None, None, None


class ARSTorchAgent:
    """
    This is a version of Augmented Random Search (https://arxiv.org/pdf/1803.07055) that uses arbitary pytorch polices. If you just want a linear policy see seagul/ars/rl/ars_np for a version which uses pure numpy but is limited to linear policies. 

    Args:
        env_name: name of the openAI gym env to solve
        seed: the random seed to use
        env_config: dictionary containing kwargs for the environment
        n_workers: number of workers to use
        n_delta: number of deltas to try at each update stepx
        n_top: number of deltas to use in the update calculation at each step
        step_size: it's the step size... alpha from the original paper.
        exp_noise: exploration noise, sigma from the paper.
        reward_stop: reward threshold to stop training at.
        postprocessor: reward post processor to use, default is none.
        step_schedule: an iterable of two step sizes to linearly interpolate between as training goes on, overrides step_size
        exp_schedule:  an iterable of two exp noises to linearly interpolate between as training goes on, overrides step_size
            
    """
    def __init__(self, env_name, model, seed, env_config=None, n_workers=24, n_delta=32, n_top=32,
                 step_size=.02, exp_noise=0.03, reward_stop=None, postprocessor=postprocess_default,
                 step_schedule=None, exp_schedule=None
                 ):
        self.env_name = env_name
        self.n_workers = n_workers
        self.step_size = step_size
        self.n_delta = n_delta
        self.n_top = n_top
        self.exp_noise = exp_noise
        self.postprocessor = postprocessor
        self.seed = seed
        self.r_hist = []
        self.raw_rew_hist = []
        self.total_epochs = 0
        self.total_steps = 0
        self.reward_stop = reward_stop
        self.model = None

        self.step_schedule = step_schedule
        self.exp_schedule = exp_schedule

        if env_config is None:
            env_config = {}
        self.env_config = env_config


        env = gym.make(self.env_name, **self.env_config)
        self.obs_size = env.observation_space.shape[0]
        self.act_size = env.action_space.shape[0]

        self.model = model
        W_torch = torch.nn.utils.parameters_to_vector(model.policy.parameters())
        self.W_flat = W_torch.detach().numpy()
        
        self.state_mean = np.zeros(self.obs_size)
        self.state_std = np.ones(self.obs_size)

        
    def learn(self, n_epochs, verbose=True):
        torch.set_grad_enabled(False)
        proc_list = []
        master_q_list = []
        worker_q_list = []
        learn_start_idx = copy.copy(self.total_epochs)

        if self.step_schedule:
            step_lookup = make_schedule(self.step_schedule, n_epochs)

        if self.exp_schedule:
            exp_lookup = make_schedule(self.exp_schedule, n_epochs)

        for i in range(self.n_workers):
            master_q = Queue()
            worker_q = Queue()
            proc = Process(target=worker_fn, args=(worker_q, master_q, self.model, self.env_name, self.env_config, self.postprocessor, self.seed))
            proc.start()
            proc_list.append(proc)
            master_q_list.append(master_q)
            worker_q_list.append(worker_q)

        n_param = self.W_flat.shape[0]

        rng = default_rng()         

        for epoch in range(n_epochs):
            if self.step_schedule:
                self.step_size = step_lookup(epoch)
            if self.exp_schedule:
                self.exp_noise = exp_lookup(epoch)

            if len(self.raw_rew_hist) > 2 and self.reward_stop:
                if self.raw_rew_hist[-1] >= self.reward_stop and self.raw_rew_hist[-2] >= self.reward_stop:
                    early_stop = True
                    break
            
            deltas = rng.standard_normal((self.n_delta, n_param))
            #import ipdb; ipdb.set_trace()
            pm_W = np.concatenate((self.W_flat+(deltas*self.exp_noise), self.W_flat-(deltas*self.exp_noise)))

            start = time.time()

            for i,Ws in enumerate(pm_W):
                master_q_list[i % self.n_workers].put((Ws ,self.state_mean,self.state_std))
                
            results = []
            for i, _ in enumerate(pm_W):
                results.append(worker_q_list[i % self.n_workers].get())

            end = time.time()
            t = (end - start)
                
            states = np.array([]).reshape(0,self.obs_size)
            p_returns = []
            m_returns = []
            l_returns = []
            top_returns = []

            for p_result, m_result in zip(results[:self.n_delta], results[self.n_delta:]):
                ps, pr, plr = p_result
                ms, mr, mlr = m_result

                states = np.concatenate((states, ms, ps), axis=0)
                p_returns.append(pr)
                m_returns.append(mr)
                l_returns.append(plr); l_returns.append(mlr)
                top_returns.append(max(pr,mr))

            top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:self.n_top]
            p_returns = np.stack(p_returns)[top_idx]
            m_returns = np.stack(m_returns)[top_idx]
            l_returns = np.stack(l_returns)[top_idx]

            if verbose and epoch % 10 == 0:
                print(f"{epoch} : mean return: {l_returns.mean()}, top_return: {l_returns.max()}, fps:{states.shape[0]/t}")

            self.raw_rew_hist.append(np.stack(top_returns)[top_idx].mean())
            self.r_hist.append((p_returns.mean() + m_returns.mean())/2)

            ep_steps = states.shape[0]
            self.state_mean = update_mean(states, self.state_mean, self.total_steps)
            self.state_std = update_std(states, self.state_std, self.total_steps)

            self.total_steps += ep_steps
            self.total_epochs += 1

            self.W_flat = self.W_flat + (self.step_size / (self.n_delta * np.concatenate((p_returns, m_returns)).std() + 1e-6)) * np.sum((p_returns - m_returns)*deltas[top_idx].T, axis=1)


        for q in master_q_list:
            q.put("STOP")
        for proc in proc_list:
            proc.join()

        torch.nn.utils.vector_to_parameters(torch.tensor(self.W_flat), self.model.policy.parameters())

        self.model.policy.state_means = torch.from_numpy(self.state_mean)
        self.model.policy.state_std = torch.from_numpy(self.state_std)

        torch.set_grad_enabled(True)
        return self.model, self.raw_rew_hist[learn_start_idx:], locals()
