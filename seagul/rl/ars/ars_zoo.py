import gym
from torch.multiprocessing import Process,Queue
import copy
from numpy.random import default_rng
import numpy as np
import torch
import time
import os
from seagul.mesh import mesh_dim
from seagul.rl.common import make_schedule
from seagul.zoo3_utils import load_zoo_agent, OFF_POLICY_ALGOS



def worker_fn(worker_q, master_q, algo, env_id, postprocess, get_trainable, seed):
    torch.set_grad_enabled(False)
    env, model = load_zoo_agent(env_id, algo)

    if seed:
        env.seed(seed)
    
    while True:
        data = master_q.get()
        W_flat, done = data
        if done == True:
            env.close()
            return
        else:
            torch.nn.utils.vector_to_parameters(torch.tensor(W_flat,requires_grad=False), get_trainable(model))
            states, returns, log_returns = do_rollout_train(env, model, postprocess)
            worker_q.put((returns, log_returns))


def do_rollout_train(env, model, postprocess):
    import time

    predict_time = 0
    step_time = 0
    before_loop = time.time()
    
    state_list = []
    act_list = []
    reward_list = []

    obs = env.reset()
    done = False

    while not done:
        state_list.append(np.copy(obs))

        before_predict = time.time()
        actions,_= model.predict(obs, deterministic=True)
        predict_time += (time.time() - before_predict)
        
        before_step = time.time()
        obs, reward, done, _ = env.step(actions)
        step_time += (time.time() - before_step)
        
        act_list.append(np.copy(actions))
        reward_list.append(reward)


    state_arr = np.stack(state_list).squeeze()
    act_arr = np.stack(act_list).squeeze()
    preprocess_sum = np.array(sum(reward_list))

    state_arr_n = state_arr

    reward_list = postprocess(state_arr_n, act_arr, np.array(reward_list))
    reward_sum = (np.sum(reward_list).item())
    total_time = time.time() - before_loop
    
    #Print(f"{len(reward_list)}: predict_time:{predict_time}------step_time:{step_time}-----unacount_time{total_time - predict_time - step_time}")

    return state_arr, reward_sum, preprocess_sum


def postprocess_default(obs,acts,rews):
    return rews

def get_on_policy_trainable(model):
    return model.policy.action_net.parameters()

def get_off_policy_trainable(model):
    return model.policy.actor.mu.parameters()

class ARSZooAgent:
    """
    This is a version of Augmented Random Search (https://arxiv.org/pdf/1803.07055) that uses arbitary pytorch polices. If you just want a linear policy see seagul/ars//ars_np for a version which uses pure numpy but is limited to linear policies. 

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
    def __init__(self, env_name, algo, seed=None, env_config=None, n_workers=24, n_delta=32, n_top=None,
                 step_size=.02, exp_noise=0.03, reward_stop=None, postprocessor=postprocess_default,
                 step_schedule=None, exp_schedule=None, 
                 ):
        self.env_name = env_name
        self.algo = algo
        self.n_workers = n_workers
        self.step_size = step_size
        self.n_delta = n_delta
        self.exp_noise = exp_noise
        self.postprocessor = postprocessor
        self.seed = seed
        self.r_hist = []
        self.raw_rew_hist = []
        self.total_epochs = 0
        self.reward_stop = reward_stop

        self.step_schedule = step_schedule
        self.exp_schedule = exp_schedule

        if n_top is None:
            n_top = n_delta
        self.n_top = n_top
        
        if env_config is None:
            env_config = {}
        self.env_config = env_config


        if algo in OFF_POLICY_ALGOS:
            self.get_trainable = get_off_policy_trainable
        else:
            self.get_trainable = get_on_policy_trainable  

        env, model = load_zoo_agent(env_name,algo)
        
        self.model = model


        if type(env.observation_space) == gym.spaces.dict.Dict:
            self.obs_size = env.observation_space['observation'].shape
        else:
            self.obs_size = env.observation_space.shape[0]

        self.act_size = env.action_space.shape[0]
        
        W_torch = torch.nn.utils.parameters_to_vector(self.get_trainable(self.model))
        self.W_flat = W_torch.detach().numpy()

        env.close()

        
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

            proc = Process(target=worker_fn, args=(worker_q, master_q, self.algo, self.env_name, self.postprocessor, self.get_trainable, self.seed))
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
            
            deltas = rng.standard_normal((self.n_delta, n_param), dtype=np.float32)
            #import ipdb; ipdb.set_trace()
            pm_W = np.concatenate((self.W_flat+(deltas*self.exp_noise), self.W_flat-(deltas*self.exp_noise)))

            start = time.time()

            for i,Ws in enumerate(pm_W):
                master_q_list[i % self.n_workers].put((Ws, False))
                
            results = []
            for i, _ in enumerate(pm_W):
                results.append(worker_q_list[i % self.n_workers].get())

            end = time.time()
            t = (end - start)
                
            p_returns = []
            m_returns = []
            l_returns = []
            top_returns = []

            for p_result, m_result in zip(results[:self.n_delta], results[self.n_delta:]):
                pr, plr = p_result
                mr, mlr = m_result

                p_returns.append(pr)
                m_returns.append(mr)
                l_returns.append(plr); l_returns.append(mlr)
                top_returns.append(max(pr,mr))

            top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:self.n_top]
            p_returns = np.stack(p_returns).astype(np.float32)[top_idx]
            m_returns = np.stack(m_returns).astype(np.float32)[top_idx]
            l_returns = np.stack(l_returns).astype(np.float32)[top_idx]

            if verbose and epoch % 10 == 0:
                from seagul.zoo3_utils import do_rollout_stable
                print(f"{epoch} : mean return: {l_returns.mean()}, top_return: {l_returns.max()}, eps:{self.n_delta*2/t}")

                env, model = load_zoo_agent(self.env_name, self.algo)
                torch.nn.utils.vector_to_parameters(torch.tensor(self.W_flat, requires_grad=False), self.get_trainable(self.model))
                for p in self.get_trainable(self.model):
                    print(p)

                o,a,r,info = do_rollout_stable(env, self.model)
                
                #                o_mdim  = o[200:]
                o_mdim = o
                try:
                    mdim, cdim, _, _ = mesh_dim(o_mdim)
                    print(f"main thread {self.env_name}:{self.algo}- reward={np.sum(r)}, mdim={mdim}, cdim={cdim}") 
                except:
                    print("naned out")
                    
            self.raw_rew_hist.append(np.stack(top_returns)[top_idx].mean())
            self.r_hist.append((p_returns.mean() + m_returns.mean())/2)

            self.total_epochs += 1
            self.W_flat = self.W_flat + (self.step_size / (self.n_delta * np.concatenate((p_returns, m_returns)).std() + 1e-6)) * np.sum((p_returns - m_returns)*deltas[top_idx].T, axis=1)

        for q in master_q_list:
            q.put((None, True))

        for proc in proc_list:
            proc.join()

        torch.nn.utils.vector_to_parameters(torch.tensor(self.W_flat, requires_grad=False), self.get_trainable(self.model))

        torch.set_grad_enabled(True)
        return self.model, self.raw_rew_hist[learn_start_idx:], locals()
