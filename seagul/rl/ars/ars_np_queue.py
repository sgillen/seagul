import gym
from multiprocessing import Process,Queue
import copy
from numpy.random import default_rng
import numpy as np
import time


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


def worker_fn(worker_q, master_q, env_name, env_config, postprocess, seed):
    env = gym.make(env_name, **env_config)
    env.seed(int(seed))
    while True:
        data = master_q.get()
        if data == "STOP":
            env.close()
            return
        else:
            W,state_mean,state_std = data
            policy = lambda x : W.T@((x - state_mean)/state_std)
            states, returns, log_returns = do_rollout_train(env, policy, postprocess)
            worker_q.put((states, returns, log_returns))


def do_rollout_train(env, policy, postprocess):
    state_list = []
    act_list = []
    reward_list = []

    obs = env.reset()
    done = False
    while not done:
        state_list.append(np.copy(obs))

        actions = policy(obs)
        obs, reward, done, _ = env.step(actions)

        act_list.append(np.array(actions))
        reward_list.append(reward)


    state_arr = np.stack(state_list)
    act_arr = np.stack(act_list)
    preprocess_sum = np.array(sum(reward_list))
    reward_list = postprocess(np.array(reward_list), state_arr, act_arr)
    reward_sum = np.asarray(sum(reward_list))

    return state_arr, reward_sum, preprocess_sum


def postprocess_default(rews, obs,acts):
    return rews


class ARSModel:
    """
    Just here to be compatible with the rest of seagul, ARS has such a simple policy that it doesn't really matter
    """
    def __init__(self, W, state_mean, state_std):
        self.W = W
        self.state_mean = state_mean
        self.state_std = state_std

    def step(self, obs):
        action = self.W.T@((obs - self.state_mean)/self.state_std)
        return action, None, None, None


class ARSAgent:
    """
    TODO
    """
    def __init__(self, env_name, seed, env_config=None, n_workers=24, n_delta=32, n_top=16,
                 step_size=.02, exp_noise=0.03, reward_stop=None, postprocessor=postprocess_default):
        self.env_name = env_name
        self.n_workers = n_workers
        self.step_size = step_size
        self.n_delta = n_delta
        self.n_top = n_top
        self.exp_noise = exp_noise
        self.postprocessor = postprocessor
        self.seed = seed
        self.r_hist = []
        self.lr_hist = []
        self.total_epochs = 0
        self.total_steps = 0
        self.reward_stop = reward_stop
        self.model = None

        if env_config is None:
            env_config = {}
        self.env_config = env_config

        env = gym.make(self.env_name, **self.env_config)

        self.W = np.zeros((env.observation_space.shape[0], env.action_space.shape[0]))
        self.state_mean = np.zeros(env.observation_space.shape[0])
        self.state_std = np.ones(env.observation_space.shape[0])

        self.W_list = []

    def learn(self, n_epochs, verbose=True):
        proc_list = []
        master_q_list = []
        worker_q_list = []
        learn_start_idx = copy.copy(self.total_epochs)

        for i in range(self.n_workers):
            master_q = Queue()
            worker_q = Queue()
            proc = Process(target=worker_fn, args=(worker_q, master_q, self.env_name, self.env_config, self.postprocessor, self.seed))
            proc.start()
            proc_list.append(proc)
            master_q_list.append(master_q)
            worker_q_list.append(worker_q)

        n_param = self.W.shape[0]*self.W.shape[1]

        rng = default_rng()         

        for epoch in range(n_epochs):

            if len(self.lr_hist) > 2 and self.reward_stop:
                if self.lr_hist[-1] >= self.reward_stop and self.lr_hist[-2] >= self.reward_stop:
                    early_stop = True
                    break
            
            W_flat = self.W.flatten()
            self.W_list.append(np.copy(W_flat))
            deltas = rng.standard_normal((self.n_delta, n_param))
            #import ipdb; ipdb.set_trace()
            pm_W = np.concatenate((W_flat+(deltas*self.exp_noise), W_flat-(deltas*self.exp_noise)))

            start = time.time()

            for i,Ws in enumerate(pm_W):
                master_q_list[i % self.n_workers].put((Ws.reshape(self.W.shape[0], self.W.shape[1]) ,self.state_mean,self.state_std))
                
            results = []
            for i, _ in enumerate(pm_W):
                results.append(worker_q_list[i % self.n_workers].get())

            end = time.time()
            t = (end - start)
                
            states = np.array([]).reshape(0,self.W.shape[0])
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

            if verbose:
                print(f"{epoch} : mean return: {np.mean(top_returns, axis=0)}, fps:{states.shape[0]/t}")

            self.lr_hist.append(l_returns.mean())
            self.r_hist.append((p_returns.mean() + m_returns.mean())/2)

            ep_steps = states.shape[0]
            self.state_mean = update_mean(states, self.state_mean, self.total_steps)
            self.state_std = update_std(states, self.state_std, self.total_steps)

            self.total_steps += ep_steps
            self.total_epochs += 1

            W_flat = W_flat + (self.step_size / (self.n_delta * np.concatenate((p_returns, m_returns)).std() + 1e-6)) * np.sum((p_returns - m_returns)*deltas[top_idx].T, axis=1)
            self.W = W_flat.reshape(self.W.shape[0], self.W.shape[1])

        for q in master_q_list:
            q.put("STOP")
        for proc in proc_list:
            proc.join()

        self.model = ARSModel(self.W, self.state_mean, self.state_std)
        return self.model, self.lr_hist[learn_start_idx:], locals()


if __name__ == "__main__":
    import pybullet_envs
    import seagul.envs
    import matplotlib.pyplot as plt

    env_name = "Humanoid-v2"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    from seagul.mesh import variation_dim

    def var_dim_post(rews):
        return rews/variation_dim(rews)

    import time
    start = time.time()
    agent = ARSAgent(env_name, seed=0, n_workers=24, n_delta=240, n_top=240, exp_noise=0.0075)
    rews = agent.learn(500)
    print(time.time() - start)
    print(rews)
    #plt.plot(rews)
    #plt.show()

    #
    # start = time.time()
    # rews = agent.learn(10)
    # print(time.time() - start)
    #
    # plt.plot(rews)
    # plt.show()


    #plt.plot(agent.lr_hist)
    #env = gym.make(env_name)
    #state_hist, act_hist, returns = do_rollout(env_name, policy)
