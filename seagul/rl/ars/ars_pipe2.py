import gym
import torch
from seagul.rl.common import update_std, update_mean
from torch.multiprocessing import Process,Pipe
import copy
import cProfile
import os


def worker_fn(worker_con, env_name, env_config, policy, postprocess, seed):
   # pr = cProfile.Profile()
   # pr.enable()
    env = gym.make(env_name, **env_config)
    env.seed(int(seed))
    while True:
        data = worker_con.recv()

        if data == "STOP":
    #        pr.disable()
    #        pr.dump_stats("pb.stats"+str(os.getpid()))
            env.close()
            return
        else:
            W,state_mean,state_std = data

            policy.state_std = state_std
            policy.state_means = state_mean

            states, returns, log_returns = do_rollout_train(env, policy, postprocess, W)
            worker_con.send((states, returns, log_returns))


def do_rollout_train(env, policy, postprocess, delta):
    torch.nn.utils.vector_to_parameters(delta, policy.parameters())

    state_list = []
    act_list = []
    reward_list = []

    obs = env.reset()
    done = False
    while not done:
        state_list.append(torch.as_tensor(obs))

        actions = policy(torch.as_tensor(obs))
        obs, reward, done, _ = env.step(actions)

        act_list.append(torch.as_tensor(actions))
        reward_list.append(reward)

    state_tens = torch.stack(state_list)
    act_tens = torch.stack(act_list)
    preprocess_sum = torch.as_tensor(sum(reward_list))
    reward_list = postprocess(torch.tensor(reward_list), state_tens, act_tens)
    reward_sum = torch.as_tensor(sum(reward_list))

    return state_tens, reward_sum, preprocess_sum


def postprocess_default(rews, obs,acts):
    return rews


class ARSAgent:
    def __init__(self, env_name, policy, seed, env_config=None, n_workers=8, step_size=.02, n_delta=32, n_top=16, exp_noise=0.03, postprocessor=postprocess_default):
        self.env_name = env_name
        self.policy = policy
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

        if env_config is None:
            env_config = {}
        self.env_config = env_config

    def learn(self, n_epochs):
        torch.autograd.set_grad_enabled(False)

        proc_list = []
        master_pipe_list = []
        learn_start_idx = copy.copy(self.total_epochs)

        for i in range(self.n_workers):
            master_con, worker_con= Pipe()
            proc = Process(target=worker_fn, args=(worker_con, self.env_name, self.env_config, self.policy, self.postprocessor, self.seed))
            proc.start()
            proc_list.append(proc)
            master_pipe_list.append(master_con)

        W = torch.nn.utils.parameters_to_vector(self.policy.parameters())
        n_param = W.shape[0]

        torch.manual_seed(self.seed)
        exp_dist = torch.distributions.Normal(torch.zeros(self.n_delta, n_param), torch.ones(self.n_delta, n_param))

        for _ in range(n_epochs):

            deltas = exp_dist.sample()
            pm_W = torch.cat((W+(deltas*self.exp_noise), W-(deltas*self.exp_noise)))

            for i,Ws in enumerate(pm_W):
                master_pipe_list[i % self.n_workers].send((Ws,self.policy.state_means,self.policy.state_std))

            results = []
            for i, _ in enumerate(pm_W):
                results.append(master_pipe_list[i % self.n_workers].recv())

            states = torch.empty(0)
            p_returns = []
            m_returns = []
            l_returns = []
            top_returns = []

            for p_result, m_result in zip(results[:self.n_delta], results[self.n_delta:]):
                ps, pr, plr = p_result
                ms, mr, mlr = m_result

                states = torch.cat((states, ms, ps), dim=0)
                p_returns.append(pr)
                m_returns.append(mr)
                l_returns.append(plr); l_returns.append(mlr)
                top_returns.append(max(pr,mr))

            top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:self.n_top]
            p_returns = torch.stack(p_returns)[top_idx]
            m_returns = torch.stack(m_returns)[top_idx]
            l_returns = torch.stack(l_returns)[top_idx]

            self.lr_hist.append(l_returns.mean())
            self.r_hist.append((p_returns.mean() + m_returns.mean())/2)

            ep_steps = states.shape[0]
            self.policy.state_means = update_mean(states, self.policy.state_means, self.total_steps)
            self.policy.state_std = update_std(states, self.policy.state_std, self.total_steps)

            self.total_steps += ep_steps
            self.total_epochs += 1

            W = W + (self.step_size / (self.n_delta * torch.cat((p_returns, m_returns)).std() + 1e-6)) * torch.sum((p_returns - m_returns)*deltas[top_idx].T, dim=1)

        for pipe in master_pipe_list:
            pipe.send("STOP")
        for proc in proc_list:
            proc.join()

        torch.nn.utils.vector_to_parameters(W, self.policy.parameters())
        return self.lr_hist[learn_start_idx:]


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    import pybullet_envs
    import seagul.envs
    import matplotlib.pyplot as plt
    from seagul.nn import MLP

    env_name = "HalfCheetahBulletEnv-v0"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    policy = MLP(in_size, out_size,0,0,bias=False)

    from seagul.mesh import variation_dim

    def var_dim_post(rews):
        return rews/variation_dim(rews)

    import time
    start = time.time()
    agent = ARSAgent(env_name, policy, seed=0, n_workers=8, n_delta=32, n_top=16)
    rews = agent.learn(5)
    print(time.time() - start)
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
