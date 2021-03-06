import gym
import torch
from seagul.rl.common import update_std, update_mean
from torch.multiprocessing import Process,Pipe
import cProfile
import os

def worker_fn(worker_con, env_name, env_config, policy, postprocess):
    # pr = cProfile.Profile()
    # pr.enable()
    env = gym.make(env_name, **env_config)
    epoch = 0

    while True:
        data = worker_con.recv()

        if data == "STOP":
            # pr.disable()
            # pr.print_stats()
            env.close()
            return
        else:
            W,state_mean,state_std = data

            policy.state_std = state_std
            policy.state_means = state_mean

            states, returns, log_returns = do_rollout_train(env, policy, postprocess, W)
            worker_con.send((states, returns, log_returns))
            epoch +=1


def do_rollout_train(env, policy, postprocess, W):
    torch.nn.utils.vector_to_parameters(W, policy.parameters())

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
    nstate_tens = (state_tens - policy.state_means) / policy.state_std
    reward_list = postprocess(torch.tensor(reward_list), nstate_tens, act_tens)
    reward_sum = torch.as_tensor(sum(reward_list))

    return state_tens, reward_sum, preprocess_sum


def postprocess_default(rews, obs,acts):
    return rews


def ars(env_name, policy, n_epochs, env_config={}, n_workers=8, step_size=.02, n_delta=32, n_top=16, exp_noise=0.03, zero_policy=True, learn_means=True, postprocess=postprocess_default):
    torch.autograd.set_grad_enabled(False)
    """
    Augmented Random Search
    https://arxiv.org/pdf/1803.07055

    Args:

    Returns:

    Example:
    """

    proc_list = []
    master_pipe_list = []

    for i in range(n_workers):
        master_con, worker_con= Pipe()
        proc = Process(target=worker_fn, args=(worker_con, env_name, env_config, policy, postprocess))
        proc.start()
        proc_list.append(proc)
        master_pipe_list.append(master_con)

    W = torch.nn.utils.parameters_to_vector(policy.parameters())
    n_param = W.shape[0]

    if zero_policy:
        W = torch.zeros_like(W)

    env = gym.make(env_name,**env_config)
    s_mean = policy.state_means
    s_std = policy.state_std
    total_steps = 0
    env.close()

    r_hist = []
    lr_hist = []

    exp_dist = torch.distributions.Normal(torch.zeros(n_delta, n_param), torch.ones(n_delta, n_param))

    for epoch in range(n_epochs):

        deltas = exp_dist.sample()
        pm_W = torch.cat((W+(deltas*exp_noise), W-(deltas*exp_noise)))

        for i,Ws in enumerate(pm_W):
            master_pipe_list[i % n_workers].send((Ws,s_mean,s_std))

        results = []
        for i, _ in enumerate(pm_W):
            results.append(master_pipe_list[i % n_workers].recv())

        states = torch.empty(0)
        p_returns = []
        m_returns = []
        l_returns = []
        top_returns = []

        for p_result, m_result in zip(results[:n_delta], results[n_delta:]):
            ps, pr, plr = p_result
            ms, mr, mlr = m_result

            states = torch.cat((states, ms, ps), dim=0)
            p_returns.append(pr)
            m_returns.append(mr)
            l_returns.append(plr); l_returns.append(mlr)
            top_returns.append(max(pr,mr))

        top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:n_top]
        p_returns = torch.stack(p_returns)[top_idx]
        m_returns = torch.stack(m_returns)[top_idx]
        l_returns = torch.stack(l_returns)[top_idx]

        lr_hist.append(l_returns.mean())
        r_hist.append((p_returns.mean() + m_returns.mean())/2)

        ep_steps = states.shape[0]
        s_mean = update_mean(states, s_mean, total_steps)
        s_std = update_std(states, s_std, total_steps)
        total_steps += ep_steps

        if epoch % 5 == 0:
            print(f"epoch: {epoch}, reward: {lr_hist[-1].item()}, processed reward: {r_hist[-1].item()} ")

        W = W + (step_size / (n_delta * torch.cat((p_returns, m_returns)).std() + 1e-6)) * torch.sum((p_returns - m_returns)*deltas[top_idx].T, dim=1)

    for pipe in master_pipe_list:
        pipe.send("STOP")
    policy.state_means = s_mean
    policy.state_std = s_std
    torch.nn.utils.vector_to_parameters(W, policy.parameters())
    return policy, r_hist, lr_hist


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    import seagul.envs
    import matplotlib.pyplot as plt
    from seagul.nn import MLP

    env_name = "HalfCheetah-v2"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    policy = MLP(in_size, out_size,0,0,bias=False)

    from seagul.mesh import variation_dim

    def var_dim_post(rews):
        return rews/variation_dim(rews)

    import time
    start = time.time()
    policy, r_hist, lr_hist = ars(env_name, policy, 20, n_workers=8, n_delta=32, n_top=16)
    print(time.time() - start)

    plt.plot(lr_hist)
    plt.show()

    #env = gym.make(env_name)
    #state_hist, act_hist, returns = do_rollout(env_name, policy)
