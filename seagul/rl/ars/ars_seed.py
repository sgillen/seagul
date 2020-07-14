import torch
import gym
import numpy as np
from seagul.rl.common import update_std, update_mean
from functools import partial
from torch.multiprocessing import Pool
from anytree import Node
import copy


def ars(env_name, n_epochs, env_config, step_size, n_delta, n_top, exp_noise, n_workers, policy, seed):
        torch.autograd.set_grad_enabled(False)  # Gradient free baby!
        pool = Pool(processes=n_workers)

        W = torch.nn.utils.parameters_to_vector(policy.parameters())
        n_param = W.shape[0]

        if env_config is None:
            env_config = {}

        env = gym.make(env_name, **env_config)

        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        total_steps = 0
        r_hist = []

        exp_dist = torch.distributions.Normal(torch.zeros(n_delta, n_param), torch.ones(n_delta, n_param))
        do_rollout_partial = partial(do_rollout_train, env_name, policy)

        for _ in range(n_epochs):

            deltas = exp_dist.sample()

            ###
            pm_W = torch.cat((W + (deltas * exp_noise), W - (deltas * exp_noise)))

            results = pool.map(do_rollout_partial, pm_W)

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
                l_returns.append(plr);
                l_returns.append(mlr)
                top_returns.append(max(pr, mr))

            top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:n_top]
            p_returns = torch.stack(p_returns)[top_idx]
            m_returns = torch.stack(m_returns)[top_idx]
            l_returns = torch.stack(l_returns)[top_idx]

            r_hist.append(l_returns.mean())
            ###

            W = W + (step_size / (n_delta * torch.cat((p_returns, m_returns)).std() + 1e-6)) * torch.sum(
                (p_returns - m_returns) * deltas[top_idx].T, dim=1)

            ep_steps = states.shape[0]
            policy.state_means = update_mean(states, policy.state_means, total_steps)
            policy.state_std = update_std(states, policy.state_std, total_steps)
            do_rollout_partial = partial(do_rollout_train, env_name, policy)

            total_steps += ep_steps

            torch.nn.utils.vector_to_parameters(W, policy.parameters())

        return policy, r_hist


def meta_ars(env_name, policy, meta_epochs, meta_seed, n_seeds=4, n_top_seeds=1, n_workers=4, mean_lookback=10,
             ars_epochs=10, env_config=None, step_size=.02, n_delta=32, n_top=16, exp_noise=0.03):

    n_children = n_seeds//n_top_seeds
    np.random.seed(meta_seed)

    W = torch.nn.utils.parameters_to_vector(policy.parameters())
    W = torch.zeros_like(W)
    torch.nn.utils.vector_to_parameters(W, policy.parameters())

    pool = Pool(processes=n_seeds)
    ars_partial = partial(ars, env_name, ars_epochs, env_config, step_size, n_delta, n_top, exp_noise, n_workers)
    #root = Node(meta_seed)
    reward_log = []

    top_policies = []
    for _ in range(n_top_seeds):
        top_policies.append(copy.deepcopy(policy))

    for epoch in range(meta_epochs):
        pols_and_seeds = []
        for pol in top_policies:
            for _ in range(n_children):
                pols_and_seeds.append((pol, int(np.random.randint(0, 2**32-1, 1))))

        results = pool.starmap(ars_partial, pols_and_seeds)

        p_list = []
        r_list = []
        for result in results:
            policy, rews = result
            p_list.append(policy)
            r = torch.stack(rews[-mean_lookback:])
            r_list.append(r.mean())

        top_idx = sorted(range(len(r_list)), key=lambda k: r_list[k], reverse=True)[:n_top_seeds]
        for i in top_idx:
            top_policies.append(p_list[i])

        reward_log.append(max(r_list))

    return top_policies, reward_log


def do_rollout_train(env, policy, W):
    torch.nn.utils.vector_to_parameters(W, policy.parameters())

    state_list = []
    reward_list = []

    obs = env.reset()
    done = False
    while not done:
        state_list.append(torch.as_tensor(obs))

        actions = policy(torch.as_tensor(obs))
        obs, reward, done, _ = env.step(actions)

        reward_list.append(reward)

    state_tens = torch.stack(state_list)
    reward_sum = torch.as_tensor(sum(reward_list))

    return state_tens, reward_sum


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    import seagul.envs
    import matplotlib.pyplot as plt
    from seagul.nn import MLP

    env_name = "HalfCheetah-v2"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    policy = MLP(in_size, out_size, 0, 0, bias=False)

    from seagul.mesh import variation_dim

    def var_dim_post(rews):
        return rews/variation_dim(rews)

    import time
    start = time.time()
    policys, r_hist = meta_ars(env_name, policy, 2, 0)
    print(time.time() - start)

    plt.plot(r_hist)
    plt.show()

    #env = gym.make(env_name)
    #state_hist, act_hist, returns = do_rollout(env_name, policy)
