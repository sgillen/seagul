import gym
import torch
from seagul.rl.common import update_std, update_mean
from torch.multiprocessing import Pool
from functools import partial


def do_rollout_train(env_name, policy, postprocess, delta):
    env = gym.make(env_name)
    torch.nn.utils.vector_to_parameters(delta, policy.parameters())

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
    reward_list = postprocess(torch.tensor(reward_list))
    preprocess_sum = sum(reward_list)
    reward_sum = torch.as_tensor(sum(reward_list))

    env.close()
    return state_tens, reward_sum, preprocess_sum


def postprocess_default(x):
    return x


def ars(env_name, policy, n_epochs, n_workers=8, step_size=.02, n_delta=32, n_top=16, exp_noise=0.03, zero_policy=True, postprocess=postprocess_default):
    torch.autograd.set_grad_enabled(False)
    """
    Augmented Random Search
    https://arxiv.org/pdf/1803.07055

    Args:

    Returns:

    Example:
    """

    pool = Pool(processes=n_workers)
    env = gym.make(env_name)
    W = torch.nn.utils.parameters_to_vector(policy.parameters())
    n_param = W.shape[0]

    if zero_policy:
        W = torch.zeros_like(W)

    r_hist = []
    s_mean = torch.zeros(env.observation_space.shape[0])
    s_stdv = torch.ones(env.observation_space.shape[0])

    total_steps = 0
    exp_dist = torch.distributions.Normal(torch.zeros(n_delta, n_param), torch.ones(n_delta, n_param))
    do_rollout_partial = partial(do_rollout_train, env_name, policy, postprocess)

    for _ in range(n_epochs):

        deltas = exp_dist.sample()
        pm_W = torch.cat((W+(deltas*exp_noise), W-(deltas*exp_noise)))

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
            l_returns.append(plr); l_returns.append(mlr)
            top_returns.append(max(pr,mr))

        top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:n_top]
        p_returns = torch.stack(p_returns)[top_idx]
        m_returns = torch.stack(m_returns)[top_idx]
        l_returns = torch.stack(l_returns)[top_idx]

        r_hist.append(l_returns.mean())

        ep_steps = states.shape[0]
        s_mean = update_mean(states, s_mean, total_steps)
        s_stdv = update_std(states, s_stdv, total_steps)
        total_steps += ep_steps

        policy.state_means = s_mean
        policy.state_std = s_stdv
        do_rollout_partial = partial(do_rollout_train, env_name, policy, postprocess)

        W = W + (step_size / (n_delta * torch.cat((p_returns, m_returns)).std() + 1e-6)) * torch.sum((p_returns - m_returns)*deltas[top_idx].T, dim=1)

    pool.terminate()
    torch.nn.utils.vector_to_parameters(W, policy.parameters())
    return policy, r_hist


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    import seagul.envs
    import matplotlib.pyplot as plt
    from seagul.nn import MLP

    env_name = "Walker2DBulletEnv-v0"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    policy = MLP(in_size, out_size,0,0,bias=False)

    from seagul.mesh import variation_dim

    def var_dim_post(rews):
        return rews/variation_dim(rews)

    import time
    start = time.time()
    policy, r_hist = ars(env_name, policy, 50, n_workers=8, n_delta=64, n_top=32)
    print(time.time() - start)

    plt.plot(r_hist)
    plt.show()

    #env = gym.make(env_name)
    #state_hist, act_hist, returns = do_rollout(env_name, policy)
