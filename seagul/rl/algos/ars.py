import gym
import torch

from seagul.nn import LinearNet



def do_rollout(policy, env):
    max_ep_length = 100
    reward_hist = torch.zeros((max_ep_length, 1))
    action_hist = torch.zeros((max_ep_length, 1))
    state_hist = torch.zeros((max_ep_length, 4))
    obs = env.reset()
    for i in range(max_ep_length):
        action = policy(torch.as_tensor(obs))
        obs, reward, done, _ = env.step(action.detach())
        action_hist[i, :] = action.clone()
        state_hist[i, :] = torch.tensor(obs).clone()
        reward_hist[i, :] = reward
        # env.render()
        if done:
            break

    return reward_hist, action_hist, state_hist



def ars(
        env_name,
        policy,
        num_epochs=1000,
        step_size = 1,
        n_delta = 10,
        exp_noise = .3,
):

    """
    Augmented Random Search
    https://arxiv.org/pdf/1803.07055

    Args:

    Returns:

    Example:


    """

    n_param = policy
    n_param = 4
    th = torch.zeros((n_param, 1))
    s_mean = torch.zeros((n_param, 1))
    s_stdv = torch.ones((n_param, 1))
    policy = LinearNet(n_param, 1)
    total_steps = 0

    exp_dist = torch.distributions.Normal(torch.zeros(n_param), torch.ones(n_param) * exp_noise)

    for _ in range(num_epochs):
        delta = exp_dist.sample().reshape(n_param, 1)

        policy.layer.weight[0, :] = (th + delta).reshape(-1);
        states_p, _, returns_p = do_rollout(policy)

        policy.layer.weight[0, :] = (th + delta).reshape(-1);
        states_n, _, returns_n = do_rollout(policy)

        returns = torch.cat((returns_p, returns_n))
        states = torch.cat((states_p, states_n))

        ep_steps = states.shape[0]
        s_mean = (states.mean(0) * ep_steps + s_mean * total_steps) / (total_steps + ep_steps)
        s_stdv = (states.std(0) * ep_steps + s_stdv * total_steps) / (total_steps + ep_steps)
        total_steps += ep_steps

        policy.state_means = s_mean
        policy.state_var = s_stdv

        # print(returns.std())
        th = th + np.array(
            step_size / (n_delta * returns.std() + 1e-6) * np.sum((returns_p - returns_n) * delta, 1)).reshape(n_param,
                                                                                                               -1)

    return th