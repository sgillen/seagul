import gym
import seagul.envs
import torch
import numpy as np
import copy

from seagul.nn import LinearNet

torch.set_default_dtype(torch.float64)


def do_rollout(env, policy):

    state_list = []
    action_list = []
    reward_list = []

    obs = env.reset()
    done = False
    while not done:
        actions = policy(torch.as_tensor(obs))
        obs, reward, done, _ = env.step(actions.detach())

        state_list.append(torch.as_tensor(obs).clone())  # Clone???
        action_list.append(actions.clone())
        reward_list.append(torch.as_tensor(reward).clone())

    state_tens = torch.stack(state_list)
    action_tens = torch.stack(action_list)
    reward_tens = torch.stack(reward_list)
    return state_tens, action_tens, reward_tens


def ars(env_name, policy, num_epochs=10, step_size=1, n_delta=10, exp_noise=0.3):
    """
    Augmented Random Search
    https://arxiv.org/pdf/1803.07055

    Args:

    Returns:

    Example:
    """

    env = gym.make(env_name)
    # env_vec = lambda x: torch.tensor([env for _ in range (n_delta)])
    env_vec = env  # TODO

    policy_list = [copy.deepcopy(policy) for _ in range(n_delta)]
    # policy_vec = lambda x: torch.tensor([p(x) for p in policy_list])
    policy_vec = policy  # TODO

    th = torch.nn.utils.parameters_to_vector(policy.parameters())
    n_param = th.shape[0]

    s_mean = torch.zeros((n_param,))
    s_stdv = torch.ones((n_param,))

    total_steps = 0
    exp_dist = torch.distributions.Normal(torch.zeros(n_param), torch.ones(n_param) * exp_noise)

    for _ in range(num_epochs):

        delta = exp_dist.sample()

        th_plus = th + delta
        torch.nn.utils.vector_to_parameters(th_plus, policy.parameters())
        states_p, _, returns_p = do_rollout(env_vec, policy_vec)

        th_minus = th + delta
        torch.nn.utils.vector_to_parameters(th_minus, policy.parameters())
        states_n, _, returns_n = do_rollout(env_vec, policy_vec)

        returns = torch.cat((returns_p, returns_n))
        states = torch.cat((states_p, states_n))

        ep_steps = states.shape[0]
        s_mean = (states.mean(0) * ep_steps + s_mean * total_steps) / (total_steps + ep_steps)
        s_stdv = (states.std(0) * ep_steps + s_stdv * total_steps) / (total_steps + ep_steps)
        total_steps += ep_steps

        policy.state_means = s_mean
        policy.state_var = s_stdv

        torch.nn.utils.vector_to_parameters(th, policy.parameters())

        # print(returns.std())
        th = th + step_size / (n_delta * returns.std() + 1e-6) * torch.sum(returns_p - returns_n, 0) * delta
    return th


if __name__ == "__main__":
    import seagul.envs

    policy = LinearNet(4, 1)
    env_name = "su_cartpole-v0"
    th = ars(env_name, policy)

    env = gym.make(env_name)
    state_hist, act_hist, returns = do_rollout(env, policy)
    print(returns)
