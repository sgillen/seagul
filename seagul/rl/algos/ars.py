import gym
import torch
import numpy as np

from seagul.nn import LinearNet

def do_rollout(policy, env):
    reward_list = []
    action_list = []
    state_list = []

    obs = env.reset()
    while not done:
        action = policy(torch.as_tensor(obs))
        obs, reward, done, _ = env.step(action.detach())
        
        action_list.append(action.clone())
        state_list.append(torch.as_tensor(obs).clone())   # Clone???
        reward_list.append(torch.as_tensor(reward).clone())



    reward_list = torch.stack(
    return reward_list, action_list, state_list

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

    th = torch.nn.utils.parameters_to_vector(model.parameters())


    
    s_mean = torch.zeros((n_param, 1)) 
    s_stdv = torch.ones((n_param, 1)) 
    
    total_steps = 0
    exp_dist = torch.distributions.Normal(torch.zeros(n_param), torch.ones(n_param) * exp_noise)

    for _ in range(num_epochs):

        delta = exp_dist.sample().reshape(n_param, 1)

        th_plus = (th + delta).reshape(-1);
        torch.nn.utils.vector_to_parameters(th, model.parameters())
        states_p, _, returns_p = do_rollout(policy)

        th_minus =  (th + delta).reshape(-1);
        torch.nn.utils.vector_to_parameters(th, model.parameters())
        states_n, _, returns_n = do_rollout(policy)

        returns = torch.cat((returns_p, returns_n))
        states = torch.cat((states_p, states_n))

        ep_steps = states.shape[0]
        s_mean = (states.mean(0) * ep_steps + s_mean * total_steps) / (total_steps + ep_steps)
        s_stdv = (states.std(0) * ep_steps + s_stdv * total_steps) / (total_steps + ep_steps)
        total_steps += ep_steps

        policy.state_means = s_mean
        policy.state_var = s_stdv

        torch.nn.utils.vector_to_parameters(th, model.parameters())
        
        # print(returns.std())
        th = th + np.array(
            step_size / (n_delta * returns.std() + 1e-6) * np.sum((returns_p - returns_n) * delta, 1)).reshape(n_param, -1)

    return th
