import torch
from torch.distributions import Normal, Categorical

# helper functions
# ============================================================================================

# takes a policy and the states and sample an action from it... (can we make this faster?)
def select_cont_action(policy, state, variance):
    means = policy(torch.as_tensor(state)).squeeze()
    m = Normal(loc=means, scale=torch.ones_like(means) * variance)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach(), logprob


# given a policy plus a state/action pair, what is the log liklihood of having taken that action?
def get_cont_logp(policy, states, actions, variance):
    means = policy(torch.as_tensor(states)).squeeze()
    m = Normal(loc=means, scale=torch.ones_like(means) * variance)
    logprob = m.log_prob(actions)
    return logprob


# takes a policy and the states and sample an action from it... (can we make this faster?)
def select_discrete_action(policy, state):
    probs = torch.tensor([policy(torch.as_tensor(state)), 1 - policy(torch.as_tensor(state))])
    m = Categorical(probs)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach(), logprob


# given a policy plus a state/action pair, what is the log liklihood of having taken that action?
def get_discrete_logp(policy, state, action):
    probs = torch.cat((policy(torch.as_tensor(state)), 1 - policy(torch.as_tensor(state))), dim=1)
    m = Categorical(probs)
    if any(torch.isnan(state).flatten()):
        print("whoops")
    logprob = m.log_prob(action)
    return logprob


# can make this faster I think?
def discount_cumsum(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards
