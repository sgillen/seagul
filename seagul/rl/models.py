import torch
from torch.distributions import Normal, Categorical


class ppoModel:
    def __init__(self, policy, value_fn, action_var):
        self.policy = policy
        self.value_fn = value_fn
        self.action_var = action_var

    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, _ = self._select_action(state)
        value = self.value_fn(torch.as_tensor(state))
        logp = self._get_logp(state, action)

        return action, value, None , logp


    def _select_action(self, state):
        return select_cont_action(self.policy, state, self.action_var)

    def _get_logp(self, states, actions):
        return get_cont_logp(self.policy, states, actions, self.action_var)



class switchedPpoModel:
    def __init__(self, policy, nominal_policy,  value_fn, gate_fn, action_var, gate_var, env):
        self.policy = policy
        self.nominal_policy = nominal_policy
        self.value_fn = value_fn
        self.action_var = action_var
        self.gate_fn = gate_fn
        self.gate_var = gate_var
        self.env = env

    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        path = self._select_path(state)
        value = self.value_fn(torch.as_tensor(state))

        if(path):
            action = self.nominal_policy(self.env, state)
            logp = 0
        else:
            action, logp = self._select_action(self.policy, state, self.action_var)



        return action, value, None , logp

    def _select_action(self, state):
        return select_cont_action(self.policy, state, self.action_var)

    def _get_action_logp(self, states, actions):
        return get_cont_logp(self.policy, states, actions, self.action_var)

    def _select_path(self, state):
        return select_cont_action(self.gate_fn, state, self.gate_var)

    def _get_path_logp(self, states, actions):
        return get_cont_logp(self.gate_fn,  states, actions, self.gate_var)



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
    logprob = m.log_prob(action)
    return logprob
