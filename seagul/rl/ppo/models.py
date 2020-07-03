import torch
from torch.distributions import Normal, Categorical
import numpy as np

class PPOModel:
    """
    Model for use with seagul's ppo algorithm
    """
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, policy, value_fn, log_action_std=-.5, fixed_std=True):
        self.policy = policy
        self.value_fn = value_fn
        self.num_acts = int(policy.output_layer.out_features / 2)
        self.num_obs = int(policy.layers[0].in_features)

        if fixed_std:
            self.select_action = self.select_action_fixed
            self.get_logp = self.get_logp_fixed
            self.log_act_std = torch.nn.Parameter(torch.as_tensor(log_action_std))
        else:
            self.select_action = self.select_action_variable
            self.get_logp = self.get_logp_variable

    def step(self, obs):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, _ = self.select_action(obs)
        value = self.value_fn(torch.as_tensor(obs))
        logp = self.get_logp(obs, action)

        return action, value, None, logp

    def select_action_fixed(self, obs):
        return select_cont_action(self.policy, obs, torch.exp(self.log_act_std))

    def get_logp_fixed(self, obs, acts):
        return get_cont_logp(self.policy, obs, acts, torch.exp(self.log_act_std))

    def select_action_variable(self, obs):
        out = self.policy(obs)
        means = out[..., :self.num_acts]
        logstds = torch.clamp(out[..., self.num_acts:], self.LOG_STD_MIN, self.LOG_STD_MAX)
        return sample_guassian(means, torch.exp(logstds).detach())

    def get_logp_variable(self, obs, acts):
        out = self.policy(obs)
        means = out[..., :self.num_acts]
        logstds = torch.clamp(out[..., self.num_acts:], self.LOG_STD_MIN, self.LOG_STD_MAX)
        return guassian_logp(acts, means, torch.exp(logstds).detach())


class PPOModelActHold:
    """
    also for use with PPO, this will "hold" each action made by the agent for hold_count time steps
    useful to downsample how often your agent takes an action without needing to do the same for your
    dynamics
    """

    def __init__(self, policy, value_fn, hold_count=5, action_var=0.1, discrete=False):
        self.policy = policy
        self.value_fn = value_fn
        self.action_var = action_var
        self.hold_count = hold_count
        self.cur_hold_count = 0

        self._select_action = select_cont_action
        self._get_logp = get_cont_logp

    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)

        action, _ = self.select_action(state)
        value = self.value_fn(torch.as_tensor(state))
        logp = self.get_logp(state, action)

        return action, value, None, logp

    def select_action(self, state):
        if self.cur_hold_count == 0:
            action, logp = self._select_action(self.policy, state, self.action_var)
            self.cur_action = action
            self.cur_logp = logp
            self.cur_hold_count += 1
        else:
            action = self.cur_action
            logp = self.cur_logp
            self.cur_hold_count += 1

        if self.cur_hold_count > self.hold_count:
            self.cur_hold_count = 0

        return action, logp

    def get_logp(self, states, actions):
        return self._get_logp(self.policy, states, actions, self.action_var)


class SwitchedPPOModel:
    def __init__(self, policy, nominal_policy, value_fn, gate_fn, env, action_var=None, gate_var=None, discrete=False):
        self.policy = policy
        self.nominal_policy = nominal_policy
        self.value_fn = value_fn
        self.action_var = action_var
        self.gate_fn = gate_fn
        self.gate_var = gate_var
        self.env = env
        self.hyst_state = 1
        self.hyst_vec = np.vectorize(self.hyst)

    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        path, gate_path = self.select_path(state)
        value = self.value_fn(torch.as_tensor(state))

        if path:
            action = self.nominal_policy(state)
            logp = 0
        else:
            action, logp = self.select_action(state)

        return action, value, None, float(logp)

    def select_action(self, state):
        return select_cont_action(self.policy, state, self.action_var)

    def get_action_logp(self, states, actions):
        return get_cont_logp(self.policy, states, actions, self.action_var)

    def select_path(self, state):
        gate_out, _ = select_cont_action(self.gate_fn, state, self.gate_var)
        return self.hyst_vec(gate_out), gate_out

    def get_path_logp(self, states, actions):
        return get_cont_logp(self.gate_fn, states, actions, self.gate_var)

    def hyst(self, x):
        if x > 0.5:
            return 1
        else:
            return 0


class SwitchedPPOModelActHold:
    """
    also for use with PPO, this will "hold" each action made by the agent for hold_count time steps
    useful to downsample how often your agent takes an action without needing to do the same for your
    dynamics
    """

    def __init__(
        self, policy, nominal_policy, value_fn, gate_fn, action_var=None, gate_var=None, thresh=.9, hold_count=5
    ):
        self.policy = policy
        self.value_fn = value_fn
        self.action_var = action_var
        self.policy = policy
        self.nominal_policy = nominal_policy
        self.value_fn = value_fn
        self.action_var = action_var
        self.gate_fn = gate_fn
        self.gate_var = gate_var
        self.hyst_state = 1
        self.hyst_vec = np.vectorize(self.hyst)
        self.sig = torch.nn.Sigmoid()

        self.hold_count = hold_count
        self.cur_hold_count = 0

        self._select_action = select_cont_action
        self._get_logp = get_cont_logp

        self.thresh = thresh

    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        value = self.value_fn(torch.as_tensor(state))

        path = self.sig(self.gate_fn(state)) > self.thresh

        if path:
            action = self.nominal_policy(state)
            logp = 0
        else:
            action, logp = self.select_policy_action(state)

        return action, value, None, logp

    def select_action(self,state):
        path = self.sig(self.gate_fn(state))
        if path > self.thresh:
            action = self.nominal_policy(state)
            logp = 0
        else:
            action, logp = self.select_policy_action(state)

        return action, logp

    def select_policy_action(self, state):
        if self.cur_hold_count == 0:
            action, logp = self._select_action(self.policy, state, self.action_var)
            self.cur_action = action
            self.cur_logp = logp
            self.cur_hold_count += 1
        else:
            action = self.cur_action
            logp = self.cur_logp
            self.cur_hold_count += 1

        if self.cur_hold_count > self.hold_count:
            self.cur_hold_count = 0

        return action, logp

    def select_path(self, state):
        gate_out, _ = select_cont_action(self.gate_fn, state, self.gate_var)
        return self.hyst_vec(gate_out), gate_out

    def get_path_logp(self, states, actions):
        return get_cont_logp(self.gate_fn, states, actions, self.gate_var)

    def get_logp(self, states, actions):
        return self.get_action_logp(states,actions)

    def get_action_logp(self, states, actions):
        return get_cont_logp(self.policy, states, actions, self.action_var)

    def select_path(self, state):
        gate_out, _ = select_cont_action(self.gate_fn, state, self.gate_var)
        return self.hyst_vec(gate_out), gate_out

    def get_path_logp(self, states, actions):
        return get_cont_logp(self.gate_fn, states, actions, self.gate_var)

    # def hyst(self, x):
    #     if x > 0.5:
    #         return 1
    #     else:
    #         return 0

    def hyst(self, x):
        """
        Unvectorized hysteris function with sharp transitions

        :param x double between 0 and 1:
        :return activation function:
        """
        if self.hyst_state == 0:
            if x > 0.65:
                self.hyst_state = 1
                return 1
            else:
                return 0
        elif self.hyst_state == 1:
            if x < 0.35:
                self.hyst_state = 0
                return 0
            else:
                return 1


# helper functions
# ============================================================================================
# takes a policy and the states and sample an action from it... (can we make this faster?)
def select_cont_action(policy, state, std):
    means = policy(state)
    m = Normal(loc=means, scale=torch.ones_like(means) * std)
    action = m.sample()
    #logprob = m.log_prob(action)
    return action.detach(), None


# given a policy plus a state/action pair, what is the log liklihood of having taken that action?
def get_cont_logp(policy, states, actions, std):
    means = policy(states)
    m = Normal(loc=means, scale=torch.ones_like(means) * std)
    logprob = m.log_prob(actions)
    return logprob


# takes a policy and the states and sample an action from it... (can we make this faster?)
def select_discrete_action(policy, state, variance=None):
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach().reshape(-1), logprob


# Selects a sample from a Gaussian with mean and stds given. Returns sample and the logprob
def sample_guassian(means, stds):
    actions = stds * torch.randn_like(stds) + means
    #logprob = -((actions - means) ** 2) / (2 * (stds ** 2)) - torch.log(stds) - np.log(np.sqrt(2 * np.pi))
    return actions.detach(), None


# what is the log liklihood of having taken a given action if it came form a Gaussian with stds and means given
def guassian_logp(actions, means, stds):
    return -((actions - means) ** 2) / (2 * (stds ** 2)) - torch.log(stds) - np.log(np.sqrt(2 * np.pi))
