import torch
from torch.distributions import Normal
import numpy as np


class RandModel:
    """
    class that just takes actions from a uniform random distribution
    """

    def __init__(self, act_limit, act_size):
        self.act_limit = act_limit
        self.act_size = act_size

    def select_action(self, state, noise):
        return (torch.rand(self.act_size) * 2 * self.act_limit - self.act_limit, 1 / (self.act_limit * 2))


class SACModel:
    """
    Model for use with seagul's sac algorithm
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, policy, value_fn, q1_fn, q2_fn, act_limit):
        self.policy = policy
        self.value_fn = value_fn
        self.q1_fn = q1_fn
        self.q2_fn = q2_fn

        self.num_acts = int(policy.output_layer.out_features / 2)
        self.act_limit = act_limit

    # Step is the deterministic evaluation of the policy
    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, logp = self.select_action(state, torch.zeros(1, 1))
        value = self.value_fn(torch.as_tensor(state))
        return action, value, None, logp

    # Select action is used internally and is the stochastic evaluation
    def select_action(self, state, noise):
        out = self.policy(state)
        means = out[:, : self.num_acts]
        logstd = torch.clamp(out[:, self.num_acts :], self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(logstd)

        # we can speed this up by reusing the same buffer but this is more readable
        samples = means + std * noise
        squashed_samples = torch.tanh(samples)
        acts = squashed_samples * self.act_limit

        # logp = -((acts - means) ** 2) / (2 * torch.pow(std,2)) - logstd - math.log(math.sqrt(2 * math.pi))
        m = torch.distributions.normal.Normal(means, std)
        logp = m.log_prob(samples)
        logp -= torch.sum(torch.log(torch.clamp(1 - torch.pow(squashed_samples, 2), 0, 1) + 1e-6), dim=1).reshape(-1, 1)

        return acts, logp

class SACModelActHold:
    """
    Model for use with seagul's sac algorithm
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, policy, value_fn, q1_fn, q2_fn, act_limit, hold_count=200):
        self.policy = policy
        self.value_fn = value_fn
        self.q1_fn = q1_fn
        self.q2_fn = q2_fn

        self.num_acts = int(policy.output_layer.out_features / 2)
        self.act_limit = act_limit
        self.hold_count = hold_count
        self.cur_hold_count = 0

    # Step is the deterministic evaluation of the policy
    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, logp = self.select_action(state, torch.zeros(1, 1))
        value = self.value_fn(torch.as_tensor(state))
        return action, value, None, logp

    # Select action is used internally and is the stochastic evaluation
    def select_action(self, state, noise):
        if self.cur_hold_count == 0:
            out = self.policy(state)
            means = out[:, : self.num_acts]
            logstd = torch.clamp(out[:, self.num_acts :], self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = torch.exp(logstd)

            # we can speed this up by reusing the same buffer but this is more readable
            samples = means + std * noise
            squashed_samples = torch.tanh(samples)
            acts = squashed_samples * self.act_limit

            # logp = -((acts - means) ** 2) / (2 * torch.pow(std,2)) - logstd - math.log(math.sqrt(2 * math.pi))
            m = torch.distributions.normal.Normal(means, std)
            logp = m.log_prob(samples)
            logp -= torch.sum(torch.log(torch.clamp(1 - torch.pow(squashed_samples, 2), 0, 1) + 1e-6), dim=1).reshape(-1, 1)

            self.cur_action = acts
            self.cur_logp = logp
            self.cur_hold_count += 1

        else:
            acts = self.cur_action
            logp = self.cur_logp
            self.cur_hold_count += 1

        if self.cur_hold_count > self.hold_count:
            self.cur_hold_count = 0

        return acts, logp


class SACModelSwitch:
    """
    Model for use with seagul's sac algorithm
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, policy, value_fn, q1_fn, q2_fn, act_limit, gate_fn, balance_controller, thresh_on=.9, thresh_off=.5, hold_count=20):
        self.policy = policy
        self.value_fn = value_fn
        self.q1_fn = q1_fn
        self.q2_fn = q2_fn
        self.gate_fn = gate_fn
        self.balance_controller = balance_controller
        self.sig = torch.nn.Sigmoid()

        self.num_acts = int(policy.output_layer.out_features / 2)
        self.act_limit = act_limit
        self.hold_count = hold_count
        self.cur_hold_count = 0
        self.thresh_on = thresh_on
        self.thresh_off = thresh_off
        self.thresh = thresh_on

        self.cur_action = None
        self.cur_logp = None

    # Step is the deterministic evaluation of the policy
    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, logp = self.select_action(state, torch.zeros(1, 1))
        value = self.value_fn(torch.as_tensor(state))
        return action, value, None, logp

    # Select action is used internally and is the stochastic evaluation
    def select_action(self, state, noise):
        path = self.sig(self.gate_fn(np.array(state, dtype=np.float32))) > self.thresh

        if path:
            self.thresh = self.thresh_off
            acts = self.balance_controller(state).reshape(-1, self.num_acts)
            logp = 0

        else:
            self.thresh = self.thresh_on

            out = self.policy(state)
            means = out[:, : self.num_acts]
            logstd = torch.clamp(out[:, self.num_acts :], self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = torch.exp(logstd)

            # we can speed this up by reusing the same buffer but this is more readable
            samples = means + std * noise
            squashed_samples = torch.tanh(samples)
            acts = squashed_samples * self.act_limit

            # logp = -((acts - means) ** 2) / (2 * torch.pow(std,2)) - logstd - math.log(math.sqrt(2 * math.pi))
            m = torch.distributions.normal.Normal(means, std)
            logp = m.log_prob(samples)
            logp -= torch.sum(torch.log(torch.clamp(1 - torch.pow(squashed_samples, 2), 0, 1) + 1e-6), dim=1).reshape(-1, 1)

        return acts, logp

    # Really just here for rollouts
    def swingup_controller(self, state, noise_c=1):
        state = torch.as_tensor(state, dtype=torch.float32)
        noise = torch.randn(self.num_acts)*noise_c
        self.thresh = self.thresh_on

        out = self.policy(state)
        means = out[:, : self.num_acts]
        logstd = torch.clamp(out[:, self.num_acts:], self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(logstd)

        # we can speed this up by reusing the same buffer but this is more readable
        samples = means + std * noise
        squashed_samples = torch.tanh(samples)
        acts = squashed_samples * self.act_limit

        # logp = -((acts - means) ** 2) / (2 * torch.pow(std,2)) - logstd - math.log(math.sqrt(2 * math.pi))
        m = torch.distributions.normal.Normal(means, std)
        logp = m.log_prob(samples)
        logp -= torch.sum(torch.log(torch.clamp(1 - torch.pow(squashed_samples, 2), 0, 1) + 1e-6), dim=1).reshape(
            -1, 1)

        return acts.detach()

    def to(self, device):
        self.policy = self.policy.to(device)
        self.value_fn = self.value_fn.to(device)
        self.q1_fn = self.q1_fn.to(device)
        self.q2_fn = self.q2_fn.to(device)
        self.gate_fn = self.gate_fn.to(device)
        return self


    # Select action is used internally and is the stochastic evaluation
    def select_action_parallel(self, state, noise):

        out = self.policy(state)
        means = out[:, :self.num_acts]
        logstd = torch.clamp(out[:, self.num_acts :], self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(logstd)

        # we can speed this up by reusing the same buffer but this is more readable
        samples = means + std * noise
        squashed_samples = torch.tanh(samples)
        acts = squashed_samples * self.act_limit

        # logp = -((acts - means) ** 2) / (2 * torch.pow(std,2)) - logstd - math.log(math.sqrt(2 * math.pi))
        m = torch.distributions.normal.Normal(means, std)
        logp = m.log_prob(samples)
        logp -= torch.sum(torch.log(torch.clamp(1 - torch.pow(squashed_samples, 2), 0, 1) + 1e-6), dim=1).reshape(-1, 1)

        return acts, logp

