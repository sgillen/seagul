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
        return torch.rand(self.act_size) * 2 * self.act_limit - self.act_limit, 1 / (self.act_limit * 2)


class TD3Model:
    """
    Model for use with seagul's sac algorithm
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, policy, q1_fn, q2_fn, act_limit):
        self.policy = policy
        self.q1_fn = q1_fn
        self.q2_fn = q2_fn

        self.num_acts = int(policy.output_layer.out_features / 2)
        self.act_limit = act_limit

    # Step is the deterministic evaluation of the policy
    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, _ = self.select_action(state, torch.zeros(1))
        return action, None, None, None

    # Select action is used internally and is the stochastic evaluation
    def select_action(self, state, noise):
        means = self.policy(state)
        acts = torch.clamp(means + noise, -self.act_limit, self.act_limit)
        return acts, None
