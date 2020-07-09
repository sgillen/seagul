import torch
from torch.distributions import Normal, Categorical
import numpy as np


class PPOModel:
    """
    Model for use with seagul's ppo algorithm
    """

    def __init__(self, policy, value_fn, init_logstd=-.5, learn_std=True):
        self.policy = policy
        self.value_fn = value_fn

        if learn_std:
            self.policy.logstds = torch.nn.Parameter(torch.ones(policy.output_layer.out_features) * init_logstd)
        else:
            self.policy.logstds = torch.ones(policy.output_layer.out_features) * init_logstd

    def step(self, obs):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, _ = self.select_action(obs)
        value = self.value_fn(torch.as_tensor(obs))
        logp = self.get_logp(obs, action)

        return action, value, None, logp

    def select_action(self, obs):
        means = self.policy(obs)
        return sample_guassian(means, torch.exp(self.policy.logstds))

    def get_logp(self, obs, acts):
        means = self.policy(obs)
        return guassian_logp(means, torch.exp(self.policy.logstds),acts)


# def sample_guassian(means, std):
#     m = Normal(loc=means, scale=torch.ones_like(means) * std)
#     action = m.sample()
#     #logprob = m.log_prob(action)
#     return action.detach(), None
#
# # given a policy plus a state/action pair, what is the log liklihood of having taken that action?
# def guassian_logp(means, std, actions):
#     m = Normal(loc=means, scale=torch.ones_like(means) * std)
#     logprob = m.log_prob(actions)
#     return logprob

# Selects a sample from a Gaussian with mean and stds given. Returns sample and the logprob
def sample_guassian(means, stds):
    actions = stds * torch.randn_like(stds) + means
    #logprob = -((actions - means) ** 2) / (2 * (stds ** 2)) - torch.log(stds) - np.log(np.sqrt(2 * np.pi))
    return actions, None


# what is the log liklihood of having taken a given action if it came form a Gaussian with stds and means given
def guassian_logp(means, stds, actions):
    return -((actions - means) ** 2) / (2 * (stds ** 2)) - torch.log(stds) - np.log(np.sqrt(2 * np.pi))
