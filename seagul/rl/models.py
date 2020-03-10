import torch
from torch.distributions import Normal, Categorical
import numpy as np


"""
'Models' used by the seaguls reinforcement learning algos. 

A model combines an agents policy, value function, and anything else needed to learn and take actions in a space

They all must implement step(state) which takes as input state and returns action, value, None, logp
"""


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
        action, logp = self.select_action_serial(state, torch.zeros(1, 1))
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
                self.cur_hold_count %= self.hold_count

        return acts, logp

    # Select action is used internally and is the stochastic evaluation
    def select_action_parallel(self, state, noise):
        path = self.sig(self.gate_fn(np.array(state, dtype=np.float32))) > self.thresh_on
        path = torch.as_tensor(path, dtype=torch.float32)

        balance_acts = self.balance_controller(state).reshape(-1, self.num_acts)
        balance_logp = 0

        out = self.policy(state)
        means = out[:, :self.num_acts]
        logstd = torch.clamp(out[:, self.num_acts :], self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(logstd)

        # we can speed this up by reusing the same buffer but this is more readable
        samples = means + std * noise
        squashed_samples = torch.tanh(samples)
        swingup_acts = squashed_samples * self.act_limit

        # logp = -((acts - means) ** 2) / (2 * torch.pow(std,2)) - logstd - math.log(math.sqrt(2 * math.pi))
        m = torch.distributions.normal.Normal(means, std)
        swingup_logp = m.log_prob(samples)
        swingup_logp -= torch.sum(torch.log(torch.clamp(1 - torch.pow(squashed_samples, 2), 0, 1) + 1e-6), dim=1).reshape(-1, 1)

        acts = path*balance_acts + (1 - path)*swingup_acts
        logp = path*balance_logp + (1 - path)*swingup_acts

        return acts, logp



class PPOModel:
    """
    Model for use with seagul's ppo algorithm
    """

    def __init__(self, policy, value_fn, action_var=None, discrete=False):
        self.policy = policy
        self.value_fn = value_fn
        self.action_var = action_var

        if discrete:
            self._select_action = select_discrete_action
            self._get_logp = get_discrete_logp
        else:
            self._select_action = select_cont_action
            self._get_logp = get_cont_logp

    def step(self, state):
        # (action, value estimate, None, negative log likelihood of the action under current policy parameters)
        action, _ = self.select_action(state)
        value = self.value_fn(torch.as_tensor(state))
        logp = self.get_logp(state, action)

        return action, value, None, logp

    def select_action(self, state):
        return self._select_action(self.policy, state, self.action_var)

    def get_logp(self, states, actions):
        return self._get_logp(self.policy, states, actions, self.action_var)


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

        if discrete:
            self._select_action = select_discrete_action
            self._get_logp = get_discrete_logp
        else:
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
def select_cont_action(policy, state, variance):
    means = policy(torch.as_tensor(state)).squeeze()
    m = Normal(loc=means, scale=torch.ones_like(means) * variance)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach().reshape(-1), logprob


# given a policy plus a state/action pair, what is the log liklihood of having taken that action?
def get_cont_logp(policy, states, actions, variance):
    means = policy(torch.as_tensor(states)).squeeze()
    m = Normal(loc=means, scale=torch.ones_like(means) * variance)
    logprob = m.log_prob(actions.squeeze())
    return logprob


# takes a policy and the states and sample an action from it... (can we make this faster?)
def select_discrete_action(policy, state, variance=None):
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    logprob = m.log_prob(action)
    return action.detach().reshape(-1), logprob


# given a policy plus a state/action pair, what is the log liklihood of having taken that action?
def get_discrete_logp(policy, state, action, variance=None):
    probs = policy(state)
    m = Categorical(probs)
    logprob = m.log_prob(action.squeeze())
    return logprob
