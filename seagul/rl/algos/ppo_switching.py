import gym

# It looks like these aren't used anywhere, but these imports a necessary to register environments with gym
import seagul.envs
import pybullet_envs

from seagul.rl.common import discount_cumsum
from seagul.rl.models import switchedPpoModel

import numpy as np
from tqdm import trange
import copy


import torch
from torch.utils import data

from numpy import pi, sin, cos

# ============================================================================================
def ppo_switch(
    env_name,
    num_epochs,
    model,
    action_var_schedule=None,
    gate_var_schedule=None,
    epoch_batch_size=2048,
    gamma=0.99,
    lam=0.99,
    eps=0.2,
    seed=0,
    policy_batch_size=1024,
    value_batch_size=1024,
    gate_batch_size=1024,
    policy_lr=1e-3,
    value_lr=1e-3,
    gate_lr=1e-3,
    p_epochs=10,
    v_epochs=1,
    use_gpu=False,
    reward_stop=None,
):

    """

    Implements a proximal policy optimization with clipping

    Args:

        env_name: name of the openAI gym environment to solve
        num_epochs: number of epochs to run_util the PPO for
        policy: policy function, must be a pytorch module
        value_fn: value function, must be a pytorch module
        gate_fn: gating function, another pytorch module, should output between 0 and 1
        action_var: variance to use for the policy network
        gate_var: variance to use for the gating network
        epoch_batch_size: number of environment steps to take per batch, total steps will be num_epochs*epoch_batch_size
        seed: seed for all the rngs
        gamma: discount applied to future rewards, usually close to 1
        lam: lambda for the Advantage estimmation, usually close to 1
        eps: epsilon for the clipping, usually .1 or .2
        policy_batch_size: batch size for policy updates
        param value_batch_size: batch size for value function updates
        policy_lr: learning rate for policy p_optimizer
        value_lr: learning rate of value function p_optimizer
        p_epochs: how many epochs to use for each policy update
        v_epochs: how many epochs to use for each value update
        use_gpu:  want to use the GPU? set to true
        reward_stop: reward value to stop if we achieve

    Returns:
        policy: the trained policy network
        value_fn: the trained value function
        gate_fn: the trained gate fn
        avg_reward_hist: list of average rewards per epoch
        arg_dict: dictionary of parameters used, used for saving runs

    Example:
        import torch.nn as nn
        from seagul.rl.ppo import ppo_switch
        from seagul.nn import CategoricalMLP, MLP
        import torch

        policy = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
        value_fn = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
        gate_fn = CategoricalMLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)

        env2, t_policy, t_val, t_gate, rewards = ppo_switch(
            "su_cartpole-v0", 1000, policy, value_fn, gate_fn, epoch_batch_size=500
        )

        print(rewards)

    """

    env = gym.make(env_name)

    # car_env = env.envs[0].env.env
    env.num_steps = 1500  # TODO

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        raise NotImplementedError(
            "Discrete action spaces are not implemented yet, why are you using PPO for a discrete action space anyway?"
        )
        # select_action = select_discrete_action
        # get_logp = get_discrete_logp
        # action_size = 1
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
        obs_size = env.observation_space.shape[0]
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)


    if action_var_schedule is not None:
        action_var_schedule = np.asarray(action_var_schedule)
        sched_length = action_var_schedule.shape[0]
        x_vals = np.linspace(0,num_epochs,sched_length)
        action_var_lookup = lambda epoch: np.interp(epoch,x_vals,action_var_schedule )
        model.action_var = action_var_lookup(0)

    if gate_var_schedule is not None:
        gate_var_schedule = np.asarray(gate_var_schedule)
        sched_length = gate_var_schedule.shape[0]
        x_vals = np.linspace(0, num_epochs, sched_length)
        gate_var_lookup = lambda epoch: np.interp(epoch, x_vals, gate_var_schedule)
        model.gate_var = gate_var_lookup(0)

    # init mean and var variables
    state_mean = torch.zeros(obs_size)
    state_var = torch.zeros(obs_size)
    num_states = 0  # tracks how many states we've seen so far, so that we can update means properly

    # need a copy of the old policy for the ppo loss
    old_model = copy.deepcopy(model)
    # intialize our optimizers
    p_optimizer = torch.optim.Adam(model.policy.parameters(), lr=policy_lr)
    v_optimizer = torch.optim.Adam(model.value_fn.parameters(), lr=value_lr)
    g_optimizer = torch.optim.Adam(model.gate_fn.parameters(), lr=gate_lr)

    # seed all our RNGs
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set defaults, and decide if we are using a GPU or not
    torch.set_default_dtype(torch.double)
    use_cuda = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    avg_reward_hist = []

    # ----------------------------------------------------------------------------------
    for epoch in trange(num_epochs):
        episode_reward_sum = []
        episode_reward_sum = []
        batch_steps = 0  # tracks steps taken in current batch
        traj_steps = 0  # tracks steps taken in current trajector
        traj_count = 1  # tracks number of trajectories in current batch

        # tensors for holding training information for each batch
        adv_tensor = torch.empty(0)
        disc_rewards_tensor = torch.empty(0)
        state_tensor = torch.empty(0)
        action_tensor = torch.empty(0)
        path_tensor = torch.empty(0, dtype=torch.long)
        gate_tensor = torch.empty(0)

        # Check if we have maxed out the reward, so that we can stop early
        if traj_count > 2:
            if avg_reward_hist[-1] >= reward_stop and avg_reward_hist[-2] >= reward_stop:
                break

        # keep doing rollouts until we fill a single batch of examples
        # -------------------------------------------------------------------------------
        while True:
            # reset the environment
            state = torch.as_tensor(env.reset())
            action_list = []
            state_list = []
            reward_list = []
            path_list = []
            gate_list = []
            traj_steps = 0

            # Do a single policy rollout
            for t in range(env.num_steps):

                state_list.append(state.clone())

                path, gate_out = model.select_path(state)

                if path:
                    action = model.nominal_policy(env, state)
                else:
                    action, logprob = model.select_action(state)

                state_np, reward, done, _ = env.step(action.numpy())
                state = torch.as_tensor(state_np)

                gate_list.append(gate_out)
                path_list.append(torch.as_tensor(path))
                reward_list.append(reward)
                action_list.append(torch.as_tensor(action, dtype=torch.double))

                batch_steps += 1
                traj_steps += 1

                if done:
                    traj_count += 1
                    break

            # if we failed at the first time step start again
            if traj_steps <= 1:
                traj_steps = 0
                break

            # Calculate advantages for this episode
            # -------------------------------------------------------------------------------

            # make a tensor storing the current episodes state, actions, and rewards
            ep_state_tensor = torch.stack(state_list).reshape(-1, env.observation_space.shape[0])
            ep_action_tensor = torch.stack(action_list).reshape(-1, action_size)
            ep_disc_rewards = torch.as_tensor(discount_cumsum(reward_list, gamma)).reshape(-1, 1)
            ep_path_tensor = torch.stack(path_list).reshape(-1, 1)
            ep_gate_tensor = torch.stack(gate_list).reshape(-1, 1)

            # calculate our advantage for this rollout
            value_preds = model.value_fn(ep_state_tensor)
            deltas = torch.as_tensor(reward_list[:-1]) + gamma * value_preds[1:].squeeze() - value_preds[:-1].squeeze()
            ep_adv = discount_cumsum(deltas.detach(), gamma * lam).reshape(-1, 1)

            # append to the tensors storing information for the whole batch
            state_tensor = torch.cat((state_tensor, ep_state_tensor[:-1]))
            action_tensor = torch.cat((action_tensor, ep_action_tensor[:-1]))
            disc_rewards_tensor = torch.cat((disc_rewards_tensor, ep_disc_rewards[:-1]))
            adv_tensor = torch.cat((adv_tensor, ep_adv))
            path_tensor = torch.cat((path_tensor, ep_path_tensor[:-1]))
            gate_tensor = torch.cat((gate_tensor, ep_gate_tensor[:-1]))

            # keep track of rewards for metrics later
            episode_reward_sum.append(sum(reward_list))

            # once we have enough data, update our policy and value function
            # -------------------------------------------------------------------------------
            if batch_steps > epoch_batch_size:

                state_mean = (torch.mean(state_tensor, 0)*state_tensor.shape[0] + state_mean*num_states)/(state_tensor.shape[0] + num_states)
                state_var = torch.var(state_tensor, 0)*state_tensor.shape[0] + state_var*num_states/(state_tensor.shape[0] + num_states)

                model.policy.state_means = state_mean
                model.policy.state_var = state_var

                model.value_fn.state_means = state_mean
                model.value_fn.state_var = state_var

                p_state_list = []
                p_action_list = []
                p_adv_list = []
                for state, action, adv, path in zip(state_tensor, action_tensor, adv_tensor, path_tensor):
                    if not path:
                        p_state_list.append(state)
                        p_action_list.append(action)
                        p_adv_list.append(adv)

                if len(p_state_list):
                    p_state_tensor = torch.stack(p_state_list).reshape(-1, env.observation_space.shape[0])
                    p_action_tensor = torch.stack(p_action_list).reshape(-1, action_size)
                    p_adv_tensor = torch.stack(p_adv_list).reshape(-1, 1)

                    # update policy
                    # ----------------------------------------------------------------------

                    # construct a training data generator
                    training_data = data.TensorDataset(p_state_tensor, p_action_tensor, p_adv_tensor)
                    training_generator = data.DataLoader(training_data, batch_size=policy_batch_size, shuffle=True)

                    # iterate through the data, doing the updates for our policy
                    for p_epoch in range(p_epochs):
                        for (local_states, local_actions, local_adv) in training_generator:
                            # Transfer to GPU (if GPU is enabled, else this does nothing)
                            local_states, local_actions, local_adv = (
                                local_states.to(device),
                                local_actions.to(device),
                                local_adv.to(device),
                            )

                            # predict and calculate loss for the batch
                            logp = model.get_action_logp(local_states, local_actions.squeeze()).reshape(
                                -1, action_size
                            )
                            old_logp = old_model.get_action_logp(local_states, local_actions.squeeze()).reshape(
                                -1, action_size
                            )
                            r = torch.exp(logp - old_logp)
                            p_loss = (
                                -torch.sum(torch.min(r * local_adv, local_adv * torch.clamp(r, (1 - eps), (1 + eps))))
                                / traj_count
                            )

                            # do the normal pytorch update
                            p_loss.backward(retain_graph=True)
                            p_optimizer.step()
                            p_optimizer.zero_grad()

                    p_loss.backward()
                if len(state_tensor):

                    # Now we do the update for our value function
                    # ----------------------------------------------------------------------

                    # construct a training data generator
                    training_data = data.TensorDataset(state_tensor, disc_rewards_tensor)
                    training_generator = data.DataLoader(training_data, batch_size=value_batch_size, shuffle=True)

                    for v_epoch in range(v_epochs):
                        for local_states, local_values in training_generator:
                            # Transfer to GPU (if GPU is enabled, else this does nothing)
                            local_states, local_values = (local_states.to(device), local_values.to(device))

                            # predict and calculate loss for the batch
                            value_preds = model.value_fn(local_states)
                            v_loss = torch.sum(torch.pow(value_preds - local_values, 2)) / value_preds.shape[0]

                            # do the normal pytorch update
                            v_optimizer.zero_grad()
                            v_loss.backward()
                            v_optimizer.step()

                    old_model = copy.deepcopy(model)

                    # update gating function
                    # ----------------------------------------------------------------------
                    # construct a training data generator
                    training_data = data.TensorDataset(state_tensor, gate_tensor, adv_tensor)
                    training_generator = data.DataLoader(training_data, batch_size=gate_batch_size, shuffle=True)
                    for p_epoch in range(p_epochs):
                        for local_states, local_gate, local_adv in training_generator:
                            # Transfer to GPU (if GPU is enabled, else this does nothing)
                            local_states, local_gate, local_adv = (
                                local_states.to(device),
                                local_gate.to(device),
                                local_adv.to(device),
                            )

                            logp = model.get_path_logp(local_states, local_gate.squeeze())
                            logp = logp.reshape(-1, action_size)

                            old_logp = old_model.get_path_logp(local_states, local_gate.squeeze())
                            old_logp = old_logp.reshape(-1, action_size)

                            r = torch.exp(logp - old_logp)

                            # gate_loss = -torch.sum(logp*local_adv)

                            gate_loss = (
                                -torch.sum(torch.min(r * local_adv, local_adv * torch.clamp(r, (1 - eps), (1 + eps))))
                                / traj_count
                            )

                            # do the normal pytorch update
                            g_optimizer.zero_grad()
                            gate_loss.backward()
                            g_optimizer.step()

                    old_model = copy.deepcopy(model)
                    if action_var_schedule is not None:
                        model.action_var = action_var_lookup(epoch)

                    if gate_var_schedule is not None:
                        model.gate_var = gate_var_lookup(epoch)

                    # keep track of rewards for metrics later
                    avg_reward_hist.append(sum(episode_reward_sum) / len(episode_reward_sum))
                    break

    return (model, avg_reward_hist, locals())

batch_steps = 0  # tracks steps taken in current batch

# ============================================================================================

if __name__ == "__main__":
    import torch.nn as nn
    from seagul.nn import CategoricalMLP, MLP, DummyNet
    import torch

    import matplotlib.pyplot as plt

    from seagul.sims.cartpole import LQRControl
    torch.set_default_dtype(torch.double)

    policy = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
    value_fn = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
    #gate_fn = CategoricalMLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
    gate_fn = DummyNet(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)

    def gate(state):
        if len(state.shape) == 1:
            return (((140 * pi / 180 < state[0] < pi) and state[1] <= 0) or (
                    (pi < state[0] < 220 * pi / 180) and state[1] >= 0))
        else:
            ret  = ((((140 * pi / 180 < state[:,0]) & (state[:,0] < pi)) & (state[:,1] <= 0))
                   | ((pi < state[:,0]) & (state[:,0] < 220 * pi / 180) & (state[:,1] >= 0)))
            return torch.as_tensor(ret,dtype=torch.double).reshape(-1,1)


    gate_fn.net_fn = gate

    env_name = "su_cartpole_push-v0"
    env = gym.make(env_name)

    model = switchedPpoModel(policy, LQRControl, value_fn, gate_fn, env=env)

    # env2, t_policy, t_val, rewards = ppo('InvertedPendulum-v2', 100, policy, value_fn)
    t_model, rewards, arg_dict = ppo_switch(
        env_name, 500, model, action_var_schedule=[10,0], gate_var_schedule=[1,0]
    )

    plt.plot(rewards)
    print(rewards)
