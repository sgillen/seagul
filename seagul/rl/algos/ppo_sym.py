import gym

from seagul.rl.models import PpoModel
from seagul.rl.common import discount_cumsum

import numpy as np
from tqdm import trange
import pickle


import torch
from torch.utils import data

from seagul.rllib.mirror_fns import *


# ============================================================================================
def ppo_sym(
    env_name,
    num_epochs,
    model,
    action_var_schedule=None,
    env_timesteps=2048,
    epoch_batch_size=2048,
    gamma=0.99,
    lam=0.99,
    eps=0.2,
    seed=0,
    policy_batch_size=1024,
    value_batch_size=1024,
    policy_lr=1e-4,
    value_lr=1e-5,
    p_epochs=10,
    v_epochs=10,
    use_gpu=False,
    reward_stop=None,
):

    """
    Implements proximal policy optimization with clipping

    Args:
        env_name: name of the openAI gym environment to solve
        num_epochs: number of epochs to run_util the PPO for
        model: model from seagul.rl.models. Contains policy and value fn
        epoch_batch_size: number of environment steps to take per batch, total steps will be num_epochs*epoch_batch_size
        seed: seed for all the rngs
        gamma: discount applied to future rewards, usually close to 1
        lam: lambda for the Advantage estimmation, usually close to 1
        eps: epsilon for the clipping, usually .1 or .2
        policy_batch_size: batch size for policy updates
        value_batch_size: batch size for value function updates
        policy_lr: learning rate for policy p_optimizer
        value_lr: learning rate of value function p_optimizer
        p_epochs: how many epochs to use for each policy update
        v_epochs: how many epochs to use for each value update
        use_gpu:  want to use the GPU? set to true
        reward_stop: reward value to stop if we achieve

    Returns:
        model: trained model
        avg_reward_hist: list with the average reward per episode at each epoch
        var_dict: dictionary with all locals, for logging/debugging purposes

    Example:
        import torch.nn as nn
        from seagul.rl.algos.ppo import ppo
        from seagul.nn import MLP, CategoricalMLP
        import torch

        torch.set_default_dtype(torch.double)

        input_size = 4; output_size = 1; layer_size = 64; num_layers = 3
        activation = nn.ReLU

        policy = MLP(input_size, output_size, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)

        model = PpoModel(policy, value_fn, action_var=4, discrete=False)
        t_model, rewards, var_dict = ppo("su_acro_drake-v0", 100, model, action_var_schedule=[3,2,1,0])
    """

    if env_name == "HumanoidBulletEnv-v0":
        mirror_act = mirror_human_act
        mirror_obs = mirror_human_obs
    elif env_name == "Walker2DBulletEnv-v0":
        mirror_act = mirror_walker_act
        mirror_obs = mirror_walker_obs
    elif env_name == "Pendulum-v0":
        mirror_act = mirror_pend_act
        mirror_obs = mirror_pend_obs
    else:
        raise NotImplementedError(
            "Passed invalid environment, symmetric PPO only supports Walker2dBulletEnv-v0 or HumanoidBulletEnv-v0"
        )

    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        action_size = 1
        action_dtype = torch.long
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
        action_dtype = torch.double
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    obs_size = env.observation_space.shape[0]

    if action_var_schedule is not None:
        action_var_schedule = np.asarray(action_var_schedule)
        sched_length = action_var_schedule.shape[0]
        x_vals = np.linspace(0, num_epochs, sched_length)
        action_var_lookup = lambda epoch: np.interp(epoch, x_vals, action_var_schedule)
        model.action_var = action_var_lookup(0)

    # init mean and var variables
    state_mean = torch.zeros(obs_size)
    state_var = torch.ones(obs_size)
    adv_mean = torch.zeros(1)
    adv_std = torch.ones(1)
    # rew_mean = torch.zeros(1)
    # rew_std  = torch.ones(1)

    num_states = 0  # tracks how many states we've seen so far, so that we can update means properly

    # need a copy of the old policy for the ppo loss
    old_model = pickle.loads(pickle.dumps(model))
    # intialize our optimizers
    p_optimizer = torch.optim.Adam(model.policy.parameters(), lr=policy_lr)
    v_optimizer = torch.optim.Adam(model.value_fn.parameters(), lr=value_lr)

    # seed all our RNGs
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set defaults, and decide if we are using a GPU or not
    # torch.set_default_dtype(torch.double)
    use_cuda = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    avg_reward_hist = []
    v_loss_hist = []
    p_loss_hist = []

    for epoch in trange(num_epochs):

        episode_reward_sum = []
        batch_steps = 0  # tracks steps taken in current batch
        traj_steps = 0  # tracks steps taken in current trajector
        traj_count = 1  # tracks number of trajectories in current batch

        # tensors for holding training information for each batch
        adv_tensor = torch.empty(0)
        disc_rewards_tensor = torch.empty(0)
        state_tensor = torch.empty(0)
        action_tensor = torch.empty(0, dtype=action_dtype)
        mean_tensor = torch.empty(0)

        # Check if we have maxed out the reward, so that we can stop early
        if traj_count > 2:
            if avg_reward_hist[-1] >= reward_stop and avg_reward_hist[-2] >= reward_stop:
                break

        # keep doing rollouts until we fill  a single batch of examples
        while True:
            # reset the environment
            state = torch.as_tensor(env.reset())
            action_list = []
            state_list = []
            reward_list = []
            mean_list = []
            traj_steps = 0

            # Do a single policy rollout
            for t in range(env_timesteps):
                state_list.append(state.clone())

                # import ipdb; ipdb.set_trace()
                action, logprob = model.select_action(state)
                mean = model.policy(state)
                state_np, reward, done, _ = env.step(action.numpy().reshape(-1))

                state = torch.as_tensor(state_np).detach()

                reward_list.append(torch.as_tensor(reward))
                action_list.append(torch.as_tensor(action.clone()))
                mean_list.append(torch.as_tensor(mean).clone())
                # import ipdb; ipdb.set_trace()
                batch_steps += 1
                traj_steps += 1

                if done:  # assume failure???

                    traj_count += 1  # TODO pretty sure this makes no sense to put here..
                    break

            # if we failed at the first time step start again
            if traj_steps <= 2:
                traj_steps = 0
                break

            # =======================================================================
            # make a tensor storing the current episodes state, actions, and rewards

            #            import ipdb; ipdb.set_trace()
            ep_state_tensor = torch.stack(state_list).reshape(-1, env.observation_space.shape[0])
            ep_action_tensor = torch.stack(action_list).reshape(-1, action_size)
            ep_length = ep_state_tensor.shape[0]

            ep_rewards_tensor = torch.stack(reward_list).reshape(-1)
            # rew_mean = (ep_rewards_tensor.mean()*ep_length + rew_mean*num_states)/(ep_length + num_states)
            # rew_std  = (ep_rewards_tensor.std()*ep_length + rew_std*num_states)/(ep_length + num_states)
            # ep_rewards_tensor = (ep_rewards_tensor - rew_mean)/(rew_std + 1e-5)

            #            if not done: # implies episode did not fail
            torch.cat((ep_rewards_tensor, model.value_fn(state)))

            ep_disc_rewards = torch.as_tensor(discount_cumsum(ep_rewards_tensor, gamma)).reshape(-1, 1)
            disc_rewards_tensor = torch.cat((disc_rewards_tensor, ep_disc_rewards[:-1]))

            # calculate our advantage for this rollout
            value_preds = model.value_fn(ep_state_tensor)
            deltas = (
                torch.as_tensor(ep_rewards_tensor[:-1]) + gamma * value_preds[1:].squeeze() - value_preds[:-1].squeeze()
            )
            ep_adv = discount_cumsum(deltas.detach(), gamma * lam).reshape(-1, 1)
            adv_mean = (ep_adv.mean() * ep_length + adv_mean * num_states) / (ep_length + num_states)
            adv_std = (ep_adv.std() * ep_length + adv_std * num_states) / (ep_length + num_states)
            ep_adv = (ep_adv - adv_mean) / (adv_std + 1e-5)

            # append to the tensors storing information for the whole batch
            state_tensor = torch.cat((state_tensor, ep_state_tensor[:-1]))
            action_tensor = torch.cat((action_tensor, ep_action_tensor[:-1]))
            mean_tensor = torch.cat((mean_tensor, torch.stack(mean_list)[:-1]))

            adv_tensor = torch.cat((adv_tensor, ep_adv))

            num_states += ep_length

            # keep track of rewards for metrics later
            episode_reward_sum.append(sum(reward_list))

            # once we have enough data, update our policy and value function
            if batch_steps > epoch_batch_size:
                # Update our mean/std preprocessors

                state_tensor = mirror_obs(state_tensor)
                action_tensor = mirror_act(action_tensor)

                state_mean = (torch.mean(state_tensor, 0) * state_tensor.shape[0] + state_mean * num_states) / (
                    state_tensor.shape[0] + num_states
                )
                state_var = (torch.var(state_tensor, 0) * state_tensor.shape[0] + state_var * num_states) / (
                    state_tensor.shape[0] + num_states
                )

                state_tensor = torch.cat((state_tensor, mirror_obs(state_tensor)))
                action_tensor = torch.cat((action_tensor, mirror_act(action_tensor)))
                adv_tensor = torch.cat((adv_tensor, adv_tensor))
                disc_rewards_tensor = torch.cat((disc_rewards_tensor, disc_rewards_tensor))

                # construct a training data generator
                training_data = data.TensorDataset(state_tensor, action_tensor, adv_tensor)
                training_generator = data.DataLoader(training_data, batch_size=policy_batch_size, shuffle=True)

                # iterate through the data, doing the updates for our policy
                for p_epoch in range(p_epochs):
                    for local_states, local_actions, local_adv in training_generator:
                        # Transfer to GPU (if GPU is enabled, else this does nothing)
                        local_states, local_actions, local_adv = (
                            local_states.to(device),
                            local_actions.to(device),
                            local_adv.to(device),
                        )

                        # predict and calculate loss for the batch
                        logp = model.get_logp(local_states, local_actions.squeeze()).reshape(-1, action_size)
                        old_logp = old_model.get_logp(local_states, local_actions.squeeze()).reshape(-1, action_size)
                        r = torch.exp(logp - old_logp)
                        p_loss = (
                            -torch.sum(torch.min(r * local_adv, local_adv * torch.clamp(r, (1 - eps), (1 + eps))))
                            / r.shape[0]
                        )

                        # do the normal pytorch update
                        p_loss.backward(retain_graph=True)
                        p_optimizer.step()
                        p_optimizer.zero_grad()

                p_loss.backward()

                if torch.isnan(p_loss):
                    import ipdb

                    ipdb.set_trace()

                # Now we do the update for our value function
                # construct a training data generator
                training_data = data.TensorDataset(state_tensor, disc_rewards_tensor)
                training_generator = data.DataLoader(training_data, batch_size=value_batch_size, shuffle=True)

                for v_epoch in range(v_epochs):
                    for local_states, local_values in training_generator:
                        # Transfer to GPU (if GPU is enabled, else this does nothing)
                        local_states, local_values = (local_states.to(device), local_values.to(device))

                        # predict and calculate loss for the batch
                        value_preds = model.value_fn(local_states)
                        v_loss = torch.sum(torch.pow(value_preds - local_values, 2)) / (value_preds.shape[0])

                        # do the normal pytorch update
                        v_optimizer.zero_grad()
                        v_loss.backward()
                        v_optimizer.step()

                if torch.isnan(v_loss):
                    import ipdb

                    ipdb.set_trace()

                # keep track of rewards for metrics later
                avg_reward_hist.append(sum(episode_reward_sum) / len(episode_reward_sum))
                v_loss_hist.append(v_loss)
                p_loss_hist.append(p_loss)

                old_model = pickle.loads(pickle.dumps(model))
                if action_var_schedule is not None:
                    model.action_var = action_var_lookup(epoch)

                model.policy.state_means = state_mean
                model.policy.state_var = state_var

                model.value_fn.state_means = state_mean
                model.value_fn.state_var = state_var

                break

    print(avg_reward_hist[-1])
    return model, avg_reward_hist, locals()


# ============================================================================================
if __name__ == "__main__":
    import torch.nn as nn
    from seagul.rl.algos.ppo import ppo
    from seagul.nn import MLP, CategoricalMLP
    import torch

    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.double)

    input_size = 4
    output_size = 1
    layer_size = 64
    num_layers = 3
    activation = nn.ReLU

    policy = MLP(input_size, output_size, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
    model = PpoModel(policy, value_fn, action_var=0.1, discrete=True)

    # Define our hyper parameters
    num_epochs = 100
    batch_size = 2048  # how many steps we want to use before we update our gradients
    num_steps = 1000  # number of steps in an episode (unless we terminate early)
    max_reward = num_steps
    p_batch_size = 1024
    v_epochs = 1
    p_epochs = 10
    p_lr = 1e-2
    v_lr = 1e-2

    gamma = 0.99
    lam = 0.99
    eps = 0.2

    # env2, t_policy, t_val, rewards = ppo('InvertedPendulum-v2', 100, policy, value_fn)
    t_model, rewards, var_dict = ppo("CartPole-v0", 100, model, action_var_schedule=[1])
    print(rewards)
    plt.plot(rewards)
    plt.show()
