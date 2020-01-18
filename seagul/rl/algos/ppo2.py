import numpy as np
import torch
from torch.utils import data
import tqdm
import gym
import pickle

from seagul.rl.common import discount_cumsum, update_mean, update_var


def ppo(
    env_name,
    total_steps,
    model,
    act_var_schedule=[0.7],
    epoch_batch_size=2048,
    gamma=0.99,
    lam=0.99,
    eps=0.2,
    seed=0,
    pol_batch_size=1024,
    val_batch_size=1024,
    pol_lr=1e-4,
    val_lr=1e-5,
    pol_epochs=10,
    val_epochs=10,
    use_gpu=False,
    reward_stop=None,
):

    """
    Implements proximal policy optimization with clipping

    Args:
        env_name: name of the openAI gym environment to solve
        total_steps: number of timesteps to run the PPO for
        model: model from seagul.rl.models. Contains policy and value fn
        act_var_schedule: schedule to set the variance of the policy. Will linearly interpolate values
        epoch_batch_size: number of environment steps to take per batch, total steps will be num_epochs*epoch_batch_size
        seed: seed for all the rngs
        gamma: discount applied to future rewards, usually close to 1
        lam: lambda for the Advantage estimation, usually close to 1
        eps: epsilon for the clipping, usually .1 or .2
        pol_batch_size: batch size for policy updates
        val_batch_size: batch size for value function updates
        pol_lr: learning rate for policy pol_optimizer
        val_lr: learning rate of value function pol_optimizer
        pol_epochs: how many epochs to use for each policy update
        val_epochs: how many epochs to use for each value update
        use_gpu:  want to use the GPU? set to true
        reward_stop: reward value to stop if we achieve

    Returns:
        model: trained model
        avg_reward_hist: list with the average reward per episode at each epoch
        var_dict: dictionary with all locals, for logging/debugging purposes

    Example:
        from seagul.rl.algos import ppo
        from seagul.nn import MLP
        from seagul.rl.models import PPOModel
        import torch

        input_size = 3
        output_size = 1
        layer_size = 64
        num_layers = 2

        policy = MLP(input_size, output_size, num_layers, layer_size)
        value_fn = MLP(input_size, 1, num_layers, layer_size)
        model = PPOModel(policy, value_fn)

        model, rews, var_dict = ppo("Pendulum-v0", 10000, model)

    """

    # init everything
    # ==============================================================================
    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        act_dtype = torch.double
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    actvar_lookup = make_variance_schedule(act_var_schedule, model, total_steps)
    model.action_var = actvar_lookup(0)

    obs_size = env.observation_space.shape[0]
    obs_mean = torch.zeros(obs_size)
    obs_var = torch.ones(obs_size)
    adv_mean = torch.zeros(1)
    adv_var = torch.ones(1)

    old_model = pickle.loads(
        pickle.dumps(model)
    )  # copy.deepcopy broke for me with older version of torch. Using pickle for this is weird but works fine
    pol_opt = torch.optim.Adam(model.policy.parameters(), lr=pol_lr)
    val_opt = torch.optim.Adam(model.value_fn.parameters(), lr=val_lr)

    # seed all our RNGs
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set defaults, and decide if we are using a GPU or not
    use_cuda = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    raw_rew_hist = []
    val_loss_hist = []
    pol_loss_hist = []

    progress_bar = tqdm.tqdm(total=total_steps)
    cur_total_steps = 0
    progress_bar.update(0)
    early_stop = False

    # Train until we hit our total steps or reach our reward threshold
    # ==============================================================================
    while cur_total_steps < total_steps:

        batch_obs = torch.empty(0)
        batch_act = torch.empty(0)
        batch_adv = torch.empty(0)
        batch_discrew = torch.empty(0)
        cur_batch_steps = 0

        # Bail out if we have met out reward threshold
        if len(raw_rew_hist) > 2 and reward_stop:
            if raw_rew_hist[-1] >= reward_stop and raw_rew_hist[-2] >= reward_stop:
                early_stop = True
                break

        # construct batch data from rollouts
        # ==============================================================================
        while cur_batch_steps < epoch_batch_size:

            ep_obs, ep_act, ep_rew, ep_steps = do_rollout(env, model)

            batch_obs = torch.cat((batch_obs, ep_obs[:-1]))
            batch_act = torch.cat((batch_act, ep_act[:-1]))

            ep_discrew = discount_cumsum(
                ep_rew, gamma
            )  # [:-1] because we appended the value function to the end as an extra reward
            batch_discrew = torch.cat((batch_discrew, ep_discrew[:-1]))

            # calculate this episodes advantages
            last_val = model.value_fn(ep_obs[-1]).reshape(-1, 1)
            # ep_rew = torch.cat((ep_rew, last_val)) # append value_fn to last reward
            ep_val = model.value_fn(ep_obs)
            # ep_val = torch.cat((ep_val, last_val))
            ep_val[-1] = last_val

            deltas = ep_rew[:-1] + gamma * ep_val[1:] - ep_val[:-1]
            ep_adv = discount_cumsum(deltas.detach(), gamma * lam)
            batch_adv = torch.cat((batch_adv, ep_adv))

            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps

        raw_rew_hist.append(sum(ep_rew))

        # make sure our advantages are zero mean and unit variance
        adv_mean = update_mean(batch_adv, adv_mean, cur_total_steps)
        adv_var = update_var(batch_adv, adv_var, cur_total_steps)
        batch_adv = (batch_adv - adv_mean) / (adv_var + 1e-6)

        # policy update
        # ========================================================================
        training_data = data.TensorDataset(batch_obs, batch_act, batch_adv)
        training_generator = data.DataLoader(training_data, batch_size=pol_batch_size, shuffle=True)

        # Update the policy using the PPO loss
        for pol_epoch in range(pol_epochs):
            for local_obs, local_act, local_adv in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_states, local_actions, local_adv = (
                    local_obs.to(device),
                    local_act.to(device),
                    local_adv.to(device),
                )

                # Compute the loss
                logp = model.get_logp(local_obs, local_act).reshape(-1, 1)
                old_logp = old_model.get_logp(local_obs, local_act).reshape(-1, 1)
                r = torch.exp(logp - old_logp)
                pol_loss = (
                    -torch.sum(torch.min(r * local_adv, local_adv * torch.clamp(r, (1 - eps), (1 + eps)))) / r.shape[0]
                )

                # do the normal pytorch update
                pol_opt.zero_grad()
                pol_loss.backward()
                pol_opt.step()

        # value_fn update
        # ========================================================================
        training_data = data.TensorDataset(batch_obs, batch_discrew)
        training_generator = data.DataLoader(training_data, batch_size=val_batch_size, shuffle=True)

        # Upadte value function with the standard L2 Loss
        for val_epoch in range(val_epochs):
            for local_obs, local_val in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs, local_val = (local_obs.to(device), local_val.to(device))

                # predict and calculate loss for the batch
                val_preds = model.value_fn(local_obs)
                val_loss = torch.sum(torch.pow(val_preds - local_val, 2)) / (val_preds.shape[0])

                # do the normal pytorch update
                val_opt.zero_grad()
                val_loss.backward()
                val_opt.step()

        old_model = pickle.loads(pickle.dumps(model))

        # update observation mean and variance
        obs_mean = update_mean(batch_obs, obs_mean, cur_total_steps)
        obs_var = update_var(batch_obs, obs_var, cur_total_steps)
        model.policy.state_means = obs_mean
        model.value_fn.state_means = obs_mean
        model.policy.state_var = obs_var
        model.value_fn.state_var = obs_var
        model.action_var = actvar_lookup(cur_total_steps)

        val_loss_hist.append(val_loss)
        pol_loss_hist.append(pol_loss)

        progress_bar.update(cur_batch_steps)

    progress_bar.close()
    return model, raw_rew_hist, locals()


# Takes list or array and returns a lambda that interpolates it for each epoch
def make_variance_schedule(var_schedule, model, num_steps):
    var_schedule = np.asarray(var_schedule)
    sched_length = var_schedule.shape[0]
    x_vals = np.linspace(0, num_steps, sched_length)
    var_lookup = lambda epoch: np.interp(epoch, x_vals, var_schedule)
    return var_lookup


def do_rollout(env, model):

    act_list = []
    obs_list = []
    rew_list = []
    num_steps = 0

    dtype = torch.get_default_dtype()
    obs = env.reset()
    done = False

    while not done:
        obs = torch.as_tensor(obs,dtype=dtype).detach()
        obs_list.append(obs.clone())

        act, logprob = model.select_action(obs)
        obs, rew, done, _ = env.step(act.numpy().reshape(-1))

        act_list.append(torch.as_tensor(act.clone()))
        rew_list.append(rew)

    ep_length = len(rew_list)
    ep_obs = torch.stack(obs_list)
    ep_act = torch.stack(act_list)
    ep_rew = torch.tensor(rew_list)
    ep_rew = ep_rew.reshape(-1, 1)

    return (ep_obs, ep_act, ep_rew, ep_length)
