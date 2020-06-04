import numpy as np
import torch
import tqdm.auto as tqdm
import gym
import pickle
from seagul.rl.common import update_mean, update_std


def ppo(
        env_name,
        total_steps,
        model,
        act_std_schedule=(0.7,),
        epoch_batch_size=2048,
        gamma=0.99,
        lam=0.99,
        eps=0.2,
        seed=0,
        pol_batch_size=1024,
        sgd_batch_size=1024,
        pol_lr=1e-4,
        val_lr=1e-4,
        pol_epochs=10,
        val_epochs=10,
        target_kl=.1,
        env_no_term_steps=0,
        use_gpu=False,
        reward_stop=None,
        normalize_return=True,
        normalize_obs=True,
        normalize_adv=True,
        env_config={}
):
    """
    Implements proximal policy optimization with clipping

    Args:
        env_name: name of the openAI gym environment to solve
        total_steps: number of timesteps to run the PPO for
        model: model from seagul.rl.models. Contains policy and value fn
        act_std_schedule: schedule to set the variance of the policy. Will linearly interpolate values
        epoch_batch_size: number of environment steps to take per batch, total steps will be num_epochs*epoch_batch_size
        seed: seed for all the rngs
        gamma: discount applied to future rewards, usually close to 1
        lam: lambda for the Advantage estimation, usually close to 1
        eps: epsilon for the clipping, usually .1 or .2
        pol_batch_size: batch size for policy updates
        sgd_batch_size: batch size for value function updates
        pol_lr: learning rate for policy pol_optimizer
        val_lr: learning rate of value function pol_optimizer
        pol_epochs: how many epochs to use for each policy update
        val_epochs: how many epochs to use for each value update
        target_kl: max KL before breaking
        use_gpu:  want to use the GPU? set to true
        reward_stop: reward value to stop if we achieve
        normalize_return: should we normalize the return?
        env_config: dictionary containing kwargs to pass to your the environment

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
    torch.set_num_threads(1)

    env = gym.make(env_name, **env_config)
    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        act_dtype = torch.double
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    actstd_lookup = make_std_schedule(act_std_schedule, model, total_steps)
    model.action_var = actstd_lookup(0)

    obs_size = env.observation_space.shape[0]
    obs_mean = torch.zeros(obs_size)
    obs_std = torch.ones(obs_size)
    adv_mean = torch.zeros(1)
    adv_var = torch.ones(1)
    rew_mean = torch.zeros(1)
    rew_var = torch.ones(1)

    # copy.deepcopy broke for me with older version of torch. Using pickle for this is weird but works fine
    old_model = pickle.loads(pickle.dumps(model))
    pol_opt = torch.optim.Adam(model.policy.parameters(), lr=pol_lr)
    val_opt = torch.optim.Adam(model.value_fn.parameters(), lr=val_lr)

    # seed all our RNGs
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set defaults, and decide if we are using a GPU or not
    use_cuda = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # init logging stuff
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
            ep_obs, ep_act, ep_rew, ep_steps, ep_term = do_rollout(env, model, env_no_term_steps)

            raw_rew_hist.append(sum(ep_rew).item())
            batch_obs = torch.cat((batch_obs, ep_obs[:-1]))
            batch_act = torch.cat((batch_act, ep_act[:-1]))

            if not ep_term:
                ep_rew[-1] = model.value_fn(ep_obs[-1]).detach()


            ep_discrew = discount_cumsum(ep_rew, gamma)

            if normalize_return:
                rew_mean = update_mean(batch_discrew, rew_mean, cur_total_steps)
                rew_var = update_std(ep_discrew, rew_var, cur_total_steps)
                ep_discrew = ep_discrew / (rew_var + 1e-6)

            batch_discrew = torch.cat((batch_discrew, ep_discrew[:-1]))

            ep_val = model.value_fn(ep_obs)

            deltas = ep_rew[:-1] + gamma * ep_val[1:] - ep_val[:-1]
            ep_adv = discount_cumsum(deltas.detach(), gamma * lam)
            batch_adv = torch.cat((batch_adv, ep_adv))

            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps

        # make sure our advantages are zero mean and unit variance
        if normalize_adv:
            adv_mean = update_mean(batch_adv, adv_mean, cur_total_steps)
            adv_var = update_std(batch_adv, adv_var, cur_total_steps)
            batch_adv = (batch_adv - adv_mean) / (adv_var + 1e-6)

        # policy update
        # ========================================================================
        num_mbatch = int(batch_obs.shape[0] / pol_batch_size)
        # Update the policy using the PPO loss
        for pol_epoch in range(pol_epochs):
            pol_loss_tmp = []
            for i in range(num_mbatch):
                cur_sample = i * sgd_batch_size

                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs = batch_obs[cur_sample:cur_sample + sgd_batch_size]
                local_act = batch_act[cur_sample:cur_sample + sgd_batch_size]
                local_adv = batch_adv[cur_sample:cur_sample + sgd_batch_size]

                # Compute the loss
                logp = model.get_logp(local_obs, local_act).reshape(-1, act_size)
                old_logp = old_model.get_logp(local_obs, local_act).reshape(-1, act_size)
                r = torch.exp(logp - old_logp)
                clip_r = torch.clamp(r, 1 - eps, 1 + eps)
                pol_loss = -torch.min(r * local_adv, clip_r * local_adv).mean()
                pol_loss_tmp.append(pol_loss)

                for t in model.policy.state_dict().values():
                    if torch.isnan(t).any():
                        print("oh no")

                #approx_kl = (torch.exp(logp)*(logp - old_logp)).mean()
                approx_kl = ((logp - old_logp)**2).mean()
                if approx_kl > target_kl:
                    break

                pol_opt.zero_grad()
                pol_loss.backward()
                pol_opt.step()

        # value_fn update
        # ========================================================================
        # Update value function with the standard L2 Loss
        for val_epoch in range(val_epochs):
            for i in range(num_mbatch):
                cur_sample = i * sgd_batch_size
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs = batch_obs[cur_sample:cur_sample + sgd_batch_size]
                local_val = batch_discrew[cur_sample:cur_sample + sgd_batch_size]

                # predict and calculate loss for the batch
                val_preds = model.value_fn(local_obs)
                val_loss = ((val_preds - local_val) ** 2).mean()

                # do the normal pytorch update
                val_opt.zero_grad()
                val_loss.backward()
                val_opt.step()

        # update observation mean and variance

        if normalize_obs:
            obs_mean = update_mean(batch_obs, obs_mean, cur_total_steps)
            obs_std = update_std(batch_obs, obs_std, cur_total_steps)
            model.policy.state_means = obs_mean
            model.value_fn.state_means = obs_mean
            model.policy.state_std = obs_std
            model.value_fn.state_std = obs_std
            model.action_std = actstd_lookup(cur_total_steps)
        
        old_model = pickle.loads(pickle.dumps(model))
        val_loss_hist.append(val_loss)
        pol_loss_hist.append(pol_loss)

        progress_bar.update(cur_batch_steps)

    progress_bar.close()
    return model, raw_rew_hist, locals()


# Takes list or array and returns a lambda that interpolates it for each epoch
def make_std_schedule(std_schedule, model, num_steps):
    std_schedule = np.asarray(std_schedule)
    sched_length = std_schedule.shape[0]
    x_vals = np.linspace(0, num_steps, sched_length)
    std_lookup = lambda epoch: np.interp(epoch, x_vals, std_schedule)
    return std_lookup


def do_rollout(env, model, n_steps_complete):
    torch.autograd.set_grad_enabled(False)

    act_list = []
    obs_list = []
    rew_list = []

    dtype = torch.float32
    obs = env.reset()
    done = False
    cur_step = 0

    while not done:
        obs = torch.as_tensor(obs, dtype=dtype).detach()
        obs_list.append(obs.clone())

        act, logprob = model.select_action(obs)
        obs, rew, done, _ = env.step(act.numpy())

        act_list.append(torch.as_tensor(act.clone()))
        rew_list.append(rew)

        cur_step += 1

    if cur_step < n_steps_complete:
        ep_term = True
    else:
        ep_term = False

    ep_length = len(rew_list)
    ep_obs = torch.stack(obs_list)
    ep_act = torch.stack(act_list)
    ep_rew = torch.tensor(rew_list)
    ep_rew = ep_rew.reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return ep_obs, ep_act, ep_rew, ep_length, ep_term


# can make this faster I think?
def discount_cumsum(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards
