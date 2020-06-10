from seagul.rl.common import ReplayBuffer, RandModel, make_schedule, update_target_fn
import numpy as np
import copy
import gym
import dill
import tqdm.auto as tqdm
import torch


def ddpg(
        env_name,
        train_steps,
        model,
        env_max_steps=0,
        min_steps_per_update=1,
        iters_per_update=float('inf'),
        replay_batch_size=64,
        seed=0,
        act_std_schedule=(.1,),
        gamma=0.95,
        polyak=0.995,
        sgd_lr=3e-4,
        exploration_steps=1000,
        replay_buf_size=int(100000),
        reward_stop=None,
        env_config=None
):
    # Initialize env, and other globals
    # ========================================================================
    if env_config is None:
        env_config = {}
    env = gym.make(env_name, **env_config)
    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        act_dtype = env.action_space.sample().dtype
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    obs_size = env.observation_space.shape[0]

    # seed all our RNGs
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    random_model = RandModel(model.act_limit, act_size)
    target_model = copy.deepcopy(model)
    replay_buf = ReplayBuffer(obs_size, act_size, replay_buf_size)

    for param in target_model.q1_fn.parameters():
        param.requires_grad = False

    for param in target_model.policy.parameters():
        param.requires_grad = False

    act_std_lookup = make_schedule(act_std_schedule, train_steps)
    act_std = act_std_lookup(0)

    pol_opt = torch.optim.Adam(model.policy.parameters(), lr=sgd_lr)
    q1_opt = torch.optim.Adam(model.q1_fn.parameters(), lr=sgd_lr)

    progress_bar = tqdm.tqdm(total=train_steps)
    cur_total_steps = 0
    progress_bar.update(0)
    early_stop = False

    raw_rew_hist = []
    pol_loss_hist = []
    q1_loss_hist = []
    q2_loss_hist = []

    # Fill the replay buffer with actions taken from a random model
    # ========================================================================
    while cur_total_steps < exploration_steps:
        ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = do_rollout(env, random_model, env_max_steps, act_std)
        replay_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

        ep_steps = ep_rews.shape[0]
        cur_total_steps += ep_steps

        progress_bar.update(ep_steps)

    # Keep training until we take train_step environment steps
    # ========================================================================
    while cur_total_steps < train_steps:
        cur_batch_steps = 0

        # Bail out if we have met out reward threshold
        if len(raw_rew_hist) > 2 and reward_stop:
            if raw_rew_hist[-1] >= reward_stop and raw_rew_hist[-2] >= reward_stop:
                early_stop = True
                break

        # collect data with the current policy
        # ========================================================================
        while cur_batch_steps < min_steps_per_update:
            ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = do_rollout(env, model, env_max_steps, act_std)
            replay_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

            ep_steps = ep_rews.shape[0]
            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps

            raw_rew_hist.append(torch.sum(ep_rews))

        progress_bar.update(cur_batch_steps)

        # Do the update
        # ========================================================================
        for _ in range(min(int(ep_steps), iters_per_update)):

            replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = replay_buf.sample_batch(replay_batch_size)
#            print(raw_rew_hist[-1])

            # Compute target Q
            with torch.no_grad():
                noise = torch.randn(replay_batch_size, act_size) * act_std
                target_acts, _ = target_model.select_action(replay_obs2, noise)
                q_in = torch.cat((replay_obs2, target_acts), dim=1)
                q_targ = replay_rews + gamma * (1 - replay_done) * target_model.q1_fn(q_in)

            # q_fn update
            # ========================================================================
            q_in = torch.cat((replay_obs1, replay_acts), dim=1)
            q_loss = ((model.q1_fn(q_in) - q_targ)**2).mean()
#            print('q_loss:', q_loss)

            q1_opt.zero_grad()
            q_loss.backward()
            q1_opt.step()

            # policy_fn update
            # ========================================================================
            for param in model.q1_fn.parameters():
                param.requires_grad = False

            local_acts = model.policy(replay_obs1)
            q_in = torch.cat((replay_obs1, local_acts), dim=1)

            pol_loss = -(model.q1_fn(q_in).mean())
#            print('pol_loss:', pol_loss)

            pol_opt.zero_grad()
            pol_loss.backward()
            pol_opt.step()

            for param in model.q1_fn.parameters():
                param.requires_grad = True

            # Update target value fn with polyak average
            # ========================================================================
            pol_loss_hist.append(pol_loss.item())
            q1_loss_hist.append(q_loss.item())

            target_model.q1_fn = update_target_fn(model.q1_fn, target_model.q1_fn, polyak)
            target_model.policy = update_target_fn(model.policy, target_model.policy, polyak)
            act_std = act_std_lookup(cur_total_steps)

    return model, raw_rew_hist, locals()


def do_rollout(env, model, num_steps, act_std):
    torch.autograd.set_grad_enabled(False)
    acts_list = []
    obs1_list = []
    obs2_list = []
    rews_list = []
    done_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs = env.reset()
    done = False
    cur_step = 0

    while not done:
        obs = torch.as_tensor(obs, dtype=dtype).detach()
        obs1_list.append(obs.clone())

        noise = torch.randn(1, act_size)*act_std
        act, _ = model.select_action(obs.reshape(1, -1), noise)
        act = act.detach()

        obs, rew, done, _ = env.step(act.numpy().reshape(-1))
        obs = torch.as_tensor(obs, dtype=dtype).detach()

        acts_list.append(torch.as_tensor(act.clone(), dtype=dtype))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        obs2_list.append(obs.clone())

        if cur_step < num_steps:
            done_list.append(torch.as_tensor(done))
        else:
            done_list.append(torch.as_tensor(False))

        cur_step += 1

    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)
    ep_obs2 = torch.stack(obs2_list)
    ep_done = torch.stack(done_list).reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done
