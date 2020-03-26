import numpy as np
import torch
from torch.utils import data
import tqdm.auto as tqdm
import gym
import dill

from seagul.rl.common import ReplayBuffer
from seagul.rl.models import RandModel
import ray

def ray_sac(
    env_name,
    total_steps,
    model,
    env_steps = 0,
    num_envs = 2,
    min_steps_per_update = 1,
    iters_per_update = 100,
    replay_batch_size=64,
    seed=0,
    gamma=0.95,
    polyak=0.995,
    alpha=0.2,
    pol_batch_size=64,
    val_batch_size=64,
    q_batch_size=64,
    pol_lr=1e-3,
    val_lr=1e-3,
    q_lr=1e-3,
    exploration_steps=100,
    replay_buf_size=int(50000),
    use_gpu=False,
    reward_stop=None,
):

    """
    Implements soft actor critic using ray for to collect data in parallel

    Args:
        env_name: name of the openAI gym environment to solve
        total_steps: number of timesteps to run the PPO for
        model: model from seagul.rl.models. Contains policy, value fn, q1_fn, q2_fn
        num_envs: number of environments to use, if you want to do this single threaded look at regular sac.py
        min_steps_per_update: minimun number of steps to take before running updates, will finish episodes before updating
        env_steps: number of steps the environment takes before finishing, if the environment emits a done signal before this we consider it a failure.
        iters_per_update: how many update steps to make every time we update
        replay_batch_size: how big a batch to pull from the replay buffer for each update
        seed: random seed for all rngs
        gamma: discount applied to future rewards, usually close to 1
        polyak: term determining how fast the target network is copied from the value function
        alpha: weighting term for the entropy. 0 corresponds to no penalty for deterministic policy
        pol_batch_size: minibatch size for policy updates
        val_batch_size: minibatch size for value fn updates
        q_batch_size: minibatch size for q fn updates
        pol_lr: initial learning rate for policy optimizer
        val_lr: initial learning rate for value optimizer
        q_lr: initial learning rate for q fn optimizer
        exploration_steps: initial number of random actions to take, aids exploration
        replay_buf_size: how big of a replay buffer to use
        use_gpu: determines if we try to use a GPU or not
        reward_stop: reward value to bail at

    Returns:
        model: trained model
        avg_reward_hist: list with the average reward per episode at each epoch
        var_dict: dictionary with all locals, for logging/debugging purposes

    Example:
        from seagul.rl.algos.sac import sac
        import torch.nn as nn
        from seagul.nn import MLP
        from seagul.rl.models import SACModel


        input_size = 3
        output_size = 1
        layer_size = 64
        num_layers = 2
        activation = nn.ReLU

        policy = MLP(input_size, output_size*2, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)
        q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
        q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation)
        model = SACModel(policy, value_fn, q1_fn, q2_fn, 1)


        model, rews, var_dict = sac("Pendulum-v0", 10000, model)
    """

    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        act_dtype = env.action_space.sample().dtype
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    obs_size = env.observation_space.shape[0]

    random_model = RandModel(model.act_limit, act_size)
    replay_buf = ReplayBuffer(obs_size, act_size, replay_buf_size)
    target_value_fn = dill.loads(dill.dumps(model.value_fn))

    pol_opt = torch.optim.Adam(model.policy.parameters(), lr=pol_lr)
    val_opt = torch.optim.Adam(model.value_fn.parameters(), lr=val_lr)
    q1_opt = torch.optim.Adam(model.q1_fn.parameters(), lr=q_lr)
    q2_opt = torch.optim.Adam(model.q2_fn.parameters(), lr=q_lr)

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
    q1_loss_hist = []
    q2_loss_hist = []

    progress_bar = tqdm.tqdm(total=total_steps)
    cur_total_steps = 0
    progress_bar.update(0)
    early_stop = False

    # Init replay buffer with random actions
    # ========================================================================
    while cur_total_steps < exploration_steps:
        env_futures = [do_rollout.remote(env, random_model, env_steps) for _ in range(num_envs)]
        results = ray.get(env_futures)
        for result in results:
            ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = result
            for obs1, obs2, acts, rews, done in zip(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done):
                replay_buf.store(obs1, obs2, acts, rews, done)

        ep_steps = sum([result[-1].shape[0] for result in results])
        cur_total_steps += ep_steps

    progress_bar.update(cur_total_steps)

    while cur_total_steps < total_steps:
        cur_batch_steps = 0

        # Bail out if we have met out reward threshold
        if len(raw_rew_hist) > 2 and reward_stop:
            if raw_rew_hist[-1] >= reward_stop and raw_rew_hist[-2] >= reward_stop:
                early_stop = True
                break

        # collect data with the current policy
        # ========================================================================
        while cur_batch_steps < min_steps_per_update:

            env_futures = [do_rollout.remote(env, model, env_steps) for _ in range(num_envs)]
            results = ray.get(env_futures)
            for result in results:
                ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = result
                for obs1, obs2, acts, rews, done in zip(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done):
                    replay_buf.store(obs1, obs2, acts, rews, done)

            ep_steps = sum([result[-1].shape[0] for result in results])
            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps

        raw_rew_hist.append(torch.sum(results[0][-2])) # sample from the last returned episode to report
        progress_bar.update(cur_batch_steps)

        for _ in range(int(ep_steps/num_envs)):
            # compute targets for Q and V
            # ========================================================================
            replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = replay_buf.sample_batch(replay_batch_size)

            q_targ = replay_rews + gamma * (1 - replay_done) * target_value_fn(replay_obs2)
            q_targ = q_targ.detach()

            noise = torch.randn(replay_batch_size, act_size)
            sample_acts, sample_logp = model.select_action(replay_obs1, noise)

            q_in = torch.cat((replay_obs1, sample_acts), dim=1)
            q_preds = torch.cat((model.q1_fn(q_in), model.q2_fn(q_in)), dim=1)
            q_min, q_min_idx = torch.min(q_preds, dim=1)
            q_min = q_min.reshape(-1, 1)

            v_targ = q_min - alpha * sample_logp
            v_targ = v_targ.detach()

            # q_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1, replay_acts, q_targ)
            training_generator = data.DataLoader(training_data, batch_size=q_batch_size, shuffle=False)

            for local_obs, local_acts, local_qtarg in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs, local_acts, local_qtarg = (
                    local_obs.to(device),
                    local_acts.to(device),
                    local_qtarg.to(device),
                )

                q_in = torch.cat((local_obs, local_acts), dim=1)
                q1_preds = model.q1_fn(q_in)
                q2_preds = model.q2_fn(q_in)
                q1_loss = torch.pow(q1_preds - local_qtarg, 2).mean()
                q2_loss = torch.pow(q2_preds - local_qtarg, 2).mean()
                q_loss = q1_loss + q2_loss

                q1_opt.zero_grad(); q2_opt.zero_grad()
                q_loss.backward()
                q1_opt.step(); q2_opt.step()

            # val_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1, v_targ)
            training_generator = data.DataLoader(training_data, batch_size=q_batch_size, shuffle=False)

            for local_obs, local_vtarg in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs, local_vtarg = (local_obs.to(device), local_vtarg.to(device))

                # predict and calculate loss for the batch
                val_preds = model.value_fn(local_obs)
                val_loss = torch.sum(torch.pow(val_preds - local_vtarg, 2)) / replay_batch_size

                # do the normal pytorch update
                val_opt.zero_grad()
                val_loss.backward()
                val_opt.step()

            # policy_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1)
            training_generator = data.DataLoader(training_data, batch_size=pol_batch_size, shuffle=False)

            for local_obs in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs = local_obs[0].to(device)

                noise = torch.randn(pol_batch_size, act_size)
                local_acts, local_logp = model.select_action(local_obs, noise)

                q_in = torch.cat((local_obs, local_acts), dim=1)
                pol_loss = torch.sum(alpha * local_logp - model.q1_fn(q_in)) / replay_batch_size

                # do the normal pytorch update
                pol_opt.zero_grad()
                pol_loss.backward()
                pol_opt.step()

            # Update target value fn with polyak average
            # ========================================================================
            val_loss_hist.append(val_loss.item())
            pol_loss_hist.append(pol_loss.item())
            q1_loss_hist.append(q1_loss.item())
            q2_loss_hist.append(q2_loss.item())

            val_sd = model.value_fn.state_dict()
            tar_sd = target_value_fn.state_dict()
            for layer in tar_sd:
                tar_sd[layer] = polyak * tar_sd[layer] + (1 - polyak) * val_sd[layer]

            target_value_fn.load_state_dict(tar_sd)

    return (model, raw_rew_hist, locals())

@ray.remote
def do_rollout(env, model, num_steps):

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

        noise = torch.randn(1, act_size)
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

        cur_step+=1

    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)
    ep_obs2 = torch.stack(obs2_list)
    ep_done = torch.stack(done_list).reshape(-1, 1)

    return (ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)
