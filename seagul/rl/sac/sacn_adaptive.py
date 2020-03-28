import numpy as np
import torch
from torch.utils import data
import tqdm.auto as tqdm
import gym
import dill

from seagul.rl.common import ReplayBuffer, update_mean, update_var
from seagul.nn import fit_model
from seagul.rl.sac.models import RandModel


def sac_switched(
        env_name,
        total_steps,
        model,
        env_steps=0,
        min_steps_per_update=1,
        iters_per_update=100,
        replay_batch_size=64,
        seed=0,
        gamma=0.95,
        polyak=0.995,
        alpha=0.2,
        sgd_batch_size=64,
        sgd_lr=1e-3,
        exploration_steps=100,
        replay_buf_size=int(100000),
        use_gpu=False,
        reward_stop=None,
        goal_state=np.array([np.pi / 2, 0, 0, 0]),
        goal_lookback=10,
        goal_thresh=1,
        needle_lookup_prob=.5,
        gate_update_freq=500,
        gate_x = None,
        gate_y = None,
        gate_lr = 1e-5,
        gate_w = 1e-2,
        gate_epochs = 1,
        env_config={},
):
    """
    Implements soft actor critic

    Args:
        env_name: name of the openAI gym environment to solve
        total_steps: number of timesteps to run the PPO for
        model: model from seagul.rl.models. Contains policy, value fn, q1_fn, q2_fn
        min_steps_per_update: minimun number of steps to take before running updates, will finish episodes before updating
        env_steps: number of steps the environment takes before finishing, if the environment emits a done signal before this we consider it a failure.
        iters_per_update: how many update steps to make every time we update
        replay_batch_size: how big a batch to pull from the replay buffer for each update
        seed: random seed for all rngs
        gamma: discount applied to future rewards, usually close to 1
        polyak: term determining how fast the target network is copied from the value function
        alpha: weighting term for the entropy. 0 corresponds to no penalty for deterministic policy
        sgd_batch_size: minibatch size for policy updates
        sgd_lr: initial learning rate for policy optimizer
        val_lr: initial learning rate for value optimizer
        q_lr: initial learning rate for q fn optimizer
        exploration_steps: initial number of random actions to take, aids exploration
        replay_buf_size: how big of a replay buffer to use
        use_gpu: determines if we try to use a GPU or not
        reward_stop: reward value to bail at
        env_config: dictionary containing kwargs to pass to your the environment

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

    args = locals() 
    torch.set_num_threads(1)

    env = gym.make(env_name, **env_config)
    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        act_dtype = env.action_space.sample().dtype
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    obs_size = env.observation_space.shape[0]

    random_model = dill.loads(dill.dumps(model))
    random_model.swingup_controller = lambda x: torch.rand(model.num_acts) * 2 * model.act_limit - model.act_limit

    replay_buf = ReplayBuffer(obs_size, act_size, replay_buf_size)
    needle_buf = ReplayBuffer(obs_size, act_size, replay_buf_size)
    target_value_fn = dill.loads(dill.dumps(model.value_fn))

    pol_opt = torch.optim.Adam(model.policy.parameters(), lr=sgd_lr)
    val_opt = torch.optim.Adam(model.value_fn.parameters(), lr=sgd_lr)
    q1_opt = torch.optim.Adam(model.q1_fn.parameters(), lr=sgd_lr)
    q2_opt = torch.optim.Adam(model.q2_fn.parameters(), lr=sgd_lr)

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
    gate_update_counter = 0
    progress_bar.update(0)
    early_stop = False

    needle_count = 0
    not_needle_count = 0
    while cur_total_steps < exploration_steps:
        ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done, ep_path = do_rollout(env, random_model, env_steps)
        in_goal = torch.sum(torch.sqrt((ep_obs2[-goal_lookback:] - goal_state) ** 2), axis=1) < goal_thresh

        if in_goal.all():
            needle_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

        replay_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

        ep_steps = ep_rews.shape[0]
        cur_total_steps += ep_steps

    progress_bar.update(cur_total_steps)

    while cur_total_steps < total_steps:
        cur_batch_steps = 0

        # Bail out if we have met out reward threshold
        if len(raw_rew_hist) > 2 and reward_stop:
            if raw_rew_hist[-1] >= reward_stop and raw_rew_hist[-2] >= reward_stop:
                early_stop = True
                break

        # Transfer back to CPU, which is faster for rollouts
        model = model.to('cpu')
        target_value_fn = target_value_fn.to('cpu')

        # collect data with the current policy
        # ========================================================================
        while cur_batch_steps < min_steps_per_update:
            ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done, ep_path = do_rollout(env, model, env_steps)

            in_goal = torch.sum(torch.sqrt((ep_obs2[-goal_lookback:] - goal_state) ** 2), axis=1) < goal_thresh

            if in_goal.all():
                needle_count += 1
                needle_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)
            else:
                not_needle_count += 1
                replay_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

            if ep_path.sum() != 0:
                reverse_obs = np.flip(ep_obs1.numpy(), 0).copy()
                reverse_obs = torch.from_numpy(reverse_obs)

                reverse_path = np.flip(ep_path.numpy(), 0).copy()
                reverse_path = torch.from_numpy(reverse_path)

                if in_goal.all():
                    for path, obs in zip(reverse_path, reverse_obs):
                        if not path:
                            break
                        else:
                            gate_x = torch.cat((gate_x, obs.reshape(1, -1)))
                            gate_y = torch.cat((gate_y, torch.ones((1,1), dtype=torch.float32)))
                else:
                    for path, obs in zip(reverse_path, reverse_obs):
                        if not path:
                            pass
                        else:
                            gate_x = torch.cat((gate_x, obs.reshape(1, -1)))
                            gate_y = torch.cat((gate_y, torch.zeros((1,1), dtype=torch.float32)))

            ep_steps = ep_rews.shape[0]
            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps
            gate_update_counter += ep_steps

            raw_rew_hist.append(torch.sum(ep_rews))

            progress_bar.update(ep_steps)


        print("needle/normal: ", str(needle_buf.size), str(replay_buf.size))

        if gate_update_counter > gate_update_freq:
            # For training, transfer model to GPU
            model = model.to('cuda:0')
            target_value_fn = target_value_fn.to('cuda:0')

            class_weight = (gate_y.shape[0] / sum(gate_y) * gate_w).to('cuda:0')
            gate_loss = fit_model(model.gate_fn, gate_x, gate_y, gate_epochs, use_tqdm=False, use_cuda=True, batch_size=8192,
                                  loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=class_weight), learning_rate=gate_lr)

            print("gate updated: " + str(gate_y.shape[0]) + "  " + str(sum(gate_y)))

            model = model.to('cpu')
            target_value_fn = target_value_fn.to('cpu')
            gate_update_counter = 0

        for _ in range(min(int(ep_steps), iters_per_update)):
            # compute targets for Q and V
            # ========================================================================

            p = np.random.random_sample(1)
            if p > needle_lookup_prob and needle_buf.size > 0:
                replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = needle_buf.sample_batch(
                    replay_batch_size)
            else:
                replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = replay_buf.sample_batch(
                    replay_batch_size)

            replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = \
            [replay_obs1.to(device),
             replay_obs2.to(device),
             replay_acts.to(device),
             replay_rews.to(device),
             replay_done.to(device)]

            q_targ = replay_rews + gamma * (1 - replay_done) * target_value_fn(replay_obs2)
            q_targ = q_targ.detach()

            noise = torch.randn(replay_batch_size, act_size).to(device)
            sample_acts, sample_logp = model.select_action_parallel(replay_obs1, noise)

            q_in = torch.cat((replay_obs1, sample_acts), dim=1)

            q_preds = torch.cat((model.q1_fn(q_in), model.q2_fn(q_in)), dim=1)
            q_min, q_min_idx = torch.min(q_preds, dim=1)
            q_min = q_min.reshape(-1, 1)

            v_targ = q_min - alpha * sample_logp
            v_targ = v_targ.detach()

            # q_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1, replay_acts, q_targ)
            training_generator = data.DataLoader(training_data, batch_size=sgd_batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=False)

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

                q1_opt.zero_grad()
                q2_opt.zero_grad()
                q_loss.backward()
                q1_opt.step()
                q2_opt.step()

            # val_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1, v_targ)
            training_generator = data.DataLoader(training_data, batch_size=sgd_batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=False)

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
            training_generator = data.DataLoader(training_data, batch_size=sgd_batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=False)

            for local_obs in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs = local_obs[0].to(device)

                noise = torch.randn(local_obs.shape[0], act_size).to(device)
                local_acts, local_logp = model.select_action_parallel(local_obs, noise)

                q_in = torch.cat((local_obs, local_acts), dim=1)
                pol_loss = torch.sum(alpha * local_logp - model.q1_fn(q_in)) / replay_batch_size

                # do the normal pytorch update
                pol_opt.zero_grad()
                pol_loss.backward()
                pol_opt.step()


            # Update target networks
            # ========================================================================
            val_sd = model.value_fn.state_dict()
            tar_sd = target_value_fn.state_dict()
            for layer in tar_sd:
                tar_sd[layer] = polyak * tar_sd[layer] + (1 - polyak) * val_sd[layer]

            target_value_fn.load_state_dict(tar_sd)

    return model, raw_rew_hist, locals()


def do_rollout(env, model, num_steps):
    acts_list = []
    obs1_list = []
    obs2_list = []
    rews_list = []
    path_list = []
    done_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

    obs = env.reset()
    model.thresh = model.thresh_on

    while not done:
        obs1_list.append(obs)
        path = model.sig(model.gate_fn(np.array(obs, dtype=np.float32))) > model.thresh

        if path:
            model.thresh = model.thresh_off
            for _ in range(model.hold_count):
                acts = model.balance_controller(obs).reshape(-1, model.num_acts)
                obs, rew, done, _ = env.step(acts.numpy())
        else:
            model.thresh = model.thresh_on
            acts = model.swingup_controller(obs.reshape(-1, obs_size)).reshape(-1, model.num_acts)
            for _ in range(model.hold_count):
                obs, rew, done, _ = env.step(acts.numpy().reshape(-1))

        if cur_step < num_steps:
            done_list.append(torch.as_tensor(done))
        else:
            done_list.append(torch.as_tensor(False))

        acts_list.append(torch.as_tensor(acts.clone()))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        path_list.append(path.clone())
        obs2_list.append(obs)

        cur_step += 1

    ep_obs1 = torch.tensor(obs1_list, dtype=dtype)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)
    ep_obs2 = torch.tensor(obs2_list, dtype=dtype)
    ep_done = torch.stack(done_list).reshape(-1, 1)
    ep_path = torch.tensor(path_list).reshape(-1,1)

    return ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done, ep_path
