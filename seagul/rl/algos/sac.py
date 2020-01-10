import numpy as np
import torch
from torch.utils import data
import tqdm 
import gym
import dill

from seagul.rl.common import ReplayBuffer, update_var, update_mean
from seagul.rl.models import RandModel

def sac(
        env_name,
        total_steps,
        model,
        epoch_batch_size = 1,
        replay_batch_size = 64,
        seed=0,
        gamma = .95,
        polyak = .995,
        alpha  =  .2,
        pol_batch_size=64,
        val_batch_size=64,
        q_batch_size  =64,
        pol_lr=1e-3,
        val_lr=1e-3,
        q_lr  = 1e-3,
        exploration_steps = 100,
        replay_buf_size = int(50000),
        use_gpu=False,
        reward_stop=None,
):


    """
    Implements soft actor critic

    Args:    

    Returns:

    Example:
    """
    
    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        act_dtype = env.action_space.sample().dtype
    else:
        raise NotImplementedError("trying to use unsupported action space", env.action_space)

    obs_size = env.observation_space.shape[0]
    obs_mean = torch.zeros(obs_size)
    obs_var  = torch.ones(obs_size)

    random_model = RandModel(model.act_limit, act_size)
    replay_buf = ReplayBuffer(obs_size, act_size, replay_buf_size)
    target_value_fn = dill.loads(dill.dumps(model.value_fn))
    
    pol_opt = torch.optim.Adam(model.policy.parameters(), lr=pol_lr)
    val_opt = torch.optim.Adam(model.value_fn.parameters(), lr=val_lr)
    q1_opt  = torch.optim.Adam(model.q1_fn.parameters(), lr=q_lr)
    q2_opt  = torch.optim.Adam(model.q2_fn.parameters(), lr=q_lr)
    
    # seed all our RNGs
    env.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    
    # set defaults, and decide if we are using a GPU or not
    use_cuda = torch.cuda.is_available() and use_gpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    raw_rew_hist = []
    val_loss_hist = []
    pol_loss_hist = []
    q1_loss_hist  = []
    q2_loss_hist = []
    
    sample_logp_hist =[]
    q_targ_hist = []
    q_preds_hist = []
    q_min_hist = []
    val_preds_hist = []
    val_targ_hist = []

    progress_bar = tqdm.tqdm(total=total_steps)
    cur_total_steps = 0
    progress_bar.update(0)
    early_stop = False

    
    while(cur_total_steps < exploration_steps):
        ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done  = do_rollout(env, random_model)
        
        # can def be made more efficient if found to be a bottleneck
        for obs1,obs2,acts,rews,done in zip(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done):
            replay_buf.store(obs1,obs2,acts,rews,done)
            
        ep_steps = ep_rews.shape[0]
        cur_total_steps += ep_steps


    progress_bar.update(cur_total_steps)

    while (cur_total_steps < total_steps):
        cur_batch_steps = 0

        # Bail out if we have met out reward threshold
        if len(raw_rew_hist) > 2:
            if raw_rew_hist[-1] >= reward_stop and raw_rew_hist[-2] >= reward_stop:
                early_stop = True
                break
            
        # collect data with the current policy
        # ========================================================================
        while cur_batch_steps < epoch_batch_size:
            
            ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done  = do_rollout(env, model)

            # can def be made more efficient if found to be a bottleneck
            for obs1, obs2, acts, rews, done in zip(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done):
                replay_buf.store(obs1,obs2,acts,rews,done)

            ep_steps = ep_rews.shape[0]
            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps

        raw_rew_hist.append(torch.sum(ep_rews))
        progress_bar.update(cur_batch_steps)

        #        import ipdb; ipdb.set_trace()

        for _ in range(ep_steps):
            # compute targets for Q and V
            # ========================================================================
            replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = replay_buf.sample_batch(replay_batch_size)

            q_targ = replay_rews + gamma*(1 - replay_done)*target_value_fn(replay_obs2)
            q_targ = q_targ.detach()

            noise = torch.randn(replay_batch_size, act_size)
            sample_acts, sample_logp = model.select_action(replay_obs1, noise)

            q_in = torch.cat((replay_obs1, sample_acts), dim=1)
            q_preds = torch.cat((model.q1_fn(q_in), model.q2_fn(q_in)),dim=1)
            q_min, q_min_idx = torch.min(q_preds, dim=1)
            q_min = q_min.reshape(-1,1)

            v_targ = q_min - alpha*sample_logp
            v_targ = v_targ.detach()

            sample_logp_hist.append(sample_logp)
            q_targ_hist.append(q_targ)
            q_preds_hist.append(q_preds)
            q_min_hist.append(q_min)

            # q_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1, replay_acts, q_targ)
            training_generator = data.DataLoader(training_data, batch_size=q_batch_size, shuffle=True)

            for local_obs, local_acts, local_qtarg in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs, local_acts, local_qtarg = (local_obs.to(device), local_acts.to(device), local_qtarg.to(device))

                q_in = torch.cat((local_obs, local_acts),dim=1)
                q1_preds = model.q1_fn(q_in)
                q2_preds = model.q2_fn(q_in)
                q1_loss = torch.sum(torch.pow(q1_preds - local_qtarg, 2))/replay_batch_size
                q2_loss = torch.sum(torch.pow(q2_preds - local_qtarg, 2))/replay_batch_size

                q1_opt.zero_grad(); q2_opt.zero_grad()
                q1_loss.backward(); q2_loss.backward()
                q1_opt.step(); q2_opt.step()

            # val_fn update
            # ========================================================================
            training_data = data.TensorDataset(replay_obs1, v_targ)
            training_generator = data.DataLoader(training_data, batch_size=q_batch_size, shuffle=True)

            for local_obs, local_vtarg in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs, local_vtarg = (local_obs.to(device), local_vtarg.to(device))


                # predict and calculate loss for the batch
                val_preds = model.value_fn(local_obs)
                val_loss = torch.sum(torch.pow(val_preds - local_vtarg, 2))/replay_batch_size

                # do the normal pytorch update
                val_opt.zero_grad()
                val_loss.backward()
                val_opt.step()


            # policy_fn update
            # ========================================================================
            val_preds_hist.append(val_preds)
            val_targ_hist.append(v_targ)

            training_data = data.TensorDataset(replay_obs1)
            training_generator = data.DataLoader(training_data, batch_size=pol_batch_size, shuffle=True)

            for local_obs in training_generator:
                # Transfer to GPU (if GPU is enabled, else this does nothing)
                local_obs = local_obs[0].to(device)

                noise = torch.randn(pol_batch_size, act_size)
                local_acts, local_logp = model.select_action(local_obs, noise)

                q_in = torch.cat((local_obs, local_acts), dim=1)
                pol_loss = torch.sum(alpha*local_logp - model.q1_fn(q_in))/replay_batch_size

                if pol_loss < -1e3:
                    print("nannanan")
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


def do_rollout(env, model):

    acts_list = []
    obs1_list = []
    obs2_list = []
    rews_list = []
    done_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs = env.reset()
    done = False
    
    while not done:
        obs = torch.as_tensor(obs,dtype=dtype).detach()

        if torch.sum(torch.isnan(obs)):
            print("nanan")
            
        obs1_list.append(obs.clone())

        noise = torch.randn(1, act_size)
        act, _  = model.select_action(obs.reshape(1,-1), noise)
        act = act.detach()
        
        obs, rew, done, _ = env.step(act.numpy().reshape(-1))
        obs = torch.as_tensor(obs,dtype=dtype).detach()
        
        acts_list.append(torch.as_tensor(act.clone(), dtype=dtype))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        obs2_list.append(obs.clone())
        #        done_list.append(torch.as_tensor(done))
        done_list.append(torch.as_tensor(False))
      
    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.stack(rews_list).reshape(-1,1)
    ep_obs2 = torch.stack(obs2_list)
    ep_done = torch.stack(done_list).reshape(-1,1)

    return (ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)
