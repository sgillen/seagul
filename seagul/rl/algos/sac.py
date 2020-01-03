import numpy as np
import torch
from torch.utils import data
import tqdm 
import gym
import dill

from seagul.rl.common import ReplayBuffer

def sac(
        env_name,
        total_steps,
        model,
        epoch_batch_size = 2048,
        sample_batch_size = 2048,
        seed=0,
        gamma = .95,
        polyak = .9,
        alpha  =  .9,
        pol_batch_size=1024,
        val_batch_size=1024,
        q_batch_size  =1024,
        pol_lr=1e-4,
        val_lr=1e-5,
        q_lr  = 1e-5,
        pol_epochs=10,
        val_epochs=10,
        q_epochs = 10,
        replay_buf_size = 10000,
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
    #    q_loss_hist  = ???

    
    progress_bar = tqdm.tqdm(total=total_steps)
    cur_total_steps = 0
    progress_bar.update(0)
    early_stop = False
    
    while (cur_total_steps < total_steps):
        batch_obs = torch.empty(0)
        batch_act = torch.empty(0)
        batch_adv = torch.empty(0)
        batch_discrew = torch.empty(0)
        cur_batch_steps = 0

        
        # Bail out if we have met out reward threshold
        if len(raw_rew_hist) > 2:
            if raw_rew_hist[-1] >= reward_stop and raw_rew_hist[-2] >= reward_stop:
                early_stop = True
                break

        
        while (cur_batch_steps < epoch_batch_size):
            
            ep_obs1, ep_acts, ep_rews, ep_obs2, ep_done  = do_rollout(env, model)

            # can def be made more efficient if found to be a bottleneck
            for obs1,acts,rews,obs2,done in zip(ep_obs1, ep_acts, ep_rews, ep_obs2, ep_done):
                replay_buf.store(obs1,acts,rews,obs2,done)
            
            batch_obs = torch.cat((batch_obs, ep_obs1[:-1]))
            batch_act = torch.cat((batch_act, ep_acts[:-1]))
            
            # ep_discrew = discount_cumsum(ep_rew, gamma) # [:-1] because we appended the value function to the end as an extra reward
            # batch_discrew = torch.cat((batch_discrew, ep_discrew[:-1]))
            
            # # calculate this episodes advantages
            # last_val = model.value_fn(ep_obs[-1]).reshape(-1,1)
            # # ep_rew = torch.cat((ep_rew, last_val)) # append value_fn to last reward
            # ep_val = model.value_fn(ep_obs)
            # # ep_val = torch.cat((ep_val, last_val))
            # ep_val[-1] = last_val
        
            # deltas = ep_rew[:-1] + gamma*ep_val[1:] - ep_val[:-1]
            # ep_adv = discount_cumsum(deltas.detach(), gamma*lam)
            # batch_adv = torch.cat((batch_adv, ep_adv))

            ep_steps = ep_rews.shape[0]
            cur_batch_steps += ep_steps
            cur_total_steps += ep_steps
            

        # compute targets for Q and V
        # ========================================================================
        progress_bar.update(cur_batch_steps)
        sample_obs1, sample_obs2, sample_acts, sample_rews, sample_done = replay_buf.sample_batch(sample_batch_size)

        q_targ = sample_rews + gamma*(1 - sample_done)*target_value_fn(sample_obs2)
        q_input = torch.cat((sample_obs1, sample_acts),dim=1)
        q_preds = torch.cat((model.q1_fn(q_input), model.q1_fn(q_input)),dim=1)
        q_min, q_min_idx = torch.min(q_preds, dim=1)

        v_targ = q_min - alpha*model.get_logp(sample_obs1, sample_acts)

        # Update target value fn with polyak average
        val_sd = model.value_fn.state_dict().values()
        tar_sd = target_value_fn.state_dict().values()
        for t_targ, t in zip(val_sd, tar_sd):
            t_targ = polyak*t_targ + (1-polyak)*t

        target_value_fn.load_state_dict(tar_sd)

        

    # q_fn update (recall we have two q functions
    # ========================================================================
    # training_data = data.TensorDataset(batch_obs, batch_discrew)
    # training_generator = data.DataLoader(training_data, batch_size=q_batch_size, shuffle=True)
    
    # # Update que function with the standard L2 Loss
    # for q_epoch in range(q_epochs):
    #     for local_obs, local_act, local_val in training_generator:
    #         # Transfer to GPU (if GPU is enabled, else this does nothing)
    #         local_obs, local_act, local_val = (local_obs.to(device), local_act.to(device), local_val.to(device))
            
    #         # predict and calculate loss for the batch
    #         q_preds = model.q_fn(local_obs, local_act)
    #         q_loss =  #TODO# torch.sum(torch.pow(q_preds - local_q, 2))/(q_preds.shape[0])
            
    #         # do the normal pytorch update
    #         q_opt.zero_grad()
    #         q_loss.backward()
    #         q_opt.step()


def do_rollout(env, model):

    acts_list = []
    obs1_list = []
    obs2_list = []
    rews_list = []
    done_list = []

    dtype = torch.float32
    obs = env.reset()
    done = False
    
    while not done:
        obs = torch.as_tensor(obs,dtype=dtype).detach()
        obs1_list.append(obs.clone())

        act  = model.select_action(obs).detach()
        obs, rew, done, _ = env.step(act.numpy().reshape(-1))
        obs = torch.as_tensor(obs,dtype=dtype).detach()

        
        acts_list.append(torch.as_tensor(act.clone(), dtype=dtype))
        rews_list.append(rew)
        obs2_list.append(obs.clone())
        done_list.append(torch.as_tensor(done))
      
    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.tensor(rews_list, dtype=dtype).reshape(-1,1)
    ep_obs2 = torch.stack(obs2_list)
    ep_done = torch.stack(done_list)


    return (ep_obs1, ep_acts, ep_rews, ep_obs2, ep_done)
