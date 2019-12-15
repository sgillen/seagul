import numpy as np
import torch
from torch.utils import data
import tqdm
import copy
import gym

from seagul.rl.common import discount_cumsum


def ppo(
        env_name,
        total_steps,
        model,
        act_var_schedule = [.7],
        epoch_batch_size = 2048,
        gamma = .99,
        lam = .99,
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
        total_steps: number of timesteps to run the PPO for
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
      obs_var  = torch.ones(obs_size)
      adv_mean = torch.zeros(1)
      adv_var  = torch.ones(1)
      rew_mean = torch.zeros(1)
      rew_var  = torch.ones(1)

      
      old_model = copy.deepcopy(model)

      policy_opt = torch.optim.Adam(model.policy.parameters(), lr=policy_lr)
      value_opt  = torch.optim.Adam(model.value_fn.parameters(), lr=value_lr)

      # seed all our RNGs
      env.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

      # set defaults, and decide if we are using a GPU or not
      #torch.set_default_dtype(torch.double)

      
      use_cuda = torch.cuda.is_available() and use_gpu
      device = torch.device("cuda:0" if use_cuda else "cpu")

      raw_rew_hist = []
      value_loss_hist = []
      policy_loss_hist = []
      
      progress_bar = tqdm.tqdm(total=total_steps)
      cur_total_steps = 0
      progress_bar.update(0)
      
      while (cur_total_steps < total_steps):

            batch_obs = torch.empty(0)
            batch_act = torch.empty(0)
            batch_adv = torch.empty(0)
            batch_discrew = torch.empty(0)
            
            cur_batch_steps = 0
            
            while (cur_batch_steps < epoch_batch_size):
                  
                  ep_obs, ep_act, ep_rew, ep_steps = do_rollout(env, model)

                  batch_obs = torch.cat((batch_obs, ep_obs))
                  batch_act = torch.cat((batch_act, ep_act))

                  raw_rew_hist.append(sum(ep_rew))
                  # rew_mean = update_mean(ep_rew, rew_mean, cur_total_steps)
                  # rew_var = update_mean(ep_rew, rew_var, cur_total_steps)
                  # ep_rew = (ep_rew - rew_mean)/(rew_var)
                  
                  ep_discrew = discount_cumsum(ep_rew, gamma) # [:-1] because we appended the value function to the end as an extra reward
                  batch_discrew = torch.cat((batch_discrew, ep_discrew))

                  # calculate this episodes advantages
                  last_val = model.value_fn(ep_obs[-1]).reshape(-1,1)
                  ep_rew = torch.cat((ep_rew, last_val)) # append value_fn to last reward
                  ep_val = model.value_fn(ep_obs)
                  ep_val = torch.cat((ep_val, last_val))

                  deltas = ep_rew[:-1] + gamma*ep_val[1:] - ep_val[:-1]
                  ep_adv = discount_cumsum(deltas.detach(), gamma*lam)
                  batch_adv = torch.cat((batch_adv, ep_adv))

                  # adv_mean = update_mean(batch_adv, adv_mean, cur_total_steps)
                  # adv_var = update_var(batch_adv, adv_var, cur_total_steps)
                  # batch_adv = (batch_adv - adv_mean) / (adv_var+1e-6)


                  cur_batch_steps += ep_steps
                  cur_total_steps += ep_steps


            #update filters and apply them



            # policy update
            # ========================================================================
            # construct a training data generator
            training_data = data.TensorDataset(batch_obs, batch_act, batch_adv)
            training_generator = data.DataLoader(training_data, batch_size=policy_batch_size, shuffle=True)
                        
            # iterate through the data, doing the updates for our policy
            for p_epoch in range(p_epochs):
                  for local_obs, local_act, local_adv in training_generator:
                        # Transfer to GPU (if GPU is enabled, else this does nothing)
                        local_states, local_actions, local_adv = (
                              local_obs.to(device),
                              local_act.to(device),
                              local_adv.to(device),
                        )
                        
                        # predict and calculate loss for the batch
                        logp = model.get_logp(local_obs, local_act)
                        old_logp = old_model.get_logp(local_obs, local_act)
                        r = torch.exp(logp - old_logp)
                        p_loss = (
                              -torch.sum(torch.min(r * local_adv, local_adv * torch.clamp(r, (1 - eps), (1 + eps))))
                              / r.shape[0]
                        )
                        
                        # do the normal pytorch update
                        policy_opt.zero_grad()
                        p_loss.backward()
                        policy_opt.step()
                        
                        
                 


            # value_fn update
            # ========================================================================
            # construct a training data generator


            training_data = data.TensorDataset(batch_obs, batch_discrew)
            training_generator = data.DataLoader(training_data, batch_size=value_batch_size, shuffle=True)

            
            for v_epoch in range(v_epochs):
                  for local_obs, local_val in training_generator:
                        # Transfer to GPU (if GPU is enabled, else this does nothing)
                        local_obs, local_val = (local_obs.to(device), local_val.to(device))
                        
                        # predict and calculate loss for the batch
                        value_preds = model.value_fn(local_obs)
                        v_loss = torch.sum(torch.pow(value_preds - local_val, 2))/(value_preds.shape[0])
                        
                        # do the normal pytorch update
                        value_opt.zero_grad()
                        v_loss.backward()
                        value_opt.step()


            old_model = copy.deepcopy(model)
                        
            obs_mean = update_mean(batch_obs, obs_mean, cur_total_steps)
            obs_var = update_var(batch_obs, obs_var, cur_total_steps)

#            import ipdb; ipdb.set_trace()
            model.policy.state_means = obs_mean
            model.value_fn.state_means = obs_mean
            model.policy.state_var = obs_var
            model.value_fn.state_var = obs_var

            model.action_var = actvar_lookup(cur_total_steps)

            value_loss_hist.append(v_loss)
            policy_loss_hist.append(p_loss)

            progress_bar.update(cur_batch_steps)
      

      progress_bar.close()
      return model, raw_rew_hist, locals()

# Takes list or array and returns a lambda that interpolates it for each epoch
def make_variance_schedule(var_schedule, model, num_steps):
      var_schedule = np.asarray(var_schedule)
      sched_length = var_schedule.shape[0]
      x_vals = np.linspace(0,num_steps,sched_length)
      var_lookup = lambda epoch: np.interp(epoch, x_vals, var_schedule)
      return var_lookup


def update_mean(data, cur_mean, cur_steps):
      new_steps = data.shape[0]
      return (torch.mean(data,0)*new_steps + cur_mean*cur_steps)/(cur_steps+new_steps)

            
def update_var(data, cur_var, cur_steps):
      new_steps = data.shape[0]
      return (torch.var(data,0)*new_steps + cur_var*cur_steps)/(cur_steps+new_steps)
            
def do_rollout(env, model):

      act_list = []
      obs_list = []
      rew_list = []
      num_steps = 0

      obs = env.reset()
      done = False
      
      while not done:
            obs = torch.as_tensor(obs).detach()
            obs_list.append(obs)
            
            act, logprob = model.select_action(obs)
            obs, rew, done, _ = env.step(act.numpy().reshape(-1))

            act_list.append(torch.as_tensor(act.clone()))
            rew_list.append(rew)
      
      ep_length = len(rew_list)
      ep_obs = torch.stack(obs_list)
      ep_act = torch.stack(act_list)
      ep_rew = torch.tensor(rew_list)
      ep_rew = ep_rew.reshape(-1,1)

      return (ep_obs, ep_act, ep_rew, ep_length)
