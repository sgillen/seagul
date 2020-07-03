import numpy as np
import torch
import tqdm.auto as tqdm
import gym
import pickle
import copy
from seagul.rl.common import update_mean, update_std, make_schedule, discount_cumsum


class PPOAgent:
    """
           Implements proximal policy optimization with clipping

           Args:
               env_name: name of the openAI gym environment to solve
               total_steps: number of timesteps to run the PPO for
               model: model from seagul.rl.models. Contains policy and value fn
               act_std_schedule: schedule to set the variance of the policy. Will linearly interpolate values
               self.epoch_batch_size: number of environment steps to take per batch, total steps will be num_epochs*epoch_batch_size
               seed: seed for all the rngs
               gamma: discount applied to future rewards, usually close to 1
               lam: lambda for the Advantage estimation, usually close to 1
               eps: epsilon for the clipping, usually .1 or .2
               sgd_batch_size: batch size for policy updates
               sgd_batch_size: batch size for value function updates
               lr_schedule: learning rate for policy pol_optimizer
               sgd_epochs: how many epochs to use for each policy update
               target_kl: max KL before breaking
               use_gpu:  want to use the GPU? set to true
               reward_stop: reward value to stop if we achieve
               normalize_return: should we normalize the return?
               env_config: dictionary containing kwargs to pass to your the environment
    """

    def __init__(self,
                 env_name,
                 model,
                 act_std_schedule=(0.7,),
                 epoch_batch_size=2048,
                 gamma=0.99,
                 lam=0.95,
                 eps=0.2,
                 seed=0,
                 entropy_coef=0.0,
                 sgd_batch_size=1024,
                 lr_schedule=(3e-4,),
                 sgd_epochs=80,
                 target_kl=float('inf'),
                 val_coef=.5,
                 clip_val=True,
                 env_no_term_steps=0,
                 use_gpu=False,
                 reward_stop=None,
                 normalize_return=True,
                 normalize_obs=True,
                 normalize_adv=True,
                 env_config={}
                 ):

        self.env_name = env_name
        self.model = model
        self.act_std_schedule = act_std_schedule
        self.epoch_batch_size = epoch_batch_size
        self.gamma = gamma
        self.lam = lam
        self.eps = eps
        self.seed = seed
        self.entropy_coef = entropy_coef
        self.sgd_batch_size = sgd_batch_size
        self.lr_schedule = lr_schedule
        self.sgd_epochs = sgd_epochs
        self.target_kl = target_kl
        self.val_coef = val_coef
        self.clip_val = clip_val
        self.env_no_term_steps = env_no_term_steps
        self.use_gpu = use_gpu
        self.reward_stop = reward_stop
        self.normalize_return = normalize_return
        self.normalize_obs = normalize_obs
        self.normalize_adv = normalize_adv
        self.env_config = env_config
        self.old_model = pickle.loads(pickle.dumps(self.model))

    def learn(self, total_steps):

        """
        The actual training loop
        Returns:
            model: trained model
            avg_reward_hist: list with the average reward per episode at each epoch
            var_dict: dictionary with all locals, for logging/debugging purposes

        """

        # init everything
        # ==============================================================================

        torch.set_num_threads(1)
        env = gym.make(self.env_name, **self.env_config)
        if isinstance(env.action_space, gym.spaces.Box):
            self.act_size = env.action_space.shape[0]
            self.act_dtype = torch.double
        else:
            raise NotImplementedError("trying to use unsupported action space", env.action_space)

        actstd_lookup = make_schedule(self.act_std_schedule, total_steps)
        lr_lookup = make_schedule(self.lr_schedule, total_steps)

        self.model.action_std = actstd_lookup(0)
        self.sgd_lr = lr_lookup(0)

        obs_size = env.observation_space.shape[0]
        obs_mean = torch.zeros(obs_size)
        obs_std = torch.ones(obs_size)
        rew_mean = torch.zeros(1)
        rew_std = torch.ones(1)

        # seed all our RNGs
        env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # set defaults, and decide if we are using a GPU or not
        use_cuda = torch.cuda.is_available() and self.use_gpu
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # init logging stuff
        raw_rew_hist = []
        val_loss_hist = []
        pol_loss_hist = []
        progress_bar = tqdm.tqdm(total=total_steps)
        cur_total_steps = 0
        progress_bar.update(0)
        early_stop = False

        self.pol_opt = torch.optim.Adam(self.model.policy.parameters(), lr=self.sgd_lr)
        self.val_opt = torch.optim.Adam(self.model.value_fn.parameters(), lr=self.sgd_lr)

        # Train until we hit our total steps or reach our reward threshold
        # ==============================================================================
        while cur_total_steps < total_steps:
            batch_obs = torch.empty(0)
            batch_act = torch.empty(0)
            batch_adv = torch.empty(0)
            batch_discrew = torch.empty(0)
            cur_batch_steps = 0

            # Bail out if we have met out reward threshold
            if len(raw_rew_hist) > 2 and self.reward_stop:
                if raw_rew_hist[-1] >= self.reward_stop and raw_rew_hist[-2] >= self.reward_stop:
                    early_stop = True
                    break

            # construct batch data from rollouts
            # ==============================================================================
            while cur_batch_steps < self.epoch_batch_size:
                ep_obs, ep_act, ep_rew, ep_steps, ep_term = do_rollout(env, self.model, self.env_no_term_steps)

                cur_batch_steps += ep_steps
                cur_total_steps += ep_steps

                #print(sum(ep_rew).item())
                raw_rew_hist.append(sum(ep_rew).item())
                batch_obs = torch.cat((batch_obs, ep_obs.clone()))
                batch_act = torch.cat((batch_act, ep_act.clone()))

                if ep_term:
                    ep_rew = torch.cat((ep_rew, torch.zeros(1, 1)))
                else:
                    ep_rew = torch.cat((ep_rew, self.model.value_fn(ep_obs[-1]).detach().reshape(1, 1).clone()))

                ep_discrew = discount_cumsum(ep_rew, self.gamma)[:-1]
                batch_discrew = torch.cat((batch_discrew, ep_discrew.clone()))

                if self.normalize_return:
                    rew_mean = update_mean(ep_discrew, rew_mean, cur_total_steps)
                    rew_std = update_std(ep_discrew, rew_std, cur_total_steps)
                    ep_discrew = ep_rew / (rew_std + 1e-6)

                with torch.no_grad():
                    ep_val = torch.cat((self.model.value_fn(ep_obs), ep_rew[-1].reshape(1, 1).clone()))
                    deltas = ep_rew[:-1] + self.gamma * ep_val[1:] - ep_val[:-1]

                ep_adv = discount_cumsum(deltas, self.gamma * self.lam)
                # make sure our advantages are zero mean and unit variance

                batch_adv = torch.cat((batch_adv, ep_adv.clone()))

            # PostProcess epoch and update weights
            # ==============================================================================
            if self.normalize_adv:
                # adv_mean = update_mean(batch_adv, adv_mean, cur_total_steps)
                # adv_var = update_std(batch_adv, adv_var, cur_total_steps)
                batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-6)

            # Update the policy using the PPO loss
            for pol_epoch in range(self.sgd_epochs):
                pol_loss, approx_kl = self.policy_update(batch_act, batch_obs, batch_adv)
                if approx_kl > self.target_kl:
                    print("KL Stop")
                    break

            for val_epoch in range(self.sgd_epochs):
                val_loss = self.value_update(batch_obs, batch_discrew)

            # update observation mean and variance

            if self.normalize_obs:
                obs_mean = update_mean(batch_obs, obs_mean, cur_total_steps)
                obs_std = update_std(batch_obs, obs_std, cur_total_steps)
                self.model.policy.state_means = obs_mean
                self.model.value_fn.state_means = obs_mean
                self.model.policy.state_std = obs_std
                self.model.value_fn.state_std = obs_std

            self.model.action_std = actstd_lookup(cur_total_steps)
            sgd_lr = lr_lookup(cur_total_steps)

            self.old_model = pickle.loads(pickle.dumps(self.model))
            val_loss_hist.append(val_loss)
            pol_loss_hist.append(pol_loss)

            progress_bar.update(cur_batch_steps)

        progress_bar.close()
        return self.model, raw_rew_hist, locals()

    # Takes list or array and returns a lambda that interpolates it for each epoch
    def policy_update(self, batch_act, batch_obs, batch_adv):
        num_mbatch = int(batch_obs.shape[0] / self.sgd_batch_size)
        for i in range(num_mbatch):
            # policy update
            # ========================================================================
            cur_sample = i * self.sgd_batch_size

            # Transfer to GPU (if GPU is enabled, else this does nothing)
            local_obs = batch_obs[cur_sample:cur_sample + self.sgd_batch_size]
            local_act = batch_act[cur_sample:cur_sample + self.sgd_batch_size]
            local_adv = batch_adv[cur_sample:cur_sample + self.sgd_batch_size]

            # Compute the loss
            logp = self.model.get_logp(local_obs, local_act).reshape(-1, self.act_size).sum(axis=1)
            old_logp = self.old_model.get_logp(local_obs, local_act).reshape(-1, self.act_size).sum(axis=1)
            mean_entropy = -(logp * torch.exp(logp)).mean()

            r = torch.exp(logp - old_logp).reshape(-1, 1)
            clip_r = torch.clamp(r, 1 - self.eps, 1 + self.eps).reshape(-1,1)

            pol_loss = -torch.min(r * local_adv, clip_r * local_adv).mean() - self.entropy_coef * mean_entropy

            approx_kl = ((logp - old_logp) ** 2).mean()

            self.pol_opt.zero_grad()
            pol_loss.backward()
            self.pol_opt.step()

        return pol_loss, approx_kl

    def value_update(self, batch_obs, batch_discrew):
        num_mbatch = int(batch_obs.shape[0] / self.sgd_batch_size)
        for i in range(num_mbatch):
            # value_fn update
            # ========================================================================
            cur_sample = i * self.sgd_batch_size
            local_obs = batch_obs[cur_sample:cur_sample + self.sgd_batch_size]
            local_val = batch_discrew[cur_sample:cur_sample + self.sgd_batch_size]
            val_preds = self.model.value_fn(local_obs)

            if self.clip_val:
                old_val_preds = self.old_model.value_fn(local_obs)
                val_preds_clipped = old_val_preds + torch.clamp(val_preds - old_val_preds, -self.eps, self.eps)
                val_loss1 = (val_preds_clipped - local_val) ** 2
                val_loss2 = (val_preds - local_val) ** 2
                val_loss = self.val_coef * torch.max(val_loss1, val_loss2).mean()
            else:
                val_loss = self.val_coef * ((val_preds - local_val) ** 2).mean()

            self.val_opt.zero_grad()
            val_loss.backward()
            self.val_opt.step()

            return val_loss


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
    ep_rew = torch.tensor(rew_list, dtype=dtype)
    ep_rew = ep_rew.reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return ep_obs, ep_act, ep_rew, ep_length, ep_term


