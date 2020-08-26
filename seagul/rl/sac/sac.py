import numpy as np
import torch
from torch.utils import data
import tqdm.auto as tqdm
import gym
import dill

from seagul.rl.common import ReplayBuffer, update_mean, update_std, RandModel


class SACAgent:
    def __init__(self, env_name, model, env_max_steps=0, min_steps_per_update=1, iters_per_update=100,
                 replay_batch_size=64, seed=0, gamma=0.95, polyak=0.995, alpha=0.2, sgd_batch_size=64,
                 sgd_lr=1e-3, exploration_steps=100, replay_buf_size=int(100000), normalize_steps = 1000,
                 use_gpu=False, reward_stop=None, env_config={}):
        """
        Implements soft actor critic

        Args:
            env_name: name of the openAI gym environment to solve
            model: model from seagul.rl.models. Contains policy, value fn, q1_fn, q2_fn
            min_steps_per_update: minimun number of steps to take before running updates, will finish episodes before updating
            env_max_steps: number of steps the environment takes before finishing, if the environment emits a done signal before this we consider it a failure.
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

        """
        self.env_name = env_name
        self.model = model
        self.env_max_steps=env_max_steps
        self.min_steps_per_update = min_steps_per_update
        self.iters_per_update = iters_per_update
        self.replay_batch_size = replay_batch_size
        self.seed = seed
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.sgd_batch_size = sgd_batch_size
        self.sgd_lr = sgd_lr
        self.exploration_steps = exploration_steps
        self.replay_buf_size = replay_buf_size
        self.normalize_steps = normalize_steps
        self.use_gpu = use_gpu
        self.reward_stop = reward_stop
        self.env_config = env_config

    def learn(self, train_steps):
        """
                runs sac for train_steps

                Returns:
                    model: trained model
                    avg_reward_hist: list with the average reward per episode at each epoch
                    var_dict: dictionary with all locals, for logging/debugging purposes
                """

        torch.set_num_threads(1) # performance issue with data loader

        env = gym.make(self.env_name, **self.env_config)
        if isinstance(env.action_space, gym.spaces.Box):
            act_size = env.action_space.shape[0]
            act_dtype = env.action_space.sample().dtype
        else:
            raise NotImplementedError("trying to use unsupported action space", env.action_space)

        obs_size = env.observation_space.shape[0]

        random_model = RandModel(self.model.act_limit, act_size)
        self.replay_buf = ReplayBuffer(obs_size, act_size, self.replay_buf_size)
        self.target_value_fn = dill.loads(dill.dumps(self.model.value_fn))

        pol_opt = torch.optim.Adam(self.model.policy.parameters(), lr=self.sgd_lr)
        val_opt = torch.optim.Adam(self.model.value_fn.parameters(), lr=self.sgd_lr)
        q1_opt = torch.optim.Adam(self.model.q1_fn.parameters(), lr=self.sgd_lr)
        q2_opt = torch.optim.Adam(self.model.q2_fn.parameters(), lr=self.sgd_lr)

        # seed all our RNGs
        env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # set defaults, and decide if we are using a GPU or not
        use_cuda = torch.cuda.is_available() and self.use_gpu
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.raw_rew_hist = []
        self.val_loss_hist = []
        self.pol_loss_hist = []
        self.q1_loss_hist = []
        self.q2_loss_hist = []

        progress_bar = tqdm.tqdm(total=train_steps + self.normalize_steps)
        cur_total_steps = 0
        progress_bar.update(0)
        early_stop = False
        norm_obs1 = torch.empty(0)



        while cur_total_steps < self.normalize_steps:
            ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = do_rollout(env, random_model, self.env_max_steps)
            norm_obs1 = torch.cat((norm_obs1, ep_obs1))

            ep_steps = ep_rews.shape[0]
            cur_total_steps += ep_steps

            progress_bar.update(ep_steps)
        if self.normalize_steps > 0:
            obs_mean = norm_obs1.mean(axis=0)
            obs_std  = norm_obs1.std(axis=0)
            obs_std[torch.isinf(1/obs_std)] = 1

            self.model.policy.state_means = obs_mean
            self.model.policy.state_std  =  obs_std
            self.model.value_fn.state_means = obs_mean
            self.model.value_fn.state_std = obs_std
            self.target_value_fn.state_means = obs_mean
            self.target_value_fn.state_std = obs_std

            self.model.q1_fn.state_means = torch.cat((obs_mean, torch.zeros(act_size)))
            self.model.q1_fn.state_std = torch.cat((obs_std, torch.ones(act_size)))
            self.model.q2_fn.state_means = self.model.q1_fn.state_means
            self.model.q2_fn.state_std = self.model.q1_fn.state_std

        while cur_total_steps < self.exploration_steps:
            ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = do_rollout(env, random_model, self.env_max_steps)
            self.replay_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

            ep_steps = ep_rews.shape[0]
            cur_total_steps += ep_steps

            progress_bar.update(ep_steps)

        while cur_total_steps < train_steps:
            cur_batch_steps = 0

            # Bail out if we have met out reward threshold
            if len(self.raw_rew_hist) > 2 and self.reward_stop:
                if self.raw_rew_hist[-1] >= self.reward_stop and self.raw_rew_hist[-2] >= self.reward_stop:
                    early_stop = True
                    break

            # collect data with the current policy
            # ========================================================================
            while cur_batch_steps < self.min_steps_per_update:
                ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done = do_rollout(env, self.model, self.env_max_steps)
                self.replay_buf.store(ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)

                ep_steps = ep_rews.shape[0]
                cur_batch_steps += ep_steps
                cur_total_steps += ep_steps

                self.raw_rew_hist.append(torch.sum(ep_rews))
                print(self.raw_rew_hist[-1])



            progress_bar.update(cur_batch_steps)

            for _ in range(min(int(ep_steps), self.iters_per_update)):
                # compute targets for Q and V
                # ========================================================================
                replay_obs1, replay_obs2, replay_acts, replay_rews, replay_done = self.replay_buf.sample_batch(self.replay_batch_size)

                q_targ = replay_rews + self.gamma * (1 - replay_done) * self.target_value_fn(replay_obs2)
                q_targ = q_targ.detach()

                noise = torch.randn(self.replay_batch_size, act_size)
                sample_acts, sample_logp = self.model.select_action(replay_obs1, noise)

                q_in = torch.cat((replay_obs1, sample_acts), dim=1)
                q_preds = torch.cat((self.model.q1_fn(q_in), self.model.q2_fn(q_in)), dim=1)
                q_min, q_min_idx = torch.min(q_preds, dim=1)
                q_min = q_min.reshape(-1, 1)

                v_targ = q_min - self.alpha * sample_logp
                v_targ = v_targ.detach()

                # q_fn update
                # ========================================================================
                num_mbatch = int(self.replay_batch_size / self.sgd_batch_size)

                for i in range(num_mbatch):
                    cur_sample = i*self.sgd_batch_size

                    q_in = torch.cat((replay_obs1[cur_sample:cur_sample + self.sgd_batch_size], replay_acts[cur_sample:cur_sample + self.sgd_batch_size]), dim=1)
                    q1_preds = self.model.q1_fn(q_in)
                    q2_preds = self.model.q2_fn(q_in)
                    q1_loss = torch.pow(q1_preds - q_targ[cur_sample:cur_sample + self.sgd_batch_size], 2).mean()
                    q2_loss = torch.pow(q2_preds - q_targ[cur_sample:cur_sample + self.sgd_batch_size], 2).mean()
                    q_loss = q1_loss + q2_loss

                    q1_opt.zero_grad()
                    q2_opt.zero_grad()
                    q_loss.backward()
                    q1_opt.step()
                    q2_opt.step()

                # val_fn update
                # ========================================================================
                for i in range(num_mbatch):
                    cur_sample = i*self.sgd_batch_size

                    # predict and calculate loss for the batch
                    val_preds = self.model.value_fn(replay_obs1[cur_sample:cur_sample + self.sgd_batch_size])
                    val_loss = torch.sum(torch.pow(val_preds - v_targ[cur_sample:cur_sample + self.sgd_batch_size], 2)) / self.replay_batch_size

                    # do the normal pytorch update
                    val_opt.zero_grad()
                    val_loss.backward()
                    val_opt.step()

                # policy_fn update
                # ========================================================================
                for param in self.model.q1_fn.parameters():
                    param.requires_grad = False

                for i in range(num_mbatch):
                    cur_sample = i*self.sgd_batch_size

                    noise = torch.randn(replay_obs1[cur_sample:cur_sample + self.sgd_batch_size].shape[0], act_size)
                    local_acts, local_logp = self.model.select_action(replay_obs1[cur_sample:cur_sample + self.sgd_batch_size], noise)

                    q_in = torch.cat((replay_obs1[cur_sample:cur_sample + self.sgd_batch_size], local_acts), dim=1)
                    pol_loss = torch.sum(self.alpha * local_logp - self.model.q1_fn(q_in)) / self.replay_batch_size

                    pol_opt.zero_grad()
                    pol_loss.backward()
                    pol_opt.step()

                for param in self.model.q1_fn.parameters():
                    param.requires_grad = True

                # Update target value fn with polyak average
                # ========================================================================
                self.val_loss_hist.append(val_loss.item())
                self.pol_loss_hist.append(pol_loss.item())
                self.q1_loss_hist.append(q1_loss.item())
                self.q2_loss_hist.append(q2_loss.item())

                val_sd = self.model.value_fn.state_dict()
                tar_sd = self.target_value_fn.state_dict()
                for layer in tar_sd:
                    tar_sd[layer] = self.polyak * tar_sd[layer] + (1 - self.polyak) * val_sd[layer]

                self.target_value_fn.load_state_dict(tar_sd)

        return self.model, self.raw_rew_hist, locals()


def do_rollout(env, model, num_steps):
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

        cur_step += 1

    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)
    ep_obs2 = torch.stack(obs2_list)
    ep_done = torch.stack(done_list).reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return (ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)
