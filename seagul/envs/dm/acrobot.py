from gym import core, spaces
from dm_control import suite
import numpy as np

class DMAcroEnv(core.Env):
    dtype = np.float32

    def __init__(self):
        self.dm_env = suite.load('acrobot', 'swingup')

        act_spec = self.dm_env.action_spec()
        self.act_high = np.array(act_spec.maximum, dtype=self.dtype)
        self.act_low = np.array(act_spec.minimum, dtype=self.dtype)
        self.action_space = spaces.Box(low=self.act_low, high=self.act_high, dtype=self.dtype)

        obs_low = np.array([-1, -1, -1, -1, -float('inf'), -float('inf')])
        obs_high = -obs_low
        self.observation_space = spaces.Box(low=obs_low-.2, high=obs_high+.2, dtype=self.dtype)

    def reset(self, init_vec=None):
        time, reward, discount, dm_obs = self.dm_env.reset()

        if init_vec is not None:
            self.dm_env.physics.set_state(init_vec)
            dm_obs = self.dm_env.task.get_observation(self.dm_env.physics)

        obs = np.concatenate((dm_obs['orientations'], dm_obs['velocity']))
        obs = np.array(obs, dtype=self.dtype)
        return obs

    def step(self, act):
        act = np.clip(act, -self.act_low, self.act_high)
        time, reward, discount, dm_obs = self.dm_env.step(act)
        obs = np.concatenate((dm_obs['orientations'], dm_obs['velocity']))
        obs = np.array(obs, dtype=self.dtype)

        done = False
        if time.last():
            done = True

        return obs, reward, done, {}

    def render(self, mode='human'):
        raise NotImplementedError
