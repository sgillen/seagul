import numpy as np
import gym


class ProbeEnv2(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(np.array([-np.inf]), np.array([np.inf]))
        self.action_space = gym.spaces.Box(np.array([-np.inf]), np.array([np.inf]))

    def step(self, act):
        obs = np.array([np.random.choice([-1.0, 1.0])])
        rew = np.copy(self.last_ob)

        done = True

        return obs, rew, done, {}
        pass

    def reset(self):
        self.last_ob = np.array([np.random.choice([-1.0, 1.0])])
        return self.last_ob

    def render(self, mode="whatever"):
        pass
