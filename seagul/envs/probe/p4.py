import numpy as np
import gym


class ProbeEnv4(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(np.array([-np.inf]), np.array([np.inf]))
        self.action_space = gym.spaces.Box(np.array([-np.inf]), np.array([np.inf]))

    def step(self, act):
        obs = np.array([0.0])
        rew = np.clip(act, 0, 1)

        done = True

        return obs, rew, done, {}
        pass

    def reset(self):
        return np.array([0.0])

    def render(self, mode="whatever"):
        pass
