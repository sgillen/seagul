from gym import core, spaces
from numpy.random import default_rng
import numpy as np


class TreeSimple(core.Env):
    def __init__(self, L=5.0, init_y=2, g=-5, dt=.1, tol=.1, N=5, seed=None):
        self.L = L
        self.init_y = init_y
        self.g = g
        self.dt = dt
        self.tol = tol
        self.N = N

        self.rng = default_rng(seed)

        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]))
        self.action_space = spaces.Box(low=np.array([-self.L]), high=np.array([self.L]))

        self.deadzone = np.array([-2, 2])
        self.xrange = np.array([-10.0, 10.0])

    def seed(self, seed):
        self.rng = default_rng(seed)

    def reset(self):
        self.x = self.rng.uniform(low=self.deadzone[0], high=self.deadzone[1])
        self.y = self.init_y
        self.X = np.array([self.x, self.y])
        self.cur_step = 0
        return self.X

    def step(self, act):
        x = self.X[0]
        y = self.X[1]
        reward = np.array([0.0], dtype=act.dtype)
        done = False

        act = np.clip(act, -self.L, self.L)

        x += act * self.dt
        x = np.clip(x, self.xrange[0], self.xrange[1]).item()
        y += self.g * self.dt

        self.X = np.array([x, y])

        reward = -0.1 * (act ** 2) + 1

        if (self.deadzone[0] < x < self.deadzone[1]) and (-self.tol < y < self.tol):
            done = True
            reward -= 25.0

        self.cur_step += 1
        if self.cur_step >= self.N:
            done = True

        return self.X, reward.item(), done, {}

    def render(self, mode=None):
        pass
