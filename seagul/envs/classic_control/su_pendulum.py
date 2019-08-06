import numpy as np
import gym
import os
from numpy import pi, sin, cos
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# TODO
# warning this is incomplete, check out su_cartpole for a working example
class SUPendulumEnv(gym.Env):
    def __init__(self, num_steps=1500):

        self.num_steps = num_steps
        self.cur_step = 0

        self.m = 1
        self.L = 1
        self.g = 9.8

        self.init_state = np.array([0, 0])
        self.state = self.init_state

    def step(self, action):
        # get newest state variables
        done = False

        # if -.2 < self.sim.data.qpos[1] < .2 :
        #    reward = 1
        # else:
        #    reward = 0

        ob = solve_ivp(lambda t, y: self._derivs(t, y, action), self.state, [0, self.dt])
        self.state = ob

        reward = np.sin(ob[0])

        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True

        return ob, reward, done, {}

    def reset_model(self):
        self.state = self.init_state  # + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        self.cur_step = 0
        return self._get_obs()

    def animate(self, t, y):
        dt = t[-1] / len(t)

        x1 = self.L * sin(y[:, 0])
        y1 = -self.L * cos(y[:, 0])

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, aspect="equal", xlim=(-3, 3), ylim=(-3, 3))
        ax.grid()

        line1, = ax.plot([], [], "o-", lw=2)
        time_template = "time = %.1fs"
        time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

        def init():
            line1.set_data([0, x1[0]], [0, y1[0]])
            time_text.set_text("")
            return [line1, time_text]

        def animate_step(i):
            line1.set_data([0, x1[i]], [0, y1[i]])
            time_text.set_text(time_template % (i * dt))
            return [line1, time_text]

        return animation.FuncAnimation(fig, animate_step, np.arange(1, len(y)), interval=5, blit=True, init_func=init)

    def _get_obs(self):
        return self.state

    def _derivs(self, t, q, action):

        d2th = (-self.g * sin(q[0]) + action) / self.L
        return np.array([q[1], d2th])
