
import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Force keras to use the CPU becuase it's actually faster for this size network
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import scipy.integrate as integrate



from numba import jit
import time
class Pendulum:
    # Define constants (geometry and mass properties):

    def __init__(self):
        self.m = 1
        self.L = 1
        self.g = 9.8


    # animation generation
    def animate(self, t, y):
        dt = t[-1]/len(t)

        x1 = self.L * sin(y[:,0])
        y1 = -self.L * cos(y[:,0])

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, aspect='equal', \
                             xlim=(-3, 3), ylim=(-3, 3))
        ax.grid()

        line1, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line1.set_data([0, x1[0]], [0 , y1[0]])
            time_text.set_text('')
            return [line1,  time_text]

        def animate_step(i):
            line1.set_data([0, x1[i]], [0, y1[i]])
            time_text.set_text(time_template % (i * dt))
            return [line1, time_text]

        return animation.FuncAnimation(fig, animate_step, np.arange(1, len(y)), interval=5, blit=True, init_func=init)


    #@jit(nopython=False)
    # default controller does nothing, overwrite this with something useful
    def control(self, t,q):
        return 0


    # state vector: q = transpose([theta, d(theta)/dt])
    # TODO, unwrap our angles
    #@jit(nopython=False)
    def derivs(self, t, q):

        u =self.control(t, q)

        d2th = (-self.g*sin(q[0]) + u)/self.L
        ret = np.array([q[1], d2th])
        return ret


if __name__ == '__main__':
    bot = Pendulum()

    theta = 0
    th_dot = 1

    # initial state
    init_state = np.array([theta, th_dot])
    dt = 0.1
    time = np.arange(0.0, 20, dt)

    # integrate the ODE using scipy.integrate.
    u_hist = []
    y = integrate.odeint(bot.derivs, init_state, time)