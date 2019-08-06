# sgillen This was Ty's original implementation of the cartpole simulation and contro

# ball: m = 1 kg
# pole: L = 1 m
# cart: M = 4 kg


import numpy as np

from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import matplotlib.animation as animation

g = 9.81  # m/s^2
L = 1.0  # length of the pole (m)
M = 4.0  # mass of the cart (kg)
m = 1  # mass of the ball at the end of the pole (kg)


# control input: u = F
# q is the state vector
def control(q):
    if (q[0] < 140 * rad) or (q[0] > 220 * rad):
        # swing up
        # energy error: Ee
        Ee = 0.5 * m * L * L * q[2] ** 2 - m * g * L * (1 + cos(q[0]))
        # energy control gain:
        k = 0.23
        # input acceleration: A (of cart)
        A = k * Ee * cos(q[0]) * q[2]
        # convert A to u (using EOM)
        delta = m * sin(q[0]) ** 2 + M
        u = A * delta - m * L * (q[2] ** 2) * sin(q[0]) - m * g * sin(q[2]) * cos(q[2])
    else:
        # balancing
        # LQR: K values from MATLAB
        k1 = 140.560
        k2 = -3.162
        k3 = 41.772
        k4 = -8.314
        u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
    return u


# state vector: q = transpose([theta, x, d(theta)/dt, dx/dt])
def derivs(q, t):
    dqdt = np.zeros_like(q)

    # control input
    u = control(q)

    delta = m * sin(q[0]) ** 2 + M

    dqdt[0] = q[2]
    dqdt[1] = q[3]

    dqdt[2] = (
        -m * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta
        - (m + M) * g * sin(q[0]) / delta / L
        - u * cos(q[0]) / delta / L
    )

    dqdt[3] = m * L * (q[2] ** 2) * sin(q[0]) / delta + m * L * g * sin(q[0]) * cos(q[0]) / delta / L + u / delta

    return dqdt


# time step
dt = 0.1
t = np.arange(0.0, 20, dt)

rad = np.pi / 180


num_trials = 100
num_states = 4
num_t = len(t)
y = np.zeros((num_t, num_states, num_trials))

for i in range(num_trials):
    # initial conditions
    theta = 0
    x = 0.0
    th_dot = 2 * (i / num_trials) - 1  # an initial velocity, triggers the swing up control
    xdot = 0.0

    # initial state
    state = np.array([theta, x, th_dot, xdot])

    # integrate the ODE using scipy.integrate.
    y[:, :, i] = integrate.odeint(derivs, state, t)


# animation generation
# x1 = y[:, 1]
# y1 = 0.0
#
# x2 = L * sin(y[:, 0]) + x1
# y2 = -L * cos(y[:, 0]) + y1
#
# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, aspect='equal', \
#                      xlim=(-3, 3), ylim=(-3, 3))
# ax.grid()
#
# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#
#
# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text
#
#
# def animate(i):
#     thisx = [x1[i], x2[i]]
#     thisy = [y1, y2[i]]
#
#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i * dt))
#     return line, time_text


# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
#    interval=40, blit=True, init_func=init)


# show the animation
plt.show()
plt.figure()

plt.plot(y[:, 1, 1])
plt.show()
