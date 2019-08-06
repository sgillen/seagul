import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit

# Define constants (geometry and mass properties):

m1 = 1
m2 = 1
L1 = 1
L2 = 1
L1c = 0.5
L2c = 0.5
J1 = 1
J2 = 1

g = 9.8
rad = np.pi / 1


# animation generation
def animate_pend(y):
    th1 = y[:, 0]
    th2 = y[:, 1]

    x1 = L1 * cos(th1)
    y1 = L1 * sin(th2)

    x2 = L2 * cos(th1 + th2) + x1
    y2 = L2 * sin(th1 + th2) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, aspect="equal", xlim=(-3, 3), ylim=(-3, 3))
    ax.grid()

    line1, = ax.plot([], [], "o-", lw=2)
    line2, = ax.plot([], [], "o-", lw=2)
    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text("")
        return line1, line2, time_text

    def animate(i):
        line1.set_data([0, x1[i]], [0, y1[i]])
        line2.set_data([x1[i], x2[i]], [y1[i], y2[i]])

        time_text.set_text(time_template % (i * dt))
        return [line1, line2, time_text]

        return animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=40, blit=True, init_func=init)


# @jit(nopython=False)
def control(self, q, t=0):
    print("1")
    return 0


# state vector: q = transpose([theta, x, d(theta)/dt, dx/dt])
# @jit(nopython=False)
def derivs(t, q):

    u = control(q, t)
    # import ipdb; ipdb.set_trace()

    # In python this will NOT be optimized away
    th1 = q[0]
    th2 = q[1]
    dth1 = q[2]
    dth2 = q[3]

    # Inertia matrix (M) and conservative torque terms (C)
    M11 = J1 + m2 * L1 ** 2 + m1 * m1 * L1c ** 2
    M12 = L2c * m2 * cos(th1 - th2) * L1
    M21 = L1 * L2c * m2 * cos(th1 - th2)
    M22 = J2 + L2c ** 2 * m2

    C1 = g * m1 * cos(th1) * L1c + L2c * m2 * sin(th1 - th2) * dth2 ** 2 * L1
    C2 = g * L2c * m2 * cos(th2 + th1) + dth1 ** 2 * L1 * L2c * m2 * sin(th1 - th2)

    M = np.array([[M11, M12], [M21, M22]])
    C = np.array([C1, C2])

    # M*d2th + C = Xi, where Xi are the non-conservative torques, i.e.,

    # TODO, unwrap our angles

    # Combine states to define parameters to be directly controlled:
    umat = np.array([0, 1])
    # Which EOMs does u affect?
    d2th = np.linalg.solve(M, (-C + umat * u))

    return np.array([q[2], q[3], d2th[0], d2th[1]])
