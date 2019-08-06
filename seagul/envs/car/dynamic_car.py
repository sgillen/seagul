import numpy as np
from numpy import cos, sin, pi
from gym.utils import seeding


import gym


from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class DynCarEnv(gym.Env):
    """
      Environment for what I'm calling a "dynamic car"
      which is a dubins car that can accelerate

      Attributes:
          m: mass of the car
          a_max: maximum control input for the car

          state: state of the car + goal state
                 = [x(m), y(m) theta(rads), dx/dt, dy/dt, dth/dt, goal_x, goal_y, goal_t]

    """

    metadata = {"render.modes": ["human"], "video.frames_per_second": 15}

    def __init__(self, dt=0.01):
        self.m = 1
        self.a_max = 10
        self.dt = dt
        self.state_noise_max = 0.0

        self.X_MAX = self.Y_MAX = 100.0
        # THETA_MAX is implict (-pi, pi)

        self.DX_MAX = self.DY_MAX = 500.0
        self.DTHETA_MAX = 100.0 * pi

        self.T_MAX = 10

        car_state_max = np.array([self.X_MAX, self.Y_MAX, pi, self.DX_MAX, self.DY_MAX, self.DTHETA_MAX])
        goal_state_max = np.array([self.X_MAX, self.Y_MAX, self.T_MAX])

        aug_state_max = np.concatenate((car_state_max, goal_state_max))
        self.observation_space = gym.spaces.Box(low=-aug_state_max, high=aug_state_max)

        self.action_space = gym.spaces.Box(-self.a_max, self.a_max, shape=(2,))

    def step(self, action):
        done = False

        # ns = rk4(self._derivs, torque, 0, self.dt, self.state)
        ns = euler(self._derivs, action, 0, self.dt, self.state)

        self.state[0] = ns[0]
        self.state[1] = ns[1]
        self.state[2] = wrap(ns[2], -pi, pi)

        self.state[3] = ns[2]
        self.state[4] = ns[3]
        self.state[5] = ns[5]

        # state 6 and 7 are the goal xy and don't change

        self.state[8] = self.state[8] - self.dt

        reward = (self.state[6] - self.state[0]) ** 2 + (self.state[7] - self.state[1]) ** 2

        if self.state[8] < 0:
            done = True

        return self.state, reward, done, {}

    def _derivs(self, t, q, a):
        """
        Defines the dynamics of our car, takes t and q and outputs qdot

        x = q[0]; y = q[1]; th = q[2]
        xdot = q[3]; ydot = q[4]; thdot = q[5]
        """

        qdot = np.zeros_like(q)

        qdot[0] = np.cos(q[2])
        qdot[1] = np.sin(q[2])
        qdot[2] = q[5]

        qdot[3] = a[0] * np.cos(q[2]) * self.m
        qdot[4] = a[0] * np.sin(q[2]) * self.m

        qdot[5] = a[1]

        return qdot

    def render(self, mode="human"):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10]) + self.np_random.uniform(
            -self.state_noise_max, self.state_noise_max, size=(9,)
        )
        return self.state

    def _get_ob(self):
        return self.state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max, size=(9,))


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def rk4(derivs, a, t0, dt, s0):
    """
    Single step of an RK4 solver, designed for control applications, so it passed an action to your
    derivs fcn

    Attributes:
        derivs: the function you are trying to integrate, should have signature:
        function(t,s,a) -> ds/dt

        a: action, should belong to the action space of your environment

        t0: float, initial time, often you can just set this to zero if all that matters for your
        derivs is the state and dt

        dt: how big of a timestep to integrate

        s0: initial state of your system, should belong to the envs obvervation space

    Returns:
        s[n+1]: I.E. the state of your system after integrating with action a for dt seconds

    Example:
        derivs = lambda t,q,a: (q+a)**2
        a =  1.0
        t0 = 0
        dt = .1
        s0 = 5
        s1 = rk4(derivs, a, t0, dt, s0)

    """

    k1 = dt * derivs(t0, s0, a)
    k2 = dt * derivs(t0 + dt / 2, s0 + k1 / 2, a)
    k3 = dt * derivs(t0 + dt / 2, s0 + k2 / 2, a)
    k4 = dt * derivs(t0 + dt, s0 + k3, a)

    return s0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(derivs, a, t0, dt, s0):
    return s0 + dt * derivs(t0 + dt, s0, a)
