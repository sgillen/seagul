import numpy as np
import gym
from numpy import cos, sin, pi

from gym.utils import seeding
import gym.spaces

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LorenzEnv(gym.Env):
    """
    Environment for the lorenz system


    Attributes:


    """

    def __init__(self, num_steps  = 100, dt = .01):

        # Lorenz system constants
        self.s = 10
        self.b = 8/3
        self.r = 28

        # Intial state that we will reset to
        self.init_state = np.array([0,1,1.05])

        # Simulation/integration parameters
        self.dt = dt
        self.num_steps = num_steps
        self.cur_step = 0
        self.state = None
        self.integrator = rk4


        # Observation (state) paramaters
        x_max = 100; y_max = 100; z_max = 100;
        self.state_max = np.array([x_max, y_max, z_max])
        self.observation_space = gym.spaces.Box(low=-self.state_max, high=self.state_max, dtype=np.float64)
        self.state_noise_max = 0.0


        # Action (Control) parameters
        ux_max = 100; uy_max = 100; uz_max = 100;
        self.action_max = np.array([ux_max, uy_max, uz_max])
        self.action_space = gym.spaces.Box(low=-self.action_max, high=self.action_max, dtype=np.float64)
        self.u_noise_max = 0.0

        self.seed()
        self.state = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.init_state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max)
        self.cur_step = 0

        return self.state

    def _get_ob(self):
        return self.state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max)

    def step(self, action):
        done = False

        # Algorithms aware of the action space won't need their inputs clipped but better to do it here than not
        action = np.clip(action, -self.action_max, self.action_max)

        # Add noise to the force action (if noise is zero this will do nothing)
        if self.u_noise_max > 0:
            action += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        ds = self._derivs(0, self.state, action)
        self.state = self.integrator(self._derivs, action, 0, self.dt, self.state)

        # Should reward be something we pass in ? I do like to mess with them a lot...
        if (0 < self.state[0] < 20  and 0 < self.state[1] < 30 and 0 < self.state[2] < 50):
            reward = 1.0
        else:
            reward = -1.0

        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True
        elif (np.abs(self.state) > self.state_max).any():
            done = True

        return self.state, reward, done, {}

    def render(self, mode='human'):
       raise NotImplementedError


    def _derivs(self, t, q, u):
        """
        Implements the dynamics for the lorenz system

        Args:
            t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
            q: numpy array of state variables [x,y,z]
            numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]

        Returns:
            dqdt: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """

        xdot = self.s * (q[1] - q[0]) - u[0]
        ydot = self.r * q[0] - q[1] - q[0]*q[2] - u[1]
        zdot = q[0] * q[1] - self.b * q[2] - u[2]

        return np.array([xdot,ydot,zdot])


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

    k1 = dt * derivs(t0, s0, a);
    k2 = dt * derivs(t0 + dt / 2, s0 + k1 / 2, a);
    k3 = dt * derivs(t0 + dt / 2, s0 + k2 / 2, a);
    k4 = dt * derivs(t0 + dt, s0 + k3, a);

    return  s0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4);


def euler(derivs, a, t0, dt, s0):
    return s0 + dt*derivs(t0+dt, s0, a)
