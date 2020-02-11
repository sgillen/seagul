import numpy as np
import gym
from numpy import cos, sin, pi

from gym.utils import seeding
import gym.spaces

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from seagul.integration import euler,rk4


class LorenzEnv(gym.core.Env):
    """
    Environment for the lorenz system


    Attributes:
    """

    def __init__(self,
                 num_steps=1000,
                 dt = 0.01,
                 s = 10,
                 b = 8/3,
                 r = 28,
                 init_state = np.array([0, 1, 1.05]),
                 xyz_max = 100,
                 u_max = 100,
                 state_noise_max = 5.0,
    ):
        
        # Lorenz system constants
        self.s = s; self.b = b; self.r = r

        # Intial state that we will reset to
        self.init_state = init_state

        # Simulation/integration parameters
        self.dt = dt
        self.num_steps = num_steps
        self.cur_step = 0
        self.state = None
        self.integrator = rk4

        # Observation (state) paramaters
        self.state_max = np.array([xyz_max, xyz_max, xyz_max,1])
        self.observation_space = gym.spaces.Box(low=-self.state_max, high=self.state_max, dtype=np.float64)
        self.state_noise_max = 5.0

        # Action (Control) parameters
        self.action_max = np.array([u_max, u_max, u_max])
        self.action_space = gym.spaces.Box(low=-self.action_max, high=self.action_max, dtype=np.float64)
        self.u_noise_max = 0.0

        self.reward_state = 1
        self.seed()
        self.state = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.init_state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max)
        self.cur_step = 0
        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))

    
        return aug_state

    def _get_ob(self):
        return self.state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max)

    def step(self, action):
        done = False

        # Algorithms aware of the action space won't need their inputs clipped but better to do it here than not
        #action = np.clip(action, -self.action_max, self.action_max)

        # Add noise to the force action (if noise is zero this will do nothing)
        if self.u_noise_max > 0:
            action += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        ds = self._derivs(0, self.state, action)

        for _ in range(10):
            self.state = self.integrator(self._derivs, action, 0, self.dt, self.state)

        # Should reward be something we pass in ? I do like to mess with them a lot...
        # if 0 < self.state[0] < 20 and 0 < self.state[1] < 30 and 0 < self.state[2] < 50:
        #     reward = 1.0
        # else:
        #     reward = -1.0


        reward = -((.01*self.state[0])**2 + (.01*self.state[1])**2 + (.01*self.state[2])**2)

        # if self.reward_state == 1:
        #     if self.state[0] > 2 and self.state[1] > 3:
        #         reward = 5.0
        #         self.reward_state = 0;
        #     else:
        #         reward = -1.0

        # elif self.reward_state == 0:
        #     if self.state[0] < -2 and self.state[1] < -3:
        #         reward = 5.0
        #         self.reward_state = 1;
        #     else:
        #         reward = -1.0

        
        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True
            # elif (np.abs(self.state) > self.state_max).any():
            #m     done = True

        
        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))
        return aug_state , reward, done, {}

    def render(self, mode="human"):
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
        ydot = self.r * q[0] - q[1] - q[0] * q[2] - u[1]
        zdot = q[0] * q[1] - self.b * q[2] - u[2]

        return np.array([xdot, ydot, zdot])
