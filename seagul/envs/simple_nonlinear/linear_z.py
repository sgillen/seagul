import numpy as np
import gym
from numpy import cos, sin, pi

from gym.utils import seeding
import gym.spaces

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from seagul.integration import euler,rk4

class LinearEnv(gym.Env):
    """
    Environment for the lorenz system


    Attributes:
    """

    def __init__(self, num_steps=50, dt=0.01):

        # Lorenz system constants
        self.s = 10
        self.b = 8 / 3
        self.r = 28

        # Intial state that we will reset to
        self.init_state = np.array([1, 1, 1])

        # Simulation/integration parameters
        self.dt = dt
        self.num_steps = num_steps
        self.cur_step = 0
        self.state = None
        self.integrator = rk4

        # Observation (state) paramaters
        # x_max = float('inf')
        # y_max = float('inf')
        # z_max = float('inf')

        x_max = 100
        y_max = 100
        z_max = 100
        self.state_max = np.array([x_max, y_max, z_max, 1])
        self.observation_space = gym.spaces.Box(low=-(self.state_max+50), high=self.state_max+50, dtype=np.float32)
        self.state_noise_max = 5.0

        # Action (Control) parameters
        ux_max = 25
        uy_max = 25
        uz_max = 25
        self.action_max = np.array([ux_max, uy_max])
        self.action_space = gym.spaces.Box(low=-self.action_max, high=self.action_max, dtype=np.float32)
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

    def step(self, action):
        done = False

        action = np.clip(action, -self.action_max, self.action_max)

        for _ in range(10):
            self.state = self.integrator(self._derivs, action, 0, self.dt, self.state)

        # Should reward be something we pass in ? I do like to mess with them a lot...
        # if 0 < self.state[0] < 20 and 0 < self.state[1] < 30 and 0 < self.state[2] < 50:
        #     reward = 1.0
        # else:
        #     reward = -1.0


        #reward = -((.01*self.state[0])**2 + (.01*self.state[1])**2 + (.01*self.state[2])**2)

        if self.reward_state == 1:
            if self.state[0] > 2 and self.state[2] > 0:
                reward = 5.0
                self.reward_state = -1;
            else:
                reward = -1.0

        elif self.reward_state == -1:
            if self.state[0] < -2 and self.state[2] < 0:
                reward = 5.0
                self.reward_state = 1;
            else:
                reward = -1.0

        
        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True
        
        
        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))

        if (np.abs(aug_state) > self.state_max).any():
            reward -= 1000
            done = True

        
        return aug_state , reward, done, {}

    def render(self, mode="human"):
        raise NotImplementedError

    def _derivs(self, t, q, u):
        """
        Implements the dynamics for the system

        Args:
            t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
            q: numpy array of state variables [x,y,z]

        Returns:
            dqdt: numpy array with the derivatives of the current state variable
        """

        xdot = u[0]
        ydot = u[1]
        zdot = q[0]

        return np.array([xdot, ydot, zdot])

