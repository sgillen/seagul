import numpy as np
import gym
from numpy import cos, sin, pi

from gym.utils import seeding
import gym.spaces

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from seagul.integration import euler,rk4


def lorenz_dynamics(t, q, u):
    """
    Implements the dynamics for the lorenz system
    
    Args:
        t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
        q: numpy array of state variables [x,y,z]
        numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
    
        Returns:
            dqdt: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
    """

    s = 10.0; b = 8/3; r = 28.0,
    
    xdot = s * (q[1] - q[0]) - u[0]
    ydot = r * q[0] - q[1] - q[0] * q[2] - u[1]
    zdot = q[0] * q[1] - b * q[2] - u[2]
    
    return np.array([xdot, ydot, zdot])


class GenEnv(gym.core.Env):
    """
    Environment for the lorenz system


    Attributes:
    """

    def __init__(self,
                 num_steps=1000,
                 dt = 0.01,
                 init_state = np.array([0, 1, 1.05]),
                 xyz_max = 100.0,
                 u_max = 100.0,
                 state_noise_max = 5.0,
                 u_noise_max = 0.0,
                 act_hold = 10,
                 dynamics = lorenz_dynamics,
                 reward_fn = lambda s: -((.01*s[0])**2 + (.01*s[1])**2 + (.01*s[2])**2),
    ):
        

        # Intial state that we will reset to
        self.init_state = init_state

        # Simulation/integration parameters
        self.dt = dt
        self.num_steps = num_steps
        self.act_hold = act_hold
        self.dynamics = lorenz_dynamics
        self.reward_fn = reward_fn
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


        action += self.np_random.uniform(self.action_space.low, self.action_space.high)
        action = np.clip(action, -self.action_max, self.action_max)
        
        for _ in range(self.act_hold):
            self.state = self.integrator(self.dynamics, action, 0, self.dt, self.state)

        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))
        reward = self.reward_fn(aug_state)
            
        self.cur_step += 1

        done = False
        if self.cur_step > self.num_steps:
            done = True
        # elif (np.abs(self.state) > self.state_max).any():
            #m     done = True
        

        return aug_state , reward, done, {}

    def render(self, mode="human"):
        raise NotImplementedError

