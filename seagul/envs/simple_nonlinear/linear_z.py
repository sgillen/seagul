import numpy as np
import gym

from gym.utils import seeding
import gym.spaces

from seagul.integration import euler,rk4

class LinearEnv(gym.Env):
    """
    Environment for the lorenz system


    Attributes:
    """

    def __init__(
        self,
        num_steps=50,
        dt=0.01,
        init_state=np.array([1, 1, 1]),
        xyz_max=float('inf'),
        u_max=25,
        state_noise_max=5.0,
        act_hold=1,
        reward_fn=lambda s: (-((.01*s[0])**2 + (.01*s[1])**2 + (.01*s[2])**2), s),
        integrator = rk4
    ):
        
        # Simulation/integration parameters
        self.dt = dt
        self.num_steps = num_steps
        self.reward_fn = reward_fn
        self.init_state = init_state
        self.act_hold = act_hold
        self.cur_step = 0
        self.integrator = integrator

        self.state_max = np.array([xyz_max, xyz_max, xyz_max, 1])
        self.observation_space = gym.spaces.Box(low=-(self.state_max+50), high=self.state_max+50, dtype=np.float32)
        self.state_noise_max = 5.0

        # Action (Control) parameters
        self.action_max = np.array([u_max, u_max])
        self.action_space = gym.spaces.Box(low=-self.action_max, high=self.action_max, dtype=np.float32)
        self.u_noise_max = 0.0

        self.reward_state = 10
        self.seed()
        self.state = None
        self.reset()  # sets self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_state=None):
        if init_state is None:
            self.state = self.init_state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max)
        else:
            self.state = init_state

        self.cur_step = 0
        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))
        return aug_state

    def step(self, action):
        action = np.clip(action, -self.action_max, self.action_max)

        for _ in range(self.act_hold):
            self.state = self.integrator(self._derivs, action, 0, self.dt, self.state)

        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))
        reward, aug_state = self.reward_fn(aug_state)
        self.reward_state = aug_state[-1]

        done = False
        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True

        return aug_state , reward, done, {}

    def render(self, mode="human"):
        raise NotImplementedError

    def _derivs(self, t, q, u):
        """
        Implements the dynamics for the system

        Args:
            t: float with the current time (may not actually be used but need to keep signature compatible with ODE solvers)
            q: numpy array of state variables 
            u: nump array of control variables

        Returns:
            dqdt: numpy array with the derivatives of the current state variable
        """

        xdot = u[0]
        ydot = u[1]
        zdot = q[0]

        return np.array([xdot, ydot, zdot])

