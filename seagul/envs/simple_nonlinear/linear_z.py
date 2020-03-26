import numpy as np
import gym

from gym.utils import seeding
import gym.spaces

from seagul.integration import euler,rk4

class LinearEnv(gym.Env):
    """
    Environment for the our "Linear Z" system.. just take a look at the dynamics. Also includes an extra
    "reward" state for the policy, in case you have a time dependend reward
    """

    def __init__(
        self,
        num_steps=50,
        dt=0.01,
        act_hold=1,
        init_state=np.array([1, 1, 1]),
        init_noise_max=5.0,
        xyz_max=float('inf'),
        u_max=25,
        reward_fn=lambda s: (-((.01*s[0])**2 + (.01*s[1])**2 + (.01*s[2])**2), s),
        integrator=rk4
    ):

        """
        num_steps: number of actions to take in an episode
        dt: integration timestep
        act_hold: how many integration steps to do between policy evals. So total time for an episode is act_hold*dt*num_steps
        init_state: on reset, the state is init_state + np.random.uniform(-init_noise_max, init_noise_max)
        init_noise_max: see above ^^
        xyz_max: values to kill the episode at, if we exceed them
        u_max: values to clip actions at
        reward_fn: lambda defining reward in terms of the augmented state (x,y,z,r)
        integrator: which integrator to use, must be from seagul.integration
        """

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
        self.init_noise_max = init_noise_max

        # Action (Control) parameters
        self.action_max = np.array([u_max, u_max])
        self.action_space = gym.spaces.Box(low=-self.action_max, high=self.action_max, dtype=np.float32)

        self.reward_fn = reward_fn
        self.reward_state = 10
        self.seed()
        self.state = None
        self.reset()  # sets self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_state=None):        
        if init_state is None:
            self.state = self.init_state + self.np_random.uniform(-self.init_noise_max, self.init_noise_max)
        else:
            self.state = init_state

        self.cur_step = 0
        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))
        return aug_state

    def step(self, action):
        action = np.clip(action, -self.action_max, self.action_max)

        full_obs = np.zeros((self.act_hold,3))
        for i in range(self.act_hold):
            self.state = self.integrator(self._derivs, action, 0, self.dt, self.state)
            full_obs[i,:] = self.state

        aug_state = np.concatenate((self.state, np.array(self.reward_state).reshape(-1)))
        reward, aug_state = self.reward_fn(aug_state)
        self.reward_state = aug_state[-1]
    
        done = False
        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True

        return aug_state , reward, done, {"full_obs": full_obs}

    def render(self, mode="human"):
        raise NotImplementedError

    def _derivs(self, t, q, u):
        """
        Implements the dynamics for the system, you could monkey patch this if you really wanted

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

