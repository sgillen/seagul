from gym import core, spaces
from gym.utils import seeding
import matlab.engine
import matlab
import numpy as np
from numpy import pi
import seagul
import seagul.envs.matlab.bball_src



class BBallEnv(core.Env):
    """ Connects to nihars bouncing ball matlab code

    states are
    [0] lower link angle (rad) (absolute?)
    [1] upper link angle (relative?)
    [2] ball x position (m)
    [3] ball y position (m)
    [4] lower link velocity (rad/s)
    [5] lower link velocity (rad/s)
    [6] ball x velocity (m/s)
    [7] ball y velocity (m/s)
    """

    def __init__(self,
                 max_torque=float('inf'),
                 dt=.02,
                 seed=None,
                 init_state=(-pi / 4, 3 * pi / 4, 0.025, .5, 0, 0, 0, 0),
                 init_state_weights=(pi, pi, 0, .5, 0, 0, 0, 0),
                 reward_fn = lambda s,a: s[3],
                 done_criteria = lambda s: s[3] < (.3*np.cos(s[0]) + .3*np.cos(s[0] + s[1]) )
                ):

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(seagul.envs.matlab.bball_src.__path__[0], nargout=0)
        self.max_torque = max_torque
        self.dt = dt
        self.init_state = matlab.single(init_state,size=(8,1))
        self.init_state_weights = matlab.single(init_state_weights,size=(8,1))
        np.random.seed(seed)

        self.reward_fn = reward_fn
        self.done_criteria = done_criteria

        low = np.array([-pi, -pi, -5, -5, -10, -30, -10, -10 ])
        self.observation_space = spaces.Box(low=low, high=-low, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-max_torque]), high=np.array ([max_torque]), dtype=np.float32)

        self.reset()

    def reset(self):
        self.t = 0
        init_state = self.init_state
        #init_state += self.eng.rand(8)*(self.init_state_weights*matlab.single([2.0])) - self.init_state_weights

        self.state = init_state
        return np.array(init_state).reshape((8,))

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)
        tout, xout = self.eng.integrateODE(matlab.single([self.t, self.t+self.dt]), self.state, self.dt, matlab.single([action.item()]), nargout=2)
        impactState, impactTime = self.eng.detectImpact(tout,xout, nargout=2)

        if impactTime == -1:  #No contact
            self.state = matlab.single(xout[-1])
            self.state.reshape((8,1))
            #self.state.reshape((8,1))
            self.t = np.array(tout[-1]).item()
        else:
            self.state = self.eng.impact(matlab.single(impactState))
            #self.state.reshape((8,1))
            self.t = np.array(impactTime).item()

        reward = self.reward_fn(np.array(self.state), action)
        done = self.done_criteria(np.array(self.state))

        return np.array(self.state).reshape((8,)), reward.item(), done, {"tout": tout, "xout": xout}

    def render(self):
        raise NotImplementedError('Frame by frame rendering not supported, call animate instead')

    def animate(self,t,x):
        self.eng.animate(t,x,nargout=0)
        