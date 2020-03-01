"""classic Acrobot task but in drake"""
import numpy as np
from numpy import sin, cos, pi

from gym import core, spaces
from gym.utils import seeding

import numpy as np
from numpy import pi
import math

from seagul.resources import getResourcePath

from pydrake.all import (
    DiagramBuilder,
    FloatingBaseType,
    LinearQuadraticRegulator,
    RigidBodyTree,
    Saturation,
    Simulator,
    WrapToSystem,
    VectorSystem,
    SignalLogger,
    RigidBodyPlant,
    RigidBodyTree,
)

from pydrake.examples.acrobot import AcrobotInput, AcrobotPlant, AcrobotState

import matplotlib.pyplot as plt
from IPython.display import HTML

# We use a global action variable to implement an action hold/frame skip...
g_action = 0

class DrakeAcroEnv(core.Env):
    
    """ A simple acrobot environment implemented in drake
    
    Why drake for such a simple environment? When I created this I had plans to extend the 
    work I was doing to walking systems. I wanted to leverage Drakes trajectory optimization 
    in those walking systems, so it seemed like a good idea to wrangle a drake sim into a gym
    environment with the much simpler acrobot first. 
        
    It's a little overkill for sure, but I do like that I can set a dynamic step integrator
    with fixed step controller update... And now that this is built there is not a compelling reason
    for me to make another version with my own dynamics or something...
        
    """
    
    def __init__(self,
                 max_torque=25,
                 init_state = [0.0, 0.0, 0.0, 0.0],
                 init_state_weights=np.array([0, 0, 0, 0]),
                 dt = .01,
                 max_t = 5,
                 act_hold = 1,
                 fixed_step = True,
                 int_accuracy = .01,
                 reward_fn = lambda ns, a: -(np.cos(ns[0]) + np.cos(ns[0] + ns[1])),
                 th1_range = [0, 2*pi],
                 th2_range = [-pi, pi],
                 max_th1dot = float('inf'),
                 max_th2dot = float('inf')
    ):
        
        """ 
        Args:
            max_torque: torque at which the controller saturates (N*m)
            init_state: initial state for the Acrobot. All zeros has the acrobot in it's stable equilibrium 
            init_state_weights: initial state is going to be init_state + np.random.random(4)*init_state_weights
            dt: dt used by the simulator
            act_hold: like a frame skip, how many dts to hold every action for
            fixed_step: whether to use the integrator in fixed step mode or with a dynamic step
            int_accuracy: if fixed_step == False, what accuracy to try and obtain with the integrator
            reward_fn: lambda that defines the reward function as a function of only the state variables 

        """
        self.init_state = init_state
        self.init_state_weights = init_state_weights
        self.dt = dt
        self.fixed_step = fixed_step
        self.max_t = max_t
        self.act_hold = act_hold
        self.int_accuracy = .01
        self.reward_fn = reward_fn

        self.max_th1dot = max_th1dot
        self.max_th2dot = max_th2dot

        # These are only used for rendering
        self.LINK_LENGTH_1 = 1.0
        self.LINK_LENGTH_2 = 1.0
        self.viewer = None

        low = np.array([th1_range[0], th2_range[0], -max_th1dot, -max_th2dot])
        high = np.array([th1_range[1], th2_range[1], max_th1dot, max_th2dot])


        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-max_torque]), high=np.array([max_torque]), dtype=np.float32)
        self.seed()

        builder = DiagramBuilder()

        tree = RigidBodyTree(getResourcePath() + "acrobot.urdf", FloatingBaseType.kFixed)
        acrobot = builder.AddSystem(RigidBodyPlant(tree))

        saturation = builder.AddSystem(Saturation(min_value=[-max_torque], max_value=[max_torque]))
        builder.Connect(saturation.get_output_port(0), acrobot.get_input_port(0))

        wrapangles = WrapToSystem(4)
        wrapangles.set_interval(0, th1_range[0], th1_range[1])
        wrapangles.set_interval(1, th2_range[0], th2_range[1])
        wrapto = builder.AddSystem(wrapangles)
        builder.Connect(acrobot.get_output_port(0), wrapto.get_input_port(0))

        controller = builder.AddSystem(StepController())
        builder.Connect(wrapto.get_output_port(0), controller.get_input_port(0))
        builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))

        state_logger = builder.AddSystem(SignalLogger(wrapto.get_output_port(0).size()))
        act_logger = builder.AddSystem(SignalLogger(saturation.get_output_port(0).size()))
        builder.Connect(wrapto.get_output_port(0), state_logger.get_input_port(0))
        builder.Connect(saturation.get_output_port(0), act_logger.get_input_port(0))

        diagram = builder.Build()

        simulator = Simulator(diagram)
        self.simulator = simulator        
        self.simulator.set_publish_every_time_step(False)
        self.simulator.get_integrator().set_fixed_step_mode(self.fixed_step)
        self.simulator.get_integrator().set_maximum_step_size(self.dt)
        self.simulator.get_integrator().set_target_accuracy(self.int_accuracy)

        context = simulator.get_mutable_context()

        self.t = 0

        self.context = context

        self.num_steps = int(self.max_t / (self.act_hold * self.dt))
        self.state_logger = state_logger
        self.act_logger = act_logger
        self.diagram = diagram

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_vec = None):
        self.simulator = Simulator(self.diagram)
        self.simulator.set_publish_every_time_step(False)
        self.simulator.get_integrator().set_fixed_step_mode(self.fixed_step)
        self.simulator.get_integrator().set_maximum_step_size(self.dt)
        self.simulator.get_integrator().set_target_accuracy(self.int_accuracy)
        
        self.context = self.simulator.get_mutable_context()

        init_state = self.InitialState().CopyToVector()

        init_state += np.random.random(4)*(self.init_state_weights*2) - self.init_state_weights

        if init_vec is not None:
            init_state[0] = init_vec[0]
            init_state[1] = init_vec[1]
            init_state[2] = init_vec[2]
            init_state[3] = init_vec[3]

        self.context.SetContinuousState(init_state)
        self.simulator.Initialize()
        self.t = 0
        return init_state

    def step(self, a):
        global g_action
        g_action = a
        
        self.t += self.dt * self.act_hold

        self.simulator.AdvanceTo(self.t)
        ns = self.state_logger.data()[:, -1]
        
        reward = self.reward_fn(ns,a)


        #I should probably do this with a wrapper...
        done = False
        if self.t > self.max_t:
            done = True

        if abs(ns[2]) > self.max_th1dot or abs(ns[2] > self.max_th2dot):
            reward -= 5
            done = True

        return (ns, reward, done, {})

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state_logger.data()[:, -1]

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]), p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
     
    def InitialState(self):
        state = AcrobotState()
        state.set_theta1(self.init_state[0])
        state.set_theta2(self.init_state[1])
        state.set_theta1dot(self.init_state[2])
        state.set_theta2dot(self.init_state[3])
        return state



# Just passes along whatever action was given to step
class StepController(VectorSystem):
    def __init__(self):
        VectorSystem.__init__(self, 4, 1)

    def DoCalcVectorOutput(self, context, state, _, output):
        output[:] = g_action
