"""classic Acrobot task but in drake"""
import numpy as np
from numpy import sin, cos, pi

from gym import core, spaces
from gym.utils import seeding

import numpy as np
from numpy import pi
import math

from underactuated import (FindResource, PlanarRigidBodyVisualizer,
                           SliderSystem)

from pydrake.all import (DiagramBuilder, FloatingBaseType,
                         LinearQuadraticRegulator, RigidBodyTree,
                         Saturation, Simulator, WrapToSystem, VectorSystem, SignalLogger, RigidBodyPlant,
                         RigidBodyTree)

from pydrake.examples.acrobot import (AcrobotInput, AcrobotPlant, AcrobotState)

import matplotlib.pyplot as plt
from IPython.display import HTML


g_action = 0

# Just passes along whatever action was given to step
class StepController(VectorSystem):
    def __init__(self):
        VectorSystem.__init__(self,4,1)
    def DoCalcVectorOutput(self, context, state, _ , output):
        output[:] = g_action
                                 


def UprightState():
    state = AcrobotState()
    state.set_theta1(math.pi)
    state.set_theta2(0.)
    state.set_theta1dot(0.)
    state.set_theta2dot(0.)
    return state

        
class DrakeAcroEnv(core.Env):
    def __init__(self):

#        import ipdb; ipdb.set_trace()
        
        high = np.array([2*pi, pi, 100, 100])
        low = np.array([0, -pi, -100, -100])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-10]), high=np.array([10]), dtype=np.float32)
        self.seed()

        
        builder = DiagramBuilder()
        
        tree = RigidBodyTree(FindResource("acrobot/acrobot.urdf"),
                             FloatingBaseType.kFixed)

        acrobot = builder.AddSystem(RigidBodyPlant(tree))

        saturation = builder.AddSystem(Saturation(min_value=[-10],max_value=[10]))
        builder.Connect(saturation.get_output_port(0), acrobot.get_input_port(0))
        
        wrapangles = WrapToSystem(4)
        wrapangles.set_interval(0, 0, 2.*math.pi)
        wrapangles.set_interval(1, -math.pi, math.pi)
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
        #simulator.set_target_realtime_rate(1.0)
        simulator.set_publish_every_time_step(True)
        simulator.get_integrator().set_fixed_step_mode(True)
        
        context = simulator.get_mutable_context()


        
        self.dt = 0.005
        self.t = 0
        self.simulator = simulator
        self.context = context
        self.max_t = 5
        self.num_steps = int(self.max_t / self.dt)
        self.state_logger = state_logger
        self.act_logger = act_logger
        self.diagram = diagram


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()

        init_state = UprightState().CopyToVector() + 0.1*np.random.randn(4,)
        self.context.SetContinuousState(init_state)

        self.simulator.Initialize()
        self.t = 0
        self.state_logger.reset()
        self.act_logger.reset()
        return init_state



    def step(self, a):
        global g_action
        g_action = a

        self.t += self.dt
        self.simulator.AdvanceTo(self.t)
        ns = self.state_logger.data()[:,-1]
        reward =  -(np.cos(ns[0]) + np.cos(ns[0] + ns[1]))

        done = False
        if self.t > self.max_t:
            done = True
            
        
        return (ns, reward, done, {})

    def render(self):
        pass
