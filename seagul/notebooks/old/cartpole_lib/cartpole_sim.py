import mujoco_py as mj
import numpy as np
from numpy import pi
import copy
import matplotlib.pyplot as plt

class CartPoleSim:
    """
    Encapsulates everything you need for simulation and simple state feedback of a cartpole in Mujoco

    See cartpole_naive_swingup.ipynb for usage
    """

    # TODO make this a decorator?

    __isfrozen = False  # freeze all the classes so I can't accidentally add new variables when i typo

    # This is obviously a hack
    abs_dir = '/Users/sgillen/work_dir/sgillen_notebooks/ICRA2019/cartpole/'

    def __init__(self, xml_file = (abs_dir + 'cartpole.xml')):
        model_xml = open(xml_file).read()
        model = mj.load_model_from_xml(model_xml)  # sim has it's own model you can access, sim.model

        self.sim = mj.MjSim(model)

        # don't make a viewer right away since it spawns a GLFW window and is expensive
        # instead the viewer will be initialized the first time you call visualize
        self.viewer = None

        self.num_steps = 400  # modify this yourself, if you want to


        # TODO be really careful with your state, some of this stuff should probably be CLASS vars
        self.default_state = copy.deepcopy(self.sim.get_state())
        self.default_state[1][1] = 0  # initial angle up

        # TODO probably move these somewhere else? not sure
        self.Kp = np.array([.1, 1])
        self.Kd = np.array([.2, .4])
        self.Ki = np.array([.1, 1])
        self.i_clip = 3
        self.total_gain = 10
        self._error_sum = 0

        self.swingup_gain = 2.5
        self.catch_range = .2

        self.q_pos_hist = np.zeros((self.num_steps, self.default_state[1].size))
        self.q_vel_hist = np.zeros((self.num_steps, self.default_state[2].size))
        self.u_val_hist = np.zeros((self.num_steps, self.sim.data.ctrl.size))
        self.c_hist = []
        self.stable_q_hist = []

        self.u = self.swingup_gain
        self.swingup_travel = .1

        self.total_clip = 10

    def controller(self, q_pos, q_vel):
        """
        Controller to be called to calculate u at every time step, you can override this if you want
        """

        # stable q_pos is the angle used by the stabilizing controller
        stable_q_pos = copy.copy(q_pos)
        stable_q_pos[1] % (2 * pi)  # make sure we are between 0 and 2*pi
        if stable_q_pos[1] > pi:
            stable_q_pos[1] = (stable_q_pos[1] - (2 * pi))


        self.stable_q_hist.append(stable_q_pos[1])
        # figure out what controller to use and apply it
        if np.abs(stable_q_pos[1]) > self.catch_range:
            # swing up control
            self.c_hist.append(1)
            #self.u =  self.swingup_gain*q_vel[1]*np.cos(q_pos[1])
            #self.u = self.swingup_gain
            #if(q_vel[1] <= 0):
                #self.u = -self.swingup_gain
            #else:
                #self.u = self.swingup_gain


            if(q_pos[0] < self.swingup_travel):
                self.u = self.swingup_gain
            elif(q_pos[0] > -self.swingup_travel ):
                self.u = -self.swingup_gain

        else:
            self.c_hist.append(-1)
            # stable control
            self._error_sum = np.clip(self._error_sum + q_pos, -self.i_clip, self.i_clip)
            self.u = self.total_gain * (np.sum(stable_q_pos * self.Kp) + np.sum(q_vel * self.Kd) + np.sum(self._error_sum * self.Ki))

        self.u = np.clip(self.u, -self.total_clip, self.total_clip)
        return self.u

    def run_sim(self):
        """
        Run the simulation exactly once, you have to call visualize() to see what the run did
        """

        # reset the sim
        self.sim.reset()
        self.sim.set_state(self.default_state)

        # reset the history variables
        self.q_pos_hist = np.zeros((self.num_steps, self.default_state[1].size))
        self.q_vel_hist = np.zeros((self.num_steps, self.default_state[2].size))
        self.u_val_hist = np.zeros((self.num_steps, self.sim.data.ctrl.size))

        for i in range(self.num_steps):

            # get newest state variables
            q_pos = self.sim.get_state()[1]
            q_vel = self.sim.get_state()[2]

            self.sim.data.ctrl[:] = self.controller(q_pos, q_vel)

            # store the data no matter what
            self.q_pos_hist[i, :] = q_pos
            self.q_vel_hist[i, :] = q_vel
            self.u_val_hist[i, :] = self.sim.data.ctrl[:]

            self.sim.step()

    def visualize(self):
        """
        uses mujoco viewer to render the results of the most recent simulation
        """

        # TODO find out how to close the window when we're done?
        if self.viewer is None:
            self.viewer = mj.MjViewer(self.sim)

        self.sim.reset()
        self.sim.set_state(self.default_state)

        for u in zip(self.u_val_hist):
            self.sim.data.ctrl[:] = u
            self.sim.step()
            self.viewer.render()

    def plot_history(self):
        """
        plots position, velocity, control, and the active controller

        """
        plt.plot(self.u_val_hist)
        plt.title('u_vals')
        plt.show()
        plt.figure()

        plt.plot(self.q_pos_hist)
        plt.title('positions')
        plt.legend(('cart x', 'theta'))
        plt.figure()

        plt.plot(self.q_vel_hist)
        plt.title('velocities')
        plt.legend(('cart x', 'theta'))
        plt.figure()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


if __name__ == "__main__":

    # TODO add tests (in main? not in main? read up on "best practices"
    cart = CartPoleSim()
    cart.run_sim()

    cart.plot_history()

    cart.default_state[1][1] = pi + .1
    cart.run_sim()

    cart.plot_history()


    #cart.visualize()
