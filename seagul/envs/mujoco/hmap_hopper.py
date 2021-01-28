from gym.envs.mujoco.hopper import HopperEnv
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py as mj
from seagul.resources import getResourcePath


class HmapHopperEnv(HopperEnv):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, getResourcePath() + "/hmap_hopper.xml", 4)
        utils.EzPickle.__init__(self)


    def viewer_setup(self):
        HopperEnv.viewer_setup(self)

        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 8.0
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = 0

        mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)
