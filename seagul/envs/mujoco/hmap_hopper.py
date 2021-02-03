from gym.envs.mujoco.hopper import HopperEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py as mj
from seagul.resources import getResourcePath


class HmapHopperEnv(HopperEnv):
    def __init__(self, slope):
        mujoco_env.MujocoEnv.__init__(self, getResourcePath() + "/hmap_hopper.xml", 4)
        utils.EzPickle.__init__(self)

        if slope == 0:
            self.model.hfield_data[:] = .5
        else:

            ramp_length = int(.5//abs(slope))

            ncol = 1000
            cur_x = int(81*(1000/400))
            cur_height = .5
            self.model.hfield_data[:] = cur_height
            
            for step in range(ramp_length):
                cur_height = np.clip(cur_height + slope, 0,1)
                self.model.hfield_data[cur_x] = cur_height
                self.model.hfield_data[ncol+cur_x] = cur_height
                cur_x +=1
                

                
            self.model.hfield_data[cur_x:ncol] = cur_height
            self.model.hfield_data[ncol+cur_x:] = cur_height

        # n_steps = 50
        # step_length = 500//n_steps
        # step_incr = 1/n_steps
        # ncol = 1000
        # cur_x = 120
        # cur_height = 0
        # self.model.hfield_data[:] = 0

        # for step in range(n_steps-1):
        #     cur_height += step_incr
        #     cur_x += step_length 
        #     self.model.hfield_data[cur_x:cur_x+step_length] = cur_height
        #     self.model.hfield_data[ncol+cur_x:ncol+cur_x+step_length] = cur_height
            
        # self.model.hfield_data[cur_x+step_length:ncol] = 1
        # self.model.hfield_data[ncol+cur_x+step_length:] = 1

    def step(self, a):
        ob, reward, done, _ = super().step(a)
        s = self.state_vector()
        posafter, height, ang = self.sim.data.qpos[0:3]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (abs(ang) < .2))
        return ob, reward, done, _
        
    def viewer_setup(self):
        HopperEnv.viewer_setup(self)

        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 8.0
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = 0

        mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)
