from gym.envs.mujoco.hopper import HopperEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py as mj
import math
from seagul.resources import getResourcePath


class HmapHopperEnv(HopperEnv):
    def __init__(self, slope=0, random=False):
        mujoco_env.MujocoEnv.__init__(self, getResourcePath() + "/hmap_hopper.xml", 4)
        utils.EzPickle.__init__(self)
        self.cur_x = int(81*(1000/400))
        self.cur_height = .5
        self.model.hfield_data[:] = self.cur_height

        if random:
            while True:
                try:
                    slope = (np.random.random() - .5)*.02
                    self.make_slope(slope,ramp_length=50)
                except:
                    break # lol

        else:
            self.make_slope(slope)

                
    def make_slope(self, slope, ramp_length=None):
        if slope == 0:
            self.model.hfield_data[:] = self.cur_height
        else:
            if ramp_length is None:
                ramp_length = int(self.cur_height//abs(slope))
                
            ncol = 1000
            
            for step in range(ramp_length):
                self.cur_height = np.clip(self.cur_height + slope, 0,1)
                self.model.hfield_data[self.cur_x] = self.cur_height
                self.model.hfield_data[ncol+self.cur_x] = self.cur_height
                self.cur_x +=1

            self.model.hfield_data[self.cur_x:ncol] = self.cur_height
            self.model.hfield_data[ncol+self.cur_x:] = self.cur_height

        if self.viewer:
            mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)


    def get_height(self, offset=0):
        pos = self.sim.data.qpos[1] + offset
        max_pos = self.model.hfield_size[0,0]*2
        nrow = self.model.hfield_nrow
        ncol = self.model.hfield_ncol
        
        max_height = self.model.hfield_size[0,2]
        init_pos = 80
        
        index = (init_pos + pos)/max_pos * ncol[0]

        li = math.floor(index)
        ui = math.ceil(index)
        a = index - int(index)

        return ((1 - a)*self.model.hfield_data[li] + a*self.model.hfield_data[ui])*max_height
    


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



    def _get_obs(self):
        
        pos = self.sim.data.qpos.flat[1:]
        pos[0] -= self.get_height(0) - 5
        vel = self.sim.data.qvel.flat

        return np.concatenate([pos, vel, np.array([self.get_height(1), self.get_height(2)])])

        
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
                
