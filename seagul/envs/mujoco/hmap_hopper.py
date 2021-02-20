from gym.envs.mujoco.hopper import HopperEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py as mj
import math
from seagul.resources import getResourcePath
import random


class HmapHopperEnv(HopperEnv):
    def __init__(self, slope_set=None, random=True):
        mujoco_env.MujocoEnv.__init__(self, getResourcePath() + "/hmap_hopper.xml", 4)
        utils.EzPickle.__init__(self)

        self.cur_hfield_val = .5
        self.model.hfield_data[:] = self.cur_hfield_val
        self.ramp_length = 5
        
        self.ncol = self.model.hfield_ncol.item()
        self.init_x = 81
        self.cur_idx = int(self.init_x * (self.ncol / 400))
        self.course_length = self.ncol - self.cur_idx
        self.random = random

        if slope_set is None:
            self.slope_set = [0.0]
        else:
            self.slope_set = slope_set

    def reset(self):
        obs = super().reset()

        self.cur_idx = int(self.init_x * (self.ncol / 400))
        self.cur_hfield_val = .5
        self.model.hfield_data[:] = self.cur_hfield_val

        if self.random:
            for _ in range(self.course_length//self.ramp_length):
                slope = random.choice(self.slope_set)
                self.make_slope(slope, ramp_length=self.ramp_length)
        else:
            slope = random.choice(self.slope_set)

            try:
                self.make_slope(slope)
            except:
                print("Error making slope set")
                print(self.slope_set)
                print()
                print(slope)
                
        if self.viewer:
            mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)

        return obs

    def make_slope(self, slope, ramp_length=None):
        ncol = self.model.hfield_ncol.item()
        if ramp_length is None and slope != 0:
            ramp_length = int(self.cur_hfield_val//abs(slope))
        elif ramp_length is None and slope == 0:
            ramp_length = ncol - self.cur_idx

        for step in range(ramp_length):
            self.cur_hfield_val = np.clip(self.cur_hfield_val + slope, 0,1)
            self.model.hfield_data[self.cur_idx] = self.cur_hfield_val
            self.model.hfield_data[ncol+self.cur_idx] = self.cur_hfield_val
            self.cur_idx += 1
            
        self.model.hfield_data[self.cur_idx:ncol] = self.cur_hfield_val
        self.model.hfield_data[ncol+self.cur_idx:] = self.cur_hfield_val
        
        if self.viewer:
            mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)

    def get_height(self, offset=0):
        pos = self.sim.data.qpos[0] + offset
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

    def _get_obs(self):
        pos = np.copy(self.sim.data.qpos.flat[1:])
        pos[0] -= (self.get_height(0) - self.model.hfield_size[0,2]/2)
        vel = self.sim.data.qvel.flat

        # rel_height1 =  self.get_height(1) - self.get_height(0)
        # rel_height2 =  self.get_height(2) - self.get_height(0)
        # measured_slope = (rel_height2 - rel_height1)/(1000/400)
        measured_slope = (self.get_height(1) - self.get_height(0))/(1000/400)
        return np.concatenate([pos, vel, np.array([measured_slope])])

    def step(self, a):
        ob, reward, done, _ = super().step(a)
        reward -= .9
        s = self.state_vector()
        height, ang = self._get_obs()[:2]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .4))

        # if done:
        #     print((np.isfinite(s).all(), (np.abs(s[2:]) < 100).all(),
        #             (height > .7), (abs(ang) < .4)))


        # set done = true if anything but the foot and ground are in contact.
        ncon = self.unwrapped.sim.data.ncon
        for i in range(ncon):
            geom1 = self.unwrapped.sim.data.contact[i].geom1
            geom2 = self.unwrapped.sim.data.contact[i].geom2
            if not (geom1 == 4 or geom1 == 3 or geom1 == 0):
                done = True
                #print(f"contact with body {geom1},{geom2}")
            if not (geom2 == 4 or geom2 == 3 or geom2 == 0):
                done = True
                #print(f"contact with body {geom1},{geom2}")

        if done:
            reward -= 1000
        # if done:
        #     print(np.isfinite(s).all())
        #     print((np.abs(s[2:]) < 100))
        #     print()

        return ob, reward, done, _
        
    def viewer_setup(self):
        HopperEnv.viewer_setup(self)

        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 8.0
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = 0

        mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)
                
