from gym.envs.mujoco.hopper import HopperEnv
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import mujoco_py as mj
import math
from seagul.resources import getResourcePath
import random


class HurdleHopperEnv(HopperEnv):
    def __init__(self, gap_length, hurdle_height=.52, gap_set=None):
        self.start_x = 84
        self.neutral_hfield_val = .5
        self.h_length = 4
        self.hurdle_height = hurdle_height

        mujoco_env.MujocoEnv.__init__(self, getResourcePath() + "/hurdle_hopper.xml", 4)
        utils.EzPickle.__init__(self)

        #if slope_set is None:
        self.ncol = self.model.hfield_ncol.item()
        self.model.hfield_data[:] = self.neutral_hfield_val

        self.ncol = self.model.hfield_ncol.item()
        self.gap_length = gap_length
        self.gap_set = gap_set

        self.start_idx = int(self.start_x * (self.ncol / 400))
        self._update_num_hurdles()

    def _update_num_hurdles(self):
        if self.gap_set:
            self.n_hurdles = (self.ncol - self.start_idx) // (max(self.gap_set) + self.h_length)
        else:
            self.n_hurdles = (self.ncol - self.start_idx) // (self.gap_length + self.h_length)

    def reset(self):
        obs = super().reset()
        self._update_num_hurdles()

        self.model.hfield_data[:] = self.neutral_hfield_val

        if self.gap_set:
            self.n_hurdles = (self.ncol - self.start_idx) // (max(self.gap_set) + self.h_length)
        else:
            self.n_hurdles = (self.ncol - self.start_idx) // (self.gap_length + self.h_length)
        offset = 0

        for h in range(self.n_hurdles):
            _from = self.start_idx + offset
            _to = self.start_idx + offset + self.h_length
            self.model.hfield_data[_from:_to] = self.hurdle_height
            self.model.hfield_data[(self.ncol + _from):(self.ncol + _to)] = self.hurdle_height

            if self.gap_set:
                gap = random.choice(self.gap_set)
                offset += (gap + self.h_length)
            else:
                offset += (self.gap_length + self.h_length)
            #print(offset)

        if self.viewer:
            mj.functions.mjr_uploadHField(self.model, self.sim.render_contexts[0].con, 0)

        return obs

    def get_height(self, offset=0):
        pos = self.sim.data.qpos[0] + offset
        max_pos = self.model.hfield_size[0,0]*2

        max_height = self.model.hfield_size[0,2]

        index = (self.start_x + pos)/max_pos * self.ncol

        li = math.floor(index)
        ui = math.ceil(index)
        a = index - int(index)

        return ((1 - a)*self.model.hfield_data[li] + a*self.model.hfield_data[ui])*max_height

    def _get_obs(self):
        self.ncol = self.model.hfield_ncol.item()
        pos = np.copy(self.sim.data.qpos.flat[1:])
        pos[0] -= (self.get_height(0) - self.model.hfield_size[0,2]/2)
        vel = self.sim.data.qvel.flat

        next_hurdle_x = 0
        if self.hurdle_height != self.neutral_hfield_val:
            cur_x = self.sim.data.qpos[0]
            max_x = self.model.hfield_size[0, 0] * 2
            cur_idx = int((self.start_x + cur_x) / max_x * self.ncol)
            hurdle_idx = np.where(self.model.hfield_data == self.hurdle_height)[0]

            if hurdle_idx.size == 0:
                next_hurdle_x = 0
            else:
                next_hurdle_idx = hurdle_idx[np.where(hurdle_idx > cur_idx)[0][0]]
                next_hurdle_x = (next_hurdle_idx - cur_idx) / self.ncol * max_x

        return np.concatenate([pos, vel, np.array([next_hurdle_x])])

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


        # print(done)
        if done:
            reward -= 500
        #     #print(reward)

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
                

#
# if __name__ == "__main__":
#     import torch.nn as nn
#     import numpy as np
#     import gym
#     from seagul.rl.ars.meta_ars import MetaARSAgent
#     import matplotlib.pyplot as plt
#     import torch
#     import dill
#     import seagul.envs
#     import pybullet_envs
#     import pickle
#     from scipy.signal import find_peaks
#
#     meta_agent = pickle.load(open("agents/meta_hmap_hopper_flat4", "rb"))
#     agent0 = meta_agent.agents[4]
#
#     env_name = "hmap_hopper-v0"
#     seed = 4
#     env = gym.make(env_name, slope_set=[0], random=True)
#     # env = gym.make("Hopper-v2")
#
#     # agent = meta_agent.agents[seed]
#     obs, act, rew, _ = do_rollout(env, agent0.model, render=True, ep_length=1000)
#     print(sum(rew))
#     plt.plot(obs);