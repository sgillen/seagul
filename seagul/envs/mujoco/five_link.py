# Nihar Talele


import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env



class FiveLinkWalkerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.qpos_cur = np.zeros([1, 7])
        self.qvel_cur = np.zeros([1, 7])
        self.impact = np.array([0, 0, 0, 0, 0, 0])
        self.fall = 0
        self.t_imp = 0
        self.set_impact = 0
        self.detect_impact_time = 10
        self.mu1 = 0.5
        self.mu2 = 0.5
        self.lfoot = 0
        self.rfoot = 0
        # print("Reached", id(self))
        mujoco_env.MujocoEnv.__init__(self,  os.path.dirname(__file__) + '/five_link.xml', 4)
        utils.EzPickle.__init__(self)
        # print("Can't reach", id(self))

        self.step_success = 0

        self.rbody_xpos = 0
        self.lbody_xpos = 0
        # print("can't reach v2.0", id(self))

# This function for training
#     def step(self, a):
#         posbefore = self.sim.data.qpos[0]
#         self.do_simulation(a, self.frame_skip)
#         posafter, height, ang = self.sim.data.qpos[0:3]
#         alive_bonus = 1.0
#         reward = ((posafter - posbefore) / self.dt)
#         # print(self.sim.data.time)
#
#         reward += alive_bonus
#         reward -= 1e-3 * np.square(a).sum()
#
#         done = not (height > -0.4 and height < 0.4 and
#                     ang > -1 and ang < 1)
#
#         ob = self._get_obs()
#         # print("from 5linkwalerEnv", ob)
#         return ob, reward, done, {}

# This function for evaluating friction
#     def step(self, a):
#         posbefore = self.sim.data.qpos[0]
#         qposcur = np.zeros([7])
#         qposcur[:] = self.sim.data.qpos[:]
#         geom5 = self.sim.data.geom_xpos[5, 0]
#         ang_lfoot = qposcur[2] + qposcur[5] + qposcur[6]
#         lfootcur = geom5 - 0.21 * np.sin(ang_lfoot)
#         geom3 = self.sim.data.geom_xpos[3, 0]
#         ang_rfoot = qposcur[2] + qposcur[3] + qposcur[4]
#         rfootcur = geom3 - 0.21 * np.sin(ang_rfoot)
#
#         # print("lfootcur: ", lfootcur, "rfootcur", rfootcur, "mu1", self.mu1, "mu2", self.mu2)
#
#         self.sim.model.geom_friction[0, :] = np.array([0.1, 0.1, 0.1])
#
#         if lfootcur < self.lfoot:
#             self.sim.model.geom_friction[5, :] = np.array([self.mu1, self.mu1, self.mu1])
#         else:
#             self.sim.model.geom_friction[5, :] = np.array([self.mu2, self.mu2, self.mu2])
#
#         if rfootcur < self.lfoot:
#             self.sim.model.geom_friction[3, :] = np.array([self.mu1, self.mu1, self.mu1])
#         else:
#             self.sim.model.geom_friction[3, :] = np.array([self.mu2, self.mu2, self.mu2])
#
#         self.do_simulation(a, self.frame_skip)
#         posafter, height, ang = self.sim.data.qpos[0:3]
#         alive_bonus = 1.0
#         reward = ((posafter - posbefore) / self.dt)
#         # print(self.sim.data.time)
#         self.fall = 0
#         reward += alive_bonus
#         reward -= 1e-3 * np.square(a).sum()
#         self.rbody_xpos = self.sim.data.body_xpos[3, 0]
#         self.lbody_xpos = self.sim.data.body_xpos[5, 0]
#         self.qpos_cur[0, :] = self.sim.data.qpos
#         self.qvel_cur[0, :] = self.sim.data.qvel
#
#
#         done_contact = 0
#         done = not (height > -0.4 and height < 0.4 and
#                     ang > -2 and ang < 2)
#         if done:
#             self.fall = 1
#         if self.sim.data.ncon and self.sim.data.time > self.detect_impact_time:
#             # print("Contact Detected")
#             for i in range(self.sim.data.ncon):
#                 # print(self.sim.data.contact[i].geom1, " ", self.sim.data.contact[i].geom2)
#                 if self.sim.data.contact[i].geom2 == 5:
#                     done_contact = 1
#                     self.step_success = 1
#                     break
#         done = np.array([done, done_contact])
#         done = done.any()
#         ob = self._get_obs()
#         # print("from 5linkwalerEnv", ob)
#         return ob, reward, done, {}


# # This function for evaluating
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        # print(self.sim.data.time)
        self.fall = 0
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        self.rbody_xpos = self.sim.data.body_xpos[3, 0]
        self.lbody_xpos = self.sim.data.body_xpos[5, 0]
        self.qpos_cur[0, :] = self.sim.data.qpos
        self.qvel_cur[0, :] = self.sim.data.qvel
        if self.sim.data.time > self.t_imp and self.sim.data.time < self.t_imp + 0.02:
            # test_force = np.array([250, 0, 0, 0, 0, 0])
            self.sim.data.xfrc_applied[1, :] = self.impact
            self.set_impact += 1

        # if self.sim.data.time > self.t_imp + 0.1 and self.sim.data.time < self.t_imp + 0.2:
        if self.set_impact > 1 and self.set_impact < 10:
            test_force = np.array([0, 0, 0, 0, 0, 0])
            self.sim.data.xfrc_applied[1, :] = test_force
            self.set_impact += 1
        done_contact = 0
        done = not (height > -0.4 and height < 0.4 and
                    ang > -2 and ang < 2)
        if done:
            self.fall = 1
        if self.sim.data.ncon and self.sim.data.time > self.detect_impact_time:
            # print("Contact Detected")
            for i in range(self.sim.data.ncon):
                # print(self.sim.data.contact[i].geom1, " ", self.sim.data.contact[i].geom2)
                if self.sim.data.contact[i].geom2 == 5:
                    done_contact = 1
                    self.step_success = 1
                    break
        done = np.array([done, done_contact])
        done = done.any()
        ob = self._get_obs()
        # print("from 5linkwalerEnv", ob)
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        # self.set_state(
        #     self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
        #     self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # )
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
