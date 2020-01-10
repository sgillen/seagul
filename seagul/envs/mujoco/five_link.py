import numpy as np
from gym import utils
from gym.envs.registration import EnvSpec
from gym.envs.mujoco import mujoco_env
from seagul.resources import getResourcePath


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
        self.evaluate = False

        # print("Reached", id(self))
        mujoco_env.MujocoEnv.__init__(self, getResourcePath() + "/five_link.xml", 4)
        utils.EzPickle.__init__(self)
        # print("Can't reach", id(self))

        self.step_success = 0
        self.spec = EnvSpec("five_link-v3")  # TODO
        self.spec.max_episode_steps = 1000

        self.rbody_xpos = 0
        self.lbody_xpos = 0
        # print("can't reach v2.0", id(self))

    # This function for training
    def step(self, a):
        if self.evaluate:
            return self.step_eval(a)
        else:
            return self.step_train(a)

    def step_train(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        # print(self.sim.data.time)

        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        done = not (-0.4 < height < 0.4 and -1 < ang < 1)

        ob = self._get_obs()
        return ob, reward, done, {}

    # This function for evaluating
    def step_eval(self, a):
        # print("actions:", a)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        # print(self.sim.data.time)
        self.fall = 0
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        self.rbody_xpos = self.sim.data.body_xpos[3, 0]
        self.lbody_xpos = self.sim.data.body_xpos[5, 0]
        self.qpos_cur[0, :] = self.sim.data.qpos
        self.qvel_cur[0, :] = self.sim.data.qvel
        if self.t_imp < self.sim.data.time < self.t_imp + 0.02:
            # test_force = np.array([250, 0, 0, 0, 0, 0])
            self.sim.data.xfrc_applied[1, :] = self.impact
            self.set_impact += 1

        # if self.sim.data.time > self.t_imp + 0.1 and self.sim.data.time < self.t_imp + 0.2:
        if 1 < self.set_impact < 10:
            test_force = np.array([0, 0, 0, 0, 0, 0])
            self.sim.data.xfrc_applied[1, :] = test_force
            self.set_impact += 1

        # if self.sim.data.time > 4 and self.sim.data.time < 4 + 0.02:
        #     print("Applying force")
        #     test_force = np.array([800, 0, 0, 0, 0, 0])
        #     self.sim.data.xfrc_applied[1, :] = test_force
        #     self.set_impact += 1
        #
        # # if self.sim.data.time > self.t_imp + 0.1 and self.sim.data.time < self.t_imp + 0.2:
        # if self.sim.data.time > 4 + 0.02:
        #     print("clearing force")
        #     test_force = np.array([0, 0, 0, 0, 0, 0])
        #     self.sim.data.xfrc_applied[1, :] = test_force
        #     self.set_impact += 1

        done_contact = 0
        done = not (-0.4 < height < 0.4 and -2 < ang < 2)
        if done:
            self.fall = 1
        if self.sim.data.ncon and self.sim.data.time > self.detect_impact_time:
            # print("Contact Detected")
            for i in range(self.sim.data.ncon):
                # print(self.sim.data.contact[i].geom1, " ", self.sim.data.contact[i].geom2)
                if self.sim.data.contact[i].geom2 == 5:
                    # print(self.sim.data.time)
                    done_contact = 1
                    self.step_success = 1
                    break
                # if self.sim.data.contact[i].geom2 == 3:
                #     print("right leg: ", self.sim.data.time)
                #     # done_contact = 1
                #     # self.step_success = 1
                #     # break
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
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
