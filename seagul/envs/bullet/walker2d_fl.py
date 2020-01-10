from pybullet_envs.robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
from pybullet_envs.robot_locomotors import WalkerBase
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from seagul.resources import getResourcePath
from numpy import pi


class Walker2DFiveLinkBase(WalkerBase):
    foot_list = ["leg", "leg_left"]

    def __init__(self):
        WalkerBase.__init__(
            self, getResourcePath() + "/Walker2d_five.xml", "torso", action_dim=4, obs_dim=18, power=0.40
        )
        self.init_pos = [-0.86647779, -5.57969548, 4.56618282, -0.86647779]
        self.init_vel = [-0.08985754, 2.59193943, -0.48066481, 1.88797459]

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)

        self.jdict["leg_joint"].set_position(self.init_pos[0] * 180 / pi)
        self.jdict["thigh_joint"].set_position(self.init_pos[2] * 180 / pi)
        self.jdict["leg_left_joint"].set_position(-self.init_pos[1] * 180 / pi)
        self.jdict["thigh_left_joint"].set_position(self.init_pos[3] * 180 / pi)

        # self.jdict['leg_joint'].set_position(-50)
        # self.jdict['leg_left_joint'].set_position(50)
        # self.jdict['thigh_joint'].set_position(50)
        # self.jdict['thigh_left_joint'].set_position(-50)

        # for n in ["foot_joint", "foot_left_joint"]:
        #    self.jdict[n].power_coef = 30.0


class Walker2DFiveLink(WalkerBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = Walker2DFiveLinkBase()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
