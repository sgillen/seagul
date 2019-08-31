"""
sgillen: This is racecarGymEnv enviroment with our own modifications. Mostly made by Guillaume
"""


import os

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from seagul.envs.bullet import rc_car
import random
from pybullet_envs.bullet import bullet_client
import pybullet_data
from pkg_resources import parse_version

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    # print(vector)
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """

    vn = unit_vector(np.array([0, 0, 1]))
    c = np.cross(np.hstack([v1, 0]), np.hstack([v2, 0]))

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    return angle * np.sign(np.dot(vn, c))


def rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class RacecarGymEnv_v1(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    # default repeat only 5 times - 0.05 like in mujoco
    def __init__(
        self,
        urdfRoot=pybullet_data.getDataPath(),
        actionRepeat=5,
        isEnableSelfCollision=True,
        isDiscrete=False,
        renders=False,
    ):  # just change to True if you want to visualize
        print("init")
        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._ballUniqueId = -1
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        if self._renders:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.seed()
        # self.reset()
        observationDim = 7  # len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)
        # observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        observation_high = np.ones(observationDim) * 1000  # np.inf
        if isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)
        self.viewer = None
        self.goal = []
        self.useSphere = False
        self.dist_to_goal_threshold = 0.2  # threshold to consider we "got to goal"

        self._hard_reset = True
        self.reset()
        # self._hard_reset = False

    def save_current_state(self):
        self.savedStateID = self._p.saveState()
        return self.savedStateID

    def restore_state(self, stateID):
        # can we do the above?
        # print(stateID)
        self._p.restoreState(stateID)

    def set_new_goal(self):
        dist = 2 + 1.0 * random.random()
        ang = 2.0 * 3.1415925438 * random.random()

        ballx = dist * math.sin(ang)
        bally = dist * math.cos(ang)
        ballz = 0.1

        # ballx = -5; bally = 2

        self.goal = [ballx, bally, ballz]

        return ballx, bally, ballz

    def reset(self):
        if self._hard_reset:
            self._p.resetSimulation()
            # p.setPhysicsEngineParameter(numSolverIterations=300)
            self._p.setTimeStep(self._timeStep)
            self._p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"))

            self._p.setGravity(0, 0, -10)
            self._racecar = rc_car.RcCar(self._p, timeStep=self._timeStep)

            for i in range(100):
                self._p.stepSimulation()

            self.starting_carpos, self.starting_carorn = self._p.getBasePositionAndOrientation(
                self._racecar.racecarUniqueId
            )

        else:
            # just reset car body to 0
            self._p.resetBasePositionAndOrientation(
                self._racecar.racecarUniqueId, self.starting_carpos, self.starting_carorn
            )
            self._p.resetBaseVelocity(self._racecar.racecarUniqueId, [0, 0, 0], [0, 0, 0])

        self._envStepCounter = 0
        ballx, bally, ballz = self.set_new_goal()

        if self.useSphere:
            self._ballUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot, "sphere2.urdf"), [ballx, bally, ballz])

        if self.useSphere:
            self._observation = self.getExtendedObservation()
        else:
            self._observation = self.getExtendedObservationGoal()
        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = []  # self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        ballpos, ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        ballPosInCar, ballOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, ballpos, ballorn)

        self._observation.extend([ballPosInCar[0], ballPosInCar[1]])
        return self._observation

    def getExtendedObservationGoal(self):
        self._observation = []

        self._observation.extend(self.get_real_obs())

        return self._observation

    def get_real_obs(self):
        # linkWorldPosition and worldLinkFramePosition are same for dummy link
        # same with linkWorldOrientation and worldLinkFrameOrientation
        linkWorldPosition, linkWorldOrientation, localInertialFramePosition, localInertialFrameOrientation, worldLinkFramePosition, worldLinkFrameOrientation, worldLinkLinearVelocity, worldLinkAngularVelocity = self._p.getLinkState(
            self._racecar.racecarUniqueId, 20, computeLinkVelocity=True
        )
        # print(p.getLinkState(car, 20, computeLinkVelocity=True))
        pos = np.array(worldLinkFramePosition)
        orn = worldLinkFrameOrientation
        linVel = worldLinkLinearVelocity
        angVel = worldLinkAngularVelocity
        roll, pitch, yaw = self._p.getEulerFromQuaternion(orn)
        goal_vec = np.array(self.goal[0:2])

        # distance between current pos and goal
        distCurrPos2Goal = np.linalg.norm(pos[0:2] - goal_vec)

        # angle to goal (from current heading)
        body_dir_vec = np.matmul(rot_mat(yaw), np.array([[1], [0]]))
        body_goal_vec = goal_vec - pos[0:2]
        body_dir_vec = body_dir_vec.reshape(2)
        body_goal_vec = body_goal_vec.reshape(2)

        ang2Goal = angle_between(body_dir_vec, body_goal_vec)

        # include information on thf, velocities in x and y directions ( linVel[0:2] )
        steering_jointStates = self._p.getJointStates(self._racecar.racecarUniqueId, self._racecar.steeringLinks)
        steering_jointPositions = np.array([x[0] for x in steering_jointStates])
        steering_jointVelocities = np.array([x[1] for x in steering_jointStates])

        thf = np.mean(steering_jointPositions)
        dthf = np.mean(steering_jointVelocities)

        dxb = linVel[0]
        dyb = linVel[1]
        dthb = angVel[2]

        # print(distCurrPos2Goal)

        return [distCurrPos2Goal, ang2Goal, thf, dxb, dyb, dthb, dthf]

    def dist2Goal(self):
        _, _, _, _, base_pos, orn = self._p.getLinkState(self._racecar.racecarUniqueId, 20)
        base_pos = np.array(base_pos[0:2])
        goal = np.array(self.goal[0:2])
        dist_to_goal = np.linalg.norm(base_pos - goal)
        return dist_to_goal

    def step(self, action):
        prev_dist_to_goal = self.dist2Goal()

        if self._renders:
            basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
            # self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)
            self._p.addUserDebugText(text="GOAL", textPosition=self.goal, textSize=2, lifeTime=1)

        if self._isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            forward = fwd[action]
            steer = steerings[action]
            realaction = [forward, steer]
        else:
            realaction = action

        realaction = np.clip(realaction, self.action_space.low, self.action_space.high)

        self._racecar.applyAction(realaction)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self.useSphere:
                self._observation = self.getExtendedObservation()
                if self._termination():
                    break
            else:
                self._observation = self.getExtendedObservationGoal()
                after_dist_to_goal = self.dist2Goal()
                if self._termination() or (after_dist_to_goal < self.dist_to_goal_threshold):
                    break

            self._envStepCounter += 1

        _, _, _, _, base_pos, orn = self._p.getLinkState(self._racecar.racecarUniqueId, 20)

        if self.useSphere:
            reward = self._reward()
            done = self._termination()
        else:
            reward = (prev_dist_to_goal - after_dist_to_goal) / (self._actionRepeat * self._timeStep)
            done = self._termination() or (after_dist_to_goal < self.dist_to_goal_threshold)

            # if done:
            #    print("counter", self._envStepCounter, "goal", self.goal, "end_pos", base_pos)

        return np.array(self._observation), reward, done, {}

    def render(self, mode="human", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        return self._envStepCounter > 1000

    def _reward(self):
        closestPoints = self._p.getClosestPoints(self._racecar.racecarUniqueId, self._ballUniqueId, 10000)

        numPt = len(closestPoints)
        reward = -1000
        # print(numPt)
        if numPt > 0:
            # print("reward:")
            reward = -closestPoints[0][8]
            # print(reward)
        return reward

    if parse_version(gym.__version__) < parse_version("0.9.6"):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
