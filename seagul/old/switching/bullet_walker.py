import pybullet as p
from seagul.resources import getSgResourcePath
import pybullet_data
from numpy import pi
import math
import os
import time

GRAVITY = -9.8
dt = 1e-3
iters = 2000

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
# p.setRealTimeSimulation(True)
p.setGravity(0, 0, GRAVITY)
p.setTimeStep(dt)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1.13]
cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0, 0])
p.setAdditionalSearchPath(getSgResourcePath())
botId = p.loadURDF("five_link.urdf", cubeStartPos, cubeStartOrientation)


init_pos = [-0.86647779, -5.57969548, 4.56618282, -0.86647779]
init_vel = [-0.08985754, 2.59193943, -0.48066481, 1.88797459]

for i in range(len(init_pos)):
    p.resetJointState(botId, i + 1, init_pos[i], init_vel[i])

p.resetBasePositionAndOrientation(botId, [0, 0, 0], p.getQuaternionFromEuler([0, pi, 0]))


# disable the default velocity motors
# and set some position control with small force to emulate joint friction/return to a rest pose
# jointFrictionForce = 1
# for joint in range(p.getNumJoints(botId)):
#  p.setJointMotorControl2(botId, joint, p.POSITION_CONTROL, force=jointFrictionForce)

# for i in range(10000):
#     p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
#     p.stepSimulation()
# import ipdb
# ipdb.set_trace()

import time

while 1:

    p.stepSimulation()
    # p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
    p.setGravity(0, 0, GRAVITY)
    time.sleep(1 / 240.0)


# time.sleep(1000)
