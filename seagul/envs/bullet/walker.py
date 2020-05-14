import gym
import numpy as np
import pybullet
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot


class SGWalkerBaseBulletEnv(MJCFBaseBulletEnv):
    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints


    def __init__(self, robot, render=False):
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        MJCFBaseBulletEnv.__init__(self, robot, render)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.stadium_scene

    def reset(self):
        if (self.stateId >= 0):
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        if (self.stateId < 0):
          self.stateId = self._p.saveState()

        return r

    def _isDone(self):
        return self._alive < 0

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
          self.robot.apply_action(a)
          self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
          print("~INF~", state)
          done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        # TODO: Maybe calculating feet contacts could be done within the robot code
        feet_collision_cost = 0.0
        for i, f in enumerate(self.robot.feet):
          contact_ids = set((x[2], x[4]) for x in f.contact_list())
          #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
          if (self.ground_ids & contact_ids):
            #see Issue 63: https://github.com/openai/roboschool/issues/63
            #feet_collision_cost += self.foot_collision_cost
            self.robot.feet_contact[i] = 1.0
          else:
            self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost = self.stall_torque_cost * float(np.square(a).mean())

        #electricity_cost = -1e-3 * float(np.square(a).sum())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
          print("alive=")
          print(self._alive)
          print("progress")
          print(progress)
          print("electricity_cost")
          print(electricity_cost)
          print("joints_at_limit_cost")
          print(joints_at_limit_cost)
          print("feet_collision_cost")
          print(feet_collision_cost)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
        ]
        if (debugmode):
          print("rewards=")
          print(self.rewards)
          print("sum rewards")
          print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.robot.body_real_xyz

        self.camera_x = x
        self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)


class SGWalkerBase(MJCFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt

class SGWalker2D(SGWalkerBase):
  foot_list = ["foot", "foot_left"]

  def __init__(self):
      SGWalkerBase.__init__(self, "walker2d.xml", "torso", action_dim=6, obs_dim=22, power=0.40)

  def alive_bonus(self, z, pitch):
      return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

  def robot_specific_reset(self, bullet_client):
      SGWalkerBase.robot_specific_reset(self, bullet_client)
      for n in ["foot_joint", "foot_left_joint"]:
          self.jdict[n].power_coef = 30.0

