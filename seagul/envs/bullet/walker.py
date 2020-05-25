import gym
import numpy as np
import pybullet as p
import pybullet_data


class PBMJWalker2dEnv(gym.Env):
    motor_joints = [4, 6, 8, 10, 12, 14]
    num_joints = 16

    def __init__(self,
                 render=False,
                 torque_limit=100,
                 init_noise=.005,
                 physics_params = None,
                 dynamics_params = None,
                 ):

        self.args = locals()
        self.torque_limit=torque_limit
        self.init_noise = init_noise

        self.cur_step = 0
        
        low = -np.ones(6)
        self.action_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)

        low = -np.ones(17)*np.inf
        self.observation_space = gym.spaces.Box(low=low, high=-low, dtype=np.float32)

        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.plane_id = p.loadSDF(pybullet_data.getDataPath() + "/plane_stadium.sdf")[0]
        self.walker_id = p.loadMJCF(pybullet_data.getDataPath() + "/mjcf/walker2d.xml")[0]
        #flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)[0] # TODO not sure the self collision needs to be here..

        p.setGravity(0, 0, -9.8)

        if physics_params is None:
            physics_params = {}
        if dynamics_params is None:
            dynamics_params = {}

        p.changeDynamics(self.plane_id, -1, **dynamics_params)

        for i in range(p.getNumJoints(self.walker_id)):
            p.changeDynamics(self.walker_id,i,**dynamics_params)

        p.changeDynamics(self.walker_id, -1, **dynamics_params)

    
        p.setPhysicsEngineParameter(**physics_params)

        self.dt = p.getPhysicsEngineParameters()['fixedTimeStep']

        self.reset()

    def step(self, a):

        a = np.clip(a,-1,1)

        #forces = a*np.array([40, 40, 12, 40, 40, 12]).tolist()
        #forces = a.tolist()
        forces = a*np.array([100, 100, 100, 100, 100, 100]).tolist()
        
        x_before  = p.getLinkState(self.walker_id, 3, computeForwardKinematics=1)[0][0]

        p.setJointMotorControlArray(self.walker_id, self.motor_joints, p.TORQUE_CONTROL, forces=forces)
        p.stepSimulation()
        
        x_after  = p.getLinkState(self.walker_id, 3, computeForwardKinematics=1)[0][0]

        base_link_info = p.getLinkState(self.walker_id, 3, computeLinkVelocity=1, computeForwardKinematics=1)
        base_pos = base_link_info[0]
        base_orn = p.getEulerFromQuaternion(base_link_info[1])
        
        height = base_pos[2]
        pitch  = base_orn[1] # Pitch

        reward = (x_after - x_before) / self.dt;
        reward += 1.0  # alive bonus
        reward -= 1e-3 * np.square(a).sum()
        
        done = not ((0.8 < (height) < 2.0) and (-1.0 < pitch < 1.0))

        self.cur_step+=1
        if self.cur_step > 1000:
            done = True

        return self._get_obs(), reward, done, {}


    def _get_obs(self):

        state = []

        base_link_info = p.getLinkState(self.walker_id, 3, computeLinkVelocity=1, computeForwardKinematics=1)
        base_pos = base_link_info[0]
        base_orn = p.getEulerFromQuaternion(base_link_info[1])
        
        state.append(base_pos[2]) # Z
        state.append(base_orn[1]) # Pitch

        for s in p.getJointStates(self.walker_id, self.motor_joints):
            state.append(s[0])


        base_linvel = base_link_info[6]
        base_angvel = base_link_info[7]


        state.append(np.clip(base_linvel[1], -10, 10))  # Y
        state.append(np.clip(base_linvel[2], -10,10)) # Z
        state.append(np.clip(base_angvel[1], -10,10)) # Pitch

        
        for s in p.getJointStates(self.walker_id, self.motor_joints):
            state.append(np.clip(s[1], -10,10))
            
        return np.array(state)


    def reset(self):

        p.resetBasePositionAndOrientation(self.walker_id, [0,0,0],[0,0,0,1])

        for i in range(p.getNumJoints(self.walker_id)):
            init_ang = np.random.uniform(low=-self.init_noise, high=self.init_noise)
            init_vel = np.random.uniform(low=-self.init_noise, high=self.init_noise)
            p.resetJointState(self.walker_id, i,init_ang,init_vel)

        init_x = np.random.uniform(low=-self.init_noise, high=self.init_noise)
        init_z = np.random.uniform(low=-self.init_noise, high=self.init_noise)
        init_pitch = np.random.uniform(low=-self.init_noise, high=self.init_noise)
        init_pos = [init_x, 0, init_z]
        init_orn = p.getQuaternionFromEuler([0, init_pitch, 0])
        p.resetBasePositionAndOrientation(self.walker_id, init_pos, init_orn)

        init_vx = np.random.uniform(low=-self.init_noise, high=self.init_noise)
        init_vz = np.random.uniform(low=-self.init_noise, high=self.init_noise)
        init_vp = np.random.uniform(low=-self.init_noise, high=self.init_noise)
        p.resetBaseVelocity(self.walker_id,[init_vx,0,init_vz], [0,init_vp,0])

        p.setJointMotorControlArray(self.walker_id,
                                    [i for i in range(p.getNumJoints(self.walker_id))],
                                    p.POSITION_CONTROL,
                                    positionGains=[0.1] * self.num_joints,
                                    velocityGains=[0.1] * self.num_joints,
                                    forces=[0 for _ in range(p.getNumJoints(self.walker_id))]
                                    )
        self.cur_step = 0
        
        return self._get_obs()




