import gym
import numpy as np
import pybullet as p


class PyBulletPhysicsWrapper(gym.Wrapper):
    """
    Wraps a pybulletgym environment, allowing us to change the physical and dynamical params on init/reset
    """
    def __init__(self, env, physics_params, dynamics_params):
        self.physics_params = physics_params
        self.dynamics_params = dynamics_params
        super().__init__(env)

    def reset(self):
        obs = super().reset()
        p.setPhysicsEngineParameter(**self.physics_params)

        for body in range(p.getNumBodies()):
            p.changeDynamics(body, -1, **self.dynamics_params)
            for joint in range(p.getNumJoints(body)):
                p.changeDynamics(body, joint)

        return obs
