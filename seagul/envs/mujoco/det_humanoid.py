from gym.envs.mujoco.humanoid import HumanoidEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class DetHumanoidEnv(HumanoidEnv):
    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()
