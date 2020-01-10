import numpy as np
import gym
from numpy import cos, sin, pi

from gym.utils import seeding

from seagul.integration import rk4, euler, wrap
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SUCartPoleEnv(gym.Env):
    """
    Environment for for a classic_control cartpole pendulum.

    mostly just a rewrite of sgillen_research/cartpole/cartpole.py

    Attributes:
        L: length of the pendulum in (m)
        mc:  mass of the kart (kg)
        mp: magnitude of pointmass at the end of the cart's pole (kg)
        g: force f gravity (N)

        state: state of the cartpole, [theta(rads), x(m), dtheta(rads/s), dx (m/s)]

    """

    metadata = {"render.modes": ["human"], "video.frames_per_second": 15}

    def __init__(self, num_steps=1500, dt=0.001):
        self.L = 2.0  # length of the pole (mg)
        self.mc = 1.0  # mass of the cart (kg)
        self.mp = 1.0  # mass of the ball at the end of the pole

        self.g = 9.8

        self.dt = dt
        self.num_steps = num_steps
        self.cur_step = 0
        self.state = None

        # THETA_MAX is implict (-pi, pi)
        self.X_MAX = 50.0

        # might impose an upper limit on these but it would only end the episode
        self.DTHETA_MAX = 100.0 * pi
        self.DX_MAX = 500.0

        self.state_noise_max = 0
        high = np.array([pi, self.X_MAX, self.DTHETA_MAX, self.DX_MAX])
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high)

        self.TORQUE_MAX = 1000.0
        self.torque_noise_max = 0.0
        self.action_space = gym.spaces.Box(-self.TORQUE_MAX, self.TORQUE_MAX, shape=(1,))

        self.viewer = None

        self.seed()

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([0, 0, 0, 0]) + self.np_random.uniform(
            -self.state_noise_max, self.state_noise_max, size=(4,)
        )
        self.cur_step = 0

        return self.state

    def _get_ob(self):
        return self.state + self.np_random.uniform(-self.state_noise_max, self.state_noise_max, size=(4,))

    def step(self, action):
        done = False

        # RL algorithms aware of the action space won't need this but things like the
        # imitation learning or energy shaping controllers might try feeding in something
        # above the torque limit
        torque = np.clip(action, -self.TORQUE_MAX, self.TORQUE_MAX)
        # torque = action
        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        for _ in range(5):
            # ns = euler(self._derivs, torque, 0, self.dt, self.state)
            ns = rk4(self._derivs, torque, 0, self.dt, self.state)

            self.state[0] = wrap(ns[0], -pi, pi)
            # self.state[0] = ns[0]
            self.state[1] = ns[1]
            # self.state[1] = np.clip(ns[1], -self.X_MAX, self.X_MAX)
            self.state[2] = ns[2]
            self.state[3] = ns[3]

        # self.state[2] = np.clip(ns[2], -self.DTHETA_MAX, self.DTHETA_MAX)
        # self.state[3] = np.clip(ns[3], -self.DX_MAX, self.DX_MAX)

        # Should reward be something we pass in ? I do like to mess with them a lot...
        #        reward = -np.cos(self.state[0]) + 2

        reward = (-5 * np.cos(self.state[0])) + 2

        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True
        elif np.abs(self.state[1]) > self.X_MAX:
            done = True

        return self.state, reward, done, {}

    # def reset_model(self):
    #   self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
    #   self.cur_step = 0
    #   return self._get_obs()

    # def render(self, mode='human', close=False):

    # basically taken from gym's classic control cartpole
    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        if self.state is None:
            return None

        screen_width = 600
        screen_height = 400

        world_width = self.X_MAX * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = scale * 0.25
        polelen = scale * self.L * 3.0  # 3.0 arbitrary just to make the length look OK
        cartwidth = scale * 2.0
        cartheight = scale * 1.0

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (-polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2)
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        x = self.state
        cartx = self.state[1] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(self.state[0] + pi)

        return self.viewer.render()

    def _derivs(self, t, q, u):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:
        y = integrate.odeint(bot.derivs, init_state, time)
        or whatever other ode solver you prefer.

        Args:
            t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
            q: numpy array of state variables [theta, x, thetadot, xdot]
            numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]

        Returns:
            dqdt: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """

        dqdt = np.zeros_like(q)

        delta = self.mp * (sin(q[0]) ** 2) + self.mc

        dqdt[0] = q[2]
        dqdt[1] = q[3]

        dqdt[2] = (
            1
            / (self.L * delta)
            * (
                -u * cos(q[0])
                - self.mp * self.L * q[2] ** 2 * cos(q[0]) * sin(q[0])
                - (self.mc + self.mp) * self.g * sin(q[0])
            )
        )

        dqdt[3] = 1 / delta * (u + self.mp * sin(q[0] * (self.L * q[2] ** 2 + self.g * cos(q[0]))))

        return dqdt
