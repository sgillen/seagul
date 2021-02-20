import pygame as pg
import pygame.freetype as ft
import sys
from gym import core, spaces
import numpy as np

class PixelTreeEnv(core.Env):
    def __init__(self, render=True):
        self.screen_size = self.screen_width, self.screen_height = 640, 480
        self.color_black = (0, 0, 0)
        self.color_white = (255, 255, 255)

        self.sprite_size = self.sprite_width, self.sprite_height = 10,10
        self.tree_size = self.tree_width, self.tree_height = 150,50

        self.yvel = 10
        self.xvel = 20
        self.ymax = self.screen_height + 50

        self.renderer_is_init = False

        # self.max_steps = 1000
        # self.cur_step = 0

        self.obs_size = self.screen_width*self.screen_height
        obs_shape = (1, self.screen_height, self.screen_width)
        self.observation_space = spaces.Box(shape=obs_shape, low=np.zeros(obs_shape,dtype=np.uint8), high=np.ones(obs_shape,dtype=np.uint8),dtype=np.uint8)
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))

        self.init_render()
        self.tree_x = self._spawn_tree()

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.x = 0.0
        self.y = self.screen_height - 50.0
        self.cur_step = 0
        self._spawn_tree()
        return self._get_obs()

    def _spawn_tree(self):
        self.tree_x = np.random.randint(100, self.screen_width-100)  - self.screen_width//2
        self.tree_y = -50
        return self.tree_x

    def step(self, act):
        self.render()
        act = np.clip(act, -1,1)
        self.x += act * self.xvel
        self.x = np.clip(self.x, 100 - self.screen_width/2, self.screen_width/2 - 100).item()
        self.tree_y = self.tree_y + self.yvel

        done = False
        # if self.cur_step >= self.max_steps:
        #     done=True

        # self.cur_step+=1

        if (self.tree_y + self.tree_height/2) > (self.y - self.sprite_height/2) and (self.tree_y - self.tree_height/2) < (self.y + self.sprite_height/2):
            #print("in y")
            if (self.tree_x + self.tree_width / 2) > (self.x - self.sprite_width / 2) and (self.tree_x - self.tree_width / 2) < (self.x + self.sprite_width/ 2):
                #print("in x")
                done = True

        if self.tree_y > self.ymax:
            self.tree_y = self.tree_y % self.ymax
            self.tree_x = self._spawn_tree()

        reward = 1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return pg.surfarray.array2d(self.screen).reshape((1, self.screen_height, self.screen_width))
        #return np.array([self.x/220, (self.tree_y - self.y)/480, (self.x - self.tree_x)/480])

    def init_render(self):
        pg.init()
        self.screen = pg.display.set_mode(self.screen_size)
        pg.key.set_repeat(1, 10)
        self.myfont = pg.font.SysFont(ft.get_default_font(), 24)

    def render(self, mode="who_cares"):
        import pygame as pg

        if not self.renderer_is_init:
            self.init_render()

        self.screen.fill(self.color_black)

        wall_left = pg.Rect(100, 0, 2, self.screen_height)
        pg.draw.rect(self.screen, self.color_white, wall_left)

        wall_right = pg.Rect(self.screen_width-100, 0, 2, self.screen_height)
        pg.draw.rect(self.screen, self.color_white, wall_right)
        self.renderer_is_init = True

        player_rect = pg.Rect(self.x + self.screen_width/2 - self.sprite_width/2, self.y, self.sprite_width, self.sprite_height)
        pg.draw.rect(self.screen, self.color_white, player_rect)

        tree_rect = pg.Rect(self.tree_x + self.screen_width/2 - self.tree_width/2, self.tree_y, self.tree_width, self.tree_height)
        pg.draw.rect(self.screen, self.color_white, tree_rect)

        # for i,obs in enumerate(self._get_obs()):
        #     text_surf = self.myfont.render(str(obs), False, self.color_white)
        #     self.screen.blit(text_surf, (0,i*20))

        pg.display.flip()


def update():
    obs, rews, done, _ = env.step(act)
    env.render()
    if done:
        print("done")
        exit()

if __name__ == "__main__":
    from seagul.rl.ars.ars_np_queue import ARSAgent, postprocess_default
    import copy
    import gym
    import time
    import xarray as xr
    import numpy as np
    import os

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    import seagul.envs

    env_name = "pixel_tree-v0"
    env = gym.make(env_name)
    # env_names = ["HalfCheetah-v2"]
    # env_names = ["Hopper-v2"]
    # post_fns = [madodiv, variodiv]

    # env_names = ["Humanoid-v2"]
    # env_names = ["MinitaurBulletEnv-v0"]

    num_steps = int(2e6)

    agent = PPO('CnnPolicy',
                env,
                verbose=0,
                # policy_kwargs={'net_arch':[{'pi':[32, 32]}, {'vf':[32, 32]}]},
                # device='cpu'
                )

    agent.learn(total_timesteps=num_steps)

    # print(f"{env_name}, {i}, {time.time() - start}")
    #
    #
    #
    # import seagul.envs
    # from seagul.rl.ars.ars_np_queue import ARSAgent
    # import gym
    #
    # # env = gym.make("tree-v0")
    # # agent = ARSAgent("tree-v0", 0)
    # # agent.learn(100)
    #
    # import pygame as pg
    #
    # env = TreeEnv()
    # env.render()
    # while True:
    #     while True:
    #         events = pg.event.get()
    #         for event in events:
    #             if event.type == pg.KEYDOWN:
    #                 if event.key == pg.K_RIGHT:
    #                     act = 1
    #                     update()
    #                 if event.key == pg.K_LEFT:
    #                     act = -1
    #                     update()