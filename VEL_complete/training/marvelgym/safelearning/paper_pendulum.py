import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
import random
from scipy.integrate import odeint

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper


class GymPP(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = 10.0
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        # print(u[0])
        # if u[0] > self.max_torque or u[0] < -self.max_torque:
        #     print("exceed")
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        # high = np.array([np.pi / 2 + 0.1, 0.1])
        # low = np.array([np.pi / 2 - 0.1, -0.11])
        high = np.array([0.1, 0.1])
        low = np.array([-0.1, -0.1])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class PP(GymWrapper):

    environment_name = 'PP'
    # what to set for threshold?
    reward_threshold = -3.75
    entry_point = "marvelgym.safelearning.paper_pendulum:GymPP"
    max_episode_steps = 200

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(PP, self).__init__(config)

    def make_summary(self, observations, name):
        pass
