import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
import random

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper



class GymOS(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.u_min = -5
        self.u_max = 5
        self.dt= 0.01
        self.gamma = 1
        self.viewer = None

        # modify from original gym env for (potential) images
        if self.image:
            assert False
        else:
            self.high = np.array([1.0,  1.0])
            self.low = np.array([-1.0, -1.0])
            self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(1,), dtype=np.float32)

        self.goal = np.array([-0.05, -0.05, 0.05, 0.05])
        self.obstacle = np.array([-0.3, 0.2, -0.25, 0.35])

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reach(self):
        return min([self.goal[2] - self.state[0],
                    self.goal[3] - self.state[1],
                    self.state[0] - self.goal[0],
                    self.state[1] - self.goal[1]])

    def avoid(self):
        return max([self.obstacle[0] - self.state[0],
                    self.obstacle[1] - self.state[1],
                    self.state[0] - self.obstacle[2],
                    self.state[1] - self.obstacle[3]])

    def step(self,u):
        x1, x2 = self.state # th := theta

        dt = self.dt

        u = u[0]
        if not self.image:
            self.last_u = u # for rendering

        # lead_a = max(self.a_min, min(self.a_max, random.gauss(0,1)))

        newx1 = x1 + x2 * dt
        newx2 = x2 + (self.gamma * (1 - x1**2) * x2 - x1 + u) * dt
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newx1, newx2])
        # reward value?
        safe_rwd = self.avoid()
        reach_rwd = self.reach()
        center = np.array([0.0, 0.0])
        vec = self.state[0:2] - center
        reward_near = - np.linalg.norm(vec)

        if safe_rwd < 0:
            safe_rwd = -100
            done = True
        else:
            safe_rwd = 0
            done = False
        if reach_rwd >= 0:
            done = True

        return self._get_obs(), safe_rwd + reward_near, done, {}

    def reset(self):
        if self.sliding_window:
            self._prev_img = None
        # modify from original gym env to fix starting state
        high = np.array([-0.49,  0.51])
        low = np.array([-0.51, 0.49])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        if self.image:
            assert False
        x1, x2 = self.state
        return np.array([x1, x2])

    def render(self, mode='human'):
        print (f'state : {self.state}')
        if self.avoid() < 0:
            print("unsafe")
        else:
            print("safe")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class OS(GymWrapper):

    environment_name = 'OS'
    # what to set for threshold?
    reward_threshold = -3.75
    entry_point = "marvelgym.safelearning.os:GymOS"
    max_episode_steps = 150

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(OS, self).__init__(config)

    def make_summary(self, observations, name):
        pass
