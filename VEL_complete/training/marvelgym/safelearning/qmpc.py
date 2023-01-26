import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
import random
from scipy.integrate import odeint

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper



class GymQMPC(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.u_min = np.array([-0.1, -0.1, 7.81])
        self.u_max = np.array([0.1, 0.1, 11.81])
        self.dt=0.2
        self.viewer = None
        self.c = 0

        # modify from original gym env for (potential) images
        if self.image:
            assert False
        else:
            self.high = np.array([2.0,  2.0, 2.0, 2.0, 2.0, 2.0])
            self.low = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
            self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, dtype=np.float32)

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

    def f1(self, state, t, u):
        x1, x2, x3, x4, x5, x6 = state
        u1, u2, u3 = u
        dx1 = x4 - 0.25
        dx2 = x5 + 0.25
        dx3 = x6
        dx4 = 9.81 * np.sin(u1) / np.cos(u1)
        dx5 = -9.81 * np.sin(u2) / np.cos(u2)
        dx6 = u3 - 9.81
        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def f2(self, state, t, u):
        x1, x2, x3, x4, x5, x6 = state
        u1, u2, u3 = u
        dx1 = x4 - 0.25
        dx2 = x5 - 0.25
        dx3 = x6
        dx4 = 9.81 * np.sin(u1) / np.cos(u1)
        dx5 = -9.81 * np.sin(u2) / np.cos(u2)
        dx6 = u3 - 9.81
        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])
    
    def f3(self, state, t, u):
        x1, x2, x3, x4, x5, x6 = state
        u1, u2, u3 = u
        dx1 = x4
        dx2 = x5 + 0.25
        dx3 = x6
        dx4 = 9.81 * np.sin(u1) / np.cos(u1)
        dx5 = -9.81 * np.sin(u2) / np.cos(u2)
        dx6 = u3 - 9.81
        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def f4(self, state, t, u):
        x1, x2, x3, x4, x5, x6 = state
        u1, u2, u3 = u
        dx1 = x4 + 0.25
        dx2 = x5 - 0.25
        dx3 = x6
        dx4 = 9.81 * np.sin(u1) / np.cos(u1)
        dx5 = -9.81 * np.sin(u2) / np.cos(u2)
        dx6 = u3 - 9.81
        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def is_done(self):
        x1, x2 = self.state
        done = x1 >= 0.0 \
                and x1 <= 0.2 \
                and x2 >= 0.05 \
                and x2 <= 0.3 
        return bool(done)

    def step(self,u):
        self.c += 1
        N = 400
        # u = np.clip(u, np.array([-0.1, -0.1, 7.81]), np.array([0.1, 0.1, 11.81]))
        # print(u)
        t = np.linspace(0, self.dt, N)
        if self.c < 10:
            self.state = odeint(self.f1, self.state, t, args=(u, ))[-1, :]
        elif self.c < 20 and self.c >= 10:
            self.state = odeint(self.f2, self.state, t, args=(u, ))[-1, :]
        elif self.c < 25 and self.c >= 20:
            self.state = odeint(self.f3, self.state, t, args=(u, ))[-1, :]
        elif self.c >= 25:
            self.state = odeint(self.f4, self.state, t, args=(u, ))[-1, :]
        x1, x2, x3, x4, x5, x6 = self.state
        reward = -1 * np.linalg.norm(np.array([x1, x2, x3]))

        return np.array(self.state), reward, False, {}

    def reset(self):
        self.c = 0
        if self.sliding_window:
            self._prev_img = None
        # modify from original gym env to fix starting state
        high = np.array([0.05, 0.025, 0.0, 0.0, 0.0, 0.0])
        low = np.array([0.025, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        if self.image:
            assert False
        x1, x2, x3, x4, x5, x6 = self.state
        return np.array([x1, x2, x3, x4, x5, x6])

    def render(self, mode='human'):
        x1, x2, x3, x4, x5, x6 = self.state
        # print (f'state {self.c}: {abs(x1)} {abs(x2)} {abs(x3)}', x4, x5, x6)
        print("state", self.c, x1, x2, x3, x4, x5, x6)
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class QMPC(GymWrapper):

    environment_name = 'QMPC'
    reward_threshold = -3.75
    entry_point = "marvelgym.safelearning.qmpc:GymQMPC"
    max_episode_steps = 30

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(QMPC, self).__init__(config)

    def make_summary(self, observations, name):
        pass
