import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
import random
from scipy.integrate import odeint

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper



class GymReachNN1(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.u_min = -5
        self.u_max = 5
        self.dt=0.2
        self.viewer = None
        self.c = 0

        # modify from original gym env for (potential) images
        if self.image:
            assert False
        else:
            self.high = np.array([2.0,  2.0])
            self.low = np.array([-2.0, -2.0])
            self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(1,), dtype=np.float32)

        self.goal = np.array([0.0, 0.05, 0.2, 0.3])

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

    def f(self, state, t, u):
        x1, x2 = state
        ff = np.array([x2, u * (x2**2) - x1])
        return ff

    def is_done(self):
        x1, x2 = self.state
        done = x1 >= 0.0 \
                and x1 <= 0.2 \
                and x2 >= 0.05 \
                and x2 <= 0.3 
        return bool(done)

    def step(self,u):
        self.c += 1
        x1, x2 = self.state
        u = u[0]
        N = 40
        t = np.linspace(0, self.dt, N)
        self.state = odeint(self.f, self.state, t, args=(u, ))[-1, :]

        x1, x2 = self.state
        reward_near = 0
        center = np.array([0.1, 0.175])
        vec = self.state - center
        reward_near = -np.linalg.norm(vec)
        # if x1 < 0 or x1 > 0.2 or x2 < 0.05 or x2 > 0.3:
        #     reward_near -= 1
        reward_safe = 0
        if x1 < -1.5 or x1 > 1.5 or x2 < -1.5 or x2 > 1.5:
            reward_safe -= 100
        # print(self.state.shape)
        # new_x1 = x1 + x2 * self.dt
        # new_x2 = x2 + (u * x2 * x2 - x1) * self.dt
        # self.state = np.array([new_x1, new_x2])
        # print(self.state)
        # center = np.array([0.1, 0.175])
        # vec = self.state - center
        # reward_near = -np.linalg.norm(vec)
        # reward_safe = 0
        # if self.state[0] >= 0.3 and self.state[0] <= 0.8 and self.state[1] >= -0.1 and self.state[1] <= 0.4:
        #     reward_safe -=  5
        # done = self.is_done()

        return np.array(self.state), reward_near + reward_safe, False, {}

    def reset(self):
        self.c = 0
        if self.sliding_window:
            self._prev_img = None
        # modify from original gym env to fix starting state
        high = np.array([0.9,  0.6])
        low = np.array([0.8, 0.5])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        if self.image:
            assert False
        x1, x2 = self.state
        return np.array([x1, x2])

    def render(self, mode='human'):
        print (f'state {self.c}: {self.state}')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class ReachNN1(GymWrapper):

    environment_name = 'ReachNN1'
    # what to set for threshold?
    reward_threshold = -3.75
    entry_point = "marvelgym.safelearning.reachnn1:GymReachNN1"
    max_episode_steps = 35

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(ReachNN1, self).__init__(config)

    def make_summary(self, observations, name):
        pass
