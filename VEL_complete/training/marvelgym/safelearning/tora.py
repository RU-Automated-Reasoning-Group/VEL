import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
from scipy.integrate import odeint

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper

__all__ = ['TORA', 'TORAEq']

class GymTORA(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, *args, **kwargs):
        self.threshold = 2

        high = np.array([self.threshold, self.threshold, self.threshold, self.threshold], dtype=np.float32)
        low = -high
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.dt = 0.1
        self.ilb = np.array([0.6, -0.7, -0.4, 0.5])
        self.iub = np.array([0.7, -0.6, -0.3, 0.6])
        self.rlb = np.array([-0.25, -0.25, -0.25, -0.25])
        self.rub = np.array([ 0.25,  0.25,  0.25,  0.25])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=self.ilb, high=self.iub)
        print("initial state", self.state)
        return np.array(self.state)

    def is_done(self):
        th1, th2, th1_dot, th2_dot = self.state
        done =      th1 < - self.threshold \
                    or th1 > self.threshold \
                    or th1_dot < - self.threshold \
                    or th1_dot > self.threshold \
                    or th2 < - self.threshold \
                    or th2 > self.threshold \
                    or th2_dot < - self.threshold \
                    or th2_dot > self.threshold
        return bool(done)


    def f(self, state, t, u):
        x1, x2, x3, x4 = state
        ff = np.array([x2, - x1 + 0.1*np.sin(x3), x4, u])
        return ff

    def step(self, action):
        u = action[0]
        N = 10
        t = np.linspace(0, self.dt, N)
        # self.state = self.state + self.dt * self.f(u)
        self.state = odeint(self.f, self.state, t, args=(u, ))[-1,:]
        #done = self.is_done()
        reward_near = - np.linalg.norm(self.state)
        return np.array(self.state), reward_near, False, {}

class GymTORAEq(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, *args, **kwargs):
        self.threshold = 2

        high = np.array([self.threshold, self.threshold, self.threshold, self.threshold], dtype=np.float32)
        low = -high
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.dt = 0.1
        self.ilb = np.array([-0.1, -0.1, -0.1, -0.1])
        self.iub = np.array([0.1, 0.1, 0.1, 0.1])
        self.rlb = np.array([-0.1, -0.1, -0.1, -0.1])
        self.rub = np.array([ 0.1,  0.1,  0.1,  0.1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=self.ilb, high=self.iub)
        # self.state = np.array([-0.1, -0.1, -0.1, -0.1])
        return np.array(self.state)

    def is_done(self):
        th1, th2, th1_dot, th2_dot = self.state
        done =      th1 < - self.threshold \
                    or th1 > self.threshold \
                    or th1_dot < - self.threshold \
                    or th1_dot > self.threshold \
                    or th2 < - self.threshold \
                    or th2 > self.threshold \
                    or th2_dot < - self.threshold \
                    or th2_dot > self.threshold
        return bool(done)


    def f(self, states, t, u):
        x1, x2, x3, x4 = self.state
        ff = np.array([x2, - x1 + 0.1*np.sin(x3), x4, u])
        return ff

    def step(self, action):
        u = action[0]
        N = 10
        t = np.linspace(0, self.dt, N)
        self.state = odeint(self.f, self.state, t, args=(u, ))[-1, :]
        done = self.is_done()

        if not done:
            reward =  1.0
        else:
            reward = -1.0

        reward_near = - np.linalg.norm(self.state)
        return np.array(self.state), reward_near+reward, done, {}

class TORA(GymWrapper):

    environment_name = 'TORA'
    entry_point = "marvelgym.safelearning.tora:GymTORA"
    max_episode_steps = 1000
    reward_threshold = -3.75

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(TORA, self).__init__(config)

    def make_summary(self, observations, name):
        pass

    def is_image(self):
        return self.image

    def image_size(self):
        if self.image:
            return [self.image_dim, self.image_dim, 3]
        return None

    def start_recording(self, video_path):
        frame_shape = (1000, 1000, 3)
        self.image_encoder = ImageEncoder(video_path, frame_shape, 30)

    def grab_frame(self):
        frame = self.render(mode='rgb_array')
        self.image_encoder.capture_frame(frame)

    def stop_recording(self):
        self.image_encoder.close()

    def torque_matrix(self):
        return 0.1 * np.eye(self.get_action_dim())
    
    def render(self, mode='human'):
        print(self.state)

    #specifications
    def training_settings(self):
        return {
            "dp": False,
            "ilqr": False,
            "trpo": True,
            "training_iters": 1,
            "learning_rate": 0.0001,
            "ars": 0.1,
            "train_safe": True,
            "train_ind": False,
            "train_reach": True,
            "train_performance": False,
            "lb_start": self.gym_env.ilb,
            "ub_start": self.gym_env.iub,
            "lb_safe": np.array([-self.gym_env.threshold, -self.gym_env.threshold, -self.gym_env.threshold, -self.gym_env.threshold]),
            "ub_safe": np.array([ self.gym_env.threshold,  self.gym_env.threshold,  self.gym_env.threshold,  self.gym_env.threshold]),
            "lb_reach": self.gym_env.rlb,
            "ub_reach": self.gym_env.rub,
            "lb_action": None, #np.array([self.gym_env.min_action]),
            "ub_action": None, #np.array([self.gym_env.max_action]),
            "lb_avoids": None,
            "ub_avoids": None,
        }

class TORAEq(TORA):

    environment_name = 'TORAEq'
    entry_point = "marvelgym.safelearning.tora:GymTORAEq"
    max_episode_steps = 20
    reward_threshold = -3.75

    #specifications
    def training_settings(self):
        return {
            "dp": False,
            "ilqr": False,
            "trpo": True,
            "training_iters": 1,
            "learning_rate": 0.0001,
            "ars": 0.1,
            "train_safe": True,
            "train_ind": True,
            "train_reach": False,
            "train_performance": False,
            "lb_start": self.gym_env.ilb,
            "ub_start": self.gym_env.iub,
            "lb_safe": np.array([-self.gym_env.threshold, -self.gym_env.threshold, -self.gym_env.threshold, -self.gym_env.threshold]),
            "ub_safe": np.array([ self.gym_env.threshold,  self.gym_env.threshold,  self.gym_env.threshold,  self.gym_env.threshold]),
            "lb_reach": self.gym_env.rlb,
            "ub_reach": self.gym_env.rub,
            "lb_action": None, #np.array([self.gym_env.min_action]),
            "ub_action": None, #np.array([self.gym_env.max_action]),
            "lb_avoids": None,
            "ub_avoids": None,
        }
