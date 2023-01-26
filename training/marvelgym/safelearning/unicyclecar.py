import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper

__all__ = ['UnicycleCar']

class GymUnicycleCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, *args, **kwargs):
        self.threshold = 10

        high = np.array([self.threshold, self.threshold, self.threshold, self.threshold], dtype=np.float32)
        low = -high
        self.action_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf,  np.inf]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.dt = 0.2
        self.c = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.c = 0
        self.state = self.np_random.uniform(low=np.array([9.5, -4.5, 2.1, 1.5]), high=np.array([9.55, -4.45, 2.11, 1.51]))
        return np.array(self.state)

    def is_done(self):
        x1, x2, x3, x4 = self.state
        done =      x1 >= -0.6 \
                    and x1 <= 0.6 \
                    and x2 >= -0.2 \
                    and x2 <= 0.2 \
                    and x3 >= -0.06 \
                    and x3 <= 0.06 \
                    and x4 >= -0.3 \
                    and x4 <= 0.3
        return bool(done)


    def f(self, u1, u2):
        self.c += 1
        x1, x2, x3, x4 = self.state
        ff = np.array([x4 * np.cos(x3), x4 * np.sin(x3), u2, u1])
        return ff

    def step(self, action):
        u1, u2 = action[0], action[1]
        self.state = self.state + self.dt * self.f(u1, u2)
        reward_near = - np.linalg.norm(self.state)
        done = self.is_done()
        return np.array(self.state), reward_near, False, {}
    
    def render(self, mode='human'):
        print(self.c, self.state)

class UnicycleCar(GymWrapper):

    environment_name = 'UnicycleCar'
    entry_point = "marvelgym.safelearning.unicyclecar:GymUnicycleCar"
    max_episode_steps = 30
    reward_threshold = -3.75

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(UnicycleCar, self).__init__(config)

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

    #specifications
    def training_settings(self):
        return {
            "dp": False,
            "ilqr": False,
            "trpo": True,
            "training_iters": 1,
            "learning_rate": 0.0001,
            "ars": 0.1,
            "train_safe": False,
            "train_ind": False,
            "train_reach": True,
            "train_performance": False,
            "lb_start": np.array([ 9.5,  -4.50, 2.10, 1.50]),
            "ub_start": np.array([ 9.55, -4.45, 2.11, 1.51]),
            "lb_safe": np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            "ub_safe": np.array([ np.inf,  np.inf,  np.inf,  np.inf]),
            "lb_reach": np.array([-0.6, -0.2, -0.06, -0.3]),
            "ub_reach": np.array([ 0.6,  0.2,  0.06,  0.3]),
            "lb_action": None, #np.array([self.gym_env.min_action]),
            "ub_action": None, #np.array([self.gym_env.max_action]),
            "lb_avoids": None,
            "ub_avoids": None,
        }
