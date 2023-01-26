import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from os import path
from scipy.integrate import odeint

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper

__all__ = ['AccCMP']

class GymAccCMP(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, *args, **kwargs):
        self.action_space = spaces.Box(
            low=np.full((1,), -np.inf),
            high=np.full((1,), np.inf),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.full((5,), -np.inf, dtype=np.float32),
            high=np.full((5,), np.inf, dtype=np.float32),
            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.dt = 0.1
        self.c = 0

        self.ilow = np.array([90, 32, 0, 10, 30, 0])
        self.ihigh = np.array([110, 32.2, 0, 11, 30.2, 0])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.c = 0
        self.state = self.np_random.uniform(low=self.ilow, high=self.ihigh)
        return self.get_obs()

    def get_obs(self):
        x1, x2, x3, x4, x5, x6 = self.state
        obs1 = 30       # V_set
        obs2 = 1.4      # T_gap
        obs3 = x5       # v_ega
        obs4 = x1 - x4  # D_rel
        obs5 = x2 - x5  # v_rel
        return np.array([obs1, obs2, obs3, obs4, obs5])

    def is_done(self):
        x1, x2, x3, x4, x5, x6 = self.state
        done = (x1 - x4 - 1.4 * x5 < 10) or (x1 - x4 > 102)
        return bool(done)

    def dynamics(self, states, t, actions):

        # unwraping states and actions
        x1, x2, x3, x4, x5, x6 = states
        a_ego = actions[0]
        mu = 0.0001
        a_lead = -2
        dx1 = x2
        dx2 = x3
        dx3 = -2 * x3 + 2 * a_lead - mu*x2**2
        dx4= x5
        dx5 = x6
        dx6 = -2 * x6 + 2 * a_ego - mu*x5**2

        dXdt = np.array([dx1, dx2, dx3, dx4, dx5, dx6])
        return dXdt

    def step(self, actions):
        # print(actions[0])
        self.c += 1
        N = 5
        t = np.linspace(0, self.dt, N)
        self.state = odeint(self.dynamics, self.state, t, args=(actions, ))[-1, :]
        done = self.is_done()
        # center = np.array([53])
        # cur = np.array([self.state[0] - self.state[3] - 1.4 * self.state[4]])
        # reward_near = -np.linalg.norm(cur - center)
        if not done:
            reward = 1.0
        else:
            reward = -100.0
        # if not done:
        #     reward =  1.0
        # else:
        #     reward = -1.0
        # reward
        return self.get_obs(), reward, False, {}
    
    def render(self, mode='human'):
        print("step", self.c, self.state[0] - self.state[3] - 1.4 * self.state[4], self.state[0] - self.state[3])

    # def step(self, action):
    #     self.state = self.state + self.dt * self.f(action)
    #     done = self.is_done()
    #     if not done:
    #         reward =  1.0
    #     else:
    #         reward = -1.0
    #     return np.array(self.state), reward, done, {}

class AccCMP(GymWrapper):

    environment_name = 'AccCMP'
    entry_point = "marvelgym.safelearning.acc_complete:GymAccCMP"
    max_episode_steps = 50
    reward_threshold = -3.75

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(AccCMP, self).__init__(config)

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
    # def training_settings(self):
    #     return {
    #         "dp": False,
    #         "ilqr": False,
    #         "trpo": True,
    #         "training_iters": 1,
    #         "learning_rate": 0.0001,
    #         "ars": 0.1,
    #         "train_safe": True,
    #         "train_ind": False,
    #         "train_reach": False,
    #         "train_performance": False,
    #         "lb_start": np.concatenate([self.gym_env.ilow, np.array([36.72, 79])]),
    #         "ub_start": np.concatenate([self.gym_env.ihigh, np.array([58.0, 100])]),
    #         "lb_safe": np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 10.,   -np.inf]),
    #         "ub_safe": np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf, 102]),
    #         "lb_reach": None,
    #         "ub_reach": None,
    #         "lb_action": None, #np.array([self.gym_env.min_action]),
    #         "ub_action": None, #np.array([self.gym_env.max_action]),
    #         "lb_avoids": None,
    #         "ub_avoids": None,
    #     }
