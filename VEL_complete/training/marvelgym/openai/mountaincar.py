import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding
from os import path

from ..utils import ImageEncoder
from ..gym_wrapper import GymWrapper

__all__ = ['MountainCar']

class GymMountainCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, *args, **kwargs):
        self.c = 0
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = 0
        self.power = 0.0015

        self.x_constraint_min = -1.15

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.c += 1
        position = self.state[0]
        velocity = self.state[1]

        cost = (position - self.goal_position)**2 + (velocity - self.goal_velocity)**2 # + 0.1*(action[0])**2

        force = min(max(action[0], self.min_action), self.max_action)
        # print(force)
        # force = action[0]
        # print("temp", force * self.power - 0.0025 * math.cos(3 * position))
        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): 
            velocity = self.max_speed
            # print("max speed clipped")
        if (velocity < -self.max_speed): 
            velocity = -self.max_speed
            # print("min speed clipped")
        position += velocity
        if (position > self.max_position): 
            position = self.max_position
        if (position < self.min_position): 
            position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        # Convert a possible numpy bool to a Python bool.
        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = -cost
        # if position >= 0.15 and position <= 0.25 and velocity <= 0.02:
        #     reward -= 1

        if position <= -1.15:
            reward += 5

        # if position < self.x_constraint_min:
        #     done = False
        #     print("x_min triggered")
        #     reward = -100.

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self):
        self.c = 0
        self.state = np.array([self.np_random.uniform(low=-0.53, high=-0.50), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        print("step", self.c, self.state[0], self.state[1])
        if self.state[0] >= 0.15 and self.state[0] <= 0.25 and self.state[1] <= 0.02:
            print("unsafe")
        # screen_width = 600
        # screen_height = 400

        # world_width = self.max_position - self.min_position
        # scale = screen_width/world_width
        # carwidth = 40
        # carheight = 20

        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        #     xs = np.linspace(self.min_position, self.max_position, 100)
        #     ys = self._height(xs)
        #     xys = list(zip((xs-self.min_position)*scale, ys*scale))

        #     self.track = rendering.make_polyline(xys)
        #     self.track.set_linewidth(4)
        #     self.viewer.add_geom(self.track)

        #     clearance = 10

        #     l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        #     car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     car.add_attr(rendering.Transform(translation=(0, clearance)))
        #     self.cartrans = rendering.Transform()
        #     car.add_attr(self.cartrans)
        #     self.viewer.add_geom(car)
        #     frontwheel = rendering.make_circle(carheight / 2.5)
        #     frontwheel.set_color(.5, .5, .5)
        #     frontwheel.add_attr(
        #         rendering.Transform(translation=(carwidth / 4, clearance))
        #     )
        #     frontwheel.add_attr(self.cartrans)
        #     self.viewer.add_geom(frontwheel)
        #     backwheel = rendering.make_circle(carheight / 2.5)
        #     backwheel.add_attr(
        #         rendering.Transform(translation=(-carwidth / 4, clearance))
        #     )
        #     backwheel.add_attr(self.cartrans)
        #     backwheel.set_color(.5, .5, .5)
        #     self.viewer.add_geom(backwheel)
        #     flagx = (self.goal_position-self.min_position)*scale
        #     flagy1 = self._height(self.goal_position)*scale
        #     flagy2 = flagy1 + 50
        #     flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        #     self.viewer.add_geom(flagpole)
        #     flag = rendering.FilledPolygon(
        #         [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        #     )
        #     flag.set_color(.8, .8, 0)
        #     self.viewer.add_geom(flag)

        # pos = self.state[0]
        # self.cartrans.set_translation(
        #     (pos-self.min_position) * scale, self._height(pos) * scale
        # )
        # self.cartrans.set_rotation(math.cos(3 * pos))

        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class MountainCar(GymWrapper):

    environment_name = 'MountainCar'
    reward_threshold = -3.75
    entry_point = "marvelgym.openai.mountaincar:GymMountainCar"
    max_episode_steps = 1000

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(MountainCar, self).__init__(config)

    def make_summary(self, observations, name):
        pass

    def is_image(self):
        return self.image

    def image_size(self):
        if self.image:
            return [self.image_dim, self.image_dim, 3]
        return None

    def start_recording(self, video_path):
        frame_shape = (800, 1200, 3)
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
            "lb_start": np.array([-0.6,0]),
            "ub_start": np.array([-0.4,0]),
            "lb_safe": np.array([self.gym_env.x_constraint_min, -5*self.gym_env.max_speed]),
            "ub_safe": np.array([5*self.gym_env.max_position, 5*self.gym_env.max_speed]),
            "lb_reach": np.array([self.gym_env.goal_position, self.gym_env.goal_velocity]),
            "ub_reach": np.array([5*self.gym_env.max_position, 5*self.gym_env.max_speed]),
            "lb_action": None, #np.array([self.gym_env.min_action]),
            "ub_action": None, #np.array([self.gym_env.max_action]),
            "lb_avoids": None,
            "ub_avoids": None,
        }
