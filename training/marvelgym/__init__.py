import contextlib
with contextlib.redirect_stdout(None):
    #from .mujoco import *
    from .openai import *
    # from .basic import *
    from .safelearning import *
    #from .mujoco import *

ENVS = [CartPole,MountainCar,TORA,TORAEq,ReachNN1, ReachNN2, ReachNN3, ReachNN4, ReachNN5, ReachNN6, AccCMP, PP, QMPC, UnicycleCar, OS]
#Reacher,Hopper,DoublePendulum,InvertedPendulum,UUV,Pusher,Thrower,Bicycle2,BicycleSteering,Academic3D,NoisyRoad,NoisyRoadBack,NoisyRoad2D,Road,Road2D,CarSafe,HalfCheetah,LunarLander,Airplane,TORA,TORAEq,UnicycleCar,SinglePendulum,DoublePendulum2,TriplePendulum,Acc2,F16,Quadrotor]

ENV_MAP = { env.environment_name : env for env in ENVS }

def from_config(env_config):
    env_config = env_config.copy()
    name = env_config.pop('environment_name')
    return make(name, **env_config)

def make(env_name, **kwargs):
    if env_name not in ENV_MAP:
        raise Exception("Environment %s does not exist" % env_name)
    return ENV_MAP[env_name](**kwargs)
