import torch
import torch.nn as nn
import numpy as np

from control.rl.utils.gym_env import GymEnv
from control.rl.policies.gaussian_prog import *
from control.rl.utils.fc_network import FCNetwork
from control.rl.baselines.mlp_baseline import MLPBaseline
from control.rl.algos.npg_cg import NPG
from control.rl.algos.trpo import TRPO
from control.rl.algos.ppo_clip import PPO
from control.rl.utils.train_agent import train_agent

from gym.utils import seeding
from timeit import default_timer as timer
from os import path
import dill as pickle

import marvelgym as gym
# from marvelgym.spec_env import SpecEnv

def createProgNetwork(prog_type, observation_dim, action_dim):
    if prog_type == 'Linear':
        prog = LinearProgNetwork(observation_dim, action_dim)
    elif prog_type == 'ITELinear':
        prog = ITELinearProgNetwork(observation_dim, action_dim)
    elif prog_type == 'NestITELinear':
        prog = NestITELinearProgNetwork(observation_dim, action_dim)
    elif prog_type == 'Nest2ITELinear':
        prog = Nest2ITELinearProgNetwork(observation_dim, action_dim)
    elif prog_type == 'ITEConstant':
        prog = ITEConstantProgNetwork(observation_dim, action_dim)
    elif prog_type == 'NestITEConstant':
        prog = NestITEConstantProgNetwork(observation_dim, action_dim)
    elif prog_type == 'Nest2ITEConstant':
        prog = Nest2ITEConstantProgNetwork(observation_dim, action_dim)
    elif prog_type == 'PendulumPID':
        prog = PendulumPIDProgNetwork(observation_dim, action_dim)
    elif prog_type == 'LunarLanderPD':
        prog = LunarLanderPDProgNetwork(observation_dim, action_dim)
    elif prog_type == 'MLP':
        prog = FCNetwork(observation_dim, action_dim, hidden_sizes=(64,64), nonlinearity='tanh', bias=True)
    else:
        assert False

    return prog

def train(env_name, prog_type, trainsteps, SEED, phaselearning=False, phase_trainiter_inc=0):
    try:
        e = GymEnv(gym.make(env_name).gym_env)
    except:
        e = GymEnv(env_name)

    if phaselearning: # learning in different phases
        time_limit = e.env._max_episode_steps / e.env.env.specs_size()
        print (f'episode_steps for each learning phase {time_limit}')
        e.env._max_episode_steps = time_limit
        e.env.env.set_timelimit(time_limit)
    else:
        prog = createProgNetwork(prog_type, e.spec.observation_dim, e.spec.action_dim)

    e.set_seed(SEED) # random search uses a fixed seed.
    # state_shape = e.spec.observation_dim
    # action_shape = e.spec.action_dim  # number of actions
    # step, noise, ndr, bdr = .1, .3, 16, 8
    # rl_agent = ARS_V1(alpha=step, noise=noise, N_of_directions=ndr, b=bdr, training_length=50)
    # rl_agent.train(e, policy, Normalizer(state_shape))
    #agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)
    #agent = PPO(e, policy, baseline, clip_coef=0.2, epochs=10, mb_size=64, learn_rate=3e-4, seed=seed, save_logs=True)

    if not phaselearning: # do we have specs to solve? No.
        policy = ProgPolicy(e.spec, prog=prog, seed=SEED) #LinearPolicy(e.spec, seed=SEED)
        baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
        agent = TRPO(e, policy, baseline, normalized_step_size=0.1, kl_dist=None, seed=seed, save_logs=True)
        train_agent(job_name=env_name, # No. Train with rewards
                agent=agent,
                seed=SEED,
                niter=trainsteps,
                gamma=0.995,
                gae_lambda=0.97,
                num_cpu=4,
                sample_mode='trajectories',
                num_traj=100,
                save_freq=5,
                evaluation_rollouts=5,
                out_dir='data/')
    else: # Yes. Train with the specs
        for i in range(e.env.env.specs_size()):
            prog = createProgNetwork(prog_type, e.spec.observation_dim, e.spec.action_dim)
            policy = ProgPolicy(e.spec, prog=prog, seed=SEED)
            baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
            agent = TRPO(e, policy, baseline, normalized_step_size=0.1, kl_dist=None, seed=seed, save_logs=True)
            print (f"------------- Training phase {i} -------------")
            env_name_ph = env_name + '_' + str(i)
            best_policy_at, best_perf = train_agent(job_name=env_name_ph,
                    agent=agent,
                    seed=SEED,
                    niter=trainsteps + (i*phase_trainiter_inc),
                    gamma=0.995,
                    gae_lambda=0.97,
                    num_cpu=1,
                    sample_mode='trajectories',
                    num_traj=100,
                    save_freq=5,
                    evaluation_rollouts=5,
                    out_dir='data/')
            pi = 'data/' + env_name_ph + '/iterations/best_policy.pickle'
            policy = pickle.load(open(pi, 'rb'))
            e.env.env.advance(policy)

# def visualize_policy(env_name, num_episodes=1, mode='exploration', discrete=False, render=True):
#     try:
#         e = GymEnv(gym.make(env_name).gym_env)
#     except:
#         e = GymEnv(env_name)
#     horizon = e._horizon

#     # if isinstance(e.env.env, SpecEnv):
#     #     for i in range(e.env.env.specs_size()):
#     #         env_name_ph = env_name + '_' + str(i)
#     #         pi = 'data/' + env_name_ph + '/iterations/best_policy.pickle'
#     #         policy = pickle.load(open(pi, 'rb'))
#     #         e.env.env.advance(policy)

#     #     time_limit = e.env._max_episode_steps / e.env.env.specs_size()
#     #     print (f'episode_steps for each learning phase {time_limit}')
#     #     e.env.env.set_timelimit(time_limit)
#     #     e.env.env.eval(horizon, num_episodes, mode, discrete)
#     # else:
#         pi = 'data/' + env_name + '/iterations/best_policy.pickle'
#         policy = pickle.load(open(pi, 'rb'))
#         print (f'policy type = {type(policy.model)}')
#         total_score = 0.
#         f = 0
#         for ep in range(num_episodes):
#             o = e.reset()
#             d = False
#             t = 0
#             score = 0.0
#             while t < horizon and d == False:
#                 if render:
#                     e.render()
#                 a = policy.get_action(o, discrete=discrete)[0] \
#                     if mode == 'exploration' else policy.get_action(o, discrete=discrete)[1]['evaluation']
#                 o, r, d, _ = e.step(a)
#                 t = t+1
#                 score = score + r
#             print("Episode score = %f" % score)
#             total_score += score
#             f = f + 1 if score != 500 else f
#         print (f'averaged score: {total_score / num_episodes}')
#         print (f'succ rate {(num_episodes - f) / num_episodes}')
#     del(e)

def interpret_policy(env_name, phaselearning=False):
    if phaselearning:
        e = GymEnv(gym.make(env_name).gym_env)
        for i in range(e.env.env.specs_size()):
            env_name_ph = env_name + '_' + str(i)
            pi = 'data/' + env_name_ph + '/iterations/best_policy.pickle'
            policy = pickle.load(open(pi, 'rb'))
            print (policy.model.interpret())
            print (f'---> {e.env.env.specs[i]}')
        del(e)
    else:
        pi = 'data/' + env_name + '/iterations/best_policy.pickle'
        policy = pickle.load(open(pi, 'rb'))
        print (policy.model.interpret())

def save_policy(env_name, phaselearning=False):
    if phaselearning:
        e = GymEnv(gym.make(env_name).gym_env)
        for i in range(e.env.env.specs_size()):
            env_name_ph = env_name + '_' + str(i)
            pi = 'data/' + env_name_ph + '/iterations/best_policy.pickle'
            policy = pickle.load(open(pi, 'rb'))
            print (policy.model.interpret())
            print (f'---> {e.env.env.specs[i]}')
        del(e)
    else:
        pi = 'data/' + env_name + '/iterations/best_policy.pickle'
        policy = pickle.load(open(pi, 'rb'))
        policy.model.save_model()

if __name__ == '__main__':
    start = timer()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',  action = 'store', dest = 'env_name',  default = 'Pendulum')
    parser.add_argument('--eval', action = 'store_true', dest = 'eval')
    parser.set_defaults(eval=False)
    parser.add_argument('--interpret', action = 'store_true', dest = 'interpret')
    parser.set_defaults(interpret=False)
    parser.add_argument('--save_model', action = 'store_true', dest = 'save_model')
    parser.set_defaults(sace_model=False)
    parser.add_argument('--seed',  action = 'store', dest = 'seed', type=int, default = 0)
    args = parser.parse_args()

    seed = args.seed
    print(f"seed is {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark = args.env_name
    eval = args.eval
    interpret = args.interpret
    save_model = args.save_model

    # Default is that we do not have phase-based training
    phaselearning = False
    phase_trainiter_inc = 0

    # Experiments for now
    if benchmark in ['Pendulum', 'Pendulum-v0', 'InvertedPendulum-v2', 'Swimmer-v3']:
        prog_type = 'ITELinear'
        trainsteps = 50
    elif benchmark in ['Acc', 'Bicycle', 'Hopper-v2', 'HalfCheetah-v2', 'BipedalWalker-v3', 'Acc2']:
        # For Bicycle (obstacle), need at least 300 iterations to converge to the target. Better to be 500.
        prog_type = 'ITELinear'
        trainsteps = 300
    elif benchmark in ['CartPole', 'MountainCarContinuous-v0', 'MountainCar', 'Acrobot', "MountainCarSpeed"]:
        prog_type = 'ITEConstant'
        trainsteps = 40
    elif benchmark in ['CarRetrieval']:
        prog_type = 'NestITEConstant'
        trainsteps = 150
    elif benchmark in ['Quad', 'QuadFull', 'QuadTest', 'QuadFullTest']:
        prog_type = 'MLP'
        trainsteps = 150
    elif benchmark in ['LunarLander', 'InvertedDoublePendulum-v2', 'Humanoid-v2', 'Ant-v2', 'Walker2d-v2', 'Reacher-v2']:
        prog_type = 'NestITELinear'
        trainsteps = 200
    elif benchmark in ['LunarLanderContinuous-v2']:
        prog_type = 'LunarLanderPD'
        trainsteps = 100
    elif benchmark in ['PendulumPID']:
        prog_type = 'PendulumPID'
        trainsteps = 100
    elif benchmark in ['Car2d2']:
        prog_type = 'ITELinear'
        trainsteps = 300
    elif benchmark in ['CarRacing','Car2d']:
        prog_type = 'Linear'
        trainsteps = 100 # each phase trained using 100 iterations.
        phaselearning = True
        phase_trainiter_inc = 20 # i-th phase trained 20 iterations more than i-1.
    elif benchmark in ['CarMaze']:
        prog_type = 'ITELinear'
        trainsteps = 100 # each phase trained using 100 iterations.
        phaselearning = True
        phase_trainiter_inc = 100 # i-th phase trained 100 iterations more than i-1.
    elif benchmark in ['CarFall', 'CarPush']: # Test on very simple environments.
        prog_type = 'NestITEConstant'
        trainsteps = 100 # each phase trained using 100 iterations.
        phaselearning = True
        phase_trainiter_inc = 100 # i-th phase trained 100 iterations more than i-1.
    elif benchmark in ["TORA", "TORAEq"]:
        prog_type = 'Linear'
        trainsteps = 200 # each phase trained using 100 iterations.
    elif benchmark in ["ReachNN1"]:
        prog_type = "Linear"
        trainsteps = 50
    elif benchmark in ["ReachNN2", "ReachNN3", "ReachNN4", "ReachNN5", "ReachNN6", "OS"]:
        prog_type = "Linear"
        trainsteps = 100
    elif benchmark in ["UnicycleCar"]:
        prog_type = "ITELinear"
        trainsteps = 100
    elif benchmark in ["AccCAV", "AccCMP"]:
        prog_type = "ITELinear"
        trainsteps = 350
    elif benchmark in ["PP"]:
        prog_type = "Linear"
        trainsteps = 150
    elif benchmark in ["QMPC"]:
        prog_type = "Nest2ITEConstant"
        trainsteps = 260
    else:
        assert False

    if eval:
        visualize_policy(benchmark, num_episodes=5, mode='evaluation', discrete=False)
    elif interpret:
        interpret_policy(benchmark, phaselearning=phaselearning)
    elif save_model:
        save_policy(benchmark, phaselearning=phaselearning)
    else:
        train(benchmark, prog_type, trainsteps, seed, phaselearning=phaselearning, phase_trainiter_inc=phase_trainiter_inc)
        save_policy(benchmark, phaselearning=phaselearning)

    print(f'Total Cost Time: {timer() - start}s')
    pass
