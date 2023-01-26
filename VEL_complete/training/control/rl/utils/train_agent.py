import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from control.rl.utils.make_train_plots import make_train_plots
from control.rl.utils.gym_env import GymEnv
from control.rl.samplers.core import sample_paths
import numpy as np
import dill as pickle
import time as timer
import os
import copy
import collections

def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                out_dir = '',
                corrector=None,
                lam_corr=None
                ):
    job_name = out_dir + job_name
    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    mean_pol_perf = 0.0
    e = GymEnv(agent.env.env_id) #fixme1.
    best_policy_at = -1

    perfs = collections.deque(25*[1], 25)

    for i in range(niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)

        curr_policy = copy.deepcopy(agent.policy)
        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu, corrector=corrector, lam_corr=lam_corr) #env=agent.env)#fixme2.
        stats = agent.train_step(**args)
        train_curve[i] = stats[0]

        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=1,
                                      env=e.env_id, #env=agent.env, #env=e.env_id, fixme3.
                                      eval_mode=True, base_seed=seed, corrector=corrector, lam_corr=lam_corr)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)

        if i % save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_pol_perf, best_perf))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

        if i >= 1 and train_curve[i] >= best_perf:
            best_policy = copy.deepcopy(curr_policy)
            best_perf = train_curve[i]
            best_policy_at = i-1 # fixme4. The final (last) policy is not evaluated.

        # Check if traning has converged.
        if i >= 1:
            perfs.append((train_curve[i] - train_curve[i-1]) / abs(train_curve[i-1]))
        # if i > 25:
        #     converged = (all(x < 0.01 for x in perfs))
        #     if converged:
        #         print (f"Training Converged because there is no improvement during the last 25 training iterations.")
        #         break

    if niter == 1:
        best_policy = copy.deepcopy(agent.policy)
        # Don't have a way evaluate its performance.
        best_policy_at = i

    # final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)
    print (f"best_policy_at {best_policy_at} with performance {best_perf}")
    return best_policy_at, best_perf
