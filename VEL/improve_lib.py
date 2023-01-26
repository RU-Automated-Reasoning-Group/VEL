from random import random
from unittest import result
import numpy as np
import time
import subprocess
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import multiprocess as mp

def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)
    pool.close()
    pool.terminate()
    pool.join()
    return results

def calculate_rewards_both_direction(controller, random_directions, noise, eval_controller):
    N_of_directions = len(random_directions)
    input_dict_list = []

    # positive sign
    for d in random_directions:
        input_dict_list.append({"controller": controller + noise * d})

    # negative sign
    for d in random_directions:
        input_dict_list.append({"controller": controller - noise * d})

    # current controller
    input_dict_list.append({"controller": controller})
    results = _try_multiprocess(eval_controller, input_dict_list, 70, 60000, 60000)
    return results[0: N_of_directions], results[N_of_directions:2*N_of_directions], results[2*N_of_directions]

def true_ars_combined_direction(controller, alpha_in, N_of_directions_in, b_in, noise_in, eval_controller):
    alpha = alpha_in
    N_of_directions = N_of_directions_in
    b = b_in
    noise = noise_in

    random_directions = [np.random.randn(controller.size) for _ in range(N_of_directions)]
    positive_rewards, negative_rewards, loss = calculate_rewards_both_direction(controller, random_directions, noise, eval_controller)
    assert len(positive_rewards) == N_of_directions
    assert len(negative_rewards) == N_of_directions

    all_rewards = np.array(positive_rewards + negative_rewards)
    reward_sigma = np.std(all_rewards)

    min_rewards = {k: min(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
    order_of_directions = sorted(min_rewards.keys(), key=lambda x:min_rewards[x])
    print(order_of_directions)
    rollouts = [(positive_rewards[k], negative_rewards[k], random_directions[k]) for k in order_of_directions]
    print([(positive_rewards[k], negative_rewards[k]) for k in order_of_directions])
    # update controller parameters
    update_step = np.zeros(shape=controller.shape)
    for positive_reward, negative_reward, direction in rollouts[:b]:
        update_step = update_step + (positive_reward - negative_reward) * direction
    controller -= alpha / (b * reward_sigma) * update_step
    return controller, loss

def local_search(controller, num_of_samples_in, error_in, eval_controller):
    input_dict_list = []
    num_of_samples = num_of_samples_in
    v = error_in
    ws = [np.random.normal(size=controller.size) for _ in range(0, num_of_samples)]
    for i in range(0, num_of_samples):
        input_dict_list.append({"controller": controller + v * ws[i]})
    input_dict_list.append({"controller": controller})
    results = _try_multiprocess(eval_controller, input_dict_list, 70, 60000, 60000)
    loss = results[-1]
    idx = 0
    min_loss = results[0]
    for i in range(1, len(results) - 1):
        if results[i] < min_loss:
            min_loss = results[i]
            idx = i
    print("loss", loss)
    print("min_loss", min_loss)
    print(input_dict_list[idx]["controller"])
    return input_dict_list[idx]["controller"], loss