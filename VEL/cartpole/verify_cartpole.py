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

def eval_controller(controller, interval):
    cmd = ["./cartpole"]
    controller = [str(x) for x in controller]
    interval = [str(x) for x in interval]
    final_cmd = cmd + controller + interval
    # print(final_cmd)
    x = subprocess.run(final_cmd, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return float(x)

def eval_all(params):
    controller = params.copy()
    # controller = [-0.16291744, -0.89586353, 10.1553135, -0.14661914, 1.3754923, 1.998532, -7.5645523] # seed 1
    #controller = [-0.04350144, -0.5160665,   9.241911,   -0.19057913, 1.0459521, 2.1423945, -6.2977405] # seed 6
    # controller = [-0.44363075, -0.53502434, 7.2539496, 0.27114174, 0.17272002, 3.570098, -4.249988] # seed 7
    x3_range = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    x4_range = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]

    input_dict_list = []
    for x3_start in x3_range:
        for x4_start in x4_range:
            interval = [x3_start, x3_start + 0.01, x4_start, x4_start + 0.01]
            input_dict_list.append({"controller": controller, "interval": interval})

    results = _try_multiprocess(eval_controller, input_dict_list, 70, 60000, 60000)
    total_loss = 0.0
    for idx, loss in enumerate(results):
        # print(input_dict_list[idx]["interval"], loss)
        total_loss += loss
        if loss > 0:
            print("unsafe for this interval")
    
    if total_loss == 0.0:
        print("verification success")
            

if __name__ == "__main__":
    import sys
    model_file = str(sys.argv[1])
    file = open(model_file, 'r')
    lines = file.readlines()
    params = lines[0].split()
    params = [float(x) for x in params]
    params = np.array(params)
    file.close()
    eval_all(params)
