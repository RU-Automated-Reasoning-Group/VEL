import time
import sys
import numpy as np
import subprocess
sys.path.append('../')
import improve_lib

def eval_controller(controller):
    cmd1 = ['./uc']
    for i in range(0, controller.size):
        cmd1.append(str(controller[i]))
    # print(cmd1) 
    x1 = subprocess.run(cmd1, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    # print(x1)
    return float(x1)

def run(params):
    loss_list = []
    # theta = np.array([1.2177143, 1.4056991, 0.31899714, 1.2948294, -1.2930791, 1.6057166, -1.5763961, -1.5473098, -1.1416376, 0.39961806, 0.44079867, -0.8414853, -0.06218815, -2.590983, 1.5875443, -1.1842718, 0.57684004, 0.42069614, -2.100871, -0.11231647, -0.10139762, -3.5183306, -0.48945636, 0.7031986, -0.47678217]) # (seed 1)
    # theta = np.array([-0.14752883, -0.6970784, 0.7970959, -0.13079157, -2.3920295, 0.9528393, -1.2708454, -0.8969389, -0.5588428, -0.05089822, -0.94710463, -1.4459893, -1.5178332, 0.2845582, -0.6163721, 0.53212786, -0.4184079, -1.2929893, -1.8578717, 1.8747214, -1.5779335, -3.3363802, 2.5128865, -0.02798443, 0.06561681]) # (seed 3)
    theta = params.copy()
    for t in range(0, 2000):
        current = time.time()
        new_theta, loss = \
            improve_lib.true_ars_combined_direction(
                theta.copy(),
                alpha_in=0.002,
                N_of_directions_in=6,
                b_in=2,
                noise_in=0.001,
                eval_controller=eval_controller
                )
        # current = time.time()
        print(f'----- iteration {t} -------------')
        print("old theta", theta)
        print("updated theta", new_theta)
        loss_list.append(loss)
        print("loss", loss)
        print("loss list", loss_list)
        if loss == 0:
            print("loss reaches 0")
            print("theta", theta)
            print("time", time.time() - current)
            print("saving loss list to loss.txt")
            with open("loss.txt", "a") as file:
                loss_str = [str(x) for x in loss_list]
                loss_str = " ".join(loss_str)
                file.write(loss_str)
                file.write('\n')
            exit(0)

        theta = new_theta
        print("time", time.time() - current)
        print('----------------------------')

if __name__ == "__main__":
    model_file = str(sys.argv[1])
    file = open(model_file, 'r')
    lines = file.readlines()
    params = lines[0].split()
    params = [float(x) for x in params]
    params = np.array(params)
    file.close()
    run(params)

