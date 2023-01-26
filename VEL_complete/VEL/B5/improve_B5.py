import time
import sys
import numpy as np
import subprocess
sys.path.append('../')
import improve_lib

def eval_controller(controller):
    cmd1 = ['./B5']
    for i in range(0, controller.size):
        cmd1.append(str(controller[i]))
    print(cmd1) 
    x1 = subprocess.run(cmd1, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(x1)
    return float(x1)

def run(params):
    loss_list = []
    # theta = np.array([19.142782, -32.163837, -4.473618, 9.800558])
    # theta = np.array([19.52251, -32.821556, -4.4587007, 9.823288])
    # theta = np.array([18.761831, -31.51923, -4.37203, 9.537802])
    theta = params.copy()
    for t in range(0, 2000):
        current = time.time()
        new_theta, loss = \
            improve_lib.true_ars_combined_direction(
                theta.copy(),
                alpha_in=0.05,
                N_of_directions_in=3,
                b_in=2,
                noise_in=0.1,
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
            print("theta", new_theta)
            print("time", time.time() - current)
            print("saving loss list to loss.txt")
            with open("loss.txt", "a") as file:
                loss_str = [str(x) for x in loss_list]
                loss_str = " ".join(loss_str)
                file.write(loss_str)
                file.write('\n')
            exit(0)

        theta = new_theta
        print("theta", theta)
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

