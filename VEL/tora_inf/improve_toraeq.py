import time
import sys
import numpy as np
import subprocess
sys.path.append('../')
import improve_lib

def eval_controller(controller):
    cmd1 = ['./toraeq']
    for i in range(0, controller.size):
        cmd1.append(str(controller[i]))
    # print(cmd1)
    interval_1 = [str(-0.1), str(0.0), str(-0.1), str(0.0)]
    interval_2 = [str(-0.1), str(0.0), str(0.0), str(0.1)]
    interval_3 = [str(0.0), str(0.1), str(-0.1), str(0.0)]
    interval_4 = [str(0.0), str(0.1), str(0.0), str(0.1)]
    # print(cmd1 + interval_1)
    x1 = subprocess.run(cmd1 + interval_1, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x2 = subprocess.run(cmd1 + interval_2, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x3 = subprocess.run(cmd1 + interval_3, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x4 = subprocess.run(cmd1 + interval_4, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')    
    print(x1, x2, x3, x4)
    return float(x1) + float(x2) + float(x3) + float(x4)

def run(arg, params):
    loss_list = []
    # theta = np.array([-0.42375514,   2.0062392,  -11.024776,   -12.117741, -0.00642696]) # seed 0
    # theta = np.array([-0.09830257,   1.3307514,  -10.342365,   -12.138152, -0.00515717]) # seed 1
    # theta = np.array([1.5908277,   1.368971,  -12.437533,  -16.894487, -0.09863488]) # seed 2
    theta = params.copy()
    if str(arg) == "--eval":
        eval_loss = eval_controller(theta)
        print("loss", eval_loss)
        exit()
    # print(theta.size)
    # exit()
    for t in range(0, 2000):
        current = time.time()
        new_theta, loss = \
            improve_lib.true_ars_combined_direction(
                theta.copy(),
                alpha_in=0.1,
                N_of_directions_in=3,
                b_in=2,
                noise_in=0.15,
                eval_controller=eval_controller
            )

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
    import sys
    task = str(sys.argv[1])
    model_file = str(sys.argv[2])
    file = open(model_file, 'r')
    lines = file.readlines()
    params = lines[0].split()
    params = [float(x) for x in params]
    params = np.array(params)
    file.close()
    run(task, params)

