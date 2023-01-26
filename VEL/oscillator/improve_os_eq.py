import time
import sys
import numpy as np
import subprocess
sys.path.append('../')
import improve_lib

def eval_controller(controller):
    cmd1 = ['./os']
    cmd2 = ['./oseq']
    for i in range(0, controller.size):
        cmd1.append(str(controller[i]))
        cmd2.append(str(controller[i]))
    print(cmd1)
    print(cmd2)
    x1 = subprocess.run(cmd1, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x2 = subprocess.run(cmd2, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(x1, x2)
    return float(x1) + float(x2)

def run(arg, params):
    loss_list = []
    theta = params.copy()
    # theta = np.array([-10.780834,   -8.6560955, -2.725309])
    if str(arg) == "--eval":
        eval_controller(theta)
        exit()
    for t in range(0, 2000):
        current = time.time()
        new_theta, loss = \
            improve_lib.true_ars_combined_direction(
                theta.copy(), 
                alpha_in=0.1,
                N_of_directions_in=3, 
                b_in=2, 
                noise_in=0.1, 
                eval_controller=eval_controller)

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

