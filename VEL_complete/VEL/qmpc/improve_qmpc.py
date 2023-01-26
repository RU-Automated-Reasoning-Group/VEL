import time
import sys
import numpy as np
import subprocess
sys.path.append('../')
import improve_lib

def parse_output(output, score):
    print(output)
    lst = output.split()
    nums = [float(x) for x in lst]
    assert len(nums[:-1]) == 12
    return score + nums[-1], nums[:-1]

def append_range(cmd, ranges):
    for x in ranges:
        cmd.append(str(x))
    return cmd

def eval_controller(controller):
    score = 0.0
    cmd1 = ['./qmpc1']
    cmd2 = ['./qmpc2']
    cmd3 = ['./qmpc3']
    cmd4 = ['./qmpc4']
    for i in range(0, controller.size):
        cmd1.append(str(controller[i]))
        cmd2.append(str(controller[i]))
        cmd3.append(str(controller[i]))
        cmd4.append(str(controller[i]))
    # print(cmd1)
    x1 = subprocess.run(cmd1, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    score, ranges = parse_output(x1, score)

    cmd2 = append_range(cmd2, ranges)
    # print(cmd2)
    assert len(cmd2) == 46
    x2 = subprocess.run(cmd2, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    score, ranges = parse_output(x2, score)

    cmd3 = append_range(cmd3, ranges)
    # print(cmd3)
    assert len(cmd3) == 46
    x3 = subprocess.run(cmd3, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    score, ranges = parse_output(x3, score)

    cmd4 = append_range(cmd4, ranges)
    # print(cmd4)
    assert len(cmd4) == 46
    x4 = subprocess.run(cmd4, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    score, ranges = parse_output(x4, score)

    print(score)
    return score

def run(arg, params):
    loss_list = []
    # theta = np.array([-0.44438407, -0.7704203, 0.04734802, 0.3202033, -2.2925243, 0.6780341, -2.2581437, 0.75513357, -2.333691, 4.009605, 0.52152324, -0.9112037, -0.15706797, -0.8776282, -0.02192774, -1.3992957, 2.551787, 0.02871716, 0.39486897, 11.796398, 1.6516205, 0.23113677, -0.33589906, 1.456676, -0.60098267, 1.141922, -0.19662783, -7.04549, -2.4876144, -0.31968763, 2.3510857, 0.8998343, 1.1119993]) # (seed 3)
    # theta = np.array([0.30794907, 1.6587877, 1.3479837, 0.35834038, 2.455599, 0.27555808, 2.6456132, -0.05013026, 0.30456668, 10.017926, 3.6687436, -1.2528447, -0.8557977, 2.9649246, 0.70612866, -0.478837, 0.14810641, -2.2453845, -6.317108, 9.361775, 0.01503912, -0.4177037, 1.4912254, 0.08106456, -1.0031562, 1.7733785, -1.6531662, 2.0150056, -0.7043396, -0.75051916, 5.787931, -2.680328, 3.4165192]) # (seed 5)
    # theta = np.array([0.17180209, 0.05602094, -1.1524299, 0.92702305, -1.1008363, -2.5272465, 0.755777, 0.11010686, 0.01299657, 12.990218, -0.46194226, -1.0311759, 0.8706603, -0.672688, -1.5178447, 0.17660241, -0.84166366, -0.989007, -1.8924142, 1.2729815, -0.58881086, 1.2469705, -0.39482218, -2.196647, 1.2465792, -0.12379479, 0.965148, 1.3669583, 1.0825276, 4.377455, -2.3166552, 0.13216619, 1.1930784]) # (seed 7)
    theta = params.copy()
    if str(arg) == "--eval":
        eval_controller(theta)
        exit()
    for t in range(0, 2000):
        current = time.time()
        new_theta, loss = \
            improve_lib.true_ars_combined_direction(
                theta.copy(),
                alpha_in=0.0005,
                N_of_directions_in=3,
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

