import time
import sys
import numpy as np
import subprocess
sys.path.append('../')
import improve_lib

def eval_controller(controller):
    cmd1 = ['./acc1']
    cmd2 = ['./acc2']
    cmd3 = ['./acc3']
    cmd4 = ['./acc4']
    for i in range(0, controller.size):
        cmd1.append(str(controller[i]))
        cmd2.append(str(controller[i]))
        cmd3.append(str(controller[i]))
        cmd4.append(str(controller[i]))
    # print(cmd1) 
    # print(cmd2) 
    # print(cmd3) 
    # print(cmd4) 
    x1 = subprocess.run(cmd1, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x2 = subprocess.run(cmd2, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x3 = subprocess.run(cmd3, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    x4 = subprocess.run(cmd4, shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(x1, x2, x3, x4)
    return (float(x1) + float(x2) + float(x3) + float(x4))

def true_ars_combined_direction(controller, alpha_in, N_of_directions_in, b_in, noise_in, eval_controller):
    alpha = alpha_in
    N_of_directions = N_of_directions_in
    b = b_in
    noise = noise_in

    random_directions = [np.random.randn(controller.size) for _ in range(N_of_directions)]
    positive_rewards, negative_rewards, loss = improve_lib.calculate_rewards_both_direction(controller, random_directions, noise, eval_controller)
    assert len(positive_rewards) == N_of_directions
    assert len(negative_rewards) == N_of_directions

    all_rewards = np.array(positive_rewards + negative_rewards)
    reward_sigma = np.std(all_rewards)

    min_rewards = {k: r_pos + r_neg for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
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

def run(arg, params):
    loss_list = []
    # theta = np.array([0.29538625, -0.33441657, 0.36445197, 0.28615144, 0.3757519, -0.34288293, -0.03436325, 0.22519949, -0.68176216, 0.21952716, -0.77294517, 0.06661368, -0.38244617, 0.14251709, 0.192307, 0.07095274, 0.43015411, 0.13430597])
    # theta = np.array([0.29538625, -0.33441657, 0.36445194, 0.2861514, 0.3757519, -0.34288293, 0.5082113, 0.250532, -1.0820371, 0.17685509, -0.8418154, 0.0847057, -0.38244617, 0.14251709, 0.192307, 0.07095277, 0.43015414, 0.13430595])
    # theta = np.array([0.23043269, -0.19739035, -0.08669749,  0.20990819, -0.42102337, 0.2682017, 0.01496948,  0.2324709,  -0.1552192,   0.04600073, -0.9487125, 0.02561859,  0.16333503, -0.1742796,  -0.0326058,  -0.04026145,  0.06482089, -0.00178653]) # seed 1
    # theta = np.array([-0.32843375, -0.01270519, 0.44195992, -0.39221716, 0.05990371, 0.17267613, 0.09753825, -0.33125156, 0.10731802, 0.4052707, -0.23098232, 0.25341126, -0.08947261, -0.02754292, -0.48896623, 0.1830583, -0.7427269, 0.23636526])
    # theta = np.array([0.13407847, 0.05662118, -0.67476386, -0.3257277, -0.10364221, -0.40967187, 0.3930277, 0.36726743, 0.07112581, 0.373062, -0.20502019, -0.2566596, -0.657944, -0.07218181, 0.6311317, 0.01073453, -0.08467028, -0.2907107])
    theta = params.copy()
    if str(arg) == "--eval":
        print(theta)
        eval_loss = eval_controller(theta)
        print("loss", eval_loss)
        exit()
    
    for t in range(0, 2000):
        current = time.time()
        new_theta, loss = \
            true_ars_combined_direction(
                theta.copy(),
                alpha_in=0.005,
                N_of_directions_in=16,
                b_in=2,
                noise_in=0.001,
                eval_controller=eval_controller
                )
        # current = time.time()
        print(f'----- iteration {t} -------------')
        loss_list.append(loss)
        print("old theta", theta)
        print("updated theta", new_theta)
        # loss_list.append(loss)
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

