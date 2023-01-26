import matplotlib.pyplot as plt
import numpy as np

def gen_plot_mean_and_std(training_logs, output_name, benchmark_name):
    len_list = [len(log) for log in training_logs]
    max_len = max(len_list)

    # fix the length of each log to the longest one by appending zeros
    for idx, log in enumerate(training_logs):
        for _ in range(0, max_len - len_list[idx]):
            log.append(0.0)
    print(training_logs)

    # calculcate mean and std curve
    vals_at_each_timestep = list(zip(*training_logs))
    print(vals_at_each_timestep)
    points = np.array([np.mean(np.array(vals)) for vals in vals_at_each_timestep])
    stds = np.array([np.std(np.array(vals)) for vals in vals_at_each_timestep])
    assert points.size == max_len
    assert stds.size == max_len

    # draw on plot
    x = np.arange(0, max_len)
    plt.plot(x, points,'k', color='#CC4F1B')
    plt.fill_between(x, points + stds, points - stds,  alpha=0.5, facecolor='#FF9848')

    # x, y labels and title
    plt.xlabel("training iterations")
    plt.ylabel("abstract loss")
    plt.title(benchmark_name)
    plt.savefig(output_name)
    

