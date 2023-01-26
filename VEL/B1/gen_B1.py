import imp
import sys
sys.path.append('../')
from gen_plot import gen_plot_mean_and_std

# log1 = [0.0679332, 0.0128328, 0.0]
# log2 = [0.0679332, 0.0114204, 0.0]
# log3 = [0.0679332, 0.0101859, 0.0]
# log4 = [0.0679332, 0.00824886, 0.0]
# log5 = [0.0679332, 0.0185038, 0.0]

# all_logs = [log1, log2, log3, log4, log5]

# read all logs
all_logs = []
with open("loss.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        log = line.split()
        log = [float(x) for x in log]
        all_logs.append(log)

gen_plot_mean_and_std(all_logs, "B1.png", "B1")