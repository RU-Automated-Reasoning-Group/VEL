import imp
import sys
sys.path.append('../')
from gen_plot import gen_plot_mean_and_std

# log1 = [0.084802, 0.049773, 0.040035, 0.028711, 0.013352, 0.0]
# log2 = [0.084802, 0.074739, 0.046279, 0.029943, 0.021532, 0.012561, 0.005213, 0.0]
# log3 = [0.084802, 0.056384, 0.040434, 0.024075, 0.013358, 0.005133, 0.0]
# log4 = [0.084802, 0.06695, 0.054942, 0.032842, 0.025288, 0.006986, 0.0]
# log5 = [0.084802, 0.07432, 0.062313, 0.039221, 0.030148, 0.020489, 0.009746, 0.0]

# all_logs = [log1, log2, log3, log4, log5]

# read all logs
all_logs = []
with open("loss.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        log = line.split()
        log = [float(x) for x in log]
        all_logs.append(log)

gen_plot_mean_and_std(all_logs, "QMPC.png", "QMPC")