import sys
import os
import time

table = []

def measure_VT(benchmark, iters):
    t1 = time.time()
    os.system(f"cd {benchmark} && bash verify.sh {iters} && cd ..")
    t2 = time.time()
    table.append(f"{benchmark} V.T. average {(t2 - t1) / iters}s\n")

def measure_TT(benchmark, iters):
    t1 = time.time()
    os.system(f"cd {benchmark} && bash vel.sh {iters} && cd ..")
    t2 = time.time()
    table.append(f"{benchmark} T.T. average {(t2 - t1) / iters}s\n")

def generate_table(iters):
    python3 = "python3.7"

    # B1
    measure_VT("B1", iters)
    measure_TT("B1", iters)

    # B2
    measure_VT("B2", iters)

    # B3
    measure_VT("B3", iters)

    # B4
    measure_VT("B4", iters)

    # B5
    measure_VT("B5", iters)
    measure_TT("B5", iters)

    # Oscillator_inf
    measure_VT("oscillator", iters)
    measure_TT("oscillator", iters)

    # Tora
    measure_VT("tora", iters)

    # tora_inf
    measure_VT("tora_inf", iters)
    measure_TT("tora_inf", iters)

    # unicycle
    measure_VT("unicycle_car", iters)
    measure_TT("unicycle_car", iters)

    # CartPole
    measure_VT("cartpole", iters)

    # pendulum
    measure_VT("pendulum", iters)

    # qmpc
    measure_VT("qmpc", iters)
    measure_TT("qmpc", iters)

    # ACC
    measure_VT("ACC", iters)
    measure_TT("ACC", iters)

    # MountainCar
    measure_VT("MountainCar", iters)

    # print table
    with open("figures/table.txt", "w") as f:
        f.writelines(table)

if __name__ == "__main__":
    iters = int(sys.argv[1])
    generate_table(iters)