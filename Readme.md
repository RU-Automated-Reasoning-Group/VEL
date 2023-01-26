# Introduction

We implement the verification-guided learning framework 
VEL in this artifact. This artifact consists of two parts.
The first part is the code for programmatic reinforcement learning
algorithm from (Qiu and Zhu) to train programmatic controllers with high rewards. The second part is our implementation of the VEL framework.
Trained controllers are used for verification and 
verification-guided learning by the VEL code. All results 
presented in the paper, including both the figures and the table, 
can be reproduced using the provided scripts.

# Installation

The following instructions are tested on Ubuntu 22.04 with Python3.7. However, the installation process for other versions should be similar.

1. Use the following command to install dependencies for flow*

```
sudo apt install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc g++ libopenmpi-dev
```

2. Compile flow* and verification code
```
bash setup.sh
```

3. Use the following command to install required python packages
```
python3.7 -m pip install -r requirements.txt
```


# Approximated Experiment Runtime

The computation needed by our tool is CPU intensive and we use multiple threads to 
evaluate various perturbed controllers simultaneously. 

The verification/training time reported in the paper is obtained by running our 
tool on a desktop computer with Intel Core I7-12700K processor. Running our tool
in a virtual machine (setting the number of CPU to 8) will produce results
that are very similar to the results reported in the paper.

Here, we provided estimated experiment runtime on a laptop with weaker performance.
The following statistics are obtained by running our tool in a virtual machine
(setting the number of CPU to 4) on a MacBook Pro with Quad-core Intel Core i5 processor.


|      Task      |  V.T  |  T.T  |
| :----:        |    :----:   |          :----: |
|       B1       |  58s  | 4m55s |
|       B2       | 0.75s |   -   |
|       B3       |  5.8s |   -   |
|       B4       |  1.4s |   -   |
|       B5       |  1.1s |  5.3s |
| Oscillator_inf |  3.3s | 1m31s |
|      Tora      |  7.8s |   -   |
|    Tora_inf    |  1.6s | 8m52s |
|    Unicycle    |  1.6s | 1m14s |
|    CartPole    | 1m24s |   -   |
|    Pendulum    |   2s  |   -   |
|      QMPC      |  4.2s | 1m22s |
|       ACC      |  5.8s |  20m  |
|   MountainCar  |  25m  |   -   |

# Reproducibility Instructions

Note: the line start with a "\$" sign means a command for linux terminal
      but the "\$" sign itself is not part of the command

1.
  - First, we introduce the structure of the implementation. The Code for each benchmark 
    are in a separated folder. In a folder, a filed called output.model contains 
    a trained controller. The verify.sh script is used to measure the verification 
    time (V.T) for the controller in output.model. This script requires an integer i as input
    and will run the verification code for i times. The vel.sh script 
    (if exists) is used to measure the verification-guided training time (T.T) for the 
    controller in output.model. This script requires an integer as input and will run the VEL
    training code for i times. The random seed used in each training process is 
    different thus all i runs of verification-guided training are different.

    The output of the verify.sh script contains the abstract symbolic rollout loss.
    The output of the vel.sh script contains the controllers verified for each iteration,
    current abstract symbolic rollout loss, and the loss history.

  - To reproduce the result for a specific benchmark, please run the following commands:
  ```
    $ cd VEL/<benchmark-name>/
    $ bash verify.sh 1 # to measure the verification time for 1 time
    $ bash vel.sh 1 # (if this script exists) to measure the training time for 1 time
  ```

  - Available benchmark names are: ACC, B1, B2, B3, B4, B5, cartpole, oscillator, pendulum, 
    qmpc, tora, tora_inf, unicycle_car, MountainCar

2.
  - To reproduce all the figures in the paper, please run the following commands
  ```
    $ cd VEL/
    $ bash figure.sh 5
  ```

  After running this script, all figures will be generated in the figures/ folder.
  5 in the commands is a integer parameter that can be adjusted, indicating the number of times
  of running each benchmark. Note: Running each benchmark for 5 times can be time consuming.
  You may consider changing 5 to 1 for evaluation.

  - To reproduce the table in the paper, the verification/training time can be 
    collected by the following commands:
  ```
    $ cd VEL/
    $ python3.7 table.py 
  ```

    After running this script, a table.txt file containing verification/training
    time for all the benchmarks will be generated. 5 in the commands is a parameter that
    can be adjusted, indicating the number of times of running each benchmark.
    Note: Running each benchmark for 5 times may be time consuming. You may consider changing 
    5 to 1 for evaluation.

3. 
  - To train a new controller for a benchmark, run the following commands
  ```
    $ bash train_new.sh <benchmark-name> <seed>
  ```

  - Available benchmark names are: ACC, B1, B2, B3, B4, B5, cartpole, oscillator, pendulum, 
    qmpc, tora, tora_inf, unicycle_car, MountainCar
   
  - The effect of the above script is to train a new controller for the input benchmark and 
    update the output.model file for this benchmark with this new controller.
