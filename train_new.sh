benchmark=$1
seed=$2
python3="python3.7"
if [ "$benchmark" = "B1" ]; then
    cd training && $python3 -m control.synthesize --env ReachNN1 --seed $seed && cp output.model ../VEL/B1 && rm output.model && cd ..
elif [ "$benchmark" = "B2" ]; then
    cd training && $python3 -m control.synthesize --env ReachNN2 --seed $seed && cp output.model ../VEL/B2 && rm output.model && cd ..
elif [ "$benchmark" = "B3" ]; then
    cd training && $python3 -m control.synthesize --env ReachNN3 --seed $seed && cp output.model ../VEL/B3 && rm output.model && cd ..
elif [ "$benchmark" = "B4" ]; then
    cd training && $python3 -m control.synthesize --env ReachNN4 --seed $seed && cp output.model ../VEL/B4 && rm output.model && cd ..
elif [ "$benchmark" = "B5" ]; then
    cd training && $python3 -m control.synthesize --env ReachNN5 --seed $seed && cp output.model ../VEL/B5 && rm output.model && cd ..
elif [ "$benchmark" = "tora" ]; then
    cd training && $python3 -m control.synthesize --env ReachNN6 --seed $seed && cp output.model ../VEL/tora && rm output.model && cd ..
elif [ "$benchmark" = "tora_inf" ]; then
    cd training && $python3 -m control.synthesize --env TORQEq --seed $seed && cp output.model ../VEL/tora_inf && rm output.model && cd ..
elif [ "$benchmark" = "oscillator" ]; then
    cd training && $python3 -m control.synthesize --env OS --seed $seed && cp output.model ../VEL/oscillator && rm output.model && cd ..
elif [ "$benchmark" = "ACC" ]; then
    cd training && $python3 -m control.synthesize --env AccCMP --seed $seed && cp output.model ../VEL/ACC && rm output.model && cd ..
elif [ "$benchmark" = "MountainCar" ]; then
    cd training && $python3 -m control.synthesize --env MountainCar --seed $seed && cp output.model ../VEL/MountainCar && rm output.model && cd ..
elif [ "$benchmark" = "qmpc" ]; then
    cd training && $python3 -m control.synthesize --env QMPC --seed $seed && cp output.model ../VEL/qmpc && rm output.model && cd ..
elif [ "$benchmark" = "cartpole" ]; then
    cd training && $python3 -m control.synthesize --env CartPole --seed $seed && cp output.model ../VEL/cartpole && rm output.model && cd ..
elif [ "$benchmark" = "unicycle_car" ]; then
    cd training && $python3 -m control.synthesize --env UnicycleCar --seed $seed && cp output.model ../VEL/unicycle_car && rm output.model && cd ..
elif [ "$benchmark" = "pendulum" ]; then
    cd training && $python3 -m control.synthesize --env PP --seed $seed && cp output.model ../VEL/pendulum && rm output.model && cd ..
fi

