python3="python3.7"
iters=$1

echo "generate figure for B1"
cd B1 && bash vel.sh $iters && $python3 gen_B1.py && cp B1.png ../figures && cd ..

echo "generate figure for B5"
cd B5 && bash vel.sh $iters && $python3 gen_B5.py && cp B5.png ../figures && cd ..

echo "generate figure for oscillator"
cd oscillator && bash vel.sh $iters && $python3 gen_os_eq.py && cp oseq.png ../figures && cd ..

echo "generate figure for QMPC"
cd qmpc && bash vel.sh $iters && $python3 gen_qmpc.py && cp QMPC.png ../figures && cd ..

echo "generate figure for tora_inf"
cd tora_inf && bash vel.sh $iters && $python3 gen_toraeq.py && cp toraeq.png ../figures && cd ..

echo "generate figure for unicycle_car"
cd unicycle_car && bash vel.sh $iters && $python3 gen_unicycle_car.py && cp unicycle.png ../figures && cd ..

echo "generate figure for ACC"
cd ACC && bash vel.sh $iters && $python3 gen_acc.py && cp ACC_training.png ../figures && cd ..