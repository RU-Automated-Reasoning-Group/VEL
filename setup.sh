# install flowstar
cd VEL/flowstar/flowstar-toolbox
make -j 4
cd ../flowstar-2.1.0/
unzip flowstar-2.1.0.zip
cd flowstar-2.1.0/
make -j 4
cd ../../../

# compile the verifier for each benchmark
# ACC
cd ACC
make
cd ..

# B1
cd B1
make
cd ..

# B2
cd B2
make
cd ..

# B3
cd B3
make
cd ..

# B4
cd B4
make
cd ..

# B5
cd B5
make
cd ..

# cartpole
cd cartpole
make
cd ..

# oscillator
cd oscillator
make
cd ..

# pendulum
cd pendulum
make
cd ..

# qmpc
cd qmpc
make
cd ..

# tora
cd tora
make
cd ..

# toraeq
cd tora_inf
make
cd ..

# unicycle
cd unicycle_car
make
cd ..
