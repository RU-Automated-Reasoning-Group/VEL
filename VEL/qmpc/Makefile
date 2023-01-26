CXX = g++
HOME= /usr/local/include
LIB_HOME = ../flowstar/flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g -L$(LIB_HOME) -L/usr/local/lib

all: qmpc1 qmpc2 qmpc3 qmpc4

qmpc1: verify_qmpc1.o
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

qmpc2: verify_qmpc2.o
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

qmpc3: verify_qmpc3.o
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

qmpc4: verify_qmpc4.o
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean: 
	rm -f *.o qmpc1 qmpc2 qmpc3 qmpc4
