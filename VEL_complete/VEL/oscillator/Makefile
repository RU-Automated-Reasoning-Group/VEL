CXX = g++
HOME= /usr/local/include
LIB_HOME = ../flowstar/flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g -L$(LIB_HOME) -L/usr/local/lib

all: os oseq

os: verify_os.o
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

oseq: verify_os_eq.o
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean: 
	rm -f *.o os oseq
