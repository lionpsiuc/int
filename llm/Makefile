# Makefile for Exponential Integral CPU + CUDA

CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3 -std=c++11

EXEC = exponentialIntegral.out
OBJS = main.o expint_gpu.o

all: $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(EXEC)

clean:
	rm -f *.o $(EXEC)
