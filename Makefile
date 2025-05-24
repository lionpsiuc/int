NVCC = nvcc
CXX  = g++

TARGET = main

SRC_DIR     = src
INCLUDE_DIR = include

CU_SOURCES  = $(SRC_DIR)/gpu.cu
CPP_SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/matrix.cpp $(SRC_DIR)/timing.cpp

CU_OBJS  = $(CU_SOURCES:.cu=.o)
CPP_OBJS = $(CPP_SOURCES:.cpp=.o)
OBJS     = $(CU_OBJS) $(CPP_OBJS)

CXXFLAGS  = -O3 -Wall -Wextra -I$(INCLUDE_DIR)
NVCCFLAGS = -O3 -arch=sm_75 -I$(INCLUDE_DIR)
LDFLAGS   = -L/usr/local/cuda/lib64 -lcudart

PRECISION_FLAG =

# Allow specifying double-precision (i.e., use make type=double)
ifeq ($(type),double)
  PRECISION_FLAG = -DDOUBLE
endif

NVCCFLAGS += $(PRECISION_FLAG)
CXXFLAGS  += $(PRECISION_FLAG)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
