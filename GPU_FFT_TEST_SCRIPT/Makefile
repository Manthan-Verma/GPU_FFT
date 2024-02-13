COMPILER ?= HIP
Nx ?= 128
Ny ?= 128
Nz ?= 128

CC_HOME?=/usr/local/cuda-12.2
CC_INC=$(CC_HOME)/include/

MPI_HOME ?= /scratch/Manthan_phy/hpc_sdk/Linux_x86_64/23.9/comm_libs/mpi
MPI_INC= $(MPI_HOME)/include/
MPI_LIB= $(MPI_HOME)/lib/

GPU_FFT_HOME ?= /scratch/Manthan_phy/FFT_testing/FFT_GPU
GPU_FFT_INC = $(GPU_FFT_HOME)/include
GPU_FFT_LIB = $(GPU_FFT_HOME)/lib

MPI_link= -lmpi
GPU_FFT_link = -lGPU_FFT

ifeq ($(COMPILER),HIP)
CC_LIB=$(CC_HOME)/lib/
CC= $(CC_HOME)/bin/hipcc -std=c++14
NVCC_FLAGS= -w -Xhip -fPIC -Xhip -O3
CUFFT_link= -lhipfft
INCLUDE_ALL= -I $(CC_INC) -I $(MPI_INC) -I $(GPU_FFT_INC)
LIB_ALL= -L $(CC_LIB) -L $(MPI_LIB) -L $(GPU_FFT_LIB)

else ifeq ($(COMPILER),NVCC)
CC_LIB=$(CC_HOME)/lib64/
CC= $(CC_HOME)/bin/nvcc -std=c++14
NVCC_FLAGS= -w -Xcompiler -fPIC -Xcompiler -O3
CUFFT_link= -lcufft
INCLUDE_ALL= -I $(CC_INC),$(MPI_INC),$(GPU_FFT_INC)
LIB_ALL= -L $(CC_LIB),$(MPI_LIB),$(GPU_FFT_LIB)
endif

Link_all= $(CUFFT_link) $(MPI_link) $(GPU_FFT_link)

ifeq ($(COMPILER),HIP)
compile:
	$(CC) test.cpp $(INCLUDE_ALL) $(LIB_ALL) $(Link_all) -o GPU_FFT_test
endif

ifeq ($(COMPILER),NVCC)
compile:
	$(CC) test.cu $(INCLUDE_ALL) $(LIB_ALL) $(Link_all) -o GPU_FFT_test
endif

test_fft: GPU_FFT_test
	LD_LIBRARY_PATH=$(MPI_LIB):$(CC_LIB):$(GPU_FFT_LIB):$(LD_LIBRARY_PATH) mpiexec -np 2 ./GPU_FFT_test $(Nx) $(Ny) $(Nz)

clean:
	rm -rf *o