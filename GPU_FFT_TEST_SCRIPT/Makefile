NVCC_FLAGS= --device-c -w
Oflag= -O3

Nx ?= 128
Ny ?= 128
Nz ?= 128

CUDA_HOME?=/usr/local/cuda-12.2/
CUDA_INC=$(CUDA_HOME)/include/
CUDA_LIB=$(CUDA_HOME)/lib64/
NVCC= $(CUDA_HOME)/bin/nvcc -std=c++14

MPI_HOME ?= /scratch/Manthan_phy/hpc_sdk/Linux_x86_64/23.9/comm_libs/mpi
MPI_INC= $(MPI_HOME)/include/
MPI_LIB= $(MPI_HOME)/lib/

GPU_FFT_HOME ?= /scratch/Manthan_phy/FFT_testing/FFT_GPU
GPU_FFT_INC = $(GPU_FFT_HOME)/include
GPU_FFT_LIB = $(GPU_FFT_HOME)/lib

CUFFT_link= -lcufft
MPI_link= -lmpi
GPU_FFT_link = -lGPU_FFT

Link_all= $(CUFFT_link) $(MPI_link) $(GPU_FFT_link)
INCLUDE_ALL= -I $(CUDA_INC),$(MPI_INC),$(GPU_FFT_INC)
LIB_ALL= -L $(CUDA_LIB),$(MPI_LIB),$(GPU_FFT_LIB)

compile: test.cu header.h
	$(NVCC)  test.cu $(INCLUDE_ALL) $(LIB_ALL) $(Link_all) -o GPU_FFT_test

test_fft: GPU_FFT_test
	LD_LIBRARY_PATH=$(MPI_LIB):$(CUDA_LIB):$(GPU_FFT_LIB):$(LD_LIBRARY_PATH) mpiexec -np 2 GPU_FFT_test $(Nx) $(Ny) $(Nz)

clean:
	rm -rf *o