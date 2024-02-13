NVCC_FLAGS= --device-c -w -Xcompiler -fPIC
Oflag= -O3

INSTALL_DIR ?= /home/manthan/

CUDA_HOME?=/usr/local/
CUDA_INC=$(CUDA_HOME)/include/
CUDA_LIB=$(CUDA_HOME)/lib64/
NVCC= $(CUDA_HOME)/bin/nvcc -std=c++14

MPI_HOME ?= /home/manthan
MPI_INC_FLAG= $(MPI_HOME)/include/
MPI_LIB_FLAG= $(MPI_HOME)/lib/

CUFFT_link= -lcufft
MPI_link= -lmpi

Link_all= $(CUFFT_link) $(MPI_link)
INCLUDE_ALL= -I $(CUDA_INC),$(MPI_INC_FLAG)
LIB_ALL= -L $(CUDA_LIB),$(MPI_LIB_FLAG)

start: FFT_implementation.o init_fft.o transpose.o
	$(NVCC)  FFT_R2C.o FFT_C2R.o FFT_def.o comm_setup.o init_fft_data.o transpose_ker.o $(INCLUDE_ALL) $(LIB_ALL) -o\
	 libGPU_FFT.so -Xcompiler -Wall -Xcompiler -O3 -Xcompiler -fPIC -shared -lc $(Link_all)

FFT_implementation.o: src/FFT_implementation/FFT_C2R.cu src/FFT_implementation/FFT_R2C.cu \
						src/FFT_implementation/FFT_definations.cu
	$(NVCC) src/FFT_implementation/FFT_R2C.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c  -o FFT_R2C.o
	$(NVCC) src/FFT_implementation/FFT_C2R.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c  -o FFT_C2R.o
	$(NVCC) src/FFT_implementation/FFT_definations.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c  -o FFT_def.o

init_fft.o: src/initialize_GPU_FFT/communications_setup.cu src/initialize_GPU_FFT/GPU_FFT_init.cu
	$(NVCC) src/initialize_GPU_FFT/communications_setup.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c -o comm_setup.o
	$(NVCC) src/initialize_GPU_FFT/GPU_FFT_init.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c -o init_fft_data.o

transpose.o: src/matrix_transpose/transpose.cu
	$(NVCC) src/matrix_transpose/transpose.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) $(Oflag) -c -o transpose_ker.o

install:
	mkdir -p $(INSTALL_DIR)/include
	mkdir -p $(INSTALL_DIR)/lib
	cp -r src/GPU_FFT $(INSTALL_DIR)/include
	cp libGPU_FFT.so $(INSTALL_DIR)/lib

clean:
	rm -rf *o