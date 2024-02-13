COMPILER ?= HIP

Oflag= -O3

INSTALL_DIR ?= /mnt/Disk_6TB/Manthan/Trash/GPU_FFT

CC_HOME?=/usr/local/cuda-12.2
CC_INC=$(CC_HOME)/include/

MPI_HOME ?= /scratch/Manthan_phy/hpc_sdk/Linux_x86_64/23.9/comm_libs/mpi
MPI_INC= $(MPI_HOME)/include/
MPI_LIB= $(MPI_HOME)/lib/

MPI_link= -lmpi

ifeq ($(COMPILER),HIP)
CC_LIB=$(CC_HOME)/lib/
CC= $(CC_HOME)/bin/hipcc -std=c++14
NVCC_FLAGS= -w -Xhip -fPIC -Xhip -O3
CUFFT_link= -lhipfft
INCLUDE_ALL= -I $(CC_INC) -I $(MPI_INC)
LIB_ALL= -L $(CC_LIB) -L $(MPI_LIB)

else ifeq ($(COMPILER),NVCC)
CC_LIB=$(CC_HOME)/lib64/
CC= $(CC_HOME)/bin/nvcc -std=c++14
NVCC_FLAGS= -w -Xcompiler -fPIC -Xcompiler -O3
CUFFT_link= -lcufft
INCLUDE_ALL= -I $(CC_INC),$(MPI_INC)
LIB_ALL= -L $(CC_LIB),$(MPI_LIB)
endif

Link_all= $(CUFFT_link) $(MPI_link)


start: FFT_implementation.o init_fft.o transpose.o
	$(CC)  FFT_R2C.o FFT_C2R.o FFT_def.o comm_setup.o init_fft_data.o transpose_ker.o $(INCLUDE_ALL) $(LIB_ALL) -o\
	 libGPU_FFT.so $(NVCC_FLAGS) -shared -lc $(Link_all)

FFT_implementation.o: src/FFT_implementation/FFT_C2R.cu src/FFT_implementation/FFT_R2C.cu \
						src/FFT_implementation/FFT_definations.cu
	$(CC) src/FFT_implementation/FFT_R2C.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c  -o FFT_R2C.o
	$(CC) src/FFT_implementation/FFT_C2R.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c  -o FFT_C2R.o
	$(CC) src/FFT_implementation/FFT_definations.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c  -o FFT_def.o

init_fft.o: src/initialize_GPU_FFT/communications_setup.cu src/initialize_GPU_FFT/GPU_FFT_init.cu
	$(CC) src/initialize_GPU_FFT/communications_setup.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c -o comm_setup.o
	$(CC) src/initialize_GPU_FFT/GPU_FFT_init.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) -c -o init_fft_data.o

transpose.o: src/matrix_transpose/transpose.cu
	$(CC) src/matrix_transpose/transpose.cu $(INCLUDE_ALL) $(LIB_ALL) $(NVCC_FLAGS) $(Oflag) -c -o transpose_ker.o

install:
	mkdir -p $(INSTALL_DIR)/include
	mkdir -p $(INSTALL_DIR)/lib
	cp -r src/GPU_FFT $(INSTALL_DIR)/include
	cp libGPU_FFT.so $(INSTALL_DIR)/lib

clean:
	rm -rf *o