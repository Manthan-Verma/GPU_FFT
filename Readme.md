# This Repository is for FFT libraray Development and Testing

Here Me and My team ( Manthan Verma , Soumyadeep chatterjee, Nishant , sashi , Mahendra kumar verma Sir ) developing and testing the FFT Library development .</br></br>

## GPU-FFT

GPU-FFT is an open source parallel C++ code to compute FFT on GPUs using CUDA.

## Getting the Source Code

GPU-FFT is hosted on GitHub. You can download the source code from the following link:
       https://github.com/Manthan-Verma/FFT_LIB.git

## Installing GPU-FFT

GPU-FFT requires several libraries (listed later in this section) for compiling. These libraries can be installed in any directory as long as the said directory is added to the user $PATH.
      
## Required Libraries

The following libraries are required for installing and running GPU-FFT:

Cuda Library is used for development of our code. This library can be installed from :
       https://developer.nvidia.com/cuda-downloads
       Then choose the system type and download the distribution.

After extracting install the software and go with the instructions. Put the path of installationn where you want to install.

An MPI (Message Passing Interface) Library - GPU-FFT uses MPI for parallelism. The software was tested using OPEN-MPI(Version 4.1.1), however, any standard MPI implementation should not be sufficient because for our program to give best scaling and timming we need MPI to be CUDA Aware. Here, we will provide instructions for installing OPEN-MPI. Download OPEN-MPI from:
    https://www.open-mpi.org/
    
After extraction, change to openmpi-4.1.1 folder and enter the following:

    ./configure --prefix=$(Path to install openmpi) --with-cuda=$(Path to cuda installation) --enable-mpi-cxx

    make all install
    
## Compilation instructions

To compile run the following command:
       nvcc start.cu data_init.cu gpu_data.cu bench.cu cufft_calls.cu -I (path to include dirctory of MPI) -L (path to library dirctory of MPI) -lmpi -lcufft -o run -std=c++17
       
       After this a object file named run will be created in the same folder 
## Detailed Instructions for running and Benchmarking the Program

       Our Code runs FFT on GPUs. We used the slab-decomposition in our code because of the lack of less no of GPUs. We used MPI_Isend, MPI_Irecv, CudaMemcpy for communication and 2 our own transpose kernels with high throughput.
       It calculates 3D FFT R2C on multi-node, multi-GPU using cuFFT in Backend.
       
       To run/ Benchmar the code :
              mpiexec -np <no_of_processes/no_of_GPUs to run on> --host hostname:<no_of_GPU_on_this_host>,hostname2<no_of_GPU_on_this_host> nic_bindings.sh run <x axis grid size> <y axis grid size> <z axis grid size> <no_of_iterations> <precision>
              
              example:
              mpiexec -np 4 --host h1:2,h2:2 nic_bindings run 512 512 512 100 double                      ---> for double preciosn  and grid size ( 512,512,512), for 100 iterations 
                                                                                                  for ingle precision put single in place of double.(Put iterations as 1 for single run)
       
## Here set the nic bindings file according to the linking architecture of your device. 
