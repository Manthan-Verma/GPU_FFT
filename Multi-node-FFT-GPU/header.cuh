/*
Copyright (c) 2022, Mahendra Verma, Manthan verma, Soumyadeep Chatterjee
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
    \brief ---> Code to compute FFT on multi-node on GPUs Scaling upto 512 GPUs for grid size upto 4096^3
    \author ---> Manthan verma, Soumyadeep Chatterjee, Gaurav Garg, Bharatkumar Sharma, Nishant Arya, Shashi Kumar, Mahendra Verma
    \dated --> Feb 2022
    \copyright New BSD License
*/

#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <mpi.h>
#include <initializer_list>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

using int64 = long long int;

// GPU VARIABLES
__constant__ int64 x_gpu, y_gpu, z_gpu;
__constant__ double dx_gpu, dy_gpu, dz_gpu;

// CUFFT DEFINATIONS
/*auto cufft_type_r2c{CUFFT_R2C};
auto cufft_type_c2r{CUFFT_C2R};
auto cufft_type_c2c{CUFFT_C2C};*/

template <typename T1, typename T2>
class header
{
    // Host data variables
    T1 *local_host_data;
    T1 *local_host_data_out;

    // Device variables
    T1 *slab_inp_data;
    T2 *slab_outp_data;
    T2 *slab_outp_data_tr;

    // Kernel Launch Parameters
    dim3 grid_slab;
    dim3 block_slab;
    dim3 grid_chunk;
    dim3 block_chunk;

    // Parameters_VARIABLES
    int n_gpu_pernode;
    int64 Nx, Ny, Nz;
    int64 Total_data_size_real_per_gpu;
    int Nx_d, Ny_d, iteration_count;
    double pi, dx, dy, dz;

    // CUFFT DATA POINTS
    cufftHandle planR2C, planC2R, planC2C;
    size_t *worksize;
    int n_r2c[2], *inembed_r2c{nullptr}, *onembed_r2c{nullptr}, istride_r2c{1}, ostride_r2c{1};
    int idist_r2c{1}, odist_r2c{1}, BATCH_r2c;
    int64 BATCHED_SIZE_R2C;

    int rank_c2c{1}, rank_r2c{2};
    int n_c2c[1], *inembed_c2c, *onembed_c2c;
    int istride_c2c, ostride_c2c, odist_c2c{1}, idist_c2c{1}, BATCH_C2C;
    int64 BATCHED_SIZE_C2C;

    // CUFFT TYPES
    cufftType_t cufft_type_r2c{CUFFT_R2C};
    cufftType_t cufft_type_c2r{CUFFT_C2R};
    cufftType_t cufft_type_c2c{CUFFT_C2C};
    // auto execution_r2c{cufftExecR2C};

    // MPI variables
    int procs, rank;
    T1 mpi_pass{0};
    //MPI_Datatype complex_type{MPI_CXX_COMPLEX};

    // TIME VARIABLES
    double start, stop, Total_time, start_comm, stop_comm, Comm_time;

public:
    header(int Nx_get, int Ny_get, int Nz_get, int iterations, int process, int MPI_rank); // Constructor initializing basic variables
    void initialize_cufft_variables();                                                     // Initialize cufft variables
    void initialize_gpu_data();                                                            // Initialize GPU data variables
    void Mem_allocation_gpu_cpu();                                                         // Memory allocation on Host and Device
    void data_init_and_copy_to_gpu();                                                      // Initialize data and copy from host to device
    void benchmarking_initialization();                                                    // pre work for benchmarking
    void becnhmarking_loop(MPI_Request requests[], int j);                                 // Benchmarking loop common
    void validate();                                                                       // Validation part 2 (end part)
    void compute_details();
    ~header();

protected:
    std::stringstream st;
};

// GPU Functions
// SINGLE PRECISION GPU KERNELS
template <typename T3>
__global__ void results_show(T3 *data, int rank, int process);

template <typename T3>
__global__ void transpose_slab(T3 *matrix_data, T3 *matrix_transpose, int64 Ny, int64 Nx);

template <typename T3>
__global__ void chunk_transpose(T3 *matrix_data, T3 *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs);

template <typename T3>
__global__ void chunk_transpose_inverse(T3 *matrix_data, T3 *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs);

template <typename T3>
__global__ void Normalize_single(T3 *data, int64 Normalization_size_c2c, int64 Normalization_size_r2c);

// CUFFT Specialization of functions
template <typename T4, typename T5>
void cufft_call_r2c(cufftHandle &plan, T4 *input_data, T5 *output_data);

template <typename T4>
void cufft_call_c2c(cufftHandle &plan, T4 *input_data, int direction);

template <typename T4, typename T5>
void cufft_call_c2r(cufftHandle &plan, T4 *input_data, T5 *output_data);

// MPI datatype template
template<typename T>
MPI_Datatype mpi_type_call(T a);

// Error checker FUNCTIONS

extern "C++" inline void mpierror(int val, int line) // MPI error checker
{
    if (val != 0)
    {
        char *error = new char[20];
        int *len = new int{};
        MPI_Error_string(val, error, len);
        printf("\n MPI_Error = %s , at line = %d", error, line);
        exit(0);
    }
}
extern "C++" inline void gpuerrcheck_cufft(cufftResult test, int line) // CUFFT ERROR CHECKER
{
    if (test != 0)
    {
        std::cout << "\n cufft error code  = " << test << " , At line " << line;
        exit(0);
    }
}
extern "C++" inline void gpuerrcheck_cudaerror(cudaError_t err, int line) // CUFFT ERROR CHECKER
{
    if (err != 0)
    {
        std::cout << "\n cuda error  = " << cudaGetErrorString(err) << " , At line " << line;
        exit(0);
    }
}