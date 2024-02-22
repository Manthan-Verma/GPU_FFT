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
    \dated --> Feb 2024
    \copyright New BSD License
*/

#pragma once
#include "../GPU_FFT/GPU_FFT.h"

namespace GPU_FFT
{
    int64 __Nx__{0}; // Nx in CPU Memory
    int64 __Ny__{0}; // Ny in CPU Memory
    int64 __Nz__{0}; // Nz in CPU Memory

    // Communication Variables
    int __procs__; // procs for CPU Memory
    int __rank__;  // rank for CPU Memory

    // MPI Communicator
    MPI_Comm __MPI_COMMUNICATOR__;

    // MPI Requests
    MPI_Request *__requests__;

    // Buffer data pointer
    template <typename T2>
    T2 *__buffer__;

    // Variable to get MPI Datatype
    template <typename T2>
    T2 __temp_variable_for_mpi_datatype__;

    // cOMMUNICATION NUMBER
    int __comm_no__{0};

    // CUFFT DATA POINTS
    TRANSITIONS::fftHandle __planR2C__, __planC2R__, __planC2C__;
    size_t *__worksize__;
    int __n_r2c__[2], *__inembed_r2c__{nullptr}, *__onembed_r2c__{nullptr}, __istride_r2c__{1}, __ostride_r2c__{1};
    int __idist_r2c__{1}, __odist_r2c__{1}, __BATCH_r2c__;
    int64 __BATCHED_SIZE_R2C__;

    int __rank_c2c__{1}, __rank_r2c__{2};
    int __n_c2c__[1], *__inembed_c2c__, *__onembed_c2c__;
    int __istride_c2c__, __ostride_c2c__, __odist_c2c__{1}, __idist_c2c__{1}, __BATCH_C2C__;
    int64 __BATCHED_SIZE_C2C__;

    // Grid and block of kernel launch
    dim3 __grid_fourier_space__;
    dim3 __block_fourier_space__;

    // CUFFT TYPES
    TRANSITIONS::ffttype_t __type_r2c__{TRANSITIONS::FFT_R2C};
    TRANSITIONS::ffttype_t __type_c2r__{TRANSITIONS::FFT_C2R};
    TRANSITIONS::ffttype_t __type_c2c__{TRANSITIONS::FFT_C2C};

    // GPU_FFT Functions
    template <typename T1, typename T2>
    void INIT_GPU_FFT_COMM(int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT); // Initialize Communications 

    template <typename T1, typename T2>
    void INIT_GPU_FFT(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT); // Initialize GPU_FFT variables and plans

    template <typename T1, typename T2>
    void GPU_FFT_R2C(T1 *input_data);

    template <typename T1, typename T2>
    void GPU_FFT_C2R(T2 *input_data);

    // Teamplate for getting MPI datatype
    template <typename T>
    MPI_Datatype __get_mpi_datatype__(T a);

} // namespace GPU_FFT