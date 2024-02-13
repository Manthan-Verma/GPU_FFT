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

#ifndef GPU_FFT_H_
#define GPU_FFT_H_

#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include "transitions/transitions.h"
#include "mpi.h"
#include "transpose_kernels/transpose.h"
#include "FFT_definations/FFT_defination.h"

using int64 = long long int;

template <typename T1, typename T2>
class GPU_FFT
{
    int64 Nx{0}; // Nx in CPU Memory
    int64 Ny{0}; // Ny in GPU Memory
    int64 Nz{0}; // Nz in GPU Memory

    // Communication Variables
    int procs; // procs for CPU Memory
    int rank;  // rank for CPU Memory

    // MPI Communicator
    MPI_Comm MPI_COMMUNICATOR;
    // MPI Requests
    MPI_Request *requests;

    // Buffer data pointer
    T2 *buffer;

    // Variable to get MPI Datatype
    T2 temp_variable_for_mpi_datatype;
    // cOMMUNICATION NUMBER
    int comm_no{0};

    // CUFFT DATA POINTS
    cufftHandle planR2C,
        planC2R, planC2C;
    size_t *worksize;
    int n_r2c[2], *inembed_r2c{nullptr}, *onembed_r2c{nullptr}, istride_r2c{1}, ostride_r2c{1};
    int idist_r2c{1}, odist_r2c{1}, BATCH_r2c;
    int64 BATCHED_SIZE_R2C;

    int rank_c2c{1}, rank_r2c{2};
    int n_c2c[1], *inembed_c2c, *onembed_c2c;
    int istride_c2c, ostride_c2c, odist_c2c{1}, idist_c2c{1}, BATCH_C2C;
    int64 BATCHED_SIZE_C2C;

    // Grid and block of kernel launch
    dim3 grid_fourier_space;
    dim3 block_fourier_space;

    // CUFFT TYPES
    cufftType_t cufft_type_r2c{CUFFT_R2C};
    cufftType_t cufft_type_c2r{CUFFT_C2R};
    cufftType_t cufft_type_c2c{CUFFT_C2C};

public:
    GPU_FFT(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT); // Constructor
    void INIT_GPU_FFT_COMM(int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT);                         // Initialize cufft variables
    void INIT_GPU_FFT();                                                                                   // Initialize cufft variables

    void GPU_FFT_R2C(T1 *input_data);
    void GPU_FFT_C2R(T2 *input_data);

    ~GPU_FFT();

protected:
    std::stringstream st;
};

// Teamplate for getting MPI datatype
template <typename T>
MPI_Datatype get_mpi_datatype(T a);

#endif