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

#if defined(_MSC_VER)
#define inline_qualifier __inline
#define _USE_MATH_DEFINES
#include <direct.h>
#else
#define inline_qualifier inline
#endif

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

namespace GPU_FFT
{
    extern "C" int64 Nx_; // Nx in CPU Memory
    extern "C" int64 Ny_; // Ny in CPU Memory
    extern "C" int64 Nz_; // Nz in CPU Memory

    // Communication Variables
    extern "C" int procs_; // procs for CPU Memory
    extern "C" int rank_;  // rank for CPU Memory

    // MPI Communicator
    extern "C" MPI_Comm MPI_COMMUNICATOR_;

    // MPI Requests
    extern "C" MPI_Request *requests_;

    // Buffer data pointer
    template <typename T2>
    extern T2 *buffer_;

    // Variable to get MPI Datatype
    template <typename T2>
    extern  T2 temp_variable_for_mpi_datatype_;

    // cOMMUNICATION NUMBER
    extern "C" int comm_no_;

    // CUFFT DATA POINTS
    extern "C" TRANSITIONS::fftHandle planR2C_, planC2R_, planC2C_;
    extern "C" size_t *worksize_;
    extern "C" int n_r2c_[2], *inembed_r2c_, *onembed_r2c_, istride_r2c_, ostride_r2c_;
    extern "C" int idist_r2c_, odist_r2c_, BATCH_r2c_;
    extern "C" int64 BATCHED_SIZE_R2C_;

    extern "C" int rank_c2c_, rank_r2c_;
    extern "C" int n_c2c_[1], *inembed_c2c_, *onembed_c2c_;
    extern "C" int istride_c2c_, ostride_c2c_, odist_c2c_, idist_c2c_, BATCH_C2C_;
    extern "C" int64 BATCHED_SIZE_C2C_;

    // Grid and block of kernel launch
    extern "C" dim3 grid_fourier_space_;
    extern "C" dim3 block_fourier_space_;

    // CUFFT TYPES
    extern "C" TRANSITIONS::ffttype_t type_r2c_;
    extern "C" TRANSITIONS::ffttype_t type_c2r_;
    extern "C" TRANSITIONS::ffttype_t type_c2c_;

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
    MPI_Datatype get_mpi_datatype_(T a);

} // namespace GPU_FFT

#endif