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
    int64 Nx_{0}; // Nx in CPU Memory
    int64 Ny_{0}; // Ny in CPU Memory
    int64 Nz_{0}; // Nz in CPU Memory

    // Communication Variables
    int procs_; // procs for CPU Memory
    int rank_;  // rank for CPU Memory

    // MPI Communicator
    MPI_Comm MPI_COMMUNICATOR_;

    // MPI Requests
    MPI_Request *requests_;

    // Buffer data pointer
    template <typename T2>
    T2 *buffer_;

    // Variable to get MPI Datatype
    template <typename T2>
    T2 temp_variable_for_mpi_datatype_;

    // cOMMUNICATION NUMBER
    int comm_no_{0};

    // CUFFT DATA POINTS
    TRANSITIONS::fftHandle planR2C_, planC2R_, planC2C_;
    size_t *worksize_;
    int n_r2c_[2], *inembed_r2c_{nullptr}, *onembed_r2c_{nullptr}, istride_r2c_{1}, ostride_r2c_{1};
    int idist_r2c_{1}, odist_r2c_{1}, BATCH_r2c_;
    int64 BATCHED_SIZE_R2C_;

    int rank_c2c_{1}, rank_r2c_{2};
    int n_c2c_[1], *inembed_c2c_, *onembed_c2c_;
    int istride_c2c_, ostride_c2c_, odist_c2c_{1}, idist_c2c_{1}, BATCH_C2C_;
    int64 BATCHED_SIZE_C2C_;

    // Grid and block of kernel launch
    dim3 grid_fourier_space_;
    dim3 block_fourier_space_;

    // CUFFT TYPES
    TRANSITIONS::ffttype_t type_r2c_{TRANSITIONS::FFT_R2C};
    TRANSITIONS::ffttype_t type_c2r_{TRANSITIONS::FFT_C2R};
    TRANSITIONS::ffttype_t type_c2c_{TRANSITIONS::FFT_C2C};

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