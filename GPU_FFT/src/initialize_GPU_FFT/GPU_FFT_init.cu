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

template <typename T1, typename T2>
void GPU_FFT::INIT_GPU_FFT(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT)
{

    // Setting the 3D dimensions of real space data
    Nx_ = Nx_in;
    Ny_ = Ny_in;
    Nz_ = Nz_in;

    // Initalizing the buffer_ data pointer
    buffer_<T2> = TRANSITIONS::Memory_allocation_gpu(buffer_<T2>, Nx_ * (Ny_ / procs) * (Nz_ / 2 + 1));

    // Setting the grid and blocks for FFT kernels
    grid_fourier_space_ = {static_cast<unsigned int>((Nx_ * (Ny_ / procs) * (Nz_ / 2 + 1)) / 256) + 1, 1, 1};
    block_fourier_space_ = {256, 1, 1};

    // Now initialiazing the FFT
    INIT_GPU_FFT_COMM<T1, T2>(procs, rank, MPI_COMMUNICATOR_INPUT);

    // ################################### Intializzing the FFT Plans ######################################
    // R2C DATA
    n_r2c_[0] = Ny_;
    n_r2c_[1] = Nz_;
    BATCHED_SIZE_R2C_ = Nz_ * Ny_;
    BATCH_r2c_ = (Nx_ / procs);

    // C2C DATA
    n_c2c_[0] = Nx_;
    inembed_c2c_ = new int[1]{static_cast<int>(Nx_)};
    onembed_c2c_ = new int[1]{static_cast<int>(Nx_)};
    BATCHED_SIZE_C2C_ = Nx_;
    istride_c2c_ = (Ny_ / procs) * (Nz_ / 2 + 1);
    ostride_c2c_ = (Ny_ / procs) * (Nz_ / 2 + 1);
    BATCH_C2C_ = (Ny_ / procs) * (Nz_ / 2 + 1);
    worksize_ = new size_t{};

    // Initializing some points for plan initialization
    if (std::is_same<T1, TRANSITIONS::T1_d>::value)
    {
        // CUFFT TYPES
        type_r2c_ = TRANSITIONS::FFT_D2Z;
        type_c2r_ = TRANSITIONS::FFT_Z2D;
        type_c2c_ = TRANSITIONS::FFT_Z2Z;
    }

    // Plans intiialization
    TRANSITIONS::fftCreate(&planR2C_);
    TRANSITIONS::fftCreate(&planC2R_);
    TRANSITIONS::fftCreate(&planC2C_);
    TRANSITIONS::fftMakePlanMany(planC2C_, rank_c2c_, n_c2c_, inembed_c2c_, istride_c2c_, idist_c2c_, onembed_c2c_, ostride_c2c_, odist_c2c_, type_c2c_, BATCH_C2C_, worksize_);
    TRANSITIONS::fftMakePlanMany(planR2C_, rank_r2c_, n_r2c_, inembed_r2c_, istride_r2c_, idist_r2c_, onembed_r2c_, ostride_r2c_, odist_r2c_, type_r2c_, BATCH_r2c_, worksize_);
    TRANSITIONS::fftMakePlanMany(planC2R_, rank_r2c_, n_r2c_, inembed_r2c_, istride_r2c_, idist_r2c_, onembed_r2c_, ostride_r2c_, odist_r2c_, type_c2r_, BATCH_r2c_, worksize_);
    // ###################################################################################################
}

// ########### Explicit instantiation of class templates ##################
template <>
TRANSITIONS::T2_f *GPU_FFT::buffer_<TRANSITIONS::T2_f>;
template <>
TRANSITIONS::T2_d *GPU_FFT::buffer_<TRANSITIONS::T2_d>;

template void GPU_FFT::INIT_GPU_FFT<TRANSITIONS::T1_f, TRANSITIONS::T2_f>(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT);
template void GPU_FFT::INIT_GPU_FFT<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT);
// ########################################################################