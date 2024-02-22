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
    __Nx__ = Nx_in;
    __Ny__ = Ny_in;
    __Nz__ = Nz_in;

    // Initalizing the __buffer__ data pointer
    __buffer__<T2> = TRANSITIONS::__Memory_allocation_gpu__(__buffer__<T2>, __Nx__ * (__Ny__ / procs) * (__Nz__ / 2 + 1));

    // Setting the grid and blocks for FFT kernels
    __grid_fourier_space__ = {static_cast<unsigned int>((__Nx__ * (__Ny__ / procs) * (__Nz__ / 2 + 1)) / 256) + 1, 1, 1};
    __block_fourier_space__ = {256, 1, 1};

    // Now initialiazing the FFT
    INIT_GPU_FFT_COMM<T1, T2>(procs, rank, MPI_COMMUNICATOR_INPUT);

    // ################################### Intializzing the FFT Plans ######################################
    // R2C DATA
    __n_r2c__[0] = __Ny__;
    __n_r2c__[1] = __Nz__;
    __BATCHED_SIZE_R2C__ = __Nz__ * __Ny__;
    __BATCH_r2c__ = (__Nx__ / procs);

    // C2C DATA
    __n_c2c__[0] = __Nx__;
    __inembed_c2c__ = new int[1]{static_cast<int>(__Nx__)};
    __onembed_c2c__ = new int[1]{static_cast<int>(__Nx__)};
    __BATCHED_SIZE_C2C__ = __Nx__;
    __istride_c2c__ = (__Ny__ / procs) * (__Nz__ / 2 + 1);
    __ostride_c2c__ = (__Ny__ / procs) * (__Nz__ / 2 + 1);
    __BATCH_C2C__ = (__Ny__ / procs) * (__Nz__ / 2 + 1);
    __worksize__ = new size_t{};

    // Initializing some points for plan initialization
    if (std::is_same<T1, TRANSITIONS::T1_d>::value)
    {
        // CUFFT TYPES
        __type_r2c__ = TRANSITIONS::FFT_D2Z;
        __type_c2r__ = TRANSITIONS::FFT_Z2D;
        __type_c2c__ = TRANSITIONS::FFT_Z2Z;
    }

    // Plans intiialization
    TRANSITIONS::fftCreate(&__planR2C__);
    TRANSITIONS::fftCreate(&__planC2R__);
    TRANSITIONS::fftCreate(&__planC2C__);
    TRANSITIONS::fftMakePlanMany(__planC2C__, __rank_c2c__, __n_c2c__, __inembed_c2c__, __istride_c2c__, __idist_c2c__, __onembed_c2c__, __ostride_c2c__, __odist_c2c__, __type_c2c__, __BATCH_C2C__, __worksize__);
    TRANSITIONS::fftMakePlanMany(__planR2C__, __rank_r2c__, __n_r2c__, __inembed_r2c__, __istride_r2c__, __idist_r2c__, __onembed_r2c__, __ostride_r2c__, __odist_r2c__, __type_r2c__, __BATCH_r2c__, __worksize__);
    TRANSITIONS::fftMakePlanMany(__planC2R__, __rank_r2c__, __n_r2c__, __inembed_r2c__, __istride_r2c__, __idist_r2c__, __onembed_r2c__, __ostride_r2c__, __odist_r2c__, __type_c2r__, __BATCH_r2c__, __worksize__);
    // ###################################################################################################
}

// ########### Explicit instantiation of class templates ##################
template <>
TRANSITIONS::T2_f *GPU_FFT::__buffer__<TRANSITIONS::T2_f>;
template <>
TRANSITIONS::T2_d *GPU_FFT::__buffer__<TRANSITIONS::T2_d>;

template void GPU_FFT::INIT_GPU_FFT<TRANSITIONS::T1_f, TRANSITIONS::T2_f>(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT);
template void GPU_FFT::INIT_GPU_FFT<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT);
// ########################################################################