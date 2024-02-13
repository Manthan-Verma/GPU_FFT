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

#include "../GPU_FFT/GPU_FFT.h"

template <typename T1, typename T2>
GPU_FFT<T1, T2>::GPU_FFT(int64 Nx_in, int64 Ny_in, int64 Nz_in, int procs, int rank, MPI_Comm &MPI_COMMUNICATOR_INPUT)
{
    // std::cout << "\n Initializing the GPU FFT Constructor with variables ";

    // Setting the 3D dimensions of real space data
    Nx = Nx_in;
    Ny = Ny_in;
    Nz = Nz_in;

    // Initalizing the buffer data pointer
    cudaMalloc(&buffer, sizeof(T2) * Nx * (Ny / procs) * (Nz / 2 + 1));

    // Setting the grid and blocks for FFT kernels
    grid_fourier_space = {((Nx * (Ny / procs) * (Nz / 2 + 1)) / 256) + 1, 1, 1};
    block_fourier_space = {256, 1, 1};

    // Now initialiazing the FFT
    INIT_GPU_FFT_COMM(procs, rank, MPI_COMMUNICATOR_INPUT);
}

template <typename T1, typename T2>
void GPU_FFT<T1, T2>::INIT_GPU_FFT()
{
    // R2C DATA
    n_r2c[0] = Ny;
    n_r2c[1] = Nz;
    BATCHED_SIZE_R2C = Nz * Ny;
    BATCH_r2c = (Nx / procs);

    // C2C DATA
    n_c2c[0] = Nx;
    inembed_c2c = new int[1]{static_cast<int>(Nx)};
    onembed_c2c = new int[1]{static_cast<int>(Nx)};
    BATCHED_SIZE_C2C = Nx;
    istride_c2c = (Ny / procs) * (Nz / 2 + 1);
    ostride_c2c = (Ny / procs) * (Nz / 2 + 1);
    BATCH_C2C = (Ny / procs) * (Nz / 2 + 1);
    worksize = new size_t{};

    // Initializing some points for plan initialization
    if (std::is_same<T1, T1_d>::value)
    {
        // CUFFT TYPES
        cufft_type_r2c = CUFFT_D2Z;
        cufft_type_c2r = CUFFT_Z2D;
        cufft_type_c2c = CUFFT_Z2Z;
    }

    // Plans intiialization
    cufftCreate(&planR2C);
    cufftCreate(&planC2R);
    cufftCreate(&planC2C);
    cufftMakePlanMany(planC2C, rank_c2c, n_c2c, inembed_c2c, istride_c2c, idist_c2c, onembed_c2c, ostride_c2c, odist_c2c, cufft_type_c2c, BATCH_C2C, worksize);
    cufftMakePlanMany(planR2C, rank_r2c, n_r2c, inembed_r2c, istride_r2c, idist_r2c, onembed_r2c, ostride_r2c, odist_r2c, cufft_type_r2c, BATCH_r2c, worksize);
    cufftMakePlanMany(planC2R, rank_r2c, n_r2c, inembed_r2c, istride_r2c, idist_r2c, onembed_r2c, ostride_r2c, odist_r2c, cufft_type_c2r, BATCH_r2c, worksize);
}

template <typename T1, typename T2>
GPU_FFT<T1, T2>::~GPU_FFT()
{
    cudaFree(buffer);
}

// ########### Explicit instantiation of class templates ##################
template class GPU_FFT<T1_f, T2_f>;
template class GPU_FFT<T1_d, T2_d>;
// ########################################################################