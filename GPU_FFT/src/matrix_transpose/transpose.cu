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

template <typename T2>
__global__ void TRANSPOSE::transpose_slab_(T2 *matrix_data, T2 *matrix_transpose, int64 Ny_current_slab, int64 Nx_current_slab, int64 Nz_current_slab)
{
    int64 i = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i < (Nx_current_slab * Ny_current_slab * (Nz_current_slab / 2 + 1)))
    {
        int z = i % (Nz_current_slab / 2 + 1);
        int y = (i / (Nz_current_slab / 2 + 1)) % Ny_current_slab;
        int x = (i / ((Nz_current_slab / 2 + 1) * Ny_current_slab));
        int64 put_no = (x * (Nz_current_slab / 2 + 1)) + (y * (Nz_current_slab / 2 + 1) * Nx_current_slab) + z;
        matrix_transpose[put_no] = matrix_data[i];
    }
}

template <typename T2>
__global__ void TRANSPOSE::chunk_transpose_(T2 *matrix_data, T2 *matrix_transpose, int64 Nx, int64 Ny, int64 Nz, int procs)
{
    int64 i = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i < ((Nx / procs) * (Ny) * (Nz / 2 + 1)))
    {
        int d_tmp_z = (Ny / procs);
        int Nx_no = i % (Nz / 2 + 1);
        int Ny_no = (i / (Nz / 2 + 1)) % (Nx / procs);
        int Nz_no = i / ((Nz / 2 + 1) * (Nx / procs));
        int odd_even = Nz_no / d_tmp_z;
        int put_odd_even = Nz_no % d_tmp_z;
        int64 put_no_slab = (odd_even * (Nx / procs) * (Nz / 2 + 1) * (Ny / procs)) + (put_odd_even * (Nz / 2 + 1)) + (Ny_no * (Nz / 2 + 1) * (Ny) / procs);
        int64 put_no_full = put_no_slab + Nx_no;
        matrix_transpose[put_no_full] = matrix_data[i];
    }
}

template <typename T2>
__global__ void TRANSPOSE::chunk_transpose_inverse_(T2 *matrix_data, T2 *matrix_transpose, int64 Nx, int64 Ny, int64 Nz, int procs)
{
    int64 i = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i < (Nx * (Ny / procs) * (Nz / 2 + 1)))
    {
        int Nx_no = i % (Nz / 2 + 1);
        int Ny_no = (i / (Nz / 2 + 1)) % (Ny / procs);
        int Nz_no = i / ((Nz / 2 + 1) * (Ny / procs));
        int odd_even = Nz_no / (Nx / procs);
        int put_odd_even = (Nz_no % (Nx / procs));
        int64 put_no_slab = (odd_even * (Nx / procs) * (Nz / 2 + 1) * (Ny / procs)) + (put_odd_even * (Nz / 2 + 1)) + (Ny_no * (Nz / 2 + 1) * (Nx / procs));
        int64 put_no_full = put_no_slab + Nx_no;
        matrix_transpose[put_no_full] = matrix_data[i];
    }
}

// ########### Explicit instantiation ###########
template __global__ void TRANSPOSE::chunk_transpose_inverse_<TRANSITIONS::T2_f>(TRANSITIONS::T2_f *matrix_data, TRANSITIONS::T2_f *matrix_transpose, int64 Nx, int64 Ny, int64 Nz, int procs);
template __global__ void TRANSPOSE::chunk_transpose_<TRANSITIONS::T2_f>(TRANSITIONS::T2_f *matrix_data, TRANSITIONS::T2_f *matrix_transpose, int64 Nx, int64 Ny, int64 Nz, int procs);
template __global__ void TRANSPOSE::transpose_slab_<TRANSITIONS::T2_f>(TRANSITIONS::T2_f *matrix_data, TRANSITIONS::T2_f *matrix_transpose, int64 Ny_current_slab, int64 Nx_current_slab, int64 Nz_current_slab);

template __global__ void TRANSPOSE::chunk_transpose_inverse_<TRANSITIONS::T2_d>(TRANSITIONS::T2_d *matrix_data, TRANSITIONS::T2_d *matrix_transpose, int64 Nx, int64 Ny, int64 Nz, int procs);
template __global__ void TRANSPOSE::chunk_transpose_<TRANSITIONS::T2_d>(TRANSITIONS::T2_d *matrix_data, TRANSITIONS::T2_d *matrix_transpose, int64 Nx, int64 Ny, int64 Nz, int procs);
template __global__ void TRANSPOSE::transpose_slab_<TRANSITIONS::T2_d>(TRANSITIONS::T2_d *matrix_data, TRANSITIONS::T2_d *matrix_transpose, int64 Ny_current_slab, int64 Nx_current_slab, int64 Nz_current_slab);
// ##############################################