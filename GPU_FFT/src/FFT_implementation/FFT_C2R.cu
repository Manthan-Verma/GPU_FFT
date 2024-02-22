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
void GPU_FFT::GPU_FFT_C2R(T2 *input_data)
{
    // Inverse 1D
    FFT_DEFINATIONS::__fft_call_c2c__(__planC2C__, input_data, __buffer__<T2>, TRANSITIONS::FFT_INVERSE);

    // Transpose within chunk, save in slab_outp_data_tr to save space
    TRANSPOSE::__chunk_transpose_inverse__<<<__grid_fourier_space__, __block_fourier_space__>>>(__buffer__<T2>, input_data, __Nx__, __Ny__, __Nz__, __procs__);
    TRANSITIONS::__Device_synchronize__();

    // Communications
    __comm_no__ = 0;
    for (int i = 0; i < __procs__; i++)
    {
        if (i != __rank__)
        {
            MPI_Irecv(__buffer__<T2> + i * ((__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1)), (__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1), __get_mpi_datatype__(__temp_variable_for_mpi_datatype__<T2>), i, 0, __MPI_COMMUNICATOR__, &__requests__[(__procs__ - 1) + __comm_no__]);
            MPI_Isend(input_data + i * ((__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1)), (__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1), __get_mpi_datatype__(__temp_variable_for_mpi_datatype__<T2>), i, 0, __MPI_COMMUNICATOR__, &__requests__[__comm_no__]);
            __comm_no__++;
        }
    }
    TRANSITIONS::__Memory_copy_gpu_to_gpu__((input_data + __rank__ * ((__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1))), (__buffer__<T2> + __rank__ * ((__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1))), (__Nx__ / __procs__) * (__Ny__ / __procs__) * (__Nz__ / 2 + 1));
    MPI_Waitall(2 * (__procs__ - 1), __requests__, MPI_STATUS_IGNORE);

    // Transpose
    TRANSPOSE::__transpose_slab__<<<__grid_fourier_space__, __block_fourier_space__>>>(__buffer__<T2>, input_data, (__Nx__ / __procs__), __Ny__, __Nz__);
    TRANSITIONS::__Device_synchronize__();

    FFT_DEFINATIONS::__fft_call_c2r__(__planC2R__, input_data, (T1 *)input_data);
}

// ########### Explicit instantiation of class templates ##################
template void GPU_FFT::GPU_FFT_C2R<TRANSITIONS::T1_f, TRANSITIONS::T2_f>(TRANSITIONS::T2_f *input_data);
template void GPU_FFT::GPU_FFT_C2R<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(TRANSITIONS::T2_d *input_data);
// ########################################################################
