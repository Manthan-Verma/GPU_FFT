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
void GPU_FFT::GPU_FFT_R2C(T1 *input_data)
{
    // 2D R2C CUFFT transform
    FFT_DEFINATIONS::fft_call_r2c(planR2C_, input_data, (T2 *)(input_data));

    // Transpose slab wise
    TRANSPOSE::transpose_slab_<<<grid_fourier_space_, block_fourier_space_>>>(((T2 *)input_data), buffer_<T2>, Ny_, (Nx_ / procs_), Nz_);
    TRANSITIONS::Device_synchronize();

    // Communication
    comm_no_ = 0;
    for (int i = 0; i < procs_; i++)
    {
        if (i != rank_)
        {
            MPI_Irecv(((T2 *)input_data) + i * ((Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1)), (Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1), get_mpi_datatype_(temp_variable_for_mpi_datatype_<T2>), i, 0, MPI_COMMUNICATOR_, &(requests_[(procs_ - 1) + comm_no_]));
            MPI_Isend(buffer_<T2> + i * ((Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1)), (Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1), get_mpi_datatype_(temp_variable_for_mpi_datatype_<T2>), i, 0, MPI_COMMUNICATOR_, &(requests_[comm_no_]));
            comm_no_++;
        }
    }
    TRANSITIONS::Memory_copy_gpu_to_gpu((buffer_<T2> + rank_ * ((Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1))), (((T2 *)input_data) + rank_ * ((Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1))), (Nx_ / procs_) * (Ny_ / procs_) * (Nz_ / 2 + 1));
    MPI_Waitall(2 * (procs_ - 1), requests_, MPI_STATUS_IGNORE);

    // Transpose within chunk, save in slab_outp_data_tr to save space
    TRANSPOSE::chunk_transpose_<<<grid_fourier_space_, block_fourier_space_>>>(((T2 *)input_data), buffer_<T2>, Nx_, Ny_, Nz_, procs_);

    // 1D FFT along X
    FFT_DEFINATIONS::fft_call_c2c(planC2C_, buffer_<T2>, ((T2 *)input_data), TRANSITIONS::FFT_FORWARD);
}

// ########### Explicit instantiation of class templates ##################
template void GPU_FFT::GPU_FFT_R2C<TRANSITIONS::T1_f, TRANSITIONS::T2_f>(TRANSITIONS::T1_f *input_data);
template void GPU_FFT::GPU_FFT_R2C<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(TRANSITIONS::T1_d *input_data);
// ########################################################################