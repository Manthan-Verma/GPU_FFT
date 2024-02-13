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
void GPU_FFT<T1, T2>::GPU_FFT_R2C(T1 *input_data)
{
    // 2D R2C CUFFT transform
    cufft_call_r2c(planR2C, input_data, (T2 *)(input_data));

    // Transpose slab wise
    transpose_slab<<<grid_fourier_space, block_fourier_space>>>(((T2 *)input_data), buffer, Ny, (Nx / procs), Nz);
    Device_synchronize();

    // Communication
    comm_no = 0;
    for (int i = 0; i < procs; i++)
    {
        if (i != rank)
        {
            MPI_Irecv(((T2 *)input_data) + i * ((Nx / procs) * (Ny / procs) * (Nz / 2 + 1)), (Nx / procs) * (Ny / procs) * (Nz / 2 + 1), get_mpi_datatype(temp_variable_for_mpi_datatype), i, 0, MPI_COMMUNICATOR, &(requests[(procs - 1) + comm_no]));
            MPI_Isend(buffer + i * ((Nx / procs) * (Ny / procs) * (Nz / 2 + 1)), (Nx / procs) * (Ny / procs) * (Nz / 2 + 1), get_mpi_datatype(temp_variable_for_mpi_datatype), i, 0, MPI_COMMUNICATOR, &(requests[comm_no]));
            comm_no++;
        }
    }
    Memory_copy_gpu_to_gpu((buffer + rank * ((Nx / procs) * (Ny / procs) * (Nz / 2 + 1))), (((T2 *)input_data) + rank * ((Nx / procs) * (Ny / procs) * (Nz / 2 + 1))), (Nx / procs) * (Ny / procs) * (Nz / 2 + 1));
    MPI_Waitall(2 * (procs - 1), requests, MPI_STATUS_IGNORE);

    // Transpose within chunk, save in slab_outp_data_tr to save space
    chunk_transpose<<<grid_fourier_space, block_fourier_space>>>(((T2 *)input_data), buffer, Nx, Ny, Nz, procs);

    // 1D FFT along X
    cufft_call_c2c(planC2C, buffer, ((T2 *)input_data), FFT_FORWARD);
}

// ########### Explicit instantiation of class templates ##################
template class GPU_FFT<T1_f, T2_f>;
template class GPU_FFT<T1_d, T2_d>;
// ########################################################################