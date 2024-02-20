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

// ############## Defination for mpi datatype collection ################
// MPI CALLS datatype
template <>
MPI_Datatype GPU_FFT::get_mpi_datatype_(TRANSITIONS::T2_f a)
{
    return MPI_C_COMPLEX;
}

template <>
MPI_Datatype GPU_FFT::get_mpi_datatype_(TRANSITIONS::T2_d a)
{
    return MPI_C_DOUBLE_COMPLEX;
}
// ######################################################################

template <typename T1, typename T2>
void GPU_FFT::INIT_GPU_FFT_COMM(int procs_in, int rank_in, MPI_Comm &MPI_COMMUNICATOR_INPUT)
{
    // Setting the MPI Communicator
    MPI_COMMUNICATOR_ = MPI_COMMUNICATOR_INPUT;

    // Setting the rank_ and procs_
    rank_ = rank_in;
    procs_ = procs_in;

    // Initiating the MPI requests here
    requests_ = new MPI_Request[2 * (procs_ - 1)];

    // Setting the variable to get mpi datatype
    temp_variable_for_mpi_datatype_<T2> = {0, 0};
}

// ################################# Explicit instantiations ##################################
template <>
TRANSITIONS::T2_f GPU_FFT::temp_variable_for_mpi_datatype_<TRANSITIONS::T2_f>;
template <>
TRANSITIONS::T2_d GPU_FFT::temp_variable_for_mpi_datatype_<TRANSITIONS::T2_d>;

template void GPU_FFT::INIT_GPU_FFT_COMM<TRANSITIONS::T1_f, TRANSITIONS::T2_f>(int procs_in, int rank_in, MPI_Comm &MPI_COMMUNICATOR_INPUT);
template void GPU_FFT::INIT_GPU_FFT_COMM<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(int procs_in, int rank_in, MPI_Comm &MPI_COMMUNICATOR_INPUT);
// ############################################################################################