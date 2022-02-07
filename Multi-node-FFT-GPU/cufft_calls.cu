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
    \author ---> Manthan verma, Soumyadeep Chatterjee, Gaurav Garg, Bharatkumar Sharma, Nishant Arya, Shashi Kumar, Mahendra K.Verma
    \dated --> Feb 2022
    \copyright New BSD License
*/

#include "header.cuh"

// Explicit initialization of CUFFT R2C
template<> void cufft_call_r2c<cufftReal,cufftComplex>(cufftHandle &plan,cufftReal* input_data,cufftComplex* output_data)
{
    gpuerrcheck_cufft(cufftExecR2C(plan, input_data, output_data), __LINE__);
}

template<> void cufft_call_r2c<cufftDoubleReal,cufftDoubleComplex>(cufftHandle &plan,cufftDoubleReal* input_data,cufftDoubleComplex* output_data)
{
    gpuerrcheck_cufft(cufftExecD2Z(plan, input_data, output_data), __LINE__);
}


// Explicit initialization of CUFFT C2C
template<> void cufft_call_c2c<cufftComplex>(cufftHandle &plan,cufftComplex* input_data, int direction)
{
    gpuerrcheck_cufft(cufftExecC2C(plan, input_data, input_data,direction), __LINE__);
}

template<> void cufft_call_c2c<cufftDoubleComplex>(cufftHandle &plan,cufftDoubleComplex* input_data,int direction)
{
    gpuerrcheck_cufft(cufftExecZ2Z(plan, input_data, input_data,direction), __LINE__);
}

// Explicit initialization of CUFFT C2R
template<> void cufft_call_c2r<cufftComplex,cufftReal>(cufftHandle &plan,cufftComplex* input_data,cufftReal* output_data)
{
    gpuerrcheck_cufft(cufftExecC2R(plan, input_data, output_data), __LINE__);
}

template<> void cufft_call_c2r<cufftDoubleComplex,cufftDoubleReal>(cufftHandle &plan,cufftDoubleComplex* input_data,cufftDoubleReal* output_data)
{
    gpuerrcheck_cufft(cufftExecZ2D(plan, input_data, output_data), __LINE__);
}

// MPI CALLS datatype

template<> MPI_Datatype mpi_type_call(float a)
{
    return MPI_CXX_COMPLEX;
}

template<> MPI_Datatype mpi_type_call(double a)
{
    return MPI_CXX_DOUBLE_COMPLEX;
}