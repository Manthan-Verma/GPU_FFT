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

// Explicit initialization of CUFFT R2C
template <>
void cufft_call_r2c<T1_f, T2_f>(fftHandle &plan, T1_f *input_data, T2_f *output_data)
{
    fftExecR2C(plan, input_data, output_data);
}

template <>
void cufft_call_r2c<T1_d, T2_d>(fftHandle &plan, T1_d *input_data, T2_d *output_data)
{
    fftExecD2Z(plan, input_data, output_data);
}

// Explicit initialization of CUFFT C2C
template <>
void cufft_call_c2c<T2_f>(fftHandle &plan, T2_f *input_data, T2_f *output_data, int direction)
{
    fftExecC2C(plan, input_data, output_data, direction);
}

template <>
void cufft_call_c2c<T2_d>(fftHandle &plan, T2_d *input_data, T2_d *output_data, int direction)
{
    fftExecZ2Z(plan, input_data, output_data, direction);
}

// Explicit initialization of CUFFT C2R
template <>
void cufft_call_c2r<T1_f, T2_f>(fftHandle &plan, T2_f *input_data, T1_f *output_data)
{
    fftExecC2R(plan, input_data, output_data);
}

template <>
void cufft_call_c2r<T1_d, T2_d>(fftHandle &plan, T2_d *input_data, T1_d *output_data)
{
    fftExecZ2D(plan, input_data, output_data);
}
