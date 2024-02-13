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

#ifndef TRANSITIONS_H_
#define TRANSITIONS_H_

#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <initializer_list>
#include <type_traits>

// GPU Includes
#ifdef __HIPCC__
#include <hipfft.h>
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
    // ########### Type Definations ###########
    typedef hipfftReal T1_f;
    typedef hipfftComplex T2_f;
    typedef hipfftDoubleReal T1_d;
    typedef hipfftDoubleComplex T2_d;
    // #########################################
    typedef hipEvent_t event_timers;
    typedef hipStream_t GPU_stream;

    // FFTs
    typedef hipfftType_t ffttype_t;
    typedef hipfftHandle_t *fftHandle;
    static const auto FFT_R2C = HIPFFT_R2C;
    static const auto FFT_D2Z = HIPFFT_D2Z;
    static const auto FFT_C2R = HIPFFT_C2R;
    static const auto FFT_Z2D = HIPFFT_Z2D;
    const auto fftCreate = hipfftCreate;
    const auto fftMakePlanMany = hipfftMakePlanMany;
    const auto fftExecC2R = hipfftExecC2R;
    const auto fftExecR2C = hipfftExecR2C;
    const auto fftExecZ2D = hipfftExecZ2D;
    const auto fftExecD2Z = hipfftExecD2Z;
    const auto fftSetStream = hipfftSetStream;
#endif

#ifdef __CUDACC__
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>

    // ########### Type Definations ###########
    typedef cufftReal T1_f;
    typedef cufftComplex T2_f;
    typedef cufftDoubleReal T1_d;
    typedef cufftDoubleComplex T2_d;
    // #########################################
    typedef cudaEvent_t event_timers;
    typedef cudaStream_t GPU_stream;

    // FFTs
    typedef cufftType_t ffttype_t;
    typedef cufftHandle *fftHandle;
    static const auto FFT_R2C = CUFFT_R2C;
    static const auto FFT_D2Z = CUFFT_D2Z;
    static const auto FFT_C2R = CUFFT_C2R;
    static const auto FFT_Z2D = CUFFT_Z2D;
    const auto fftCreate = cufftCreate;
    const auto fftMakePlanMany = cufftMakePlanMany;
    const auto fftExecC2R = cufftExecC2R;
    const auto fftExecR2C = cufftExecR2C;
    const auto fftExecZ2D = cufftExecZ2D;
    const auto fftExecD2Z = cufftExecD2Z;
    const auto fftSetStream = cufftSetStream;
#endif

#endif