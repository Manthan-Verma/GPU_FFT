/*
Copyright (c) 2022, Manthan-Verma
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

#include "header.cuh"

template <typename T1, typename T2>
void header<T1, T2>::initialize_gpu_data()
{
    // Data INITIALIZATION For GPUS
    grid_slab = {((Nz / 2 + 1) * Ny * Nx_d / 256), 1, 1};
    block_slab = {256, 1, 1};
    grid_chunk = {(Nx * (Nz / 2 + 1) * Ny_d / 256), 1, 1};
    block_chunk = {256, 1, 1};
    cudaMemcpyToSymbol(x_gpu, &Nx, sizeof(int64));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    cudaMemcpyToSymbol(y_gpu, &Ny, sizeof(int64));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    cudaMemcpyToSymbol(z_gpu, &Nz, sizeof(int64));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    cudaMemcpyToSymbol(dx_gpu, &dx, sizeof(double));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    cudaMemcpyToSymbol(dy_gpu, &dy, sizeof(double));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    cudaMemcpyToSymbol(dz_gpu, &dz, sizeof(double));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    //std::cout<<"\n GPU data variables initialized";
}

template <typename T3>
__global__ void results_show<T3>(T3 *data, int rank, int process)
{
    int64 i = threadIdx.x + (blockDim.x * blockIdx.x);
    int z = (i % (z_gpu / 2 + 1));
    int y = (i / (z_gpu / 2 + 1)) % (y_gpu / process);
    int x = (i / ((z_gpu / 2 + 1) * (y_gpu / process))) % x_gpu;
    if ((abs(data[(x * (z_gpu / 2 + 1) * (y_gpu / process)) + (y * (z_gpu / 2 + 1)) + z].x) > 1e-1) || (abs(data[(x * (z_gpu / 2 + 1) * (y_gpu / process)) + (y * (z_gpu / 2 + 1)) + z].y) > 1e-1))
    {
        printf("\n rank = %d , data at (%d,%lld,%d)  ------>  %f,%f", rank, x, ((rank * (y_gpu / process)) + y) , z, data[(x * (z_gpu / 2 + 1) * (y_gpu / process)) + (y * (z_gpu / 2 + 1)) + z].x, data[(x * (z_gpu / 2 + 1) * (y_gpu / process)) + (y * (z_gpu / 2 + 1)) + z].y);
    }
}

template <typename T3>
__global__ void transpose_slab<T3>(T3 *matrix_data, T3 *matrix_transpose, int64 Ny, int64 Nx)
{
    int64 i = threadIdx.x + (blockDim.x * blockIdx.x);
    int z = i % (z_gpu / 2 + 1);
    int y = (i / (z_gpu / 2 + 1)) % Ny;
    int x = (i / ((z_gpu / 2 + 1) * Ny));
    int64 put_no = (x * (z_gpu / 2 + 1)) + (y * (z_gpu / 2 + 1) * Nx) + z;
    matrix_transpose[put_no] = matrix_data[i];
}

template <typename T3>
__global__ void chunk_transpose<T3>(T3 *matrix_data, T3 *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs)
{
    int64 i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int d_tmp_z = Nz / procs;
    int Nx_no = i % Nx;
    int Ny_no = (i / Nx) % Ny;
    int Nz_no = i / (Nx * Ny);
    int odd_even = Nz_no / d_tmp_z;
    int put_odd_even = Nz_no % d_tmp_z;
    int64 put_no_slab = (odd_even * Ny * Nx * Nz / procs) + (put_odd_even * Nx) + (Ny_no * Nx * Nz / procs);
    int64 put_no_full = put_no_slab + Nx_no;
    matrix_transpose[put_no_full] = matrix_data[i];
}

template <typename T3>
__global__ void chunk_transpose_inverse<T3>(T3 *matrix_data, T3 *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs)
{
    int64 i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int Nx_no = i % Nx;
    int Ny_no = (i / Nx) % (Nz / procs);
    int Nz_no = i / (Nx * (Nz / procs));
    int odd_even = Nz_no / Ny;
    int put_odd_even = (Nz_no % Ny);
    int64 put_no_slab = (odd_even * Ny * Nx * Nz / procs) + (put_odd_even * Nx) + (Ny_no * Nx * Ny);
    int64 put_no_full = put_no_slab + Nx_no;
    matrix_transpose[put_no_full] = matrix_data[i];
}

template <typename T3>
__global__ void Normalize_single<T3>(T3 *data, int64 Normalization_size_c2c, int64 Normalization_size_r2c)
{
    int64 i = threadIdx.x + (blockDim.x * blockIdx.x);
    data[i].x /= (Normalization_size_c2c * Normalization_size_r2c);
    data[i].y /= (Normalization_size_c2c * Normalization_size_r2c);
}


//Explicit instantiation of template class
template class header<float,float2>;
template class header<double,double2>;

// Instantiation of functions in template
template __global__ void Normalize_single<cufftComplex>(cufftComplex* data,int64 Normalization_size_c2c, int64 Normalization_size_r2c);
template __global__ void chunk_transpose_inverse<cufftComplex>(cufftComplex *matrix_data, cufftComplex *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs);
template __global__ void chunk_transpose<cufftComplex>(cufftComplex *matrix_data, cufftComplex *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs);
template __global__ void transpose_slab<cufftComplex>(cufftComplex *matrix_data, cufftComplex *matrix_transpose, int64 Ny, int64 Nx);
template __global__ void results_show<cufftComplex>(cufftComplex *data, int rank, int process);

template __global__ void Normalize_single<cufftDoubleComplex>(cufftDoubleComplex* data,int64 Normalization_size_c2c, int64 Normalization_size_r2c);
template __global__ void chunk_transpose_inverse<cufftDoubleComplex>(cufftDoubleComplex *matrix_data, cufftDoubleComplex *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs);
template __global__ void chunk_transpose<cufftDoubleComplex>(cufftDoubleComplex *matrix_data, cufftDoubleComplex *matrix_transpose, int64 Nz, int64 Ny, int64 Nx, int procs);
template __global__ void transpose_slab<cufftDoubleComplex>(cufftDoubleComplex *matrix_data, cufftDoubleComplex *matrix_transpose, int64 Ny, int64 Nx);
template __global__ void results_show<cufftDoubleComplex>(cufftDoubleComplex *data, int rank, int process);