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

template <typename T1, typename T2>
header<T1, T2>::header(int Nx_get, int Ny_get, int Nz_get, int iterations, int process, int MPI_rank) : Nx{Nx_get}, Ny{Ny_get}, Nz{Nz_get}, pi{2 * asin(1.0)}, dx{(2 * pi) / Nx}, dy{(2 * pi) / Ny}, dz{(2 * pi) / Nz},
                                                                                                  Nx_d{Nx / process}, Ny_d{Ny / process}, iteration_count{iterations},
                                                                                                  procs{process},rank{MPI_rank},start{0},stop{0},Total_time{0},start_comm{0},stop_comm{0},Comm_time{0}
{
    Total_data_size_real_per_gpu = Nx_d * Ny * Nz;
    //std::cout << "\n Constructor of class initialized";
}

template <typename T1, typename T2>
void header<T1, T2>::initialize_cufft_variables()
{
    //R2C DATA
    
    n_r2c[0] = Ny;
    n_r2c[1] = Nz;
    BATCHED_SIZE_R2C = Nz * Ny;
    BATCH_r2c = Nx_d;

    // C2C DATA
    n_c2c[0] = Nx;
    inembed_c2c = new int[1]{static_cast<int>(Nx)};
    onembed_c2c = new int[1]{static_cast<int>(Nx)};
    BATCHED_SIZE_C2C = Nx;
    istride_c2c = Ny_d * (Nz / 2 + 1);
    ostride_c2c = Ny_d * (Nz / 2 + 1);
    BATCH_C2C = Ny_d * (Nz / 2 + 1);
    worksize = new size_t{};

    // Initializing some points for plan initialization
    if (std::is_same<T1, cufftDoubleReal>::value)
    {
        // CUFFT TYPES
        cufft_type_r2c = CUFFT_D2Z;
        cufft_type_c2r = CUFFT_Z2D;
        cufft_type_c2c = CUFFT_Z2Z;

        //MPI Datatype
        //complex_type = MPI_CXX_DOUBLE_COMPLEX;

        
    }
    // Plans intiialization
    gpuerrcheck_cufft(cufftCreate(&planR2C), __LINE__);
    gpuerrcheck_cufft(cufftCreate(&planC2R), __LINE__);
    gpuerrcheck_cufft(cufftCreate(&planC2C), __LINE__);
    gpuerrcheck_cufft(cufftMakePlanMany(planC2C, rank_c2c, n_c2c, inembed_c2c, istride_c2c, idist_c2c, onembed_c2c, ostride_c2c, odist_c2c, cufft_type_c2c, BATCH_C2C, worksize), __LINE__);
    gpuerrcheck_cufft(cufftMakePlanMany(planR2C, rank_r2c, n_r2c, inembed_r2c, istride_r2c, idist_r2c, onembed_r2c, ostride_r2c, odist_r2c, cufft_type_r2c, BATCH_r2c, worksize), __LINE__);
    gpuerrcheck_cufft(cufftMakePlanMany(planC2R, rank_r2c, n_r2c, inembed_r2c, istride_r2c, idist_r2c, onembed_r2c, ostride_r2c, odist_r2c, cufft_type_c2r, BATCH_r2c, worksize), __LINE__);

    //std::cout << "\n CUFFT Plan variables initialized";
}

template <typename T1, typename T2>
void header<T1, T2>::Mem_allocation_gpu_cpu()
{
    cudaMalloc(&slab_outp_data, (Nx_d)*Ny * (Nz / 2 + 1) * sizeof(T2));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    cudaMalloc(&slab_outp_data_tr, Nx * Ny_d * (Nz / 2 + 1) * sizeof(T2));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    cudaMalloc(&slab_inp_data, (Nx_d)*Ny * Nz * sizeof(T1));
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    local_host_data = (T1 *)malloc((Nx_d)*Ny * Nz * sizeof(T1));

    local_host_data_out = (T1 *)malloc((Nx_d)*Ny * Nz * sizeof(T1));

    //std::cout<<"\n Memory allocated in host and device";
}

template <typename T1, typename T2>
void header<T1, T2>::data_init_and_copy_to_gpu()
{
    for (int64 i = 0, l = (Nx_d * rank); i < Nx_d, l < (Nx_d * (rank + 1)); i++, l++)
    {
        for (int64 j = 0; j < Ny; j++)
        {
            for (int64 k = 0; k < Nz; k++)
            {
                local_host_data[(i * Ny * Nz) + (j * Nz) + k] = 8 * ((sin(1.0 * l * dx) * sin(2.0 * j * dy) * sin(3.0 * k * dz)) + (sin(4.0 * l * dx) * sin(5.0 * j * dy) * sin(6.0 * k * dz)));
                //std::cout<<"\n local = "<<local_host_data[(i * Ny * Nz) + (j * Nz) + k];
            }
        }
    }

    cudaMemcpy(slab_inp_data, local_host_data, (Nx_d)*Ny * Nz * sizeof(T1), cudaMemcpyHostToDevice);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    
    //std::cout<<"\n data initialized and copied to gpu ";
}


template<typename T1,typename T2>
header<T1,T2>::~header()
{
    cufftDestroy(planC2C);
    cufftDestroy(planC2R);
    cufftDestroy(planR2C);
    free(local_host_data);
    free(local_host_data_out);
    cudaFree(slab_inp_data);
    cudaFree(slab_outp_data);
    cudaFree(slab_outp_data_tr);
    local_host_data = nullptr;
    local_host_data_out = nullptr;
    slab_inp_data = nullptr;
    slab_outp_data = nullptr;
    slab_outp_data_tr = nullptr;
}

//Explicit instantiation of template class
template class header<float,float2>;
template class header<double,double2>;