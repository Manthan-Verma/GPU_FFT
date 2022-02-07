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
    \dated --> Feb 2022
    \copyright New BSD License
*/

#include "header.cuh"

template <typename T1, typename T2>
void header<T1, T2>::benchmarking_initialization()
{
    // std::cout << "\n This is mpi";
    MPI_Request requests[2 * (procs - 1)];
    int j{0};

    if (iteration_count == 1)
    {
        // 1 call to iteration loop
        becnhmarking_loop(requests, j);
        //std::cout<<"\n loop done ";
        if (rank == 0)
        {
            cudaMemcpy(local_host_data_out, slab_inp_data, Nx_d * Ny * Nz * sizeof(T1), cudaMemcpyDeviceToHost);
            gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
            
            validate();
        }
    }
    else
    {
        // Warmup calls
        becnhmarking_loop(requests, j);
        becnhmarking_loop(requests, j);

        cudaDeviceSynchronize();
        gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

        // Main loop calls

        start = MPI_Wtime();
        for (int i = 0; i < iteration_count; i++)
        {
            becnhmarking_loop(requests, j);
        }
        cudaDeviceSynchronize();
        stop = MPI_Wtime();

        if (rank == 0)
        {
            Total_time = (((stop - start) * 1000) / iteration_count); // Assiging total time.
            Comm_time = Comm_time / iteration_count;

            // Calls for computation details calculations
            compute_details();
        }
    }
}

template <typename T1, typename T2>
void header<T1, T2>::becnhmarking_loop(MPI_Request requests[], int j)
{
    // 2D R2C CUFFT transform
    cufft_call_r2c(planR2C, slab_inp_data, slab_outp_data);
    // gpuerrcheck_cufft(cufftExecR2C(planR2C, slab_inp_data, slab_outp_data), __LINE__);

    // Transpose slab wise
    transpose_slab<<<grid_slab, block_slab>>>(slab_outp_data, slab_outp_data_tr, Ny, Nx_d);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
    cudaDeviceSynchronize();

    // Communication
    start_comm = MPI_Wtime();
    for (int i = 0; i < procs; i++)
    {
        if (i != rank)
        {
            MPI_Irecv(slab_outp_data + i * (Nx_d * Ny_d * (Nz / 2 + 1)), Nx_d * Ny_d * (Nz / 2 + 1), mpi_type_call(mpi_pass), i, 0, MPI_COMM_WORLD, &requests[(procs - 1) + j]);
            MPI_Isend(slab_outp_data_tr + i * (Nx_d * Ny_d * (Nz / 2 + 1)), Nx_d * Ny_d * (Nz / 2 + 1), mpi_type_call(mpi_pass), i, 0, MPI_COMM_WORLD, &requests[j]);
            j++;
        }
    }
    cudaMemcpy((slab_outp_data + rank * (Nx_d * Ny_d * (Nz / 2 + 1))), (slab_outp_data_tr + rank * (Nx_d * Ny_d * (Nz / 2 + 1))), sizeof(T2) * Nx_d * Ny_d * (Nz / 2 + 1), cudaMemcpyDeviceToDevice);
    MPI_Waitall(2 * (procs - 1), requests, MPI_STATUS_IGNORE);

    stop_comm = MPI_Wtime();
    Comm_time += ((stop_comm - start_comm) * 1000);
    j = 0;

    // Transpose within chunk, save in slab_outp_data_tr to save space
    chunk_transpose<<<grid_chunk, block_chunk>>>(slab_outp_data, slab_outp_data_tr, Ny, Nx_d, (Nz / 2 + 1), procs);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    // 1D FFT along X
    cufft_call_c2c(planC2C, slab_outp_data_tr, CUFFT_FORWARD);
    // gpuerrcheck_cufft(cufftExecZ2Z(planC2C, slab_outp_data_tr, slab_outp_data_tr, CUFFT_FORWARD), __LINE__);

    /*************************************************************************/
    // Output checking

    Normalize_single<<<grid_chunk, block_chunk>>>(slab_outp_data_tr, BATCHED_SIZE_C2C, BATCHED_SIZE_R2C); // Normalizing for Both R2C and C2C in single kernel
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    if (iteration_count == 1)
    {
        cudaDeviceSynchronize();
        results_show<<<grid_chunk, block_chunk>>>(slab_outp_data_tr, rank, procs); // Results Checking
        gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);
        cudaDeviceSynchronize();
    }

    //  /*************************************************************************/             IFFT

    // Inverse 1D
    cufft_call_c2c(planC2C, slab_outp_data_tr, CUFFT_INVERSE);
    // gpuerrcheck_cufft(cufftExecZ2Z(planC2C, slab_outp_data_tr, slab_outp_data_tr, CUFFT_INVERSE), __LINE__);

    // Transpose within chunk, save in slab_outp_data_tr to save space
    chunk_transpose_inverse<<<grid_chunk, block_chunk>>>(slab_outp_data_tr, slab_outp_data, Ny, Nx_d, Nz / 2 + 1, procs);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    cudaDeviceSynchronize();
    start_comm = MPI_Wtime();
    for (int i = 0; i < procs; i++)
    {
        if (i != rank)
        {
            MPI_Irecv(slab_outp_data_tr + i * (Nx_d * Ny_d * (Nz / 2 + 1)), Nx_d * Ny_d * (Nz / 2 + 1), mpi_type_call(mpi_pass), i, 0, MPI_COMM_WORLD, &requests[(procs - 1) + j]);
            MPI_Isend(slab_outp_data + i * (Nx_d * Ny_d * (Nz / 2 + 1)), Nx_d * Ny_d * (Nz / 2 + 1), mpi_type_call(mpi_pass), i, 0, MPI_COMM_WORLD, &requests[j]);
            j++;
        }
    }
    cudaMemcpy((slab_outp_data_tr + rank * (Nx_d * Ny_d * (Nz / 2 + 1))), (slab_outp_data + rank * (Nx_d * Ny_d * (Nz / 2 + 1))), sizeof(T2) * Nx_d * Ny_d * (Nz / 2 + 1), cudaMemcpyDeviceToDevice);
    MPI_Waitall(2 * (procs - 1), requests, MPI_STATUS_IGNORE);

    stop_comm = MPI_Wtime();
    Comm_time += ((stop_comm - start_comm) * 1000);
    j = 0;

    // Transpose

    transpose_slab<<<grid_slab, block_slab>>>(slab_outp_data_tr, slab_outp_data, Nx_d, Ny);
    gpuerrcheck_cudaerror(cudaGetLastError(), __LINE__ - 1);

    cufft_call_c2r(planC2R, slab_outp_data, slab_inp_data);
    // gpuerrcheck_cufft(cufftExecZ2D(planC2R, slab_outp_data, slab_inp_data), __LINE__);
}

template <typename T1, typename T2>
void header<T1, T2>::compute_details()
{
    std::cout << "\n\n This is compute details function ";
    double Size_fft{((double)Nx * (double)Ny * (double)Nz)};
    double Flop_rating = ((std::log(Size_fft)) * 10.0 * (Size_fft)) / (std::log(2.0) * Total_time * (double)1e9); // FLOP RATINGS calculations
    std::cout << "\n Grid Size = " << Nx << " * " << Ny << " * " << Nz << " \t iterations = " << iteration_count << "\t no of process = " << procs;
    std::cout << "\n Total Communication time = " << Comm_time << " ms \t Avg communication time = " << Comm_time / 2 << " ms";
    std::cout << "\n Total time total = " << Total_time << " ms \t Avg time (forward+backward)/2 per iteration = " << Total_time / 2 << " ms";
    std::cout << "\n Flop Ratings = " << Flop_rating << " TFLOPS\n";
}

template <typename T1, typename T2>
void header<T1, T2>::validate()
{
    double total_error{0}, avg_error{0}, tes{0}, maxerr{0};

    for (int64 i = 0; i < Total_data_size_real_per_gpu; i++)
    {
        tes = std::abs(local_host_data_out[i] - local_host_data[i]);
        if (tes > maxerr)
        {
            maxerr = tes;
        }
    }
    // std::cout<<"\n data = "<<local_host_data[2560]<<" , --> "<<local_host_data_out[2560];
    // avg_error = total_error / (Total_data_size_real_per_gpu);
    //std::cout << "\n avg error = " << total_error;
    std::cout << "\n max error = " << maxerr;
}

// Explicit instantiation of template class
template class header<float, float2>;
template class header<double, double2>;