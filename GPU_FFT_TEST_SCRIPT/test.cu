#include "header.h"

// ###################################### Set GPU device #########################################
extern "C" inline_qualifier void __set_gpu_device__(int device_id)
{
#ifdef __HIPCC__
    hipSetDevice(device_id);
#endif

#ifdef __CUDACC__
    cudaSetDevice(device_id);
#endif
}
// ##############################################################################################

__global__ void normalize(TRANSITIONS::T2_d *data, int64 Nx_gpu, int64 Ny_gpu, int64 Nz_gpu, int procs_gpu)
{
    int64 i = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i < (Nx_gpu * (Ny_gpu / procs_gpu) * (Nz_gpu / 2 + 1)))
    {
        TRANSITIONS::T2_d data_local = data[i];
        data_local.x /= (Nx_gpu * Ny_gpu * Nz_gpu);
        data_local.y /= (Nx_gpu * Ny_gpu * Nz_gpu);

        data[i] = data_local;
    }
}

__global__ void results_show(TRANSITIONS::T2_d *data, int64 Nx_gpu, int64 Ny_gpu, int64 Nz_gpu, int procs_gpu, int rank_gpu)
{
    int64 i = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i < (Nx_gpu * (Ny_gpu / procs_gpu) * (Nz_gpu / 2 + 1)))
    {
        int z = (i % (Nz_gpu / 2 + 1));
        int y = (i / (Nz_gpu / 2 + 1)) % (Ny_gpu / procs_gpu);
        int x = (i / ((Nz_gpu / 2 + 1) * (Ny_gpu / procs_gpu))) % Nx_gpu;
        if ((abs(data[(x * (Nz_gpu / 2 + 1) * (Ny_gpu / procs_gpu)) + (y * (Nz_gpu / 2 + 1)) + z].x) > 1e-1) || (abs(data[(x * (Nz_gpu / 2 + 1) * (Ny_gpu / procs_gpu)) + (y * (Nz_gpu / 2 + 1)) + z].y) > 1e-1))
        {
            TRANSITIONS::T1_d real_part = data[(x * (Nz_gpu / 2 + 1) * (Ny_gpu / procs_gpu)) + (y * (Nz_gpu / 2 + 1)) + z].x;
            TRANSITIONS::T1_d imag_part = data[(x * (Nz_gpu / 2 + 1) * (Ny_gpu / procs_gpu)) + (y * (Nz_gpu / 2 + 1)) + z].y;

            printf("\n rank = %d , data at (%d,%lld,%d)  ------>  %f,%f", rank_gpu, x, ((rank_gpu * (Ny_gpu / procs_gpu)) + y), z, real_part, imag_part);
        }
    }
}

int main(int argc, char *argv[])
{
    Nx = atoi(argv[1]);
    Ny = atoi(argv[1]);
    Nz = atoi(argv[1]);

    MPI_COMMUNICATOR = MPI_COMM_WORLD;

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMMUNICATOR, &rank);
    MPI_Comm_size(MPI_COMMUNICATOR, &procs);

    __set_gpu_device__(rank);

    if (rank == 0)
    {
        std::cout << "\n Nx, Ny, Nz = " << Nx << "," << Ny << "," << Nz;
    }

    dx = {(2 * pi) / Nx};
    dy = {(2 * pi) / Ny};
    dz = {(2 * pi) / Nz};

    // initialize the Memory
    data_cpu_in = (TRANSITIONS::T2_d *)malloc((Nx / procs) * Ny * (Nz / 2 + 1) * sizeof(TRANSITIONS::T2_d));
    data_cpu_out = (TRANSITIONS::T2_d *)malloc((Nx / procs) * Ny * (Nz / 2 + 1) * sizeof(TRANSITIONS::T2_d));
    data_gpu_in = TRANSITIONS::__Memory_allocation_gpu__(data_gpu_in, (Nx / procs) * Ny * (Nz / 2 + 1));

    grid_basic = {static_cast<uint32_t>((Nx * (Ny / procs) * (Nz / 2 + 1) / 256) + 1), 1, 1};
    block_basic = {256, 1, 1};

    // Initialize the data
    for (int64 i = 0, l = ((Nx / procs) * rank); i < (Nx / procs), l < ((Nx / procs) * (rank + 1)); i++, l++)
    {
        for (int64 j = 0; j < Ny; j++)
        {
            for (int64 k = 0; k < Nz; k++)
            {
                ((TRANSITIONS::T1_d *)data_cpu_in)[(i * Ny * (Nz + 2)) + (j * (Nz + 2)) + k] = 8 * ((sin(1.0 * l * dx) * sin(2.0 * j * dy) * sin(3.0 * k * dz)) + (sin(4.0 * l * dx) * sin(5.0 * j * dy) * sin(6.0 * k * dz)));
            }
        }
    }

    for (int64 i = 0, l = ((Nx / procs) * rank); i < (Nx / procs), l < ((Nx / procs) * (rank + 1)); i++, l++)
    {
        for (int64 j = 0; j < Ny; j++)
        {
            ((TRANSITIONS::T1_d *)data_cpu_in)[(i * Ny * (Nz + 2)) + (j * (Nz + 2)) + (Nz + 0)] = 0;
            ((TRANSITIONS::T1_d *)data_cpu_in)[(i * Ny * (Nz + 2)) + (j * (Nz + 2)) + (Nz + 1)] = 0;
        }
    }

    // Copying the data to GPU
    TRANSITIONS::__Memory_copy_cpu_to_gpu__(data_cpu_in, data_gpu_in, (Nx / procs) * Ny * (Nz / 2 + 1));

    // Initializing the Object for FFT
    GPU_FFT::INIT_GPU_FFT<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(Nx, Ny, Nz, procs, rank, MPI_COMMUNICATOR);

    // Performing the FFT
    GPU_FFT::GPU_FFT_R2C<TRANSITIONS::T1_d, TRANSITIONS::T2_d>((TRANSITIONS::T1_d *)(data_gpu_in));
    TRANSITIONS::__Device_synchronize__();

    // Normalizing the output
    normalize<<<grid_basic, block_basic, 0, 0>>>(data_gpu_in, Nx, Ny, Nz, procs);

    // Checking the output
    results_show<<<grid_basic, block_basic, 0, 0>>>(data_gpu_in, Nx, Ny, Nz, procs, rank);
    TRANSITIONS::__Device_synchronize__();

    // Doing inverse FFT
    GPU_FFT::GPU_FFT_C2R<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(data_gpu_in);
    TRANSITIONS::__Device_synchronize__();

    // Copying the output back to CPU
    TRANSITIONS::__Memory_copy_gpu_to_cpu__(data_gpu_in, data_cpu_out, (Nx / procs) * Ny * (Nz / 2 + 1));

    // Now checking the output of inverse FFT
    // TRANSITIONS::T1_d total_error{0}, avg_error{0}, tes{0}, maxerr{0};
    TRANSITIONS::T1_d max_err{0}, tes{0};

    for (int64 i = 0; i < (Nx * (Ny / procs) * Nz); i++)
    {
        tes = std::abs(((TRANSITIONS::T1_d *)data_cpu_in)[i] - ((TRANSITIONS::T1_d *)data_cpu_out)[i]);
        if (tes > max_err)
        {
            max_err = tes;
        }
    }
    std::cout << "\n max error = " << max_err << std::endl;

    MPI_Finalize();
    return 0;
}