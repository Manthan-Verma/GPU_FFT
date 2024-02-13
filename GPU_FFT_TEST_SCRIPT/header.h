#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <initializer_list>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>
#include <GPU_FFT/GPU_FFT.h>

int64 Nx{0};
int64 Ny{0};
int64 Nz{0};

double2 *data_cpu_in;
double2 *data_cpu_out;
double2 *data_gpu_in;

dim3 grid_basic;
dim3 block_basic;

int rank;
int procs;

double pi = M_PI;
double dx;
double dy;
double dz;

MPI_Comm MPI_COMMUNICATOR;