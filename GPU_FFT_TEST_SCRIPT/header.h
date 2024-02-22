#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include <GPU_FFT/GPU_FFT.h>

int64 Nx{0};
int64 Ny{0};
int64 Nz{0};

TRANSITIONS::T2_d *data_cpu_in;
TRANSITIONS::T2_d *data_cpu_out;
TRANSITIONS::T2_d *data_gpu_in;

dim3 grid_basic;
dim3 block_basic;

int rank;
int procs;

TRANSITIONS::T1_d pi = M_PI;
TRANSITIONS::T1_d dx;
TRANSITIONS::T1_d dy;
TRANSITIONS::T1_d dz;


MPI_Comm MPI_COMMUNICATOR;