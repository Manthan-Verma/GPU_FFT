/* This is a code that compute FFT on multi-nodes on GPUs*/

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


void calls_to_program(std::initializer_list<int> variables, std::string precision)
{
    // PROGRAME STARTS
    // INITIALIZING CUFFT DATA
    
    if (precision == "single" || precision == "Single")
    {
        if(variables.begin()[5] == 0)
        printf("\n this is Single precision");

        header<cufftReal,cufftComplex> data{variables.begin()[0],variables.begin()[1],variables.begin()[2],variables.begin()[3],variables.begin()[4],variables.begin()[5]};

        data.initialize_cufft_variables();

        data.initialize_gpu_data();

        data.Mem_allocation_gpu_cpu();

        data.data_init_and_copy_to_gpu();

        data.benchmarking_initialization();

    }
    else if(precision == "double" || precision == "Double")
    {
        if(variables.begin()[5] == 0)
        printf("\n this is double precision");

        header<cufftDoubleReal,cufftDoubleComplex> data{variables.begin()[0],variables.begin()[1],variables.begin()[2],variables.begin()[3],variables.begin()[4],variables.begin()[5]};

        data.initialize_cufft_variables();

        data.initialize_gpu_data();

        data.Mem_allocation_gpu_cpu();

        data.data_init_and_copy_to_gpu();

        data.benchmarking_initialization();
    }
    else
    std::cout<<"\n Invalid INPUT Parameters ";
}
int main(int argc, char **argv)
{
    int procs, rank, len;
    char name[10];
    // Initialize MPI
    mpierror(MPI_Init(nullptr, nullptr), __LINE__);
    mpierror(MPI_Comm_size(MPI_COMM_WORLD, &procs), __LINE__);
    mpierror(MPI_Comm_rank(MPI_COMM_WORLD, &rank), __LINE__);
    mpierror(MPI_Get_processor_name(name, &len), __LINE__);

    // Calls to start program 
    std::string precision{argv[5]};
    calls_to_program({atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),atoi(argv[4]),procs,rank},precision);

    MPI_Finalize();
    return 0;
}