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