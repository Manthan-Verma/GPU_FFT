#include "header.cuh"

// Explicit initialization of CUFFT R2C
template<> void cufft_call_r2c<cufftReal,cufftComplex>(cufftHandle &plan,cufftReal* input_data,cufftComplex* output_data)
{
    gpuerrcheck_cufft(cufftExecR2C(plan, input_data, output_data), __LINE__);
}

template<> void cufft_call_r2c<cufftDoubleReal,cufftDoubleComplex>(cufftHandle &plan,cufftDoubleReal* input_data,cufftDoubleComplex* output_data)
{
    gpuerrcheck_cufft(cufftExecD2Z(plan, input_data, output_data), __LINE__);
}


// Explicit initialization of CUFFT C2C
template<> void cufft_call_c2c<cufftComplex>(cufftHandle &plan,cufftComplex* input_data, int direction)
{
    gpuerrcheck_cufft(cufftExecC2C(plan, input_data, input_data,direction), __LINE__);
}

template<> void cufft_call_c2c<cufftDoubleComplex>(cufftHandle &plan,cufftDoubleComplex* input_data,int direction)
{
    gpuerrcheck_cufft(cufftExecZ2Z(plan, input_data, input_data,direction), __LINE__);
}

// Explicit initialization of CUFFT C2R
template<> void cufft_call_c2r<cufftComplex,cufftReal>(cufftHandle &plan,cufftComplex* input_data,cufftReal* output_data)
{
    gpuerrcheck_cufft(cufftExecC2R(plan, input_data, output_data), __LINE__);
}

template<> void cufft_call_c2r<cufftDoubleComplex,cufftDoubleReal>(cufftHandle &plan,cufftDoubleComplex* input_data,cufftDoubleReal* output_data)
{
    gpuerrcheck_cufft(cufftExecZ2D(plan, input_data, output_data), __LINE__);
}

// MPI CALLS datatype

template<> MPI_Datatype mpi_type_call(float a)
{
    return MPI_CXX_COMPLEX;
}

template<> MPI_Datatype mpi_type_call(double a)
{
    return MPI_CXX_DOUBLE_COMPLEX;
}