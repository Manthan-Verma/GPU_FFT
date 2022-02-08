# This Repository is for FFT libraray Development and Testing

Here Me and My team ( Manthan Verma , Soumyadeep chatterjee, Nishant , sashi , Mahendra kumar verma Sir ) developing and testing the FFT Library development .</br></br>

## Name_of_code

Name_of_code is an open source parallel C++ code to compute FFT on GPUs using CUDA.

## Getting the Source Code

Name_of_code is hosted on GitHub. You can download the source code from the following link:
       https://github.com/Manthan-Verma/FFT_LIB.git

## Installing Name_of_code

Name_of_code requires several libraries (listed later in this section) for compiling. These libraries can be installed in any directory as long as the said directory is added to the user $PATH.
      
## Required Libraries

The following libraries are required for installing and running fastSF:

    Blitz++ (Version 1.0.2)- All array manipulations are performed using the Blitz++ library. Download Blitz++ from here. After downloading, change to the blitz-master directory and enter the following commands

    CC=gcc CXX=g++ ./configure --prefix=$HOME/local

    make install

    YAML-cpp(Version 0.3.0) - The input parameters are stored in the para.yaml file which needs the YAML-cpp library to parse. Download YAML-cpp from here. Extract the zip/tar file and change the yaml-cpp-release-0.3.0 directory. Important: Please ensure that CMake is installed in your system. Enter the following commands:

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=$HOME/local

    make install

    An MPI (Message Passing Interface) Library - fastSF uses MPI for parallelism. The software was tested using MPICH(Version 3.3.2), however, any standard MPI-1 implementation should be sufficient. Here, we will provide instructions for installing MPICH. Download MPICH from here. After extraction, change to mpich-3.3.2 folder and enter the following:

    CC=gcc CXX=g++ ./configure --prefix=$HOME/local

    make install

    HDF5(Version 1.8.20) - The output files are written in HDF5 format. Download HDF5 from here. After extracting the tar file, change to hdf5-1.8.20 and enter the following:

    CC=mpicc CXX=mpicxx ./configure --prefix=$HOME/local --enable-parallel --without-zlib

    make install

    H5SI(Version 1.1.1) - This library is used for simplifying the input-output operations of HDF5. Download H5SI from here. After downloading the zip file, extract it and change to h5si-master/trunk. Important: Please ensure that CMake is installed in your system. Enter the following:

    CXX=mpicxx cmake . -DCMAKE_INSTALL_PREFIX=$HOME/local

    make

    make install
