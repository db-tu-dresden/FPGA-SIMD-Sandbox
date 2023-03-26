/*
###############################
## Created: 
##          TU Dresden
##          2023
## 
###############################

 * This is a hashbased group count implementation using the linear probing approach.
 * The Intel Intrinsics from the previous AVX512-based implementation were re-implemented without AVX512.
 * This (actually serial) code is intended to be able to run it again later in parallel with the Intel OneAPI on FPGAs.
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <array>
#include <iomanip>
#include <numeric>
#include <vector>
#include <time.h>
#include <tuple>
#include <utility>

// Time
#include <sys/time.h>
// Sleep
#include <unistd.h>


#include "../config/global_settings.hpp"
#include "kernel.hpp"
#include "../helper/helper_main.hpp"
#include "LinearProbing_scalar.hpp"

using namespace std::chrono;

/**
 * Compile code with:    see notes in build.sh script
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	OVERVIEW about functions in kernel.cpp
//
//	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
// 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
//	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
//	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
// 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of 
//																					process each item individually, even though it occurs multiple times in the currently loaded data		
// 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// Forward declare functions
struct MyException : public exception {
   const char * what () const throw () {
      return "C++ Exception";
   }
};
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int  main(int argc, char** argv){

    /**
     * calculate parameters for memory allocation
     *
     * If a second parameter is passed when running the main.fpga file, 
     * use this as "size", otherwise define the parameter "size" using the value of
     * variable dataSize, which is defined in global_settings.hpp.
    */ 
    long size = 0;
    if ( argc != 2 ) { // argc should be 2 for correct execution
        size = dataSize;
	} else {
		size = atoi(argv[1]);
	}
    printf("Input vector length (atoi(argv[1])): %zd \n", size);

    size_t number_CL_buckets = 0;
    size_t number_CL = 0;
	
	if(size % (4096) == 0)
	{
		number_CL_buckets = size / (4096);
	}
	else 
	{
		number_CL_buckets = size / (4096) + 1;
	}
	
    number_CL = number_CL_buckets * (4096/multiplier);
    
	printf("Number CL buckets: %zd \n", number_CL_buckets);
    printf("Number CLs: %zd \n", number_CL);


    // print global settings
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"============= Program Start =================="<<std::endl; 
    std::cout <<"=============================================="<<std::endl;    
    std::cout << "Global configuration:"<<  std::endl;
    std::cout << "distinctValues | scale-facor | dataSize : "<<distinctValues<<" | "<<scale<<" | "<<dataSize<< std::endl;
    // print hashsize of current settings
    std::cout << "Configured HSIZE : " << HSIZE << std::endl;
    std::cout << "Configured DATATYPE within registers : " << typeid(Type).name() << std::endl;
    std::cout << "Configured register size (regSize) for data transfer : " << regSize << " byte (= " << (regSize*8) << " bit)" << std::endl;

    /**
     * allocate memory for data input array and fill with random numbers
     */
    uint32_t *arr;
    
    arr = (uint32_t *) aligned_alloc(64,number_CL*multiplier * sizeof(uint32_t));
    if (arr != NULL) {
        std::cout << "Memory allocated - " << dataSize << " values, between 1 and " << distinctValues << std::endl;
    } else {
        std::cout << "Memory not allocated!" << std::endl;
    }
    generateData(arr);     
    std::cout <<"Generation of initial data done."<< std::endl; 

    /**
     * allocate memory for hash array
     */
    uint32_t *hashVec, *countVec; 
    hashVec = (uint32_t *) aligned_alloc(64, HSIZE * sizeof (uint32_t));
    countVec = (uint32_t *) aligned_alloc(64, HSIZE * sizeof (uint32_t));

    if (hashVec != NULL ||  countVec != NULL) {
        std::cout << "HashTable allocated - " <<HSIZE<< " values" << std::endl;
    } else {
        std::cout << "HashTable not allocated" << std::endl;
    }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingScalar()               //scalar version
    // track timing information, in ms
    double pcie_time_v0=0.0;
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : Linear Probing for FPGA - SIMD Variant SCALAR:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run
        initializeHashMap(hashVec,countVec);
        LinearProbingScalar(arr, hashVec, countVec);

        // measured run
        initializeHashMap(hashVec,countVec);
        auto begin_v0 = std::chrono::high_resolution_clock::now();
        LinearProbingScalar(arr, hashVec, countVec);
        auto end_v0 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v0 = end_v0 - begin_v0;

        std::cout<<"Kernel runtime of function LinearProbingScalar(): "<< (diff_v0.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v0=diff_v0.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (std::exception const& e) {
        std::cout << "Caught a exception: " << e.what() << "\n";
        std::terminate();
    }    
       
    // check result for correctness
    validate(hashVec,countVec);
    validate_element(arr, hashVec, countVec);
    std::cout<< " " <<std::endl;

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v0 = dataSize * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v0 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v0 / (pcie_time_v0 * 1e-3)) << " MB/s\n";
        // note: Value is not to be taken seriously in pure host execution!

    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant SCALAR ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
//// end of LinearProbingScalar()
////////////////////////////////////////////////////////////////////////////////    

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant1()    //SIMD for FPGA function v1 
    // track timing information, in ms
    double pcie_time_v1=0.0;
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA Variant v1:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run
        initializeHashMap(hashVec,countVec);
        LinearProbingFPGA_variant1(arr, hashVec, countVec, number_CL*multiplier);

        // measured run
        initializeHashMap(hashVec,countVec);
        auto begin_v1 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant1(arr, hashVec, countVec, number_CL*multiplier);
        auto end_v1 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v1 = end_v1 - begin_v1;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant1(): "<< (diff_v1.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v1=diff_v1.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (std::exception const& e) {
        std::cout << "Caught a exception: " << e.what() << "\n";
        std::terminate();
    }        
    // check result for correctness
    validate(hashVec,countVec);
    validate_element(arr, hashVec, countVec);
    std::cout<< " " <<std::endl;

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v1 = size * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v1 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v1 / (pcie_time_v1 * 1e-3)) << " MB/s\n";
    // note: Value is not to be taken seriously in pure host execution!

    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 1 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;

//// end of LinearProbingFPGA_variant1()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant2()    //SIMD for FPGA function v2 
    // track timing information, in ms
    double pcie_time_v2=0.0;
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA Variant v2:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run
        initializeHashMap(hashVec,countVec);
        LinearProbingFPGA_variant2(arr, hashVec, countVec, number_CL*multiplier);

        // measured run
        initializeHashMap(hashVec,countVec);
        auto begin_v2 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant2(arr, hashVec, countVec, number_CL*multiplier);
        auto end_v2 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v2 = end_v2 - begin_v2;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant2(): "<< (diff_v2.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v2=diff_v2.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (std::exception const& e) {
        std::cout << "Caught a exception: " << e.what() << "\n";
        std::terminate();
    }        
    // check result for correctness
    validate(hashVec,countVec);
    validate_element(arr, hashVec, countVec);
    std::cout<< " " <<std::endl;

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v2 = size * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v2 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v2 / (pcie_time_v2 * 1e-3)) << " MB/s\n";
    // note: Value is not to be taken seriously in pure host execution!
    
    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 2 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;

//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant3()    //SIMD for FPGA function v3 
   // track timing information, in ms
    double pcie_time_v3=0.0;
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA Variant v3:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run
        initializeHashMap(hashVec,countVec);
        LinearProbingFPGA_variant3(arr, hashVec, countVec, number_CL*multiplier);

        // measured run
        initializeHashMap(hashVec,countVec);
        auto begin_v3 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant3(arr, hashVec, countVec, number_CL*multiplier);
        auto end_v3 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v3 = end_v3 - begin_v3;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant3(): "<< (diff_v3.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v3=diff_v3.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (std::exception const& e) {
        std::cout << "Caught a exception: " << e.what() << "\n";
        std::terminate();
    }        
    // check result for correctness
    validate(hashVec,countVec);
    validate_element(arr, hashVec, countVec);
    std::cout<< " " <<std::endl;

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v3 = size * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v3 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v3 / (pcie_time_v3 * 1e-3)) << " MB/s\n";
    // note: Value is not to be taken seriously in pure host execution!
    
    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 3 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;

//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant4()    //SIMD for FPGA function v4 (SoAoV_v1) 
    // track timing information, in ms
    double pcie_time_v4=0.0;
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run
        initializeHashMap(hashVec,countVec);
        LinearProbingFPGA_variant4(arr, hashVec, countVec, number_CL*multiplier);

        // measured run
        initializeHashMap(hashVec,countVec);
        auto begin_v4 = std::chrono::high_resolution_clock::now();
       LinearProbingFPGA_variant4(arr, hashVec, countVec, number_CL*multiplier);
        auto end_v4 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v4 = end_v4 - begin_v4;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant4(): "<< (diff_v4.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v4=diff_v4.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (std::exception const& e) {
        std::cout << "Caught a exception: " << e.what() << "\n";
        std::terminate();
    }        
    // check result for correctness
    validate(hashVec,countVec);
    validate_element(arr, hashVec, countVec);
    std::cout<< " " <<std::endl;

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v4 = size * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v4 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v4 / (pcie_time_v4 * 1e-3)) << " MB/s\n";
    // note: Value is not to be taken seriously in pure host execution!
    
    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 4 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;

//// end of LinearProbingFPGA_variant4()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant5()    //SIMD for FPGA function v5 (SoA_conflict_v1) 

    
    // allocate match_32bit array and fill with values
    // fill the match_32bit array with 32 32bit elements to represent all possible matches found in an register up to 32 elements
	// allocate the array with malloc_device
	// copy this data to the FPGA     
    /*                                  << array is currently implemented directly inside the primitiv-function, declaration here isn't necessary anymore
    Type *match_32bit;
    match_32bit = (Type *) aligned_alloc(64, 32 * sizeof (Type));

	match_32bit[0] =    0x00000001;		match_32bit[1] = 	0x00000002;		match_32bit[2] = 	0x00000004;		match_32bit[3] = 	0x00000008;		
	match_32bit[4] = 	0x00000010;		match_32bit[5] = 	0x00000020;		match_32bit[6] = 	0x00000040;		match_32bit[7] = 	0x00000080;			
	match_32bit[8] = 	0x00000100;		match_32bit[9] = 	0x00000200;		match_32bit[10] = 	0x00000400;		match_32bit[11] = 	0x00000800;		
	match_32bit[12] = 	0x00001000;		match_32bit[13] = 	0x00002000;		match_32bit[14] = 	0x00004000;		match_32bit[15] = 	0x00008000;		
	match_32bit[16] = 	0x00010000;		match_32bit[17] = 	0x00020000;		match_32bit[18] = 	0x00040000;		match_32bit[19] = 	0x00080000;		
	match_32bit[20] = 	0x00100000;		match_32bit[21] = 	0x00200000;		match_32bit[22] = 	0x00400000;		match_32bit[23] = 	0x00800000;		
	match_32bit[24] = 	0x01000000;		match_32bit[25] = 	0x02000000;		match_32bit[26] = 	0x04000000;		match_32bit[27] = 	0x08000000;		
	match_32bit[28] = 	0x10000000;		match_32bit[29] = 	0x20000000;		match_32bit[30] = 	0x40000000;		match_32bit[31] = 	0x80000000;		
    
    if (match_32bit == NULL) {
        std::cout << "match_32bit array not allocated" << std::endl;
    } 
    */
    
    // track timing information, in ms
    double pcie_time_v5=0.0;
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run
        initializeHashMap(hashVec,countVec);
        LinearProbingFPGA_variant5(arr, hashVec, countVec, dataSize);  //difference value for size parameter compared to v1-v4

        // measured run
        initializeHashMap(hashVec,countVec);
        auto begin_v5 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant5(arr, hashVec, countVec, dataSize);  //difference value for size parameter compared to v1-v4
        auto end_v5 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v5 = end_v5 - begin_v5;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant5(): "<< (diff_v5.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v5=diff_v5.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (std::exception const& e) {
        std::cout << "Caught a exception: " << e.what() << "\n";
        std::terminate();
    }        

    // check result for correctness
    validate(hashVec,countVec);
    validate_element(arr, hashVec, countVec);
    std::cout<< " " <<std::endl;

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v5 = size * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v5 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v5 / (pcie_time_v5 * 1e-3)) << " MB/s\n";
    // note: Value is not to be taken seriously in pure host execution!
    
    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 5 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;

//// end of LinearProbingFPGA_variant5()
////////////////////////////////////////////////////////////////////////////////

    return 0;
}