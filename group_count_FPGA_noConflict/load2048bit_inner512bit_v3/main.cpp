/*
###############################
## Created: Eric Stange
##          TU Dresden
##          January 2023
## 
## Used template from:
##          Intel Corporation 
##          Christian Faerber
##          PSG CE EMEA TS-FAE 
##          June 2022
###############################

* This is a hashbased group count implementation using the linear probing approach.
* The Intel Intrinsics from the previous AVX512-based implementation were re-implemented without AVX512.
* This code is intended to be able to run in parallel with the Intel OneAPI on FPGAs.
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

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Time
#include <sys/time.h>
// Sleep
#include <unistd.h>

#include "../config/global_settings.hpp"
#include "kernel.hpp"
#include "../helper/helper_main.hpp"
#include "../helper/datagen.hpp"

using namespace sycl;
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////
//// Board globals. Can be changed from command line.
// default to values in pac_s10_usm BSP
                         
#ifndef DDR_CHANNELS
#define DDR_CHANNELS 4
#endif

#ifndef DDR_WIDTH
#define DDR_WIDTH 64 // bytes (512 bits)
#endif

#ifndef PCIE_WIDTH
#define PCIE_WIDTH 64 // bytes (512 bits)
#endif

#ifndef DDR_INTERLEAVED_CHUNK_SIZE
#define DDR_INTERLEAVED_CHUNK_SIZE 4096 // bytes
#endif

constexpr size_t kDDRInterleavedChunkSize = DDR_INTERLEAVED_CHUNK_SIZE;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// Forward declare functions
template<typename T>
bool validate(T *in_host, T *out_host, size_t size);
void exception_handler(exception_list exceptions);
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// main
int  main(int argc, char** argv){

    // make default input size enough to hide overhead
    #ifdef FPGA_EMULATOR
    long size = kDDRInterleavedChunkSize * 4;
    #else
    long size = kDDRInterleavedChunkSize * 16384;
    #endif

    // the device selector
    #ifdef FPGA_EMULATOR
    ext::intel::fpga_emulator_selector selector;
    #else
    ext::intel::fpga_selector selector;
    #endif

    // create the device queue
    // auto props = property_list{property::queue::enable_profiling()};
    auto props = property_list{};
    queue q(selector, exception_handler, props);

    // make sure the device supports USM device allocations
    device d = q.get_device();
    if (!d.get_info<info::device::usm_device_allocations>()) {
        std::cerr << "ERROR: The selected device does not support USM device"
                << " allocations\n";
        std::terminate();
    }
    if (!d.get_info<info::device::usm_host_allocations>()) {
        std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
        std::terminate();
    }

    /**
     * calculate parameters for memory allocation
     *
     * If a second parameter is passed when running the main.fpga file, 
     * use this as "size", otherwise define the parameter "size" using the value of
     * variable dataSize, which is defined in global_settings.hpp.
    */ 
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
   
    // Define for Allocate input/output data in pinned host memory
    // Used in all three tests, for convenience
    Type *arr_h, *arr_d; 
    Type *hashVec_h, *hashVec_d;
    Type *countVec_h, *countVec_d;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant3()
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
	printf("\n \n ### START of Linear Probing for FPGA - SIMD Variant 3 ### \n\n");

    // Host buffer 
    if ((arr_h = malloc_host<Type>(number_CL*multiplier, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((hashVec_h = malloc_host<Type>(HSIZE, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'hashVec_h'\n";
        std::terminate();
    }
    if ((countVec_h = malloc_host<Type>(HSIZE, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'countVec_h'\n";
        std::terminate();
    }  

    // Device buffer  
    if ((arr_d = malloc_device<Type>(number_CL*multiplier, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_d'\n";
        std::terminate();
    }
    if ((hashVec_d = malloc_device<Type>(HSIZE, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'hashVec_d'\n";
        std::terminate();
    }
    if ((countVec_d = malloc_device<Type>(HSIZE, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'countVec_d'\n";
        std::terminate();
    }  

    // check if memory for input array and HashTable (hashVec and countVec) is allocated correctly (on host)
    if (arr_h != NULL) {
        std::cout << "Memory allocated - " << dataSize << " values, between 1 and " << distinctValues << std::endl;
    } else {
        std::cout << "Memory not allocated!" << std::endl;
    }
    if (hashVec_h != NULL ||  countVec_h != NULL) {
        std::cout << "HashTable allocated - " <<HSIZE<< " values" << std::endl;
    } else {
        std::cout << "HashTable not allocated" << std::endl;
    }

    // Init input buffer with data, that contains NO conflicts!
    // For this we use generate_data_p0 to create an input array with zero conflicts!
    // Due to this manipulated data, we can ignore the while(1) loop inside the kernel.cpp
    size_t data_size = dataSize;
    size_t distinct_values = distinctValues;    
    uint64_t seed = 13;
    size_t (*functionPtr)(Type,size_t);
    functionPtr=&hashx_forNoConflicts;
    generate_data_p0<Type>(arr_h, data_size, distinct_values, functionPtr, 0 , 0 , seed);    
    // generateData<Type>(arr_h);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    // Copy input host buffer to input device buffer
    q.memcpy(arr_d, arr_h, number_CL*multiplier * sizeof(Type));
    q.wait();	

    // init HashMap
    initializeHashMap(hashVec_h,countVec_h);
    
    // Copy with zero initialized HashMap (hashVec, countVec) from host to device
    q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(Type));
    q.wait();
    q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(Type));
    q.wait();

    // track timing information, in ms
    double pcie_time_v3=0.0;

//SIMD for FPGA function v3
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA Variant v3:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run to program FPGA, dont care first run for measurement
        LinearProbingFPGA_variant3(q, arr_d, hashVec_d, countVec_d, number_CL*multiplier);

        // Re-Initialize HashMap after dummy run
        initializeHashMap(hashVec_h,countVec_h);
        q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(Type));
        q.wait();
        q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(Type));
        q.wait();

        // measured run on FPGA
        auto begin_v3 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant3(q, arr_d, hashVec_d, countVec_d, number_CL*multiplier);
        auto end_v3 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v3 = end_v3 - begin_v3;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant3(): "<< (diff_v3.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v3=diff_v3.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (sycl::exception const& e) {
        std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
        std::terminate();
    }   

    // Copy output device buffer to output host buffer 
    q.memcpy(hashVec_h, hashVec_d, HSIZE * sizeof(Type));
    q.wait();  
    q.memcpy(countVec_h, countVec_d, HSIZE * sizeof(Type));
    q.wait();  
    
    std::cout << "Value in variable dataSize: " << dataSize << std::endl;
    std::cout<< " " <<std::endl;

    // check result for correctness
    validate(hashVec_h, countVec_h);
//  validate_element(arr_h, hashVec_h, countVec_h);
    std::cout<< " " <<std::endl;

    // free USM memory
    sycl::free(arr_h, q);
    sycl::free(hashVec_h, q);
    sycl::free(countVec_h, q);
    
    sycl::free(arr_d, q);
    sycl::free(hashVec_d, q);
    sycl::free(countVec_d, q);   

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v3 = size * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v3 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v3 / (pcie_time_v3 * 1e-3)) << " MB/s\n";

    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 3 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////

}
// end of main()

void exception_handler (exception_list exceptions) {                     
  for (std::exception_ptr const& e : exceptions) {
    try {
        std::rethrow_exception(e);
    } catch(sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
            << e.what() << std::endl;
    }
  }
}