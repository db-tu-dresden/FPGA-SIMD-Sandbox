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
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <array>
#include <iomanip>
#include <chrono>
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

#include "kernel.hpp"
#include "kernel.cpp"
#include "LinearProbing_scalar.cpp"
#include "helper_main.cpp"

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

constexpr size_t kDDRChannels = DDR_CHANNELS;
constexpr size_t kDDRWidth = DDR_WIDTH;
constexpr size_t kDDRInterleavedChunkSize = DDR_INTERLEAVED_CHUNK_SIZE;
constexpr size_t kPCIeWidth = PCIE_WIDTH;
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// Forward declare functions
//// Forward declar kernel names to reduce name mangling
template<typename T>
bool validate(T *in_host, T *out_host, size_t size);

void exception_handler(exception_list exceptions);

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/**
 * define global parameters for data generation
 * @param distinctValues determines the generated values between 1 and distinctValues
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
//uint64_t distinctValues = 8000;
uint64_t distinctValues = 128;
uint64_t dataSize = 16*10000000;
float scale = 1.4;
uint64_t HSIZE = distinctValues*scale;
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
    auto props = property_list{property::queue::enable_profiling()};
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

    if ( argc != 2 ) { // argc should be 2 for correct execution
        size = 1024;
	} else {
		size = atoi(argv[1]);
	}
    printf("Vector length: %zd \n", size);

    // print global settings
     std::cout << "Global configuration:"<<  std::endl;
     std::cout << "distinctValues | scale-facor | dataSize : "<<distinctValues<<" | "<<scale<<" | "<<dataSize<< std::endl;
    // print hashsize of current settings
    std::cout << "Configured HSIZE : " << HSIZE << std::endl;


    // Define for Allocate input/output data in pinned host memory
    // Used in all three tests, for convenience
    using Type = uint32_t;  // type to use for the test
    Type *arr_h, *arr_d; 
    Type *hashVec_d, *countVec_d;
    long *out_v1_h, *out_v1_d

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant1()

	printf("\n \n ### aggregation ### \n\n");
	size_t number_CL_buckets = 0;
    size_t number_CL = 0;
	
	if(size % (4096) == 0) {
		number_CL_buckets = size / (4096);
	} else {
		number_CL_buckets = size / (4096) + 1;
	}
	
    number_CL = number_CL_buckets * (4096/16);
    
	printf("Number CL buckets: %zd \n", number_CL_buckets);
    printf("Number CLs: %zd \n", number_CL);

//  Allocate input/output data in pinned host memory
//  arr = (uint32_t *) aligned_alloc(64,dataSize * sizeof(uint32_t));
//  hashVec = (uint32_t *) aligned_alloc(64, HSIZE * sizeof (uint32_t));
//  countVec = (uint32_t *) aligned_alloc(64, HSIZE * sizeof (uint32_t));
    
    // Host buffer 
    if ((arr_h = malloc_host<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_v1_h = malloc_host<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_v1_h'\n";
        std::terminate();
    }
    if ((hashVec_h = malloc_device<Type>(HSIZE * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'hashVec_h'\n";
        std::terminate();
    }
    if ((countVec_h = malloc_device<Type>(HSIZE * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'countVec_h'\n";
        std::terminate();
    }  

    // Device buffer  
    if ((arr_h = malloc_device<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_v1_d = malloc_device<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_v1_d'\n";
        std::terminate();
    }
    if ((hashVec_d = malloc_device<Type>(HSIZE * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'hashVec_d'\n";
        std::terminate();
    }
    if ((countVec_d = malloc_device<Type>(HSIZE * sizeof(uint32_t), q)) == nullptr) {
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

    // Init input buffer
    generateData(arr, distinctValues, dataSize);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    // Copy input host buffer to input device buffer
 // *4 Why?   	
    q.memcpy(arr_d, arr_h, dataSize * sizeof(uint32_t)*4);
    q.wait();	

    // init HashMap
    initializeHashMap(hashVec,countVec,HSIZE);
    
    // Copy with zero initialized HashMap (hashVec, countVec) from host to device
    q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
    q.wait();
    q.memcpy(countVev_d, countVev_h, HSIZE * sizeof(uint32_t));
    q.wait();

    // track timing information, in ms
    double pcie_time_v1=0.0;

//SIMD for FPGA function v1 
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Linear Probing for FPGA - SIMD Variant 1:"<<std::endl;
        std::cout << "Running on FPGA Hardware with an dataSize of " << dataSize << " values!" << std::endl;

        // dummy run to program FPGA, dont care first run for measurement
        // old LinearProbingFPGA_variant1(q, arr, hashVec, countVec, dataSize, HSIZE, number_CL*16);
        LinearProbingFPGA_variant1(q, arr_d, hashVec_d, countVec_d, out_v1_d, dataSize, HSIZE, number_CL*16);

        // measured run on FPGA
        auto begin = chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant1(q, arr_d, hashVec_d, countVec_d, out_v1_d, dataSize, HSIZE, number_CL*16);
        auto end = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff = end - start;

        std::cout<<"Kernel runtime of function_v1 in ms: "<< (diff.count()) <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v1=diff.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (exception const& e) {
        std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
        std::terminate();
    }   

    // Copy output device buffer to output host buffer 
    q.memcpy(out_v1_h, out_v1_d, 8);
    q.wait();  
    
//    // check result for correctness
//    validate(dataSize, hashVec,countVec, HSIZE);
//    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
     
    std::cout << "Value in variable dataSize: " << dataSize << std::endl;
    std::cout << "Result value in out_v1_h[0]: " << out_v1_h[0] << std::endl;




    // free USM memory
    sycl::free(arr_h, q);
    sycl::free(hashVec_h, q);
    sycl::free(countVec_h, q)
    sycl::free(out_v1_h, q);
    
    sycl::free(arr_d, q);
    sycl::free(hashVec_d, q);
    sycl::free(countVec_d, q);
    sycl::free(out_v1_d, q);
    

    // print result
    double input_size_mb = dataSize * sizeof(Type) * 1e-6;
	printf("Input_size_mb: %lf \n", input_size_mb);

    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb / (pcie_time_v1 * 1e-3)) << " MB/s\n";



//// end of LinearProbingFPGA_variant1()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant2()


//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////

/*
//SIMD for FPGA function v2 
    initializeHashMap(hashVec,countVec,HSIZE);
    std::cout <<"=============================="<<std::endl;
    std::cout <<"Linear Probing for FPGA - SIMD Variant 2:"<<std::endl;
    begin = chrono::high_resolution_clock::now();
//  LinearProbingFPGA_variant2(q, arr, hashVec, countVec, dataSize, HSIZE, number_CL*16);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));
    std::cout<<mis<<std::endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE); 
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant3()

/*
//SIMD for FPGA function v3 
    initializeHashMap(hashVec,countVec,HSIZE);
    std::cout <<"=============================="<<std::endl;
    std::cout <<"Linear Probing for FPGA - SIMD Variant 3:"<<std::endl;
    begin = chrono::high_resolution_clock::now();
//  LinearProbingFPGA_variant3(q, arr, hashVec, countVec, dataSize, HSIZE, number_CL*16);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));
    std::cout<<mis<<std::endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
*/

//// end of LinearProbingFPGA_variant3()
////////////////////////////////////////////////////////////////////////////////
}


void exception_handler (exception_list exceptions) {                     
  for (std::exception_ptr const& e : exceptions) {
    try {
        std::rethrow_exception(e);
    } catch(exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
            << e.what() << std::endl;
    }
  }
}


/** 	scalar version currently not used

    //scalar version
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - scalar:"<<endl;
    auto begin = chrono::high_resolution_clock::now();
    LinearProbingScalar(arr, dataSize, hashVec, countVec, HSIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    auto mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));
    cout<<mis<<endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);


*/