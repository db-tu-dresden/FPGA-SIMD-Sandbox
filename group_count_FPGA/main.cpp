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

#include "global_settings.hpp"
#include "kernel.hpp"
#include "helper_main.hpp"


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

//// parameter is currently not used
// constexpr size_t kDDRChannels = DDR_CHANNELS;
//// parameter is currently not used
// constexpr size_t kDDRWidth = DDR_WIDTH;
constexpr size_t kDDRInterleavedChunkSize = DDR_INTERLEAVED_CHUNK_SIZE;
//// parameter is currently not used
// constexpr size_t kPCIeWidth = PCIE_WIDTH;

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
//  printf("Vector length: %zd \n", size);

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
    long *out_h, *out_d;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant1()
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
	printf("\n \n ### START of Linear Probing for FPGA - SIMD Variant 1 ### \n\n");
	size_t number_CL_buckets = 0;
    size_t number_CL = 0;
	
	if(size % (4096) == 0) {
		number_CL_buckets = size / (4096);
	} else {
		number_CL_buckets = size / (4096) + 1;
	}
	
    number_CL = number_CL_buckets * (4096/16);
    
//	printf("Number CL buckets: %zd \n", number_CL_buckets);
//  printf("Number CLs: %zd \n", number_CL);

    // Host buffer 
    if ((arr_h = malloc_host<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_h = malloc_host<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_h'\n";
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
    if ((arr_d = malloc_device<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_d = malloc_device<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_d'\n";
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
    generateData(arr_h, distinctValues, dataSize);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    // Copy input host buffer to input device buffer
    q.memcpy(arr_d, arr_h, dataSize * sizeof(uint32_t));
    q.wait();	

    // init HashMap
    initializeHashMap(hashVec_h,countVec_h,HSIZE);
    
    // Copy with zero initialized HashMap (hashVec, countVec) from host to device
    q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
    q.wait();
    q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(uint32_t));
    q.wait();

    // track timing information, in ms
    double pcie_time_v1=0.0;

//SIMD for FPGA function v1 
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : Linear Probing for FPGA - SIMD Variant 1:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run to program FPGA, dont care first run for measurement
        LinearProbingFPGA_variant1(q, arr_d, hashVec_d, countVec_d, out_d, dataSize, HSIZE, number_CL*16);

        // Re-Initialize HashMap after dummy run
        initializeHashMap(hashVec_h,countVec_h,HSIZE);
        q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
        q.wait();
        q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(uint32_t));
        q.wait();

        // measured run on FPGA
        auto begin_v1 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant1(q, arr_d, hashVec_d, countVec_d, out_d, dataSize, HSIZE, number_CL*16);
        auto end_v1 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v1 = end_v1 - begin_v1;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant1(): "<< (diff_v1.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v1=diff_v1.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (sycl::exception const& e) {
        std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
        std::terminate();
    }   

    // Copy output device buffer to output host buffer 
    q.memcpy(out_h, out_d, 8);
    q.wait();  
    q.memcpy(hashVec_h, hashVec_d, HSIZE * sizeof(uint32_t));
    q.wait();  
    q.memcpy(countVec_h, countVec_d, HSIZE * sizeof(uint32_t));
    q.wait();  
    

    std::cout << "Value in variable dataSize: " << dataSize << std::endl;
    std::cout << "Result value in out_h[0]: " << out_h[0] << std::endl;
    std::cout<< " " <<std::endl;

    // check result for correctness
    validate(dataSize, hashVec_h, countVec_h, HSIZE);
    validate_element(arr_h, dataSize, hashVec_h, countVec_h, HSIZE);
    std::cout<< " " <<std::endl;

    // free USM memory
    sycl::free(arr_h, q);
    sycl::free(hashVec_h, q);
    sycl::free(countVec_h, q);
    sycl::free(out_h, q);
    
    sycl::free(arr_d, q);
    sycl::free(hashVec_d, q);
    sycl::free(countVec_d, q);
    sycl::free(out_d, q);
    

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v1 = dataSize * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v1 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v1 / (pcie_time_v1 * 1e-3)) << " MB/s\n";

    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 1 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
//// end of LinearProbingFPGA_variant1()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant2()
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
	printf("\n \n ### START of Linear Probing for FPGA - SIMD Variant 2 ### \n\n");

    /**                 unused - @TODO : adjust and use  
	if(size % (4096) == 0) {
		number_CL_buckets = size / (4096);
	} else {
		number_CL_buckets = size / (4096) + 1;
	}
    number_CL = number_CL_buckets * (4096/16);
    */
//	printf("Number CL buckets: %zd \n", number_CL_buckets);
//  printf("Number CLs: %zd \n", number_CL);
  
    // Host buffer 
    if ((arr_h = malloc_host<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_h = malloc_host<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_h'\n";
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
    if ((arr_d = malloc_device<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_d = malloc_device<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_d'\n";
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
    generateData(arr_h, distinctValues, dataSize);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    // Copy input host buffer to input device buffer
    q.memcpy(arr_d, arr_h, dataSize * sizeof(uint32_t));
    q.wait();	

    // init HashMap
    initializeHashMap(hashVec_h,countVec_h,HSIZE);
    
    // Copy with zero initialized HashMap (hashVec, countVec) from host to device
    q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
    q.wait();
    q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(uint32_t));
    q.wait();

    // track timing information, in ms
    double pcie_time_v2=0.0;

//SIMD for FPGA function v2
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : Linear Probing for FPGA - SIMD Variant 2:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run to program FPGA, dont care first run for measurement
        LinearProbingFPGA_variant2(q, arr_d, hashVec_d, countVec_d, out_d, dataSize, HSIZE, number_CL*16);
    	
        // Re-Initialize HashMap after dummy run
        initializeHashMap(hashVec_h,countVec_h,HSIZE);
        q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
        q.wait();
        q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(uint32_t));
        q.wait();

        // measured run on FPGA
        auto begin_v2 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant2(q, arr_d, hashVec_d, countVec_d, out_d, dataSize, HSIZE, number_CL*16);
        auto end_v2 = std::chrono::high_resolution_clock::now();
        duration<double, std::milli> diff_v2 = end_v2 - begin_v2;

        std::cout<<"Kernel runtime of function LinearProbingFPGA_variant2(): "<< (diff_v2.count()) << " ms." <<std::endl;
        std::cout <<"=============================="<<std::endl;
        pcie_time_v2=diff_v2.count();
        ////////////////////////////////////////////////////////////////////////////
    } 
    catch (sycl::exception const& e) {
        std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
        std::terminate();
    }   

    // Copy output device buffer to output host buffer 
    q.memcpy(out_h, out_d, 8);
    q.wait();  
    q.memcpy(hashVec_h, hashVec_d, HSIZE * sizeof(uint32_t));
    q.wait();  
    q.memcpy(countVec_h, countVec_d, HSIZE * sizeof(uint32_t));
    q.wait();  
    

    std::cout << "Value in variable dataSize: " << dataSize << std::endl;
    std::cout << "Result value in out_h[0]: " << out_h[0] << std::endl;
    std::cout<< " " <<std::endl;

    // check result for correctness
    validate(dataSize, hashVec_h, countVec_h, HSIZE);
    validate_element(arr_h, dataSize, hashVec_h, countVec_h, HSIZE);
    std::cout<< " " <<std::endl;

    // free USM memory
    sycl::free(arr_h, q);
    sycl::free(hashVec_h, q);
    sycl::free(countVec_h, q);
    sycl::free(out_h, q);
    
    sycl::free(arr_d, q);
    sycl::free(hashVec_d, q);
    sycl::free(countVec_d, q);
    sycl::free(out_d, q);
    

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v2 = dataSize * sizeof(Type) * 1e-6;
	std::cout << "Input_size_mb: " << input_size_mb_v2 <<std::endl;
    std::cout << "HOST-DEVICE Throughput: " << (input_size_mb_v2 / (pcie_time_v2 * 1e-3)) << " MB/s\n";

    std::cout <<" ### End of Linear Probing for FPGA - SIMD Variant 2 ### "<<std::endl;
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;

//// end of LinearProbingFPGA_variant2()
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Forward declare LinearProbingFPGA_variant3()
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"=============================================="<<std::endl;
	printf("\n \n ### START of Linear Probing for FPGA - SIMD Variant 3 ### \n\n");

    /**                 unused - @TODO : adjust and use  
	if(size % (4096) == 0) {
		number_CL_buckets = size / (4096);
	} else {
		number_CL_buckets = size / (4096) + 1;
	}
    number_CL = number_CL_buckets * (4096/16);
    */
//	printf("Number CL buckets: %zd \n", number_CL_buckets);
//  printf("Number CLs: %zd \n", number_CL);
  
    // Host buffer 
    if ((arr_h = malloc_host<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_h = malloc_host<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_h'\n";
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
    if ((arr_d = malloc_device<Type>(dataSize * sizeof(uint32_t), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'arr_h'\n";
        std::terminate();
    }
    if ((out_d = malloc_device<long>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out_d'\n";
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
    generateData(arr_h, distinctValues, dataSize);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    // Copy input host buffer to input device buffer
    q.memcpy(arr_d, arr_h, dataSize * sizeof(uint32_t));
    q.wait();	

    // init HashMap
    initializeHashMap(hashVec_h,countVec_h,HSIZE);
    
    // Copy with zero initialized HashMap (hashVec, countVec) from host to device
    q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
    q.wait();
    q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(uint32_t));
    q.wait();

    // track timing information, in ms
    double pcie_time_v3=0.0;

//SIMD for FPGA function v3
    try {
        ////////////////////////////////////////////////////////////////////////////
        std::cout <<"=============================="<<std::endl;
        std::cout <<"Kernel-Start : Linear Probing for FPGA - SIMD Variant 3:"<<std::endl;
        std::cout << "Running on FPGA Hardware with a dataSize of " << dataSize << " values!" << std::endl;

        // dummy run to program FPGA, dont care first run for measurement
        LinearProbingFPGA_variant3(q, arr_d, hashVec_d, countVec_d, out_d, dataSize, HSIZE, number_CL*16);

        // Re-Initialize HashMap after dummy run
        initializeHashMap(hashVec_h,countVec_h,HSIZE);
        q.memcpy(hashVec_d, hashVec_h, HSIZE * sizeof(uint32_t));
        q.wait();
        q.memcpy(countVec_d, countVec_h, HSIZE * sizeof(uint32_t));
        q.wait();

        // measured run on FPGA
        auto begin_v3 = std::chrono::high_resolution_clock::now();
        LinearProbingFPGA_variant3(q, arr_d, hashVec_d, countVec_d, out_d, dataSize, HSIZE, number_CL*16);
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
    q.memcpy(out_h, out_d, 8);
    q.wait();  
    q.memcpy(hashVec_h, hashVec_d, HSIZE * sizeof(uint32_t));
    q.wait();  
    q.memcpy(countVec_h, countVec_d, HSIZE * sizeof(uint32_t));
    q.wait();  
    

    std::cout << "Value in variable dataSize: " << dataSize << std::endl;
    std::cout << "Result value in out_h[0]: " << out_h[0] << std::endl;
    std::cout<< " " <<std::endl;

    // check result for correctness
    validate(dataSize, hashVec_h, countVec_h, HSIZE);
    validate_element(arr_h, dataSize, hashVec_h, countVec_h, HSIZE);
    std::cout<< " " <<std::endl;

    // free USM memory
    sycl::free(arr_h, q);
    sycl::free(hashVec_h, q);
    sycl::free(countVec_h, q);
    sycl::free(out_h, q);
    
    sycl::free(arr_d, q);
    sycl::free(hashVec_d, q);
    sycl::free(countVec_d, q);
    sycl::free(out_d, q);
    

    // print result
    std::cout << "Final Evaluation of the Throughput: " <<std::endl;
    double input_size_mb_v3 = dataSize * sizeof(Type) * 1e-6;
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