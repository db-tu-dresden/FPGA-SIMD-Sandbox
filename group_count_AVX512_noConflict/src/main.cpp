#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
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

#include "global_settings.hpp"
#include "LinearProbing_avx512.hpp"
#include "helper.hpp"
#include "datagen.hpp"

using namespace std::chrono;

/*
*   This is a proprietary AVX512 implementation of Bala Gurumurthy's LinearProbing approach (Version 1). 
 
*   The logical approach refers to the procedure described in the paper.
*   Link to paper "SIMD Vectorized Hashing for Grouped Aggregation"
*   https://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/Gurumurthy:ADBIS18.pdf
*
*   Version 2-5 are modified modified versions with different approaches of a hash-based LinearProbing algorithm.
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	OVERVIEW about functions in LinearProbing_avx512.cpp
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

int  main(int argc, char** argv){
   // print global settings
    std::cout <<"=============================================="<<std::endl;
    std::cout <<"============= Program Start =================="<<std::endl; 
    std::cout <<"=============================================="<<std::endl;    
    std::cout << "Global configuration:"<<  std::endl;
    std::cout << "distinctValues | scale-facor | dataSize : "<<distinctValues<<" | "<<scale<<" | "<<dataSize<< std::endl;
    // print hashsize of current settings
    std::cout << "Configured HSIZE : " << HSIZE << std::endl;

    /**
     * allocate memory for data input array and fill with random numbers
     */
    uint32_t *arr;
    
    arr = (uint32_t *) aligned_alloc(64,dataSize * sizeof(uint32_t));
    if (arr != NULL) {
        cout << "Memory allocated - " << dataSize << " values, between 1 and " << distinctValues << endl;
    } else {
        cout << "Memory not allocated!" << endl;
    }

    // Init input buffer with data, that contains NO conflicts!
    // For this we use generate_data_p0 to create an input array with zero conflicts!
    // Due to this manipulated data, we can ignore the while(1) loop inside the kernel.cpp
    size_t data_size = dataSize;
    size_t distinct_values = distinctValues;    
    uint64_t seed = 13;
    size_t (*functionPtr)(Type,size_t);
    functionPtr=&hashx_duplicate;
    generate_data_p0<Type>(arr, data_size, distinct_values, functionPtr, 0 , 0 , seed);    
    // generateData<Type>(arr);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    /**
     * allocate memory for hash array
     */
    uint32_t *hashVec, *countVec; 
    hashVec = (uint32_t *) aligned_alloc(64, HSIZE * sizeof (uint32_t));
    countVec = (uint32_t *) aligned_alloc(64, HSIZE * sizeof (uint32_t));

    if (hashVec != NULL ||  countVec != NULL) {
        cout << "HashTable allocated - " <<HSIZE<< " values" << endl;
    } else {
        cout << "HashTable not allocated" << endl;
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD AVX512 - v1 (SoA)
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - SIMD Variant 1:"<<endl;
    auto begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant1(arr, dataSize, hashVec, countVec, HSIZE);
    auto end = std::chrono::high_resolution_clock::now();
    duration<double, std::milli> diff_v1 = end - begin;
    cout<<"Elapsed time for LinearProbing algorithm - SIMD Variant 1: "<<(diff_v1.count()) << " ms."<<endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD AVX512 - v2 (SoA)
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - SIMD Variant 2:"<<endl;
    begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant2(arr, dataSize, hashVec, countVec, HSIZE);
    end = std::chrono::high_resolution_clock::now();
    duration<double, std::milli> diff_v2 = end - begin;
    cout<<"Elapsed time for LinearProbing algorithm - SIMD Variant 2: "<<(diff_v2.count()) << " ms."<<endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD AVX512 - v3 (SoA)
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - SIMD Variant 3:"<<endl;
    begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant3(arr, dataSize, hashVec, countVec, HSIZE);
    end = std::chrono::high_resolution_clock::now();
    duration<double, std::milli> diff_v3 = end - begin;
    cout<<"Elapsed time for LinearProbing algorithm - SIMD Variant 3: "<<(diff_v3.count()) << " ms."<<endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD AVX512 - v4 (SoAoV_v1)
    /**
	 * due to this approach, the hash_map and count_map can have overall more slots than the value of HSIZE
     * especially in hash_map[m_HSIZE_v-1]; some of these elements would simply be dropped and not written back with hashVec and countVec;
	 * to avoid this case, we simple set the HSIZE big enough and allocate more space for hashVec and countVec to prevent this error
     *
     * in FPGA-version of this function, we calculate an overflow correction in last register of hash_map and count_map, 
     * to prevent errors from storing elements in hash_map[m_HSIZE_v-1] in positions that are >HSIZE 
	 * --> see FPGA-SIMD-Sandbox/group_count_FPGA/    
	 */
    cout <<"=============================="<<endl;
    cout <<"PREPARING FOR Linear Probing - SIMD Variant 4:"<<endl;
    // calculate necessary HSIZE for v4
    const size_t m_elements_per_vector = (512 / 8) / sizeof(uint32_t);
    const size_t m_HSIZE_v = (HSIZE + m_elements_per_vector - 1) / m_elements_per_vector;
    const size_t HSIZE_v4 = m_HSIZE_v * m_elements_per_vector;

    // re-initialize input arr_v4
    distinct_values = m_HSIZE_v;              // ! We are not allowed to use =distinctValues here, due to the special structure of v4 we have to use =m_HSIZE_v.   
    generate_data_p0<Type>(arr, data_size, distinct_values, functionPtr, 0 , 0 , seed);    
    // generateData<Type>(arr_v4);    
    std::cout <<"Generation of initial data done."<< std::endl; 

    // allocate big enough hashVec and countVec
    uint32_t *hashVec_v4, *countVec_v4; 
    hashVec_v4 = (uint32_t *) aligned_alloc(64, HSIZE_v4 * sizeof (uint32_t));
    countVec_v4 = (uint32_t *) aligned_alloc(64, HSIZE_v4 * sizeof (uint32_t));
    if (hashVec_v4 != NULL ||  countVec_v4 != NULL) {
        cout << "HashTable for v4 allocated - " <<HSIZE<< " values, with modified HSZIE for v4 of: "<<HSIZE_v4<< endl;
    } else {
        cout << "HashTable for v4 not allocated" << endl;
    }

    // begin with execution of algorithm
    initializeHashMap(hashVec_v4,countVec_v4,HSIZE_v4);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - SIMD Variant 4:"<<endl;
    begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant4(arr, dataSize, hashVec_v4, countVec_v4, HSIZE);
    end = std::chrono::high_resolution_clock::now();
    duration<double, std::milli> diff_v4 = end - begin;
    cout<<"Elapsed time for LinearProbing algorithm - SIMD Variant 4 (SoAoV_v1): "<<(diff_v4.count()) << " ms."<<endl;
    validate(dataSize, hashVec_v4, countVec_v4, HSIZE_v4);
    validate_element(arr, dataSize, hashVec_v4, countVec_v4, HSIZE_v4);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////





    
    return 0;

}
