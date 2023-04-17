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
#include "LinearProbing_scalar.hpp"
#include "helper.hpp"

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
    generateData(arr, distinctValues, dataSize);     
    cout <<"Generation of initial data done."<<endl; 

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
// scalar version
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - scalar:"<<endl;
    auto begin = chrono::high_resolution_clock::now();
    LinearProbingScalar(arr, dataSize, hashVec, countVec, HSIZE);
    auto end = std::chrono::high_resolution_clock::now();
    // new time calculation
    duration<double, std::milli> diff_scal = end - begin;
    cout<<"Elapsed time for LinearProbing algorithm - scalar version: "<<(diff_scal.count()) << " ms."<<endl;

    // old time calculation
    // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    // auto mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));
    
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD AVX512 - v1 (SoA)
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - SIMD Variant 1:"<<endl;
    begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant1(arr, dataSize, hashVec, countVec, HSIZE);
    end = std::chrono::high_resolution_clock::now();
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
    validate(dataSize, hashVec_v4,countVec_v4, HSIZE_v4);
    validate_element(arr, dataSize, hashVec_v4, countVec_v4, HSIZE_v4);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SIMD AVX512 - v5 (SoA_conflict_v1)
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing - SIMD Variant 5:"<<endl;
    begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant5(arr, dataSize, hashVec, countVec, HSIZE);
    end = std::chrono::high_resolution_clock::now();
    duration<double, std::milli> diff_v5 = end - begin;
    cout<<"Elapsed time for LinearProbing algorithm - SIMD Variant 5 (SoA_conflict_v1): "<<(diff_v5.count()) << " ms."<<endl;
    validate(dataSize, hashVec,countVec, HSIZE);
    validate_element(arr, dataSize, hashVec, countVec, HSIZE);
    cout <<"=============================="<<endl;
    cout<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////





    /*
    //__m512i a = _mm512_setzero_epi32();
    //__m512i b = _mm512_setzero_epi32();
    //__m512i b = _mm512_setr_epi32 (0,0,0,0,12,12,12,12,12,12,12,12,12,12,12,12);
    __m512i b = _mm512_setr_epi32 (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    __m512i a = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    
    //b[0] = (uint32_t)12;
    __mmask16 mask = _mm512_cmpeq_epi32_mask(a,b);
    //__mmask16 mask1 = _mm512_knot(mask);

    cout <<32-__builtin_clz(mask)<<endl;
    */
    // Ausgabe des HashTable
   /* uint32_t sum=0;
    for (int i=0; i<distinctValues * scale; i++) {
        if (hashVec[i]>0) {
            sum+=countVec[i];
        }
    }
    cout << "Final result check: compare parameter dataSize against sum of all count values in countVec:"<<endl;
    cout << dataSize <<" "<<sum<<endl;*/
    
    return 0;

}
