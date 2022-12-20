#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <chrono>

#include "LinearProbing_avx512.cpp"
//#include "LinearProbing_avx512-v1.cpp"
#include "LinearProbing_scalar.cpp"
/*
*   This is a proprietary AVX512 implementation of Bala Gurumurthy's LinearProbing approach. 
*   The logical approach refers to the procedure described in the paper.
*   Link to paper "SIMD Vectorized Hashing for Grouped Aggregation"
*   https://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/Gurumurthy:ADBIS18.pdf
*/

/**
 * define global parameters for data generation
 * @param distinctValues determines the generated values between 1 and distinctValues
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
uint64_t distinctValues = 8000;
uint64_t dataSize = 16*10000000;
float scale = 1.1;
uint64_t HSIZE = distinctValues * scale;


int  main(int argc, char** argv){
    // print hashsize of current settings
    cout << "Configured HSIZE : " << HSIZE << endl;

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
    
    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"Linear Probing with AVX512"<<endl;
    auto begin = chrono::high_resolution_clock::now();
    LinearProbingAVX512Variant2(arr, dataSize, hashVec, countVec, HSIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    auto mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));
    cout<<mis<<endl;
     
    
    initializeHashMap(hashVec,countVec,HSIZE);
    begin = chrono::high_resolution_clock::now();
    cout <<"Linear Probing scalar"<<endl;
    LinearProbingScalar(arr, dataSize, hashVec, countVec, HSIZE);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));

    cout <<mis<<endl;
    
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
    uint32_t sum;
    for (int i=0; i<distinctValues * scale; i++) {
        if (hashVec[i]>0) {
            sum+=countVec[i];
        }
    }
    cout << "Final result check: compare parameter dataSize against sum of all count values in countVec:"<<endl;
    cout << dataSize <<" "<<sum<<endl;
    
    return 0;

}
