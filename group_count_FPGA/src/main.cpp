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
#include <chrono>
#include <numeric>
#include <vector>
#include <time.h>
#include <tuple>
#include <utility>

// Time
#include <sys/time.h>
// Sleep
#include <unistd.h>

#include "kernels.hpp"
#include "helper.cpp"


using namespace std::chrono;

/**
 * This is a hashbased group count implementation using the linear probing approach.
 * The Intel Intrinsics from the previous AVX512-based implementation were re-implemented without AVX512.
 * This (actually serial) code is intended to be able to run it again later in parallel with the Intel OneAPI on FPGAs.
*/

/**
 * define global parameters for data generation
 * @param distinctValues determines the generated values between 1 and distinctValues
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */
uint64_t distinctValues = 60;
uint64_t dataSize = 16*10000000;
float scale = 1.4;
uint64_t HSIZE = distinctValues*scale;

int  main(int argc, char** argv){
    // print hashsize of current settings
    std::cout << "Configured HSIZE : " << HSIZE << std::endl;

    /**
     * allocate memory for data input array and fill with random numbers
     */
    uint32_t *arr;
    
    arr = (uint32_t *) aligned_alloc(64,dataSize * sizeof(uint32_t));
    if (arr != NULL) {
        std::cout << "Memory allocated - " << dataSize << " values, between 1 and " << distinctValues << std::endl;
    } else {
        std::cout << "Memory not allocated!" << std::endl;
    }
    generateData(arr, distinctValues, dataSize);     
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

    initializeHashMap(hashVec,countVec,HSIZE);
    cout <<"=============================="<<endl;
    cout <<"Linear Probing for FPGA - SIMD Variant1"<<endl;
    auto begin = chrono::high_resolution_clock::now();
    LinearProbingFPGA_variant1(arr, dataSize, hashVec, countVec, HSIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

    auto mis = (dataSize/1000000)/((double)duration/(double)((uint64_t)1*(uint64_t)1000000000));
    cout<<mis<<endl;
    validate(dataSize, hashVec,countVec, HSIZE);


    return 0;

}