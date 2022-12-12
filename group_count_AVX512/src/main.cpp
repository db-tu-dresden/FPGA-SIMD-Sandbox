#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

#include "LinearProbing_avx512.cpp"
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
uint64_t distinctValues = 10;
uint64_t dataSize = 1000000;
float scale = 1.6;
uint64_t HSIZE = distinctValues * scale;


int  main(int argc, char** argv){
    // print hashsize of current settings
    cout << "Configured HSIZE : " << HSIZE << endl;

    /**
     * allocate memory for data input array and fill with random numbers
     */
    uint32_t *arr;
    arr = (uint32_t *) aligned_alloc(32,dataSize * sizeof(uint32_t));
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
    hashVec = (uint32_t *) aligned_alloc(32, HSIZE * sizeof (uint32_t));
    countVec = (uint32_t *) aligned_alloc(32, HSIZE * sizeof (uint32_t));

    if (hashVec != NULL ||  countVec != NULL) {
        cout << "HashTable allocated - " <<HSIZE<< " values" << endl;
    } else {
        cout << "HashTable not allocated" << endl;
    }
    

    LinearProbingAVX512(arr, dataSize, hashVec, countVec, HSIZE);

    // Ausgabe des HashTable
    uint32_t sum;
    for (int i=0; i<HSIZE; i++) {
        if (hashVec[i]>0) {
            sum+=countVec[i];
        }
    }
    cout << "Final result check: compare parameter dataSize against sum of all count values in countVec:"<<endl;
    cout << dataSize <<" "<<sum<<endl;
    
    return 0;

}
