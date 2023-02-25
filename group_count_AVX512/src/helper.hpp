#ifndef HELPER_HPP
#define HELPER_HPP

#include<stdio.h>
#include<stdlib.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "global_settings.hpp"

using namespace std;

unsigned int hashx(int key, int HSIZE);
void print512_num(__m512i var);
void printBits(size_t const size, void const * const ptr);
void initializeHashMap(uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);
void validate(uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE);
void validate_element(uint32_t *data, uint64_t dataSize, uint32_t*hashVec, uint32_t* countVec, uint64_t HSIZE);

/**
  *  Generate a data array with random values between 1 and #distinctValues
  *  The array is dynamically sized. The number of elements corresponds to the value in dataSize.
  * @todo : change data generation function to a function with real random values
  */
template <typename T>
void generateData(T* arr, uint64_t distinctValues, uint64_t dataSize) {
    /**
     * Create an array with #distinctValues elements which contains the count of all generated numbers
     * This only serves for testing the code and the result of the LinearProbing algorithm.
     */
    int countArrayForComparison[distinctValues] = {0};

    int i;    
    for(i=0;i<dataSize;i+=1){
        arr[i] = 1+ (rand() % distinctValues);

        /**
         *   Update the count of occurence of the generated value. Therefore, read the 
         *   previous count of occurence, add 1 and store the result in the right place 
         *   inside the "countArrayForComparison" array.
         *   This only serves for testing the code and the result of the LinearProbing algorithm.
         */
        int currentValue = arr[i];
        countArrayForComparison[currentValue-1] = countArrayForComparison[currentValue-1] + 1;
    }

    /**
     *  Function to print all generated integer values and their number of occurence inside the 
     *  data array which is passed to the vectorizedLinearProbing() function. 
     *  This function and print only serves to compare the result of the AVX512 implementation 
     *  against the initially generated data. It is not necessary for the logical flow of the algorithm.
     */
    /*printf("###########################\n");
    printf("Helper function during development process :\n");
    printf("Generated values and their count of occurence inside the input data array:\n");
    int j;
    for(j=0;j<distinctValues;j+=1) {
        printf("Count of value '%i' inside data input array: %i\n", j+1, countArrayForComparison[j]); 
    }
    printf("###########################\n");*/
}


#endif  // HELPER_HPP