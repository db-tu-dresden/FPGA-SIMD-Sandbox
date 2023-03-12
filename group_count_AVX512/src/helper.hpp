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
uint32_t exponentiation_primitive_uint32_t(int x, int a);
uint64_t exponentiation_primitive_uint64_t(int x, int a);

/**
  *  Generate a data array with random values between 1 and #distinctValues
  *  The array is dynamically sized. The number of elements corresponds to the value in dataSize.
  * @todo : change data generation function to a function with real random values
  */
template <typename T>
void generateData(T* arr, uint64_t distinctValues, uint64_t dataSize) {
    int i;    
    for(i=0;i<dataSize;i+=1){
        arr[i] = 1+ (rand() % distinctValues);
    }
} 


#endif  // HELPER_HPP