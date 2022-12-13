#include<stdio.h>
#include<stdlib.h>

using namespace std;


// simple multiplicative hashing function
unsigned int hashx(int key, int HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

// print function for vector of type __m512i
void print512_num(__m512i var) {
    uint32_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Content of __m512i Array: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n", 
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}

/** 
 * helper function to print an integer in bit-wise notation 
 * Assumes little endian
 * print-result:    p16, p15, p14, p13, p12, p11, p10, p09, p08, p07, p06, p05, p04, p03, p02, p01
 * usage: printBits(sizeof(nameOfMask), &nameOfMask);
 */
void printBits(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u ", byte);
        }
    }
    puts("");
}



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
    printf("###########################\n");
    printf("Helper function during development process :\n");
    printf("Generated values and their count of occurence inside the input data array:\n");
    int j;
    for(j=0;j<distinctValues;j+=1) {
        printf("Count of value '%i' inside data input array: %i\n", j+1, countArrayForComparison[j]); 
    }
    printf("###########################\n");
}