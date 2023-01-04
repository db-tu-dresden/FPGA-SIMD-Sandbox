#include<stdio.h>
#include<stdlib.h>

using namespace std;

// simple multiplicative hashing function
unsigned int hashx(int key, int HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

void initializeHashMap(uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    //initalize hash array with zeros
    for (int i=0; i<HSIZE;i++) {
        hashVec[i]=0;
        countVec[i]=0;
    }
}

void validate(uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    uint64_t sum=0;
    for (int i=0; i<HSIZE; i++) {
        if (hashVec[i]>0) {
            sum+=countVec[i];
        }
    }
    cout << "Final result check: compare parameter dataSize against sum of all count values in countVec:"<<endl;
    cout << dataSize <<" "<<sum<<endl;
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
    }
}

/** 
 * helper function to print an integer in bit-wise notation 
 * Assumes little endian
 * print-result:    p16, p15, p14, p13, p12, p11, p10, p09, p08, p07, p06, p05, p04, p03, p02, p01
 * usage: printBits(sizeof(nameOfMask), &nameOfMask);
 */
/*void printBits(size_t const size, void const * const ptr) {
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
}*/