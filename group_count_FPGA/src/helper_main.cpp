#include<stdio.h>
#include<stdlib.h>

using namespace std;

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
    cout <<" "<<endl;
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