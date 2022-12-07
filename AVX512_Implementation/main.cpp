#include "src/LinearProbing.cpp"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

/*
*   This is a proprietary AVX512 implementation of Bala Gurumurthy's LinearProbing approach. 
*   The logical approach refers to the procedure described in the paper.
*
*   Link to paper "SIMD Vectorized Hashing for Grouped Aggregation"
*   https://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/Gurumurthy:ADBIS18.pdf
*
*   Link to the Gitlab-repository of Bala Gurumurthy:
*   https://git.iti.cs.ovgu.de/bala/SIMD-Parallel-Hash-Based-Aggregation
*/

// define global variables for data generation
int initialSize = 50000;
int totalSize = 500000;
int dataSize = initialSize;
const int biggestRndNumber = 10;

unsigned int *arr = (unsigned int *)calloc(dataSize,sizeof(unsigned int));

/*
 *   Create an array with 10 elements which contains the count of all generated numbers
 *   This only serves for testing the code and the result of the LinearProbing algorithm.
 */
int countArrayForComparison[biggestRndNumber] = {0};

/*
 *  Generate a data array with random values between 1 and 10
 *  The array is dynamically sized. The number of elements corresponds to the value in dataSize.
 */
void generateData() {
    int i;    
    for(i=0;i<dataSize;i+=1){
        arr[i] = 1+ (rand() % biggestRndNumber);

        /*
        *   Update the count of occurence of the generated value. Therefore, read the 
        *   previous count of occurence, add 1 and store the result in the right place 
        *   inside the "countArrayForComparison" array.
        *   This only serves for testing the code and the result of the LinearProbing algorithm.
        */
        int currentValue = arr[i];
        countArrayForComparison[currentValue-1] = countArrayForComparison[currentValue-1] + 1;
    }
// DELETE - only for testing
arr[0] = 2;
arr[1] = 2;
arr[2] = 2;
}

/*
 *  Function to print all generated integer values and their number of occurence inside the 
 *  data array which is passed to the vectorizedLinearProbing() function. 
 *  This function and print only serves to compare the result of the AVX512 implementation 
 *  against the initially generated data. It is not necessary for the logical flow of the algorithm.
 */
void printCountOfInputOccurence() {
    printf("###########################\n");
    printf("Generated values and their count of occurence inside the input data array:\n");
    int j;
    for(j=0;j<biggestRndNumber;j+=1) {
        printf("Count of value '%i' inside data input array: %i\n", j+1, countArrayForComparison[j]); 
    }
    printf("###########################\n");
}   


int  main(int argc, char** argv){
    generateData();     
    printCountOfInputOccurence();
    vectorizedLinearProbing(arr, dataSize);

}
