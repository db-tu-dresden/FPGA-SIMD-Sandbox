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


void initializeHashMap(uint32_t* hashVec, uint32_t* countVec, uint64_t HSIZE) {
    //initalize hash array with zeros
    for (int i=0; i<HSIZE;i++) {
        hashVec[i]=0;
        countVec[i]=0;
    }
}

//validates only total count
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

//validates if every entry has the right number of elements and if elements are missing.
void validate_element(uint32_t *data, uint64_t dataSize, uint32_t*hashVec, uint32_t* countVec, uint64_t HSIZE) {
    std::cout << "Element Validation\n";
    size_t errors_found = 0;
    uint32_t lowest = 0;
    size_t m_id = 0;
    uint32_t *nr_list = new uint32_t[HSIZE];
    uint32_t *nr_count = new uint32_t[HSIZE];

    for(size_t nr = 0; nr < dataSize; nr++){
        uint32_t value = data[nr];
        bool found = false;
        for(size_t i = 0; i < m_id; i++){
            if(nr_list[i] == value){
                found = true;
                nr_count[i]++;
            }
        }
        if(!found){
            nr_list[m_id] = value;
            nr_count[m_id] = 1;
            m_id++;
        }
    }

    for(size_t val_id = 0; val_id < m_id; val_id++){
        uint32_t validation_val = nr_list[val_id];
        bool found = false;
        for(size_t hash_id = 0; hash_id < HSIZE; hash_id++){
            uint32_t hash_val = hashVec[hash_id];
            if(hash_val == validation_val){
                found = true;
                if(countVec[hash_id] != nr_count[val_id]){
                    std::cout << "\tERROR\tCount\t\t" << hash_val << "\thas a count of " 
                        << countVec[hash_id] << "\tbut should have a count of " 
                        << nr_count[val_id] << std::endl;
                    errors_found++;
                }
                break;
            }
        }
        if(!found){
            std::cout << "\tERROR\tMissing\t\t" << validation_val << "\tis missing. It has a count of " 
                << nr_count[val_id] << std::endl; 
            errors_found++;
        }
    }
    if(errors_found == 1){
        std::cout << "Element Validation found " << errors_found << " Error\n";
    }
    else if(errors_found > 1){
        std::cout << "Element Validation found " << errors_found << " Errors\n";
    }else{
        std::cout << "Element Validation didn't find any Errors\n";
    }
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
    /*printf("###########################\n");
    printf("Helper function during development process :\n");
    printf("Generated values and their count of occurence inside the input data array:\n");
    int j;
    for(j=0;j<distinctValues;j+=1) {
        printf("Count of value '%i' inside data input array: %i\n", j+1, countArrayForComparison[j]); 
    }
    printf("###########################\n");*/
}