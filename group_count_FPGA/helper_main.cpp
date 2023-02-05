#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
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
using namespace std;

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
    std::cout << "Final result check: compare parameter dataSize against sum of all count values in countVec:" << std::endl;
    std::cout << dataSize <<" " << sum << std::endl;
    std::cout <<" " << std::endl;
}

//validates if every entry has the right number of elements and if elements are missing.
void validate_element(uint32_t *data, uint64_t dataSize, uint32_t*hashVec, uint32_t* countVec, uint64_t HSIZE) {
    std::cout << "Element Validation\n";
    size_t errors_found = 0;
    // uint32_t lowest = 0;         // variable not used
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
    int i;    
    for(i=0;i<dataSize;i+=1){
        arr[i] = 1+ (rand() % distinctValues);
    }
}