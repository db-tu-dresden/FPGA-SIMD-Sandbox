#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <cstring>


#include "global_settings.hpp"
#include "helper.hpp"

using namespace std;


// simple multiplicative hashing function
unsigned int hashx(int key, int HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
}

// simple multiplicative hashing function
// we use our hash function from helper_kernel.hpp, but here we use it for the data generation without data conflicts in generate_data_p0()
size_t hashx_duplicate(Type key, size_t selectable_HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* selectable_HSIZE)>>32;
}

// print function for vector of type __m512i
void print512_num(__m512i var) {
    uint32_t val[16];
    std::memcpy(val, &var, sizeof(val));
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
* adaption of c++ pow-function from cmath:
* pow(double base, double exponent);
*
* own function calculate : result = x^a
* return an uint32_t value -> in this project suitable for use within 64-element registers
*/
uint32_t exponentiation_primitive_uint32_t(int x, int a) {
	uint32_t res = 1;
	if (a == 0) {
		return res;
	} else {
		for (int i=1; i<=a; i++) {
			res = res * x;
		}
		return res;
	}
}

/**	
* adaption of c++ pow-function from cmath:
* pow(double base, double exponent);
*
* own function calculate : result = x^a
* return an uint64_t value -> in this project suitable for use within 64-element registers
*/
uint64_t exponentiation_primitive_uint64_t(int x, int a) {
	uint64_t res = 1;
	if (a == 0) {
		return res;
	} else {
		for (int i=1; i<=a; i++) {
			res = res * x;
		}
		return res;
	}
}