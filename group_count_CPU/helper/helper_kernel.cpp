#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

#include "helper_kernel.hpp"

using namespace std;

// simple multiplicative hashing function
unsigned int hashx(int key, int selectable_HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* selectable_HSIZE)>>32;
}

// ####################
// OLD FUNCTIONS - in current version not used anymore
// ####################

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