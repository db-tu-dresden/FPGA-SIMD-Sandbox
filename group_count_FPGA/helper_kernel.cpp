#include<stdio.h>
#include<stdlib.h>

using namespace std;

// simple multiplicative hashing function
unsigned int hashx(int key, int HSIZE) {
    return ((unsigned long)((unsigned int)1300000077*key)* HSIZE)>>32;
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