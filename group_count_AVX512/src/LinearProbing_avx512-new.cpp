#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>

#include "helper.cpp"


/**
 * declare some (global) basic masks and arrays
 */ 
__mmask16 oneMask = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
__mmask16 zeroMask = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
__mmask16 testMask = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0);
__m512i zeroM512iArray = _mm512_setzero_epi32();
__m512i oneM512iArray = _mm512_setr_epi32 (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

void unorderedMap(uint32_t* input, uint64_t dataSize, uint64_t HSIZE) {

    unordered_map<int, int> hmap;

    uint64_t p = 0;
    unordered_map<int,int>::iterator iter;
    while (p < dataSize) {
        uint32_t inputValue = input[p];

        // If key not found in map iterator
        // to end is returned
        iter = hmap.find(inputValue);
        if (iter == hmap.end()) {
            hmap.insert(make_pair(inputValue,1));
        } else {
            iter->second++;
        }
        p++;
    }
}

void InRegister(uint32_t* input, uint64_t dataSize, const uint64_t countRegisters)  {

    /*
    * initialize in-register hash-map
    *
    * struct of arrays of SIMD registers
    * size for 32-bit data
    *   16 element per register
    *   5 registers --> 80 elements
    */
    __m512i hashVecs[countRegisters];
    __m512i countVecs[countRegisters];

    for (int i=0; i<countRegisters; i++) {
        hashVecs[i] = _mm512_setzero_epi32();
        countVecs[i] = _mm512_setzero_epi32();
    }

    __mmask16 masks[17];
    masks[0] = _cvtu32_mask16(0);
    masks[1] = _cvtu32_mask16(1); //2^0
    masks[2] = _cvtu32_mask16(2); //2^1
    masks[3] = _cvtu32_mask16(4); //2^2
    masks[4] = _cvtu32_mask16(8); //2^3
    masks[5] = _cvtu32_mask16(16); //2^4
    masks[6] = _cvtu32_mask16(32); //2^5
    masks[7] = _cvtu32_mask16(64); //2^6
    masks[8] = _cvtu32_mask16(128); //2^7
    masks[9] = _cvtu32_mask16(256); //2^8
    masks[10] = _cvtu32_mask16(512); //2^9
    masks[11] = _cvtu32_mask16(1024); //2^10
    masks[12] = _cvtu32_mask16(2048); //2^11
    masks[13] = _cvtu32_mask16(4096); //2^12
    masks[14] = _cvtu32_mask16(8192); //2^13
    masks[15] = _cvtu32_mask16(16384); //2^14
    masks[16] = _cvtu32_mask16(32768); //2^15
    

    int p = 0;
    while (p < dataSize) {

        // get current input value
        uint32_t inputValue = input[p];

        // determine hash_key based on sizeRegister*16
        //uint32_t hash_key = hashx(inputValue,sizeRegister*16);
        uint32_t hash_key = hashx(inputValue,countRegisters);

        // determine array position of the hash_key
        //uint32_t arrayPos = (hash_key/16);
        uint32_t arrayPos = hash_key;

        /**
        * broadcast element p of input[] to vector of type __m512i
        * broadcastCurrentValue contains sixteen times value of input[i]
        **/
        __m512i broadcastCurrentValue = _mm512_set1_epi32(inputValue);

        while(1) {
        
        // compare vector with broadcast value against vector with following elements for equality
        __mmask16 compareRes = _mm512_cmpeq_epi32_mask(broadcastCurrentValue, hashVecs[arrayPos]);
  
        uint32_t matchPos = (32-__builtin_clz(compareRes)); 

       // cout <<p<<" "<<inputValue<<" + matchPos="<<matchPos<<endl;
       // cout <<inputValue<<" "<<hash_key<<" "<<arrayPos<<" "<<matchPos<<endl;

        // found match
        if (matchPos>0) {
            //cout <<"Update key"<<endl;
            countVecs[arrayPos] = _mm512_mask_add_epi32( countVecs[arrayPos], compareRes ,  countVecs[arrayPos], oneM512iArray);
          //  print512_num(countVecs[arrayPos]);
            p++;
            break;

        } else { // no match found
            // deterime free position within register
            __mmask16 checkForFreeSpace = _mm512_cmpeq_epi32_mask(_mm512_setzero_epi32(),hashVecs[arrayPos]);
            if(checkForFreeSpace != 0) {                // CASE B1    
              //  cout <<"Input new key: "<<endl;
                __mmask16 mask1 = _mm512_knot(checkForFreeSpace);   
                uint32_t pos = (32-__builtin_clz(mask1))%16;
                pos = pos+1;

                //store key
                hashVecs[arrayPos] = _mm512_mask_set1_epi32(hashVecs[arrayPos],masks[pos],inputValue);

                //set count to one
                countVecs[arrayPos] = _mm512_mask_set1_epi32(countVecs[arrayPos],masks[pos],1);
                //print512_num(countVecs[arrayPos]);

                p++;
                break;
            }   else    { 
                //cout<<"B2"<<endl;
                arrayPos = (arrayPos++)%countRegisters;
            }
        }
        }

    }
}