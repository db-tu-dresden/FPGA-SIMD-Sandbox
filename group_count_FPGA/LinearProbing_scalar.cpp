#include<stdio.h>
#include<stdlib.h>
#include<iostream>

//#include "helper.cpp"

void LinearProbingScalar(uint32_t* input, uint64_t dataSize, uint32_t* hashVec, uint32_t* countVec, int HSIZE) {

    //iterate over input
    int p = 0;
    while (p<dataSize) {
        uint32_t hash_key = hashx(input[p],HSIZE);
        //cout <<input[p]<<" "<<hash_key<<endl;

        // get hash entry at the position hash_key
        uint32_t value = hashVec[hash_key];

        // entry does match the key (key match)
        // just increment count entry
        if (value == input[p]) {
             countVec[hash_key]++;
        } 
        // entry does not match the key (key mismatch)
        else {
            // option 1: entry is zero --> that means entry is empty
            // insert key and increment count entry
            if (value == 0) {
                hashVec[hash_key] = input[p];
                countVec[hash_key]++;
            } 
            // option 2: entry is not zero --> entry is occupied by another key (key conflict)
            // start a linear search from position hash_key
            else {
                //cout <<"Conflict detected"<<endl;
                // HSIZE-1 possibilites
                while (1) {
                    // increment hash_key by 1
                    // compute modulo HSIZE to prevent overflow
                    // if overflow starting at pos 0                    
                    hash_key = (hash_key+1)%HSIZE;
                    value = hashVec[hash_key];

                    // option 2.1: entry does match the key (key match)
                    // just increment count entry
                    if (value == input[p]) {
                         countVec[hash_key]++;
                         break;
                    } 
                    // option 2.2: entry is zero --> that means entry is empty
                    // we know: key is new
                    // insert key at that position and increment count entry at the corresponding position
                    if (value == 0) {
                        hashVec[hash_key] = input[p];
                        countVec[hash_key]++;
                        break;
                    }
                }
            }
        }  
        p++;
    } 
}