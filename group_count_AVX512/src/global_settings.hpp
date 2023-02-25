#ifndef GLOBAL_SETTINGS_H_
#define GLOBAL_SETTINGS_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Define global parameters (on host) for data generation
/**
 * @param distinctValues determines the generated values between 1 and distinctValues
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues) 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 */

    //static uint64_t distinctValues = 8000;
    static const uint64_t distinctValues = 128;
    static const uint64_t dataSize = 16*10000000;
    static const float scale = 1.4;
    static const uint64_t HSIZE = distinctValues*scale;



 	
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#endif