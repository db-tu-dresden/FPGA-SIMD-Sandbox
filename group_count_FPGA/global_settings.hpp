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
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 * @param Type define datatype which is used within all registers
 * @param regSize define register-size (in byte), which defines the amount of data that is load within one clock cycle :: (64=512bit; 128=1024bit; 192=1536bit; 256=2048bit;)
 */

    //static uint64_t distinctValues = 8000;
    static uint64_t distinctValues = 128;
    static uint64_t dataSize = 16*10000000;
    static float scale = 1.4;
    static uint64_t HSIZE = distinctValues*scale;

// define datatype which is used within all registers
    using Type = uint32_t;     
    using TypeSigned = int32_t;

// define register-size (in byte), which defines the amount of data that is load within one clock cycle
// Note: 64=512bit; 128=1024bit; 192=1536bit; 256=2048bit;
// Note: At the moment, please use ONLY 64, 128 OR 256 byte!! NOT 192!
    constexpr int regSize = 256; 

// DON'T CHANGE!
    const Type loops = (dataSize / (regSize/sizeof(Type)));
    const Type elementCount = (regSize/sizeof(Type));
    // @ TODO : check, if Type & regSize match regarding max 2048 bit for FPGA with 4x DDR4 memory controller
 	
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#endif