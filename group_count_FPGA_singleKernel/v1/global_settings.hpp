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
 * @param multiplier
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 * @param Type define datatype which is used within all registers
 * @param regSize define register-size (in byte), which defines the amount of data that is load within one clock cycle :: (64=512bit; 128=1024bit; 192=1536bit; 256=2048bit;)
 */
    //static uint64_t distinctValues = 8000;
    static const uint64_t distinctValues = 128;

    // change of multiplier not really necessary, but when: only in steps of 16 => e.g. 16, 32, 64 ...
    // and : multiplier should be equal with value of kValuesPerLSU in kernel.cpp
    static const int multiplier = 16;
    static const uint64_t dataSize = multiplier*10000000;
    static const float scale = 1.4;
    static const uint64_t HSIZE = distinctValues*scale;
    
//////// Up to this point the parameters can be adjusted.
////////////////////////////////////////////////////////////////////////////////
  

////////////////////////////////////////////////////////////////////////////////
//////// DO NOT CHANGE THE FOLLOWING SETTINGS :
    /**
     * define datatype which is used within all registers
     * NOTE: DON'T CHANGE these parameters!
     */ 
    using Type = uint32_t;     
    using TypeSigned = int32_t;

    /**
    * define register-size (in byte), which defines the amount of data that is load within one clock cycle
    * NOTE: 64=512bit; 128=1024bit; 192=1536bit; 256=2048bit;
    * NOTE: DON'T CHANGE - PLEASE USE ONLY 256 byte !
    * NOTE: 	Due to current data loading approach, regSize must be 256 byte, so that
    *           every register has a overall size of 2048 bit so that it can be loaded in one cycle using the 4 memory controllers
    */
    constexpr int regSize = 256; 

// DO NOT CHANGE!
    // const Type loops = (dataSize / (regSize/sizeof(Type)));
        // We don't need loops anymore, since we calculate the iterations using the DDR4-parameter and the calculated offset due to 4k pages
    const Type elementCount = (regSize/sizeof(Type));
    // @ TODO : check, if Type & regSize match regarding max 2048 bit for FPGA with 4x DDR4 memory controller
 	
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#endif