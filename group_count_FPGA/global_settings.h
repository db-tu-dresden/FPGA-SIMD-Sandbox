#ifndef GLOBAL_SETTINGS_H_
#define GLOBAL_SETTINGS_H_

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// Define global parameters (on host) for data generation
/**
 * @param distinctValues determines the generated values between 1 and distinctValues
 * @param dataSize number of tuples respectively elements in hashVec[] and countVec[]
 * @param scale multiplier to determine the value of the HSIZE (note "1.6" corresponds to 60% more slots in the hashVec[] than there are distinctValues 
 * @param HSIZE HashSize (corresponds to size of hashVec[] and countVec[])
 * @param Type define datatype which is used within all registers
 * @param byteSize define byte-size, which defines the amount of data that is load within one clock cycle :: (64=512bit; 128=1024bit; 192=1536bit; 256=2048bit;)
 */

    //uint64_t distinctValues = 8000;
    uint64_t distinctValues = 128;
    uint64_t dataSize = 16*10000000;
    float scale = 1.4;
    uint64_t HSIZE = distinctValues*scale;

// define datatype which is used within all registers
    using Type = uint32_t;     
    

// define byte-size, which defines the amount of data that is load within one clock cycle
// Note: 64=512bit; 128=1024bit; 192=1536bit; 256=2048bit;
    constexpr int byteSize = 128; 

// DON'T CHANGE!
    const int loops = (dataSize / (byteSize/sizeof(Type)));
    const int elementCount = (byteSize/sizeof(Type));
    // @ TODO : check, if Type & byteSize match regarding max 2048 bit for FPGA with 4x DDR4 memory controller
 	
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#endif