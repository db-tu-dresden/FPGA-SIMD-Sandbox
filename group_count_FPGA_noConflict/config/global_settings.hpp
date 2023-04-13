#ifndef GLOBAL_SETTINGS_HPP_
#define GLOBAL_SETTINGS_HPP_

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
    //define distinctValues (uint64_t) 8000
    #define distinctValues (uint64_t) 250000

    // change of multiplier not really necessary, but when: only in steps of 16 => e.g. 16, 32, 64 ...
    // and : multiplier should be equal with value of kValuesPerLSU in kernel.cpp
    #define multiplier (int) 16
    #define dataSize (uint64_t) (multiplier*10240000)               // 10240000*16 = 163840000 --> MOD 16 = 0; MOD 4096 = 0
    #define scale (float) (1.0)
    #define HSIZE (uint64_t) (distinctValues*scale)

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
    #define regSize (int) 256               // bytes
    #define inner_regSize (int) 64          // bytes

    #define elements_per_register (int) (regSize/sizeof(Type))                      // old variable name : "elementCount" 
    #define elements_per_inner_register (int) (inner_regSize/sizeof(Type))          // old variable name : "inner_elementCount" 


    // define additional variables and datastructures - only for LinearProbingFPGA_variant4()
	#define m_elements_per_vector (size_t) (elements_per_inner_register) 			// should be equivalent to (regSize)/sizeof(Type);		
	#define m_HSIZE_v (size_t) ((HSIZE + m_elements_per_vector - 1) / m_elements_per_vector)
    #define HSIZE_hashMap_v4 (size_t) (m_elements_per_vector * m_HSIZE_v)
	#define m_HSIZE (size_t) (HSIZE)

    // only for v4_2048bit
	#define m_elements_per_vector_v4_2048bit (size_t) (elements_per_register) 			// should be equivalent to (regSize)/sizeof(Type);		
	#define m_HSIZE_v_v4_2048bit (size_t) ((HSIZE + m_elements_per_vector_v4_2048bit - 1) / m_elements_per_vector_v4_2048bit)
    #define HSIZE_hashMap_v4_v4_2048bit (size_t) (m_elements_per_vector_v4_2048bit * m_HSIZE_v_v4_2048bit)	

    // define additional unroll factor vor load2048bit_virtual_work_4x16_ v1-v5
    #define outerUnroll (int) (regSize/inner_regSize)               // should be 4  in current config
    #define innerUnroll (int) (elements_per_inner_register)         // should be 16 in current config
			
/////////////////////////////////////////////////////////////




 	
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#endif      // GLOBAL_SETTINGS_HPP_