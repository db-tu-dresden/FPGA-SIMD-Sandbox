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
#include <stdexcept>

#include "scalar_remainder.hpp"
#include "../config/global_settings.hpp"
#include "../helper/helper_kernel.hpp"
#include "../primitives/primitives.hpp"

#include "lib/lib.hpp"

#define EMPTY_SPOT 0

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	
//  Within this file, the remaining steps of LinearProbing_v5 are executed.
//  Because these steps are scalar, they aren't executed on the FPGA.
// -> Execution of the remaining scalar steps on the host.
// 
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//// declaration of the classes
class kernelV5_scalar_remainder;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void scalar_remainder_variant5(uint32_t *arr_h, uint32_t *hashVec_h, uint32_t *countVec_h, size_t *p_h) {

    //scalar remainder
	while(p_h[0] < dataSize){
		
		// get the possible possition of the element.
		Type currentValue = arr_h[p_h[0]];                 //arr_d = input_data array
		Type hash_key = hashx(currentValue, HSIZE);

		while(1){
			// get the value of this position
			Type value = hashVec_h[hash_key];
			
			// Check if it is the correct spot
			if(value == currentValue){
				countVec_h[hash_key]++;
				break;
				
			// Check if the spot is empty
			}else if(value == EMPTY_SPOT){
				hashVec_h[hash_key] = currentValue;
				countVec_h[hash_key] = 1;
				break;		
			}
			else{
				//go to the next spot
				hash_key = (hash_key + 1);
				if (hash_key >= HSIZE) {
					hash_key = hash_key-HSIZE;
				}	
				//we assume that the hash_table is big enough
			}
		}
		p_h[0] += 1;
	}
}