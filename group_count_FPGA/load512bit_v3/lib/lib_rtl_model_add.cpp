//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
###############################
## Created: Intel Corporation 
##          Christian Faerber
##          PSG CE EMEA TS-FAE 
##          June 2022
###############################
*/

#include <CL/sycl.hpp>

SYCL_EXTERNAL extern "C" unsigned AdderUint(unsigned int a, unsigned int b) {
  return a + b; 
}
