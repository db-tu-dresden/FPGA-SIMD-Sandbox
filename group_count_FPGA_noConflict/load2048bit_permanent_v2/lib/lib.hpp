//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// SYCL functions
SYCL_EXTERNAL float SyclSquare(float);
SYCL_EXTERNAL float SyclSqrt(float);
SYCL_EXTERNAL int SyclMult(int x, int y);

SYCL_EXTERNAL std::array<int, 16> SyclBubbleSort( std::array<int, 16> a );


// RTL functions
SYCL_EXTERNAL extern "C" unsigned RtlByteswap(unsigned x);
SYCL_EXTERNAL extern "C" unsigned AdderUint(unsigned int a, unsigned int b );
