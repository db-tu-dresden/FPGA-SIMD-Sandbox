# Transformed FPGA-Code of hasbased group_count AVX512 implementation 

This directory contains the transformed code of hasbased group_count which uses the converted primitives (formerly Intel Intrinsics).
Execution is still serial. The code skeleton for compiling on real FPGA hardware is still missing in this folder (see: /group_count_FPGA).

(1) Compile Code
-   `chmod +x build.sh`
-   `./build.sh`

(2) Execute
-   `chmod +x execute.sh`
-   `./execute.sh`



## List of all needed Intel intrinsics
- Analysis based on own AVX512 implementation (Link: https://github.com/db-tu-dresden/FPGA-SIMD-Sandbox/tree/main/group_count_AVX512)

## List of all used AVX512 intrinsics:
- current status as of 04/01/2023

| # | Return value | Intel Intrinsics | List of Arguments | used in version of function LinearProbingAVX512 Variantx() | associated function name in FPGA-Code |
| ------------- | ------------- | ------------- |------------- | ------------- | ------------- |
| 1 | __m512i | _mm512_setzero_epi32() | no arguments | global | #1 setzero() |
| 2 | __m512i | _mm512_setr_epi32() | (int e15, int e14, ... int e1, int e0) | global | #2 setr_16slot() |
| 3 | __m512i | _mm512_set1_epi32() | (int a) | v1, v2, v3 | #3 set1() |
| 4 | __mmask16 | _cvtu32_mask16() | (int a) | v1 | #4 cvtu32_mask16() |
| 5 | __m512i | _mm512_maskz_loadu_epi32() | (__mmask16 k, void const* mem_addr) | v1 | #5 mask_loadu() |
| 6 | __mask16 | _mm512_mask_cmpeq_epi32_mask() | (__mmask16 k1, __m512i a, __m512i b) | v1 | #6 mask_cmpeq_epi32_mask()  |
| 7 | __m512i | _mm512_mask_loadu_epi32() | (__m512i src, __mmask16 k, void const* mem_addr) | v1 | #5 mask_loadu()  |
| 8 | __m512i | _mm512_mask_add_epi32() | (__m512i src, __mmask16 k, __m512i a, __m512i b) | v1 | #7 mask_add_epi32() |
| 9 | __m512i | _mm512_mask_storeu_epi32() | (void* mem_addr, __mmask16 k, __m512i a) | v1 | #8 mask_storeu_epi32() |
| 10 | uint32_t | _mm512_mask2int() | (__mmask16 k1) | v1, v2, v3 | #9 mask2int() |
| 11 | __mask16 | _mm512_knot() | (__mmask16 a) | v1, v2, v3 | #10 knot() |
| 12 | int | __builtin_clz() | (unsigned int x) | v1, v2, v3 | #11 clz_onceBultin() |
| | | | | | |
| | | | | | |
| 13 | __m512i | _mm512_load_epi32() | (void const* mem_addr) | v2, v3 | #12 load_epi32()) |
| 14 | __mask16 | _mm512_cmpeq_epi32_mask() | (__m512i a, __m512i b) | v2, v3 | #13 cmpeq_epi32_mask() |
| | | | | | |
| | | | | | |
| 15 | __m512i | _mm512_permutexvar_epi32() | (__m512i idx, __m512i a) | v3 | #14 permutexvar_epi32()  |
| | | | | | |
| | | | | | |
| 16 | int | __builtin_ctz() | (unsigned int x) | v1, v2, v3 | #15 ctz_onceBultin() |

--> _mm512_maskz_loadu_epi32() and _mm512_mask_loadu_epi32() are combined in only one function: #5 mask_loadu() 
