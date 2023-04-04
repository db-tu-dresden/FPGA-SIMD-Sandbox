# Transformed FPGA-Code of hasbased group_count AVX512 implementation 

This directory contains the transformed code of hasbased group_count which uses the converted primitives (formerly Intel Intrinsics).
Execution is still serial. The code skeleton for compiling on real FPGA hardware is still missing in this folder (see: /group_count_FPGA).

(1) Compile Code
-   `chmod +x build.sh`
-   `./build.sh`

(2) Execute
-   `chmod +x execute.sh`
-   `./execute.sh`


## overview about functions in kernel.cpp
-	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
- 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
-	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
-	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type std::array<fpvec<Type, regSize>, m_HSIZE_v> as hash_map structure "around" the registers
- 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of 
                                                        process each  item individually, even though it occurs multiple times in the currently loaded data	

## overview about the different sub-directories:

-   src_load512bit/
        --> Per clock cycle, we load one register à 512bit (= 16 32bit-elements). After loading this register, we do all steps of the respective algorithm for every element of this register.
            This is repeated until all elements of the input array have been processed. 
            Due to the fact that the CPU does not address 4 DMA controllers in parallel, we only use this approach to measure the execution time for the evaluation. 
            (The other approaches, e.g. loading 2048bit per clock cycle) are only evaluated on the FPGA.


The two following approaches were also debugged on the CPU for development purposes, but not evaluated on the CPU for the above reasons, only on the FPGA. -> see folder /group_count_FPGA/
-   z_old_projects/src_inner512bits/   
        --> after loading 2048bit (one register with 64 32-bit elements) within one clock cycle, 
            these projects works through all steps of the respective version of the algorithm with 512-bit registers (four register each 16 32-bit elements) 
-   z_old_projects/src_2048bit_permanent/  
        --> after loading 2048bit (one register with 64 32-bit elements) within one clock cycle, 
            these projects works through all steps of the respective version of the algorithm with the same 2048bit register (one register with 64 32-bit elements) 

## overview about additional sub-directories for source and header files:
-   /config/       --> contain the global_settings.hpp file, which contains all global settings (equal for all projects)
-   /helper/       --> contain all helper files for main and kernel (equal for all projects)
-   /primitives/   --> contain the primitives.hpp file, which contains all primitiv-functions (converted Intel Intrinsics)



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

!! --> Currently used intrinsics have been continuously expanded and some were partially adapted to the circumstances of the approaches, see functional descriptions in /group_count_FPGA/primitives/primitives.hpp !!
