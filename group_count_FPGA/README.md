# Transformed FPGA-Code of hasbased group_count AVX512 implementation 

## Emulator
(1) Build
`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_fpga_emu.sh -l walltime=23:00:00`

(2) 	
`source /opt/intel/inteloneapi/setvars.sh`

(3) Execute
`./main.fpga_emu`

## Compile and execute on FPGA hardware

(1) Build
`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_fpga_hw.sh -l walltime=23:00:00`

(2) Execute
- Connect to FPGA
(a) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(b) `devcloud_login`

- Run
`./main.fpga`




## List of all needed Intel intrinsics
- Analysis based on own AVX512 implementation (Link: https://github.com/db-tu-dresden/FPGA-SIMD-Sandbox/tree/main/group_count_AVX512)

## List of all used AVX512 intrinsics:
- current status as of 04/01/2023

| # | Line of code (tbc) | Return value | Intel Intrinsics | List of Arguments | used in version of function LinearProbingAVX512 Variantx() | associated function name in FPGA-Code |
| ------------- | ------------- | ------------- | ------------- |------------- | ------------- | ------------- |
| 1 | 25, 99, 202, 297 | __m512i | _mm512_setzero_epi32() | no arguments | global | #1 setzero() |
| 2 | 26 | __m512i | _mm512_setr_epi32() | (int e15, int e14, ... int e1, int e0) | global | #2 setr_16slot() |
| 3 | 53, 174, 250 | __m512i | _mm512_set1_epi32() | (int a) | v1, v2, v3 | #3 set1() |
| 4 | 61 | __mmask16 | _cvtu32_mask16() | (int a) | v1 | #4 cvtu32_mask16() |
| 5 | 64 | __m512i | _mm512_maskz_loadu_epi32() | (__mmask16 k, void const* mem_addr) | v1 | #5 mask_loadu() |
| 6 | 67, 110 | __mask16 | _mm512_mask_cmpeq_epi32_mask() | (__mmask16 k1, __m512i a, __m512i b) | v1 | #6 mask_cmpeq_epi32_mask()  |
| 7 | 81 | __m512i | _mm512_mask_loadu_epi32() | (__m512i src, __mmask16 k, void const* mem_addr) | v1 | #5 mask_loadu()  |
| 8 | 84 | __m512i | _mm512_mask_add_epi32() | (__m512i src, __mmask16 k, __m512i a, __m512i b) | v1 | #7 mask_add_epi32() |
| 9 | 87 | __m512i | _mm512_mask_storeu_epi32() | (void* mem_addr, __mmask16 k, __m512i a) | v1 | #8 mask_storeu_epi32() |
| 10 | 111, 216, 318 | uint32_t | _mm512_mask2int() | (__mmask16 k1) | v1, v2, v3 | #9 mask2int() |
| 11 | 114, 219, 321 | __mask16 | _mm512_knot() | (__mmask16 a) | v1, v2, v3 | #10 knot() |
| 12 | 117, 185, 220, 287, 322 | int | __builtin_clz() | (unsigned int x) | v1, v2, v3 | #11 clz_onceBultin() |
| | | | | | | |
| | | | | | | |
| 13 | 178, 263, 280 | __m512i | _mm512_load_epi32() | (void const* mem_addr) | v2, v3 | #12 load_epi32()) |
| 14 | 181, 215, 283, 317 | __mask16 | _mm512_cmpeq_epi32_mask() | (__m512i a, __m512i b) | v2, v3 | #13 cmpeq_epi32_mask() |
| | | | | | | |
| | | | | | | |
| 15 | 270 | __m512i | _mm512_permutexvar_epi32() | (__m512i idx, __m512i a) | v3 | #14 permutexvar_epi32()  |
| | | | | | |  |

--> _mm512_maskz_loadu_epi32() and _mm512_mask_loadu_epi32() are combined in only one function: #5 mask_loadu() 
