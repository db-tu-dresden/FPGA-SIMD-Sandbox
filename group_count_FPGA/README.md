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

| # | Line of code (tbc) | Return value | Intel Intrinsics | List of Arguments | used in implementation version |
| ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
| 1 | 25, 99, 202, 297 | __m512i | _mm512_setzero_epi32() | no arguments | global | 
| 2 | 26 | __m512i | _mm512_setr_epi32() | (int e15, int e14, ... int e1, int e0) | global | 
| 3 | 53, 161, 250 | __m512i | _mm512_set1_epi32() | (int a) | v1, v2, v3 | 
| 4 | 56 | __m512i | _mm512_maskz_loadu_epi32() | (__mmask16 k, void const* mem_addr) | v1 | 
| 5 | 59, 99, 168, 202, 263, 297 | __mask16 | _mm512_cmpeq_epi32_mask() | (__m512i a, __m512i b) | v1, v2, v3 | 
| 6 | 71 | __m512i | _mm512_mask_loadu_epi32() | (__m512i src, __mmask16 k, void const* mem_addr) | v1 | 
| 7 | 74 | __m512i | _mm512_mask_add_epi32() | (__m512i src, __mmask16 k, __m512i a, __m512i b) | v1 | 
| 8 | 77 | __m512i | _mm512_mask_storeu_epi32() | (void* mem_addr, __mmask16 k, __m512i a) | v1 | 
| 9 | 100, 203, 298 | uint32_t | _mm512_mask2int() | (__mmask16 k1) | v1, v2, v3 | 
| 10 | 103, 206, 301 | __mask16 | _mm512_knot() | (__mmask16 a) | v1, v2, v3 | 
| 11 | 106, 172, 207, 267, 302 | int | __builtin_clz() | (unsigned int x) | v1, v2, v3 | 
| | | | | | | 
| 12 | 165, 243, 260 | __m512i | _mm512_load_epi32() | (void const* mem_addr) | v2, v3 | 
| | | | | | | 
| 13 | 250 | __m512i | _mm512_permutexvar_epi32() | (__m512i idx, __m512i a) | v3 | 
| | | | | | | 
