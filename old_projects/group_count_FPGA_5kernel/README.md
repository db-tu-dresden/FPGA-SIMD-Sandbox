# Transformed FPGA-Code of hasbased group_count AVX512 implementation 

> monitor running batch-jobs via
> `watch -n 1 qstat -n -1`

## Global parameters
-   All parameters that can be configured are in "global_parameters.h" and should be adjusted only there at the beginning.

#########################################
(Intel DevCloud with new accounts)
FIX FOR ENV-ERROR WHILE COMPILING "Error: Failed to open quartus_sh_compile.log" 

-   From the login-2 node, there are .bash_* files in the path /etc/skel. If you copy these over to the root home directory, these files will provide the paths you need. 
-   cp /etc/skel/.bash_* ~/

#########################################

## Emulator
### Compile and run as Batchjob
(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/old_projects/group_count_FPGA_5kernel` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a file is created

(1) Build
`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_fpga_emu.sh -l walltime=23:00:00`

(2)
`source /data/intel_fpga/devcloudLoginToolSetup.sh`

(3)
`devcloud_login`
> select "5) Compilation (Command Line) Only" or 6

(4) 	
`source /opt/intel/inteloneapi/setvars.sh`

(5) Execute
`./main.fpga_emu`

### Compile and run interactive directly on a FPGA node
(1)
`source /data/intel_fpga/devcloudLoginToolSetup.sh`

(2)
`devcloud_login`
> select "4) Stratix 10 - OneAPI, OpenVINO"

(3) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/old_projects/group_count_FPGA_5kernel` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a file is created

(4) make Emulation
-   `cd ~/FPGA-SIMD-Sandbox/old_projects/group_count_FPGA_5kernel`
-   `source /data/intel_fpga/devcloudLoginToolSetup.sh`
-   `tools_setup -t S10OAPI`
-   `make emu`

(5) Execute
-   `./main.fpga_emu`


## Compile and execute on FPGA hardware

--> see instructions in file "HOW_TO_RUN" or use the following steps:

(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/old_projects/group_count_FPGA_5kernel` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a or lib/lib_rtl.a file is created

(1) Build
-   `qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . build_hw -l walltime=23:59:00`
-	after 3-4 hours the main.fpga file is created
-	`cd ~/FPGA-SIMD-Sandbox/old_projects/group_count_FPGA_5kernel`

(2) Execute main.fpga on FPGA via batchjob:
-   `qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_hw -l walltime=00:05:00`
-   After the execution ist finished, check the log files run_hw.e... and run_hw.o...

(3) Alternative: Execute (interactive on a FPGA node)
- Connect to FPGA
(a) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(b) `devcloud_login`
(c) > select "4 STRATIX10 with OneAPI"

- Run
(a) `cd ~/FPGA-SIMD-Sandbox/old_projects/group_count_FPGA_5kernel`
(b) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(c) `tools_setup -t S10OAPI`
(d) `aocl initialize acl0 pac_s10_usm`
(e) `./main.fpga 40960000` 


## overview about functions in kernel.cpp
-	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
- 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
-	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
-	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
- 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of process each item individually, even though it occurs multiple times in the currently loaded data	



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
--> Currently used intrinsics have been continuously expanded, see functional description in primitives.hppo