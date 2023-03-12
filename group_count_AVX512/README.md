# Own AVX512-Implementation of a hashbased group_count algorithm with five different LinearProbing approach

(1) Compile Code

    (a) `cmake .` or `cmake -DCMAKE_CXX_COMPILER=... .`

    (b) `make`

(2) Execute
`bin/main`

#####################################################################


## overview about functions in LinearProbing_avx512.cpp
-	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
- 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
-	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
-	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
- 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of process 
                                                        each item individually, even though it occurs multiple times in the currently loaded data	


#####################################################################

-   Within Intel Devcloud: DON'T execute project on login-2 node!
-   Login interactive on a computing node:
        (1) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
        (2) `devcloud_login`
        (3) select option 6
        (4) select a node of type "Nodes with no attached hardware"

#####################################################################