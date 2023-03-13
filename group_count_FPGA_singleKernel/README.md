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

## A notice:

In this project folder, the individual versions of the LinearProbing algorithm (v1 to v5) are each in a separate subfolder in a single FPGA project. 
This is used to compile FPGA HW files, which each contain only one version of the algorithm. 
Each version must be compiled and run separately. 
Please note the "HOW_TO_RUN" file in the respective subfolder!
The following compile- and execution steps are only generally formulated and do not contain e.g. the exact specific folder path.

#########################################

## overview of directory structures and differences in the respective projects
-   /config/       --> contain the global_settings.hpp file, which contains all global settings (equal for all projects)
-   /helper/       --> contain all helper files for main and kernel (equal for all projects)
-   /primitives/   --> contain the primitives.hpp file, which contains all primitiv-functions (converted Intel Intrinsics)

-   /v1_inner512bits/   --> contain project with kernelV1, which contains function LinearProbingFPGA_variant1()
-   /v2_inner512bits/   --> contain project with kernelV2, which contains function LinearProbingFPGA_variant2()                           
-   /v3_inner512bits/   --> contain project with kernelV3, which contains function LinearProbingFPGA_variant3()
-   /v4_inner512bits/   --> contain project with kernelV4, which contains function LinearProbingFPGA_variant4()
-   /v5_inner512bits/   --> contain project with kernelV5, which contains function LinearProbingFPGA_variant5()

        --> after loading 2048bit (one register with 64 32-bit elements) within one clock cycle, 
            these projects works through all steps of the algorithm with 512-bit registers (four register each 16 32-bit elements) 


-   /v1_2048bit_permanent/  --> contain project with kernelV1, which contains function LinearProbingFPGA_variant1()
-   /v2_2048bit_permanent/  --> contain project with kernelV2, which contains function LinearProbingFPGA_variant2()                           
-   /v3_2048bit_permanent/  --> contain project with kernelV3, which contains function LinearProbingFPGA_variant3()

        --> after loading 2048bit (one register with 64 32-bit elements) within one clock cycle, 
            these projects works through all steps of the algorithm with the same 2048bit register (one register with 64 32-bit elements) 


-   /zOld_notWorking/v4_2048bit_permanent/  --> contain project with kernelV4, which contains function LinearProbingFPGA_variant4()         
-   /zOld_notWorking/v5_2048bit_permanent/  --> contain project with kernelV5, which contains function LinearProbingFPGA_variant5() 

     -->    Within these projects, the initially loaded 2048-bit register (one register with 64 32-bit elements) should also be used directly 
            in all steps of the algorithm. Due to the greater complexity, it has not been possible up to now to run this on the FPGA STRATIX10, 
            which is why the solutions /v*_inner512bits/ were designed.


## overview about functions in kernel.cpp
-	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
- 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
-	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
-	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type <fpvec<Type, regSize> as hash_map structure "around" the registers
- 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of process 
                                each item individually, even though it occurs multiple times in the currently loaded data	



#########################################

## Emulator
### Compile and run as Batchjob
(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel/ ..` 
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
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel ..` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a file is created

(4) make Emulation
-   `cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel ..`
-   `source /data/intel_fpga/devcloudLoginToolSetup.sh`
-   `tools_setup -t S10OAPI`
-   `make emu`

(5) Execute
-   `./main.fpga_emu`


## Compile and execute on FPGA hardware

--> see instructions in file "HOW_TO_RUN" or use the following steps:

(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel ..` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a or lib/lib_rtl.a file is created

(1) Build
-   `qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . build_hw -l walltime=23:59:00`
-	after 3-4 hours the main.fpga file is created
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel ..`

(2) Execute main.fpga on FPGA via batchjob:
-   after devCloud Update 08-03-2023: Use fpga_compile node instead of fpga_runtime nodes ! 
-   new : `qsub -l nodes=1:fpga_compile:ppn=2 -d . build_hw -l walltime=23:59:00`
-	old : `qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . build_hw -l walltime=23:59:00`
-   After the execution ist finished, check the log files run_hw.e... and run_hw.o...

(3) Alternative: Execute (interactive on a FPGA node)
- Connect to FPGA
(a) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(b) `devcloud_login`
(c) > select "4 STRATIX10 with OneAPI"

- Run
(a) `cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel ..`
(b) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(c) `tools_setup -t S10OAPI`
(d) `aocl initialize acl0 pac_s10_usm`
(e) `./main.fpga` 






## List of all needed Intel intrinsics
- Analysis based on own AVX512 implementation (Link: https://github.com/db-tu-dresden/FPGA-SIMD-Sandbox/tree/main/group_count_AVX512)
