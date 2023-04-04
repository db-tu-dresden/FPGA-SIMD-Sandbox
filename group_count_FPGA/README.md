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

-   /load512bit_v1/     --> contain project with kernelV1, which contains function LinearProbingFPGA_variant1()
-   /load512bit_v2/     --> contain project with kernelV2, which contains function LinearProbingFPGA_variant2()
-   /load512bit_v3/     --> contain project with kernelV3, which contains function LinearProbingFPGA_variant3()
-   /load512bit_v4/     --> contain project with kernelV4, which contains function LinearProbingFPGA_variant4()
-   /load512bit_v5/     --> contain project with kernelV5, which contains function LinearProbingFPGA_variant5()

        --> Per clock cycle, we load one register à 512bit (= 16 32bit-elements). After loading this register, we do all steps of the respective algorithm for every element of this register.
            This is repeated until all elements of the input array have been processed. 
            Since the FPGA can load 512 bits per clock cycle per DMA controller, we only use one of the four available DMA controllers for this approach.


-   /load2048bit_permanent_v1/  --> contain project with kernelV1, which contains function LinearProbingFPGA_variant1()
-   /load2048bit_permanent_v2/  --> contain project with kernelV2, which contains function LinearProbingFPGA_variant2()                           
-   /load2048bit_permanent_v3/  --> contain project with kernelV3, which contains function LinearProbingFPGA_variant3()

        --> Per clock cycle, we load one register à 2048bit (= 64 32bit-elements). After loading this register, we do all steps of the respective algorithm for every element of this register.
            These projects works through all steps of the algorithm with the same 2048bit register (one register with 64 32-bit elements).
            Since the FPGA can load 512 bits per clock cycle per DMA controller, we use all four available DMA controllers in parallel for this approach.  
            This approach is probably not optimal for use on the FPGA.      (v4 and v5 of the algorithm aren't )


-   /load2048bit_inner512bit_v1/   --> contain project with kernelV1, which contains function LinearProbingFPGA_variant1()
-   /load2048bit_inner512bit_v2/   --> contain project with kernelV2, which contains function LinearProbingFPGA_variant2()                           
-   /load2048bit_inner512bit_v3/   --> contain project with kernelV3, which contains function LinearProbingFPGA_variant3()
-   /load2048bit_inner512bit_v4/   --> contain project with kernelV4, which contains function LinearProbingFPGA_variant4()
-   /load2048bit_inner512bit_v5/   --> contain project with kernelV5, which contains function LinearProbingFPGA_variant5()

        --> Per clock cycle, we load one register à 2048bit (= 64 32bit-elements). After loading, we split the 2048bit register into 4 seperate 512bit registers.
            After that, the algorithm processes these four registers one by one. All steps of the algorithm are carried out for each element of the respective register.
            This approach probably shows a significantly increased performance. However, this also requires an intervention in the algorithm code, since an additional for-loop must be inserted, 
            which splits the loaded 2048 bit register into the 4 512 bit registers and then processes them.

-   /load2048bit_virtual_work_4x16_v1/
-   /load2048bit_virtual_work_4x16_v2/
-   /load2048bit_virtual_work_4x16_v3/
-   /load2048bit_virtual_work_4x16_v4/
-   /load2048bit_virtual_work_4x16_v5/
        --> Per clock cycle, we load one register à 2048bit (= 64 32bit-elements).
        However, unlike in the approach (/load2048bit_inner512bit/), this register is not split up. Instead, the modeled intrinsic functions are adjusted to be designed to roll up a 64-element register 4x with 16x parallel steps. 
        The compiler should consider this register as 4 individual registers with 16 elements each. 
        The 64-element register is considered as "virtual register". While it exists physically to store the work data. However, each intrinsic function works in parallel on four individual parts of this register.
        We hope that this will give us the same speed advantages as with the previous approach, but without any necessary intervention in the program code compared to the standard C++ code.


## overview about functions in kernel.cpp
-	LinearProbingFPGA_variant1() == SoA_v1 -- SIMD for FPGA function v1 -  without aligned_start; version descbribed in paper
- 	LinearProbingFPGA_variant2() == SoA_v2 -- SIMD for FPGA function v2 - first optimization: using aligned_start
-	LinearProbingFPGA_variant3() == SoA_v3 -- SIMD for FPGA function v3 - with aligned start and approach of using permutexvar_epi32
-	LinearProbingFPGA_variant4() == SoAoV_v1 -- SIMD for FPGA function v4 - use a vector with elements of type std::array<fpvec<Type, regSize>, m_HSIZE_v> as hash_map structure "around" the registers
- 	LinearProbingFPGA_variant5() == SoA_conflict_v1 -- SIMD for FPGA function v5 - 	search in loaded data register for conflicts and add the sum of occurences per element to countVec instead of process 
                                each item individually, even though it occurs multiple times in the currently loaded data	

#########################################

## Emulator
### Compile and run as Batchjob
(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/ ..` 
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
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/ ..` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a file is created

(4) make Emulation
-   `cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/ ..`
-   `source /data/intel_fpga/devcloudLoginToolSetup.sh`
-   `tools_setup -t S10OAPI`
-   `make emu`

(5) Execute
-   `./main.fpga_emu`


## Compile and execute on FPGA hardware

--> see instructions in file "HOW_TO_RUN" or use the following steps:

(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/ ..` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a or lib/lib_rtl.a file is created

(1) Build
-   `qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . build_hw -l walltime=23:59:00`
-	after 3-4 hours the main.fpga file is created
-	`cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/ ..`

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
(a) `cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/ ..`
(b) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(c) `tools_setup -t S10OAPI`
(d) `aocl initialize acl0 pac_s10_usm`
(e) `./main.fpga` 






## List of all needed Intel intrinsics
- Analysis based on own AVX512 implementation (Link: https://github.com/db-tu-dresden/FPGA-SIMD-Sandbox/tree/main/group_count_AVX512)
!! --> Currently used intrinsics have been continuously expanded and some were partially adapted to the circumstances of the approaches, see functional descriptions in /group_count_FPGA/primitives/primitives.hpp !!