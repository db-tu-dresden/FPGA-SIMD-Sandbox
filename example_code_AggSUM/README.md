# FPGA-SIMD-Sandbox

> monitor running batch-jobs via
> `watch -n 1 qstat -n -1`

#########################################
(Intel DevCloud with new accounts)
FIX FOR ENV-ERROR WHILE COMPILING "Error: Failed to open quartus_sh_compile.log" 

-   From the login-2 node, there are .bash_* files in the path /etc/skel. If you copy these over to the root home directory, these files will provide the paths you need. 
-   cp /etc/skel/.bash_* ~/

#########################################

## Emulator
### Compile and run as Batchjob
(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/example_code_AggSUM` 
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
-	`cd ~/FPGA-SIMD-Sandbox/example_code_AggSUM` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a file is created

(4) make Emulation
-   `cd ~/FPGA-SIMD-Sandbox/example_code_AggSUM`
-   `source /data/intel_fpga/devcloudLoginToolSetup.sh`
-   `tools_setup -t S10OAPI`
-   `make emu`

(5) Execute
-   `./main.fpga_emu`


## Compile and execute on FPGA hardware

--> see instructions in file "HOW_TO_RUN" or use the following steps:

(0) Build lib 
-	`cd ~/FPGA-SIMD-Sandbox/example_code_AggSUM` 
-	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
-	`qsub -l nodes=1:fpga_compile:ppn=2 -d . build_lib -l walltime=00:10:00`
-	after 2-4 minutes the lib/lib.a or lib/lib_rtl.a file is created

(1) Build
-   `qsub -l nodes=1:fpga_compile:ppn=2 -d . build_hw -l walltime=23:59:00`
-	after 3-4 hours the main.fpga file is created
-	`cd ~/FPGA-SIMD-Sandbox/example_code_AggSUM`

(2) Execute main.fpga on FPGA via batchjob:
-   `qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_hw -l walltime=00:05:00`
-   After the execution ist finished, check the log files run_hw.e... and run_hw.o...

(3) Alternative: Execute (interactive on a FPGA node)
- Connect to FPGA
(a) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
(b) `devcloud_login`
(c) > select "4 STRATIX10 with OneAPI"

- Run
(a) `cd ~/FPGA-SIMD-Sandbox/example_code_AggSUM`
(b) `./main.fpga`
