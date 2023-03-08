#!/bin/bash

# # run this by
# after devCloud Update 08-03-2023: Use fpga_compile node instead of fpga_runtime nodes ! 
# new : qsub -l nodes=1:fpga_compile:ppn=2 -d . build_hw -l walltime=23:59:00
# old : qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . build_hw -l walltime=23:59:00
# # monitor job via
# watch -n 1 qstat -n -1

# Initial Setup
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t S10OAPI
export PATH=/glob/intel-python/python2/bin:$PATH


cd ~/FPGA-SIMD-Sandbox/group_count_FPGA_singleKernel/v5 

# Running project in FPGA Hardware Mode (this takes approximately 1 hour)
printf "\\n%s\\n" "Running in FPGA Hardware compile Mode:"


 make hw
# error_check