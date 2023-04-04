#!/bin/bash


# # run this by
# qsub -l nodes=1:fpga_compile:ppn=2 -d . build_fpga_emu.sh -l walltime=23:00:00
# # monitor job via
# watch -n 1 qstat -n -1


source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t S10OAPI
# source /opt/intel/inteloneapi/setvars.sh 
# source /glob/development-tools/versions/oneapi/2022.3/oneapi/setvars.sh --force

export PATH=/glob/intel-python/python2/bin:$PATH

cd ~/FPGA-SIMD-Sandbox/group_count_FPGA/load2048bit_inner512bit_v2

make emu

# error_check