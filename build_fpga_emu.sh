#!/bin/bash
tools_setup -t S10OAPI
source /data/intel_fpga/devcloudLoginToolSetup.sh
source /opt/intel/inteloneapi/setvars.sh 
source /glob/development-tools/versions/oneapi/2022.3/oneapi/setvars.sh --force
make emu 
