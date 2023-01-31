#!/bin/bash
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t S10OAPI
export PATH=/glob/intel-python/python2/bin:$PATH
source /opt/intel/inteloneapi/setvars.sh 
source /glob/development-tools/versions/oneapi/2022.3/oneapi/setvars.sh --force
make hw
