# FPGA-SIMD-Sandbox

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
