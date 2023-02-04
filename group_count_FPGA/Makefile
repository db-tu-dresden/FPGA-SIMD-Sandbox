###############################
## Created: Eric Stange
##          TU Dresden
##          January 2023
## 
## Based on Template by:
##          Intel Corporation 
##          Christian Faerber
##          PSG CE EMEA TS-FAE 
##          June 2022
###############################

ALL_CXXFLAGS=-Wall $(CXXFLAGS)
SRC=main.cpp
TARGET=main
TARGET_KERNEL=kernel.a
BOARD=/opt/intel/oneapi/intel_s10sx_pac:pac_s10_usm


# Directories
INC_DIRS := ../common/inc
LIB_DIRS := 

# Files
INCS := $(wildcard )
SRC := $(wildcard *.cpp)
LIBS :=


## NOTE: you may need to remove the '--gcc-toolchain=...' from the dpcpp commands below

## ## ## ## ##
## Libs
## ## ## ## ##

sycl_lib: $(SRC) Makefile
	fpga_crossgen lib/lib_sycl.cpp --source sycl --target sycl -o lib/lib_sycl.o
	
rtl_lib: $(SRC) Makefile
	fpga_crossgen lib/lib_rtl_spec.xml --emulation_model lib/lib_rtl_model.cpp --target sycl -o lib/lib_rtl.o
	
rtl_lib_add: $(SRC) Makefile
	fpga_crossgen lib/lib_rtl_spec_add.xml --emulation_model lib/lib_rtl_model_add.cpp --target sycl -o lib/lib_rtl_add.o
	
lib_a: $(SRC) Makefile
	fpga_libtool lib/lib_rtl.o lib/lib_rtl_add.o lib/lib_sycl.o --target sycl --create lib/lib.a
	 

build_lib: $(SRC) Makefile
	fpga_crossgen -v -fPIC lib/lib_sycl.cpp --source sycl --target sycl -o lib/lib_sycl.o
	fpga_libtool -v lib/lib_sycl.o --target sycl --create lib/lib_sycl.a
	fpga_crossgen -v -fPIC lib/lib_rtl_spec.xml --emulation_model lib/lib_rtl_model.cpp --target sycl -o lib/lib_rtl.o
	fpga_crossgen -v -fPIC lib/lib_rtl_spec_add.xml --emulation_model lib/lib_rtl_model_add.cpp --target sycl -o lib/lib_rtl_add.o
	fpga_libtool -v lib/lib_rtl.o lib/lib_rtl_add.o --target sycl --create lib/lib_rtl.a

build_hls: $(SRC) Makefile
	i++ lib/HLS/bubbleSort.cpp -march="Stratix10" --simulator "none" --fpga-only -v -o lib/HLS/bubbleSort


## ## ## ## ##
## Emulation
## ## ## ## ##

emu: $(SRC) Makefile
	icpx -fsycl $(ALL_CXXFLAGS) -fintelfpga -DFPGA_EMULATOR $(SRC) lib/lib_sycl.a lib/lib_rtl.a -fsycl-device-code-split=off -o $(TARGET).fpga_emu $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L)
	



## ## ## ## ##
## Report
## ## ## ## ##

report: $(SRC) Makefile
	icpx -fsycl $(ALL_CXXFLAGS)  -fintelfpga $(SRC) lib/lib_sycl.a lib/lib_rtl.a -fsycl-device-code-split=off -Xshardware -Xsboard=$(BOARD) -Xssave-temps  -fsycl-link -o $(TARGET)_report




## ## ## ## ##
## Hardware build complete
## ## ## ## ##

hw: $(SRC) Makefile
	icpx -fsycl $(ALL_CXXFLAGS) -v -fintelfpga -Xshardware -fsycl-device-code-split=off -Xsprofile -Xssave-temps -reuse-exe=$(TARGET).fpga -Xsoutput-report-folder=$(TARGET).prj -Xsboard=$(BOARD) -o $(TARGET).fpga $(SRC) lib/lib_sycl.a lib/lib_rtl.a $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L)





## ## ## ## ##
## Hardware kernel only 
## ## ## ## ##

hw_a: $(SRC) Makefile
	icpx -fsycl $(ALL_CXXFLAGS) -v -fPIC -fintelfpga -fsycl-link=image -Xshardware -fsycl-device-code-split=off -Xsprofile -reuse-exe=$(TARGET_KERNEL) -Xsboard=$(BOARD) kernel.cpp lib/lib_sycl.a lib/lib_rtl.a -o $(TARGET_KERNEL) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L) -Wl,--unresolved-symbols=ignore-all





## ## ## ## ##
## Host code only
## ## ## ## ##

host_o: $(SRC) Makefile
	icpx -fsycl $(ALL_CXXFLAGS) -v -fintelfpga -Xsboard=$(BOARD) -c -DFPGA=1 -o $(TARGET)_host.o so_example_host.cpp lib/lib_sycl.a lib/lib_rtl.a $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L)





## ## ## ## ##
## Link host and kernel
## ## ## ## ##

link: $(SRC) Makefile
	icpx -fsycl $(ALL_CXXFLAGS) -v -fintelfpga lib/lib_rtl_add.o lib/lib_sycl.o $(TARGET)_host.o $(TARGET_KERNEL) -o $(TARGET)_linked.fpga





clean:
	rm -rf *.a *.prj *.fpga_emu *.log $(TARGET)_report

clean_hw: clean
	rm -rf *.aoco *.aocr *.mon

clean_all: clean_hw
	rm -rf $(TARGET)/ *.spv *.o *.aocx *.fpga


## ## ## ## ##
## Reset folders
## ## ## ## ##

reset_main:
	rm -rf *.e* *.o* *.o *.prj kernel.a so_example.fpga so_example.fpga_emu so_example_linked.fpga *report opt.log .ipynb_checkpoints

reset_lib:
	rm -rf lib/*.o lib/*.a lib/.ipynb_checkpoints lib/HLS/*.prj lib/HLS/.ipynb_checkpoints
    
reset_so:
	rm -rf so/*.e* so/*.o* so/*.o so/libFPGAfunctions.so so/gcc_application so/env.sh so/.ipynb_checkpoints
    
reset_all:
	rm -rf *.e* *.o* *.o *.prj kernel.a so_example.fpga so_example.fpga_emu so_example_linked.fpga *report opt.log .ipynb_checkpoints
	rm -rf lib/*.o lib/*.a lib/.ipynb_checkpoints lib/HLS/*.prj lib/HLS/.ipynb_checkpoints
	rm -rf so/*.e* so/*.o* so/*.o so/libFPGAfunctions.so so/gcc_application so/env.sh so/.ipynb_checkpoints
    
    
.PHONY: emu build_lib report hw hw_quick clean clean_hw