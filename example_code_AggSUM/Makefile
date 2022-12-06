ALL_CXXFLAGS=-Wall $(CXXFLAGS)
SRC=main.cpp
TARGET=main
BOARD=/opt/intel/oneapi/intel_s10sx_pac:pac_s10_usm


# Directories
INC_DIRS := ../common/inc
LIB_DIRS :=

# Files
INCS := $(wildcard *.hpp)
SRC := $(wildcard *.cpp)
LIBS :=


## NOTE: you may need to remove the '--gcc-toolchain=...' from the dpcpp commands below

emu: $(SRC) Makefile
	dpcpp $(ALL_CXXFLAGS) -fintelfpga -DFPGA_EMULATOR $(SRC) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L) -o $(TARGET).fpga_emu


emu_lib: $(SRC) Makefile
	dpcpp $(ALL_CXXFLAGS) -fintelfpga -DFPGA_EMULATOR -fsycl-link=image kernels.cpp -o kernels_emu.a 
	dpcpp $(ALL_CXXFLAGS) -fintelfpga -DFPGA_EMULATOR main.cpp -c -o main_emu.o
	dpcpp $(ALL_CXXFLAGS) -fintelfpga -DFPGA_EMULATOR main_emu.o kernels_emu.a -o $(TARGET).fpga_emu

report: $(SRC) Makefile
	dpcpp $(ALL_CXXFLAGS) -fintelfpga $(SRC) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L) -Xshardware -Xsboard=$(BOARD) -Xssave-temps  -fsycl-link -o $(TARGET)_report

hw: $(SRC) Makefile
	dpcpp $(ALL_CXXFLAGS) -v -fintelfpga -Xshardware -Xssave-temps -reuse-exe=$(TARGET).fpga -Xsoutput-report-folder=$(TARGET).prj -Xsboard=$(BOARD) -o $(TARGET).fpga $(SRC) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L)

clean:
	rm -rf *.a *.prj *.fpga_emu *.log $(TARGET)_report

clean_hw: clean
	rm -rf *.aoco *.aocr *.mon

clean_all: clean_hw
	rm -rf $(TARGET)/ *.spv *.o *.aocx *.a *.fpga

.PHONY: emu report hw clean clean_hw
