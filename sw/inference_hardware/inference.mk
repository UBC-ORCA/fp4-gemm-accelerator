###############################################################################
# Makefile for inference test program
#
# This Makefile performs two operations:
#
#   1. Compile and link all source files into a RISC-V ELF executable.
#   2. Convert the ELF into a Verilog-compatible HEX file for simulation.
#
# Usage:
#
#   make
#       Builds both inference.elf and inference.hex
#
#   make elf
#       Builds only the ELF executable
#
#   make hex
#       Builds only the HEX image
#
#   make clean
#       Removes generated files
#
###############################################################################

###############################################################################
# Toolchain
#
# Assumes the RISC-V GCC toolchain is available in PATH.
###############################################################################
RISCV_PREFIX ?= riscv-none-elf

CC      := $(RISCV_PREFIX)-gcc
OBJCOPY := $(RISCV_PREFIX)-objcopy
DATASET ?= mnist

###############################################################################
# Output files
#
# DEV=1 builds inference_dev.c to inference_dev.* so it does not clobber the
# stock inference.* while that build is running.
###############################################################################

NAME     := inference
MAIN_SRC := inference_hardware.c

ELF_FILE := $(NAME).elf
HEX_FILE := $(NAME).hex
MAP_FILE := $(NAME).map

###############################################################################
# Source files
#
# start.S
#   Startup code / reset handler.
#
# uart.c
#   UART driver and printf support.
#
# inference.c
#   Main test program.
#
# matmul8_vec.S
#   Hand-written vector/matrix multiplication assembly.
###############################################################################
SRCS := \
	../generic/uart.c \
	../generic/image.c \
	$(MAIN_SRC)

###############################################################################
# Linker script
#
# Controls memory placement of code/data.
###############################################################################
LINKER_SCRIPT := inference_shared_link.ld

###############################################################################
# Compiler and linker flags
#
# -march=rv32im
#     RV32I base ISA + M extension (multiply/divide).
#
# -mabi=ilp32
#     32-bit integer ABI.
#
# -nostdlib
#     Do not link against standard C runtime libraries.
#
# -ffreestanding
#     Build as bare-metal firmware.
#
# -Wl,-Map=<file>
#     Generate a linker map file.
###############################################################################
CFLAGS := \
	-O3 \
	-march=rv32im \
	-mabi=ilp32 \
	-nostdlib \
	-ffreestanding

# FPGA=1 uses start_fpga.S + auto-incrementing UART pointer; FPGA=0 (default)
# uses the simulator start.S / UART.
FPGA ?= 0
ifeq ($(FPGA),1)
CFLAGS += -DFPGA
SRCS += ../generic/start_fpga.S
else
SRCS += ../generic/start.S
endif

LDFLAGS := \
	-I ../generic \
	-I ../headers/$(DATASET) \
	-T $(LINKER_SCRIPT) \
	-Wl,-Map=$(MAP_FILE)

###############################################################################
# Default target
#
# Running "make" builds the HEX image, which automatically depends on the ELF.
###############################################################################
all: $(HEX_FILE)

###############################################################################
# ELF build
#
# Compiles and links all sources into a single executable.
###############################################################################
$(ELF_FILE): $(SRCS) $(LINKER_SCRIPT)
	$(CC) \
	$(LDFLAGS) \
	$(CFLAGS) \
	$(SRCS) \
	-lgcc \
	-o $(ELF_FILE)

###############################################################################
# HEX build
#
# Converts ELF into a Verilog memory initialization file suitable for
# loading into instruction memory during simulation.
###############################################################################
$(HEX_FILE): $(ELF_FILE)
	$(OBJCOPY) \
		-O verilog \
		$(ELF_FILE) \
		$(HEX_FILE)

###############################################################################
# Convenience targets
###############################################################################
elf: $(ELF_FILE)

hex: $(HEX_FILE)

###############################################################################
# Cleanup
###############################################################################
clean:
	rm -f \
		$(ELF_FILE) \
		$(HEX_FILE) \
		$(MAP_FILE)

.PHONY: all elf hex clean
