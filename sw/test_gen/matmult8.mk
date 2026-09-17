###############################################################################
# Makefile for matmul8 vector test
#
# This Makefile performs two operations:
#
#   1. Compile and link all source files into a RISC-V ELF executable.
#   2. Convert the ELF into a Verilog-compatible HEX file for simulation.
#
# Usage:
#
#   make
#       Builds both matmul8_vec.elf and matmul8_vec.hex
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
#RISCV_PREFIX := riscv32-unknown-elf
RISCV_PREFIX ?= riscv-none-elf

CC      := $(RISCV_PREFIX)-gcc
OBJCOPY := $(RISCV_PREFIX)-objcopy

###############################################################################
# Output files
###############################################################################
ELF_FILE := matmul8_vec.elf
HEX_FILE := matmul8_vec.hex
MAP_FILE := matmul8.map

###############################################################################
# Source files
#
# start.S
#   Startup code / reset handler.
#
# uart.c
#   UART driver and printf support.
#
# matmul8_vec_test.c
#   Main test program.
#
# matmul8_vec.S
#   Hand-written vector/matrix multiplication assembly.
###############################################################################
SRCS := \
	start.S \
	uart.c \
	matmul8_vec_test.c \
	matmul8_vec.S

###############################################################################
# Linker script
#
# Controls memory placement of code/data.
###############################################################################
LINKER_SCRIPT := matmul8_shared_link.ld

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
	-march=rv32im \
	-mabi=ilp32 \
	-nostdlib \
	-ffreestanding

LDFLAGS := \
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
