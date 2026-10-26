###############################################################################
# Makefile for the ISA unit tests
#
# Each test builds into build/<test>/:
#
#   <test>.elf       linked with link.ld (preprocessed from link.ld.in + mmio.h)
#   <test>_imem.hex  .text + .rodata, loaded into the program memory region
#   <test>_dmem.hex  .data, loaded into the data memory region
#   <test>.map       linker map
#
# Usage (from sw/isa_unit_test):
#
#   make -f unit_tests.mk bram_test_gen2
#       Builds the firmware hex files
#
#   make -f unit_tests.mk tb
#       Verilates the cve2_top RTL with the C++ tb (needs `make -f sim.mk
#       fuse gen-vc` in rtl/ first, for the patched .vc file list)
#
#   make -f unit_tests.mk run_bram_test_gen2 [SEED=n] [MAX_CYCLES=n]
#       Builds both, then runs the test
#
#   make -f unit_tests.mk clean
#
###############################################################################

###############################################################################
# Toolchain
###############################################################################
RISCV_PREFIX ?= riscv32-unknown-elf

CC      := $(RISCV_PREFIX)-gcc
OBJCOPY := $(RISCV_PREFIX)-objcopy

BUILD_DIR := build
GENERIC   := ../generic

###############################################################################
# Firmware flags
###############################################################################
CFLAGS := \
	-O2 \
	-Wall \
	-march=rv32im \
	-mabi=ilp32 \
	-nostdlib \
	-ffreestanding \
	-I include \
	-I src

COMMON_SRCS := \
	$(GENERIC)/start.S \
	src/logging.c

COMMON_DEPS := $(wildcard include/*.h src/*.h)

###############################################################################
# Linker script
#
# link.ld.in includes mmio.h, so the memory regions always match the tb.
###############################################################################
LINKER_SCRIPT := $(BUILD_DIR)/link.ld

$(LINKER_SCRIPT): link.ld.in include/mmio.h
	@mkdir -p $(BUILD_DIR)
	$(CC) -E -P -x c -D__LINKER__ -I include $< -o $@

###############################################################################
# Firmware build rules, shared by every test
#
#   $(call fw_rules,<test name>,<main source>)
###############################################################################
define fw_rules
$(BUILD_DIR)/$(1)/$(1).elf: $(2) $(COMMON_SRCS) $(COMMON_DEPS) $(LINKER_SCRIPT)
	@mkdir -p $$(@D)
	$(CC) $(CFLAGS) \
		-T $(LINKER_SCRIPT) \
		-Wl,-Map=$(BUILD_DIR)/$(1)/$(1).map \
		$(COMMON_SRCS) $(2) \
		-lgcc \
		-o $$@

$(BUILD_DIR)/$(1)/$(1)_imem.hex: $(BUILD_DIR)/$(1)/$(1).elf
	$(OBJCOPY) -O verilog -j .text -j .rodata $$< $$@

$(BUILD_DIR)/$(1)/$(1)_dmem.hex: $(BUILD_DIR)/$(1)/$(1).elf
	$(OBJCOPY) -O verilog -j .data $$< $$@

.PHONY: $(1)
$(1): $(BUILD_DIR)/$(1)/$(1)_imem.hex $(BUILD_DIR)/$(1)/$(1)_dmem.hex
endef

###############################################################################
# Tests
###############################################################################
$(eval $(call fw_rules,bram_test_gen2,src/bram_test_gen2.c))

###############################################################################
# C++ tb
#
# Reuses the patched .vc file list from the rtl FuseSoC flow, but builds into
# our own directory so it doesn't clobber the inference tb.
###############################################################################
RTL_DIR  := ../../rtl
VC_DIR   := $(abspath $(RTL_DIR)/build/openhwgroup_cve2_cve2_top_0.1/lint-verilator)
VC_FILE  := $(VC_DIR)/openhwgroup_cve2_cve2_top_0.1_patched.vc

TB_DIR   := $(abspath $(BUILD_DIR)/tb)
TB_VC    := $(TB_DIR)/cve2_top.vc
TB_BIN   := $(TB_DIR)/Vcve2_top
TB_SRCS  := $(abspath tb/bram_test.cpp)
TB_DEPS  := $(wildcard tb/*.hpp tb/*.cpp include/*.h)

# The .vc paths are relative to VC_DIR; make them absolute so the generated
# Makefile can find the extra C++ sources from TB_DIR
$(TB_VC): $(VC_FILE)
	@mkdir -p $(TB_DIR)
	sed 's|\.\./src/|$(abspath $(VC_DIR)/../src)/|g' $< > $@

$(TB_BIN): $(TB_DEPS) $(TB_VC)
	verilator -f $(TB_VC) \
		-Wno-fatal \
		--cc --exe --build \
		--top-module cve2_top \
		--Mdir $(TB_DIR) \
		-CFLAGS "-std=c++17 -I$(abspath include) -I$(abspath tb)" \
		-LDFLAGS "-lelf" \
		$(TB_SRCS)

.PHONY: tb
tb: $(TB_BIN)

###############################################################################
# Run
###############################################################################
SEED       ?= 1
MAX_CYCLES ?= 2000000

.PHONY: run_bram_test_gen2
run_bram_test_gen2: bram_test_gen2 $(TB_BIN)
	$(TB_BIN) \
		--imem $(BUILD_DIR)/bram_test_gen2/bram_test_gen2_imem.hex \
		--dmem $(BUILD_DIR)/bram_test_gen2/bram_test_gen2_dmem.hex \
		--uart-log $(BUILD_DIR)/bram_test_gen2/uart.log \
		--seed $(SEED) \
		--max-cycles $(MAX_CYCLES)

###############################################################################
# Cleanup
###############################################################################
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
