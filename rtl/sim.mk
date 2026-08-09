###############################################################################
# CVE2 + Matmul8 Verilator Testbench Flow
#
# Pipeline:
#   fuse      -> run FuseSoC (generates RTL + .vc)
#   gen-vc    -> patch .vc file
#   build-sim -> run Verilator + C++ TB
#   run       -> full pipeline
###############################################################################

SHELL := /bin/bash

###############################################################################
# Configuration
###############################################################################

CVE2_CONFIG ?= small

TB_BASE := ../../../../sw/tb
MATMUL_TB_CPP := $(TB_BASE)/inference_tb/min_tb_inference.cpp
MAC_CELL_TB_CPP := $(TB_BASE)/arith_tb/mac_cell.cpp
BF16_ADDER_TB_CPP := $(TB_BASE)/arith_tb/bf16_add_tb.cpp
#MATMUL_TB_CPP := ../../../../sw/tb/inference_tb/min_tb_inference.cpp

UNIT_TEST_DIR := build/unit_test

TOP_MODULE := cve2_top

VC_NAME := openhwgroup_cve2_cve2_top_0.1.vc
VC_PATCHED := openhwgroup_cve2_cve2_top_0.1_patched.vc

VIVADO_DIR := build/openhwgroup_cve2_cve2_top_0.1/syn-vivado

RTL := rtl

###############################################################################
# RTL simulation-debug displays (all OFF by default -> clean inference runs)
# Enable a subsystem's $display dumps by setting its var, e.g.:
#   make -f sim.mk run MAC_DEBUG=1 BRAM_DEBUG=1
# Subsystems: MAC (mac unit/array/controller/mac8x8), BRAM (accumulator bram),
#             CF (cf dispatch unit), ADDER (bf16 adder), CPU (core/alu trace)
###############################################################################
DBG_DEFINES := $(if $(MAC_DEBUG),+define+MAC_DEBUG) \
               $(if $(BRAM_DEBUG),+define+BRAM_DEBUG) \
               $(if $(CF_DEBUG),+define+CF_DEBUG) \
               $(if $(ADDER_DEBUG),+define+ADDER_DEBUG) \
               $(if $(CPU_DEBUG),+define+CPU_DEBUG)

###############################################################################
# Default
###############################################################################

all: run

# Generate FP4 multiply LUT
.PHONY: fp4_mul_int9
$(RTL)/fp4_mul_int9.sv: $(wildcard scripts/rtl_gen/*)
	@echo "Generating FP4 Multiplier LUT Configuration"
	python3 scripts/rtl_gen/gen_fp4_mul.py $(RTL)/fp4_mul_int9.sv

###############################################################################
#	Unit Block Testing
###############################################################################

.PHONY: mac_cell
mac_cell: $(RTL)/fp4_mul_int9.sv
	@mkdir -p $(UNIT_TEST_DIR)/mac_cell
	@echo "Building Vmac_cell in $(UNIT_TEST_DIR)/mac_cell"
	cd $(UNIT_TEST_DIR)/mac_cell && \
	verilator --cc --exe --build \
		-sv \
		-Wno-fatal \
		--coverage \
		$(DBG_DEFINES) \
		--top-module mac_cell \
		../../../$(RTL)/fp4_pkg.sv \
		../../../$(RTL)/mac_cell.sv \
		../../../$(RTL)/sat16_adder.sv \
		../../../$(RTL)/accumulator_reg.sv \
		../../../$(RTL)/fp4_mul_int9.sv \
		$(MAC_CELL_TB_CPP)

	@echo "=================================================="
	@echo "MAC CELL SIM BUILD COMPLETE"
	@echo "=================================================="
	@echo "Generated executable:"
	@find build -name Vmac_cell 2>/dev/null || true
	@echo "=================================================="


.PHONY: clean_mac_cell
clean_mac_cell: 
	rm -rf $(UNIT_TEST_DIR)/mac_cell


.PHONY: test_bf16_add
test_bf16_add: $(RTL)/parameterized_adder.sv
	@mkdir -p $(UNIT_TEST_DIR)/bf16_adder
	@echo "Building Vbf16_adder in $(UNIT_TEST_DIR)/bf16_adder"
	cd $(UNIT_TEST_DIR)/bf16_adder && \
	verilator --cc --exe --build \
		-sv \
		-Wno-fatal \
		--coverage \
		$(DBG_DEFINES) \
		--top-module parameterized_adder \
		../../../$(RTL)/fp4_pkg.sv \
		../../../$(RTL)/parameterized_adder.sv \
		$(BF16_ADDER_TB_CPP)
	@echo "=================================================="
	@echo "BF16 ADDER SIM BUILD COMPLETE"
	@echo "=================================================="
	@echo "Generated executable:"
	@find build -name Vparameterized_adder 2>/dev/null || true
	@echo "=================================================="


.PHONY: clean_bf16_adder
clean_bf16_adder: 
	rm -rf $(UNIT_TEST_DIR)/bf16_adder



###############################################################################
# STEP 1: Generate FuseSoC file list (.vc)
###############################################################################
.PHONY: fuse
fuse: rtl/fp4_mul_int9.sv
	@echo "Running FuseSoC setup..."
	VERILATOR_OPTIONS="-Wno-fatal" \
	PATH="$(PWD)/venv_cve2/bin:$$PATH" \
	fusesoc --cores-root=. run \
		--target=lint \
		--tool=verilator \
		--setup \
		openhwgroup:cve2:cve2_top:0.1 \
		$$(./util/cve2_config.py $(CVE2_CONFIG) fusesoc_opts)


################################################################################
#  VIVADO MAKE COMMAND
################################################################################
.PHONY: vivado
vivado:
	@echo "Running Vivado synthesis..."
	VERILATOR_OPTIONS="-Wno-fatal" \
		PATH="$(PWD)/venv_cve2/bin:$$PATH" \
		fusesoc --cores-root=. run \
			--target=syn \
			--tool=vivado \
			--setup \
			openhwgroup:cve2:cve2_top:0.1 \
			$$(./util/cve2_config.py $(CVE2_CONFIG) fusesoc_opts)
		# Make the .xpr file
		cd $(VIVADO_DIR) && \
		sed -i 's/-force$$/-force -part xczu7ev-ffvc1156-2-e/' openhwgroup_cve2_cve2_top_0.1.tcl \
		openhwgroup_cve2_cve2_top_0.1.tcl && \
		$(MAKE) openhwgroup_cve2_cve2_top_0.1.xpr




###############################################################################
# STEP 2: Patch VC file
###############################################################################
.PHONY: gen-vc
gen-vc: fuse
	@VC_FILE=$$(find build -name "$(VC_NAME)" | head -n 1); \
	if [ -z "$$VC_FILE" ]; then \
		echo "ERROR: VC file not found. Run 'make fuse' first."; \
		exit 1; \
	fi; \
	VC_DIR=$$(dirname "$$VC_FILE"); \
	echo "Patching VC file in $$VC_DIR"; \
	cd "$$VC_DIR" && \
	cp $(VC_NAME) $(VC_PATCHED) && \
	sed -i \
		-e '/--lint-only/d' \
		-e '/dpi_memutil.cc/d' \
		-e '/ecc32_mem_area.cc/d' \
		-e '/mem_area.cc/d' \
		-e '/sv_scoped.cc/d' \
		-e '/scrambled_ecc32_mem_area.cc/d' \
		$(VC_PATCHED)

###############################################################################
# STEP 3: Build simulation
###############################################################################
.PHONY: build-sim
build-sim: gen-vc
	@VC_FILE=$$(find build -name "$(VC_NAME)" | head -n 1); \
	if [ -z "$$VC_FILE" ]; then \
		echo "ERROR: VC file not found. Run 'make fuse' first."; \
		exit 1; \
	fi; \
	VC_DIR=$$(dirname "$$VC_FILE"); \
	echo "Building simulator in $$VC_DIR"; \
	cd "$$VC_DIR" && \
	verilator -f $(VC_PATCHED) \
		$(DBG_DEFINES) \
		-Wall \
		-Wno-fatal \
		--cc --exe --build \
		--top-module $(TOP_MODULE) \
		-LDFLAGS "-lelf" \
		$(MATMUL_TB_CPP)

###############################################################################
# STEP 4: Full pipeline
###############################################################################
.PHONY: run
run: fuse gen-vc build-sim
	@echo "=================================================="
	@echo "CVE2 + Matmul8 build complete"
	@echo "=================================================="
	@echo "Generated executable:"
	@find build -name Vcve2_top 2>/dev/null || true
	@echo "=================================================="

###############################################################################
# CLEAN
###############################################################################
.PHONY: clean
clean: clean_mac_cell
	rm -rf build/*/lint-verilator/obj_dir
	rm -f build/*/lint-verilator/*_patched.vc
