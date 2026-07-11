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

# RTL debug $display gating (each OFF by default -> silent, fast; needs a rebuild).
#   VEC_DEBUG=1    VMAC64 datapath: MAC_VRF / MAC_MEM / MAC_LOAD_RETURN
#   MV_DEBUG=1     mv readback:     MAC_MV / tile-move + scalar writeback
#   TILE_DEBUG=1   full 8x8 MAC tile dump every cycle (very verbose)
#   CPU_DEBUG=1    CPU-side custom-function dispatch: [CF] / [ID->WB] / [REGFILE]
# e.g.:  make -f sim.mk build-sim MV_DEBUG=1
VL_DEFINES :=
ifeq ($(VEC_DEBUG),1)
VL_DEFINES += +define+VEC_DEBUG
endif
ifeq ($(MV_DEBUG),1)
VL_DEFINES += +define+MV_DEBUG
endif
ifeq ($(TILE_DEBUG),1)
VL_DEFINES += +define+TILE_DEBUG
endif
ifeq ($(CPU_DEBUG),1)
VL_DEFINES += +define+CPU_DEBUG
endif

TB_CPP := ../../../../sw/tb/inference_tb/min_tb_inference.cpp

TOP_MODULE := cve2_top

VC_NAME := openhwgroup_cve2_cve2_top_0.1.vc
VC_PATCHED := openhwgroup_cve2_cve2_top_0.1_patched.vc

###############################################################################
# Default
###############################################################################

all: run

###############################################################################
# STEP 1: Generate FuseSoC file list (.vc)
###############################################################################
.PHONY: fuse
fuse:
	@echo "Running FuseSoC setup..."
	VERILATOR_OPTIONS="-Wno-fatal" \
	PATH="$(PWD)/venv_cve2/bin:$$PATH" \
	fusesoc --cores-root=. run \
		--target=lint \
		--tool=verilator \
		--setup \
		openhwgroup:cve2:cve2_top:0.1 \
		$$(./util/cve2_config.py $(CVE2_CONFIG) fusesoc_opts)

###############################################################################
# STEP 2: Patch VC file
###############################################################################
.PHONY: gen-vc
gen-vc:
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
build-sim:
	@VC_FILE=$$(find build -name "$(VC_NAME)" | head -n 1); \
	if [ -z "$$VC_FILE" ]; then \
		echo "ERROR: VC file not found. Run 'make fuse' first."; \
		exit 1; \
	fi; \
	VC_DIR=$$(dirname "$$VC_FILE"); \
	echo "Building simulator in $$VC_DIR"; \
	cd "$$VC_DIR" && \
	verilator -f $(VC_PATCHED) \
		-Wall \
		-Wno-fatal \
		$(VL_DEFINES) \
		--cc --exe --build \
		--top-module $(TOP_MODULE) \
		-LDFLAGS "-lelf" \
		$(TB_CPP)

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
clean:
	rm -rf build/*/lint-verilator/obj_dir
	rm -f build/*/lint-verilator/*_patched.vc
