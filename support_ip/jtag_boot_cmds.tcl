
# GLOBAL PATH CONFIGURATION

# source user_env.tcl

# CHANGE THIS
set REPO_ROOT /home/khaditio/fp4-gemm-accelerator
set PLATFORM_NAME fp4_gemm
set PLATFORM_DIR "$REPO_ROOT/platform/$PLATFORM_NAME"


connect
tar 1
rst -por
exec sleep 1

targets -set -filter {name =~ "PSU"}
mwr 0xffca0038 0x1ff
rst -srst
exec sleep 1

targets -set -nocase -filter {name =~ "*PS TAP*"}
# bitstream
fpga "$PLATFORM_DIR/hw/accelerator_top.bit"
exec sleep 1

# Write pmufw
targets -set -nocase -filter {name =~ "*PSU*"}
mask_write 0xFFCA0038 0x1C0 0x1C0
targets -set -nocase -filter {name =~ "*MicroBlaze PMU*"}
dow "$PLATFORM_DIR/export/$PLATFORM_NAME/sw/$PLATFORM_NAME/boot/pmufw.elf"
con
exec sleep 1
targets -set -nocase -filter {name =~ "*PSU*"}
mask_write 0xFFCA0038 0x1C0 0x0

# Configuring PSU
targets -set -nocase -filter {name =~ "*APU*"}
mwr 0xffff0000 0x14000000
mask_write 0xFD1A0104 0x501 0x0
targets -set -nocase -filter {name =~ "PSU"}
source "$PLATFORM_DIR/hw/psu_init.tcl"
psu_init
after 500
psu_post_config
after 500
psu_ps_pl_reset_config
after 500
psu_ps_pl_isolation_removal
after 500

#download fsbl
# targets -set -nocase -filter {name =~ "*Cortex-A53 #0*"}
# dow /local/disk2/blankp1/vitis/mataccel/export/mataccel/sw/mataccel/boot/fsbl.elf
# con
# exec sleep 1
