connect
tar 1
rst -por
exec sleep 1

targets -set -filter {name =~ "PSU"}
mwr 0xffca0038 0x1ff
rst -srst
exec sleep 1

targets -set -nocase -filter {name =~ "*PS TAP*"}
fpga /local/disk2/blankp1/mataccel/fp4-gemm-accelerator/rtl/build/openhwgroup_cve2_cve2_top_0.1/default-vivado/openhwgroup_cve2_cve2_top_0.1.runs/impl_1/ps_wrapper.bit
exec sleep 1

# Write pmufw
targets -set -nocase -filter {name =~ "*PSU*"}
mask_write 0xFFCA0038 0x1C0 0x1C0
targets -set -nocase -filter {name =~ "*MicroBlaze PMU*"}
dow /local/disk2/blankp1/vitis/mataccel/export/mataccel/sw/mataccel/boot/pmufw.elf
con
exec sleep 1
targets -set -nocase -filter {name =~ "*PSU*"}
mask_write 0xFFCA0038 0x1C0 0x0

# Configuring PSU
targets -set -nocase -filter {name =~ "*APU*"}
mwr 0xffff0000 0x14000000
mask_write 0xFD1A0104 0x501 0x0
targets -set -nocase -filter {name =~ "PSU"}
source /local/disk2/blankp1/mataccel/fp4-gemm-accelerator/rtl/build/openhwgroup_cve2_cve2_top_0.1/default-vivado/openhwgroup_cve2_cve2_top_0.1.ip_user_files/mem_init_files/psu_init.tcl
psu_init
after 500
psu_post_config
after 500
psu_ps_pl_reset_config
after 500
psu_ps_pl_isolation_removal
after 500

#download fsbl
targets -set -nocase -filter {name =~ "*Cortex-A53 #0*"}
dow /local/disk2/blankp1/vitis/mataccel/export/mataccel/sw/mataccel/boot/fsbl.elf
con
exec sleep 1