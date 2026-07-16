source jtag_boot_cmds.tcl

while {[mrd -fo -size w 0xA0007FFC] < 1} {
    after 1000
}
mrd -bin -file uartdump.txt -fo -size b 0xA0000000 0x7FFC