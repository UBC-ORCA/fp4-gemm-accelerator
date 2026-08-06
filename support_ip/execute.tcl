source jtag_boot_cmds.tcl

after 5000
mrd -fo -size w 0xA0000FFC
mrd -bin -file uartdump.txt -fo -size b 0xA0000000 0xFFC