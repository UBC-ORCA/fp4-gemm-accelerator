
Evaluation Plaform Setup
---------

This document assumes the following pre-requisites:
    - Vivado 2023.1 is used to perform P&R and bitstream generation
    - Vitis 2022 is used to generate the platform files
    - risc-unknown-gcc is added to the system path
    - Dependencies from the `rtl` folder have already been satisfied.

The evaluation platform consists of program memory and a UART I/O buffer
for serial output from the program. Additionally, the system includes
a hard processor 

To setup the system:

1. Perform an implementation run for the design, go to the `rtl` folder, 
and run the following:
(Make sure the proper toolchain is setup). 
```
    make -f sim.mk platform_impl    
```

This will generate some outputs in `output`, which should include 
a `.xsa` file for Vitis handoff, `.bit`, and hardware utilization reports in 
`rpts`.

2. **[NOTE: You only need to perform this step once!]**

Open up vitis with the `.xsa` as the project configuration. Then build 
the projects. For simplicity, the diretory of the Vitis project 
should be set to `support_ip/platform`.

If the project directory and name is different, you can change the path 
in `execute.tcl`, under the variables `PLATFORM_NAME` and `PLATFORM_DIR`.

Make sure `REPO_ROOT` points to the root of the repostiory.

3. Make sure `xsct` is included in `PATH`, now run `source execute.tcl`. 
Results should be dumped in `uartdump.txt`



Configuration Parameters
------------------------


## FMAX
Fmax is set within `ps.tcl`, look for the following attribute
```
CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} 
```
This sets the hard processor clock constraint to be 100 MHz. Note that 
the actual implemented frequency often only take a discrete amount of 
values, due to limitations of the PLL.

## System Memory Size [WIP -- Will need to make a single source of truth for setting this]
Currently, the system memory is controlled in `sw/inference *`,  `sw/generic/image.h`. 
This sets the amount of memory that is used to contain the program.

Make sure that the changes are reflected in `sim.mk` (search for `program.mem`), and 
`memory_system.sv` uses the same amount of memory.









