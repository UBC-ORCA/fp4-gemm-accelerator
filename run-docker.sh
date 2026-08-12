#!/bin/bash

# Mount RTL and SW dirs instead of copying \
docker run -it \
    -v "$(pwd)/support_ip:/work/support_ip" \
    -v "$(pwd)/rtl:/work/rtl" \
    -v "$(pwd)/sw:/work/sw" \
-v /tools/Xilinx/Vivado/2023.1:/tools/Xilinx/Vivado/2023.1 \
-v /tools/Xilinx/Vitis/2022.2:/tools/Xilinx/Vitis/2022.2 \
rtl-freeze:09-08-26 /bin/bash \
-c 'unset LC_ALL; export LANG=C.UTF-8; export PATH=/tools/Xilinx/Vivado/2023.1/bin:/tools/Xilinx/Vitis/2022.2/bin:$PATH; exec /bin/bash'
