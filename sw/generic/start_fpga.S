.section .text.init
.globl _start

_start:
  la   sp, _stack_top
  call main

  # Write DONE flag for testbench (last word in mmio)
  li   t0, 0x10000FFC
  li   t1, 1
  sw   t1, 0(t0)

1:
  j 1b