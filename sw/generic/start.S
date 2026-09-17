.section .text.init
.globl _start

_start:
  la   sp, _stack_top
  call main

  # Write DONE flag for testbench (MMIO at 0xFFFF0000)
  li   t0, 0xFFFF0000
  li   t1, 1
  sw   t1, 0(t0)

1:
  j 1b