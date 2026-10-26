#pragma once

/*
    Also preprocessed into the linker script (link.ld.in) with -D__LINKER__,
    so only plain integer #defines may be used for the memory regions.
*/
#ifndef __LINKER__
#include <stdint.h>
#endif

/*
    Address map, shared by the firmware, linker script and the tb:
        0x0000_0000  program memory  512 KB  .text, .rodata
        0x0008_0000  data memory     256 KB  .data, .bss, stack
        0x0010_0000  staging memory   64 KB  tb <-> firmware buffers, not in the image
        0x1000_0000  UART            (uart_logging.h)
        0xFFFF_0000  DONE            start.S writes 1 when main returns
        0xFFFF_0004  STATUS          exit code, valid once DONE is set
*/
#define PROGRAM_MEM_BASE  0x00000000
#define PROGRAM_MEM_WORDS 0x20000

#define DATA_MEM_BASE  0x00080000
#define DATA_MEM_WORDS 0x10000

#define STAGE_MEM_BASE  0x00100000
#define STAGE_MEM_WORDS 0x4000

/* 
    Number of (32-bit) words in the BRAM
*/
#define BRAM_WORDS  ((BRAM_TILES * BRAM_ROWS * BRAM_COLS)/2)

/* TODO: vmac staging buffers, still placeholders; give each its own offset */
#define ACTIVATIONS_ADDR ((uint32_t *) STAGE_MEM_BASE)
#define ACTIVATIONS_WORDS

#define WEIGHTS_ADDR ((uint32_t *) STAGE_MEM_BASE)


#define INPUT_SCALE_ADDR
#define INPUT_SCALE_WORDS

#define ACT_SCALE_ADDR
#define ACT_SCALE_WRODS

#define C_ADDR  ((uint32_t *) STAGE_MEM_BASE)
#define C_WORDS

#define DONE_MMIO_ADDR    0xFFFF0000u
#define STATUS_MMIO_ADDR  0xFFFF0004u

#define DONE_MMIO     ((volatile uint32_t *) DONE_MMIO_ADDR)
#define STATUS_MMIO   ((volatile int32_t *) STATUS_MMIO_ADDR)
#define PUTCHAR_MMIO












