#pragma once

#include <stdint.h>
#include "mmio.h"

/* Tile Dimensions, in half words */
#define TT           8
#define BRAM_NTILES  32
#define BRAM_ROWS    TT
#define BRAM_COLS    TT

/*
    Staging buffers, carved out of the staging memory region.
    Each *_BASE is a byte address the tb can use; *_ADDR is the
    firmware pointer to the same place.
*/

/*
    Size in uint32_t words.
    Packed 2D bf16 array of size BRAM_NTILES, BRAM_COLS
*/
#define BIAS_IN_BASE (STAGE_MEM_BASE + 0x0000)
#define BIAS_IN_ADDR ((uint32_t *) BIAS_IN_BASE)
#define BIAS_IN_SIZE (BRAM_NTILES * BRAM_COLS/2)


/*
    Size in uint32_t words.
    Packed 3D bf16 array of size BRAM_NTILES, BRAM_ROWS, BRAM_COLS
*/
#define READOUT_BF16_BASE (STAGE_MEM_BASE + 0x1000)
#define READOUT_BF16_ADDR ((uint32_t *) READOUT_BF16_BASE)
#define READOUT_BF16_SIZE (BRAM_NTILES * BRAM_ROWS * BRAM_COLS/2)

/*
    Size in uint32_t words.
    2D array of size BRAM_NTILES, BRAM_COLS; all rows packed
    in a word as FP4 (nibble r = row r)
*/
#define READOUT_FP4_BASE (STAGE_MEM_BASE + 0x2000)
#define READOUT_FP4_ADDR ((uint32_t *) READOUT_FP4_BASE)
#define READOUT_FP4_SIZE (BRAM_NTILES * BRAM_COLS)
