
/**
 *  BRAM instuctions program binary source code. 
 * 
 *  This program tests functionalities of the following
 *  instructions: 
 *      - macBias
 *      - bramRd
 *      - bramRdFP4
 */

#include "bram_test_mmio.h"
#include "isa.h"

#include <stdint.h>



/*
    Seeds the bram from program memory to hardware bram.    

    Bias shape is (N_TILES, TT). This packs the column 
    dimension (output feature dimension) into tiles 
    of size 8.

    Column is folded into hi and lo pairs, and we have
    8 broadcasted rows.

    This stimulus is similar to the gemm kernel.
*/
static void seed_bram1()
{
    for (uint32_t t = 0; t < BRAM_NTILES; ++t){
        for (uint32_t c = 0; c < BRAM_COLS/2; ++c){

            uint32_t mem_offset = t * BRAM_COLS/2 + c;
            uint32_t mem_word = BIAS_IN_ADDR[mem_offset];
            uint16_t lo, hi;
            lo = (uint16_t)(mem_word & 0xFFFF);
            hi = (uint16_t)(mem_word >> 16);

            // Broadcast over rows
            #pragma GCC unroll 4
            for (uint32_t r = 0; r < BRAM_ROWS; r += 2){
                mac_bias(t, r, 2*c, lo);
                mac_bias(t, r, 2*c+1, hi);
            }
        }
    }
}

/*
    Dumps the BRAM content to another buffer in main program memory
    as BF16 values.

    The layout of the output matrix is NTILES * N_COLS words
*/
static void dump_bram()
{

    uint32_t *bf16_readout_buf = READOUT_BF16_ADDR;

    for (uint32_t t = 0; t < BRAM_NTILES; ++t){
        for (uint32_t c = 0; c < BRAM_COLS/2; ++c){
            for (uint32_t r = 0; r < BRAM_ROWS; r += 2){
                uint32_t bram_rd_packed;
                uint32_t readout_offset;
                uint16_t lo, hi;

                /* 
                    This contains the data from the coordinates
                    (t, c, r), and (t, c, r+1) 
                */
                bram_rd_packed = bram_rd(t, r, c);
                readout_offset = t * BRAM_COLS/2 * BRAM_ROWS
                                + c * BRAM_ROWS
                                + r;
                
                hi = (bram_rd_packed >> 16) & 0xFFFF;
                lo = bram_rd_packed & 0xFFFF;

                bf16_readout_buf[readout_offset]    = lo;
                bf16_readout_buf[readout_offset+1]  = hi;                
            }
        }
    }
}

/*
    Dumps the BRAM content to another buffer in the main 
    program memory as FP4 values.
*/
static void dump_bram_fp4()
{
    /* 
        Results should be in the format of 
        NTILES * TT
    */  
    uint32_t *fp4_readout_buf = READOUT_FP4_ADDR;

    for (unsigned t = 0; t < BRAM_NTILES; ++t){
        #pragma GCC unroll TT // TT
        for (unsigned c = 0; c < BRAM_COLS; ++c){
            // Collapses all TT rows back to a single row.
            BRAM_RD_FP4(fp4_readout_buf[t * TT + c], 
                BRAM_ADDR(t, 0, c));
        }
    }
}

int main() {

    seed_bram1();
    dump_bram();
    dump_bram_fp4();

    exit(0);
}

















