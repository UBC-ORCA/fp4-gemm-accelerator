
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
#include "common.h"
#include "uart_logging.h"
#include <stdint.h>

/**
    Applies hardtanh activation to bf16 values and quantize to
    FP4 values. Bit-exact model of rtl/chief/bf16_to_fp4.sv.

    Readout shift is 3, so the grid floor is exp 122 and
    anything with exp > 125 clamps to the max magnitude (6).
    Returns {sign, mag[2:0]} in the low nibble.
*/
static uint8_t bf16_to_fp4_activation (
    uint16_t bf16
){
    const uint32_t LUT_FLOOR = 122;

    uint32_t sign = (bf16 >> 15) & 0x1;
    uint32_t exp  = (bf16 >> 7) & 0xFF;
    uint32_t man  = bf16 & 0x7F;

    /* Rounding thresholds on 1.man, see the RTL for the tie rules */
    uint32_t man_non_zero = man != 0;
    uint32_t man_33       = man > 32;
    uint32_t man_64       = man >= 64;
    uint32_t man_96       = man >= 96;

    uint32_t grid;
    switch ((exp - LUT_FLOOR) & 0x3){
        case 0:  grid = man_non_zero;                  break;
        case 1:  grid = 1 + man_64;                    break;
        case 2:  grid = 2 + man_33 + man_96;           break;
        default: grid = 4 + man_33 + man_96;           break;
    }

    uint32_t mag = (exp < LUT_FLOOR) ? 0
                 : (exp > 125)       ? 6
                 : grid;

    return (uint8_t)(((sign && mag) << 3) | mag);
}

/*
    Tests whether the bramRd operation broadcasts
    properly.
    
    Seed the bram with bias values, and immediately 
    perform a read. Then checks if the read values 
    match the corresponding tile/row/col.

    bf16_readout_arr layout is an array of size NTILES x NCOLS
    words.

    bf16_expected_arr is an array of layout [NTILES x NROWS x NCOLS]
*/
static void seed_bias_hw (
    
    /* 
        Input array layout is (NTILES, NCOLS)
    */
    uint16_t *bf16_readout_arr,
    uint32_t bias_tile,  
    uint32_t bias_rows,
    uint32_t bias_cols
) {

    for (uint32_t t = 0; t < bias_tile; ++t){
        for (uint32_t c = 0; c < bias_cols; ++c){

            uint16_t rdValue = bf16_readout_arr[t * bias_cols + c];
            MAC_BIAS(BRAM_ADDR(t, 0, c), rdValue);
        }
    }
}

static void seed_bias_sw (
    
    /* 
        Input array layout is (NTILES, NCOLS)
    */
    uint16_t *bf16_readout_arr,
    uint16_t *bf16_expected_arr,
    uint32_t bias_tile,  
    uint32_t bias_rows,
    uint32_t bias_cols
) {

    for (uint32_t t = 0; t < bias_tile; ++t){
        for (uint32_t c = 0; c < bias_cols; ++c){

            uint16_t rdValue = bf16_readout_arr[t * bias_cols + c];
            
            for (uint32_t r = 0; r < bias_rows; ++r){
                bf16_expected_arr[t * bias_rows * bias_cols 
                    + r * bias_cols + c] = rdValue;
            }
        }
    }
}

static int bram_rd_bf16_hw (
    
    /* 
        Input array layout is (NTILES, NROWS, NCOLS)
    */
    uint16_t *dest_arr,
    uint32_t bias_tile,  
    uint32_t bias_rows,
    uint32_t bias_cols
) {
    for (uint32_t t = 0; t < bias_tile; ++t){
        for (uint32_t c = 0; c < bias_cols; ++c){

            /* Row pair access: {row r+1, row r} per read */
            #pragma GCC unroll 4
            for (uint32_t r = 0; r < bias_rows; r += 2){
                uint32_t readoutVal = bram_rd(t, r, c);
                dest_arr[
                    t * bias_rows * bias_cols
                    + r * bias_cols
                    + c
                ] = readoutVal & 0xFFFF;

                dest_arr[
                    t * bias_rows * bias_cols
                    + (r+1) * bias_cols
                    + c
                ] = (readoutVal >> 16) & 0xFFFF;
            }
        }
    }

    return 0;
}


/*
    Smoketest 1: Test expected behaviour of 
    macBias, and bramRd instructions
*/
static int bram_rd_broadcast_test( 
    uint16_t *bf16_readout_arr,
    uint16_t *bf16_dest_arr,
    uint16_t *bf16_expected_arr,
    uint32_t bias_tile,  
    uint32_t bias_rows,
    uint32_t bias_cols
){

    
    /*
        Step 1: Seed the bram based on the source array
    */
    seed_bias_hw(bf16_readout_arr, 
                bias_tile,
                bias_rows, 
                bias_cols);

    seed_bias_sw(bf16_readout_arr, 
                bf16_expected_arr,
                bias_tile,
                bias_rows, 
                bias_cols);

    /*
        Step 2: Read out the values from the mac unit bram, 
        and compare to the reference.
    */
    bram_rd_bf16_hw(
        bf16_dest_arr, 
        bias_tile, 
        bias_rows, 
        bias_cols
    );

    for (uint32_t t = 0; t < bias_tile; ++t){
        for (uint32_t r = 0; r < bias_rows; ++r){
            for (uint32_t c = 0; c < bias_cols; ++c){
                uint64_t expected_idx = 
                    t * bias_rows * bias_cols
                    + r * bias_cols
                    + c;


                if (bf16_expected_arr[expected_idx]
                        != bf16_dest_arr[expected_idx]){

                    /* TODO: Diagnosic Message*/
                    return -1;
                }
            }
        }
    }

    return 0;
}


/*
    Reads every (tile, col) out of the bram as packed FP4.

    dest_arr layout is (NTILES, NCOLS) words, all bias_rows (== TT)
    rows packed into one word: nibble r = row r.

    Gen2 BRAM_RD_FP4 also moves a bias into the cell it reads;
    this seeds 0, so the bram is cleared as it is read out.
*/
static int bram_rd_fp4_hw (
    uint32_t *dest_arr,
    uint32_t bias_tile,
    uint32_t bias_rows,
    uint32_t bias_cols
) {
    (void)bias_rows;
    for (uint32_t t = 0; t < bias_tile; ++t){
        #pragma GCC unroll 8
        for (uint32_t c = 0; c < bias_cols; ++c){
            BRAM_RD_FP4(dest_arr[t * bias_cols + c],
                BRAM_ADDR(t, 0, c), 0);
        }
    }

    return 0;
}

/*
    Software model of bram_rd_fp4_hw.

    bf16_expected_arr layout is (NTILES, NROWS, NCOLS), as
    produced by seed_bias_sw. Each column of rows is quantized
    and packed into fp4_expected_arr with the same (NTILES, NCOLS)
    word layout as the hw variant.
*/
static int bram_rd_fp4_sw (
    uint16_t *bf16_expected_arr,
    uint32_t *fp4_expected_arr,
    uint32_t bias_tile,
    uint32_t bias_rows,
    uint32_t bias_cols
) {
    for (uint32_t t = 0; t < bias_tile; ++t){
        for (uint32_t c = 0; c < bias_cols; ++c){

            uint32_t packed = 0;
            for (uint32_t r = 0; r < bias_rows; ++r){
                uint16_t bf16 = bf16_expected_arr[
                    t * bias_rows * bias_cols
                    + r * bias_cols
                    + c
                ];
                packed |= (uint32_t)bf16_to_fp4_activation(bf16) << (4 * r);
            }

            fp4_expected_arr[t * bias_cols + c] = packed;
        }
    }

    return 0;
}


/*
    Smoketest 2: Test expected behaviour of
    macBias, and bramRdFP4 instructions

    Seeds the bram with the same bias source as smoketest 1,
    reads it back as packed FP4 and compares against the
    software quantizer. The readout clears the bram.

    fp4_dest_arr and fp4_expected_arr layout is (NTILES, NCOLS) words.
*/
static int bram_rd_fp4_test(
    uint16_t *bf16_readout_arr,
    uint16_t *bf16_expected_arr,
    uint32_t *fp4_dest_arr,
    uint32_t *fp4_expected_arr,
    uint32_t bias_tile,
    uint32_t bias_rows,
    uint32_t bias_cols
){

    /*
        Step 1: Seed the bram based on the source array
    */
    seed_bias_hw(bf16_readout_arr,
                bias_tile,
                bias_rows,
                bias_cols);

    seed_bias_sw(bf16_readout_arr,
                bf16_expected_arr,
                bias_tile,
                bias_rows,
                bias_cols);

    /*
        Step 2: Read out the bram as FP4, and quantize
        the reference with the software model.
    */
    bram_rd_fp4_hw(
        fp4_dest_arr,
        bias_tile,
        bias_rows,
        bias_cols
    );

    bram_rd_fp4_sw(
        bf16_expected_arr,
        fp4_expected_arr,
        bias_tile,
        bias_rows,
        bias_cols
    );

    for (uint32_t t = 0; t < bias_tile; ++t){
        for (uint32_t c = 0; c < bias_cols; ++c){
            uint32_t idx = t * bias_cols + c;

            if (fp4_expected_arr[idx] != fp4_dest_arr[idx]){
                print_str("  fp4 mismatch t=");
                putdec(t);
                print_str(" c=");
                putdec(c);
                print_str(" expected=");
                print_u32_hex(fp4_expected_arr[idx]);
                print_str(" got=");
                print_u32_hex(fp4_dest_arr[idx]);
                print_str("\n");
                return -1;
            }
        }
    }

    return 0;
}

/* Software references; readouts go to the staging buffers so the tb can dump them */
static uint16_t bf16_expected_arr[BRAM_NTILES * BRAM_ROWS * BRAM_COLS];
static uint32_t fp4_expected_arr [BRAM_NTILES * BRAM_COLS];

int main() {
    /* Bias source, layout (NTILES, NCOLS), filled in by the tb */
    uint16_t *bias_in       = (uint16_t *)BIAS_IN_ADDR;
    uint16_t *bf16_dest_arr = (uint16_t *)READOUT_BF16_ADDR;
    uint32_t *fp4_dest_arr  = READOUT_FP4_ADDR;
    int status = 0;

    if (bram_rd_broadcast_test(bias_in,
                               bf16_dest_arr,
                               bf16_expected_arr,
                               BRAM_NTILES,
                               BRAM_ROWS,
                               BRAM_COLS) != 0){
        print_str("Error: Broadcast BRAM Test failed\n");
        status = -1;
    }

    if (bram_rd_fp4_test(bias_in,
                         bf16_expected_arr,
                         fp4_dest_arr,
                         fp4_expected_arr,
                         BRAM_NTILES,
                         BRAM_ROWS,
                         BRAM_COLS) != 0){
        print_str("Error: FP4 Readout BRAM Test failed\n");
        status = -1;
    }

    exit(status);
}

















