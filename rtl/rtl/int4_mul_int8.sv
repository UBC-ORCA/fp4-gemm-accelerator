`timescale 1ns/1ps

/**
 * INT4 × INT4 multiplier.
 *
 * Inputs:
 *   - Signed 4-bit integers (two's complement)
 *
 * Output:
 *   - Signed 8-bit product
 *
 * Product range:
 *   -8 × -8 = +64
 *    7 ×  7 = +49
 *   -8 ×  7 = -56
 *
 * An 8-bit signed output is sufficient for all possible products.
 */
module int4_mul_int8
import fp4_pkg::*; (

    input  int4_t act_i,
    input  int4_t wt_i,

    output logic signed [7:0] product_o

);

    assign product_o = act_i * wt_i;

endmodule
