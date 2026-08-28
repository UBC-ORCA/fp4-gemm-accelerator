`timescale 1ns/1ps

module mac_cell
import fp4_pkg::*; (

    input  logic clk,

    // control
    input  logic mac_en_i,
    input  logic clear_i,

    // FP4 inputs
    input  int4_t act_i,
    input  int4_t wt_i,

    //-----------------------------------------
    // Accumulator output
    //-----------------------------------------

    output logic signed [15:0] accum_o

);

    //-----------------------------------------
    // Internal signals
    //-----------------------------------------
    logic signed [15:0] accum_next;
    //logic [7:0] fp4_mul_mag;
    //logic fp4_mul_sign;

    // INT4 Multiply -> {sign, mantissa} HW block
/*
    fp4_mul_int9 mul_int9 (
        .fp4_a_i(act_i),
        .fp4_b_i(wt_i),
        .sign_o(fp4_mul_sign),
        .u8_mag_o(fp4_mul_mag)
    );
*/

	logic signed [7:0] product;

	int4_mul_int8 mul_int8  (
    		.act_i(act_i),
    		.wt_i(wt_i),
    		.product_o(product)
	);


    //-----------------------------------------
    // Saturating adder
    //-----------------------------------------
    sat16_adder u_add (

        .accum_i(accum_o),
  //      .product_mag_i(fp4_mul_mag),
  //      .product_sign_i(fp4_mul_sign),
        .product_mag_i(product[6:0]),
        .product_sign_i(product[7]),
        .accum_next_o(accum_next)

    );

    //-----------------------------------------
    // Accumulator register
    //-----------------------------------------

    accumulator_reg u_acc (

        .clk(clk),

        .clear_i(clear_i),
        .we_i(mac_en_i),

        .d_i(accum_next),

        .q_o(accum_o)

    );

endmodule
