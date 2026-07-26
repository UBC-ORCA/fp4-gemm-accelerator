`timescale 1ns/1ps

module mac_cell (

    input  logic clk,
    input  logic rst_n,

    //-----------------------------------------
    // Control
    //-----------------------------------------

    input  logic mac_en_i,
    input  logic clear_i,
	input logic mv_clear_i, //for mv

    //-----------------------------------------
    // FP4 operands (INT5 rep)
    //-----------------------------------------

    input  logic signed [4:0] act_i,
    input  logic signed [4:0] wt_i,

    //-----------------------------------------
    // Accumulator output
    //-----------------------------------------

    output logic signed [15:0] accum_o

);

    //-----------------------------------------
    // Internal signals
    //-----------------------------------------

    logic signed [9:0] product;
    logic signed [15:0] accum_next;


    //-----------------------------------------
    // Multiplier
    //-----------------------------------------

    fp4_multiplier u_mult (

        .a_i(act_i),
        .b_i(wt_i),
        .product_o(product)

    );

    //-----------------------------------------
    // Saturating adder
    //-----------------------------------------

    sat16_adder u_add (

        .accum_i(accum_o),
        .product_i(product),
        .accum_next_o(accum_next)

    );

    //-----------------------------------------
    // Accumulator register
    //-----------------------------------------

    accumulator_reg u_acc (

        .clk(clk),
        .rst_n(rst_n),

        .clear_i(clear_i || mv_clear_i),
        .we_i(mac_en_i),

        .d_i(accum_next),

        .q_o(accum_o)

    );

endmodule
