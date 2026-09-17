`timescale 1ns / 1ps

module e4m3_scale 
import fp4_pkg::*;
(
    input logic signed [15:0] acc_q14_2_in, 
    input fp4_scaler_e4m3_t a_scale_in,
    input fp4_scaler_e4m3_t w_scale_in,
    input bf16_t bf16_in, 
    output bf16_t bf16_out
);
    /* e7m6 prod */
    bf16_t prod_abc;
    logic isNaN, isZero;

    e4m3_mul mul (
        .A8(a_scale_in), 
        .B8(w_scale_in),
        .q14_2_C_in(acc_q14_2_in),
        .PABC(prod_abc),
        .isNaN(isNaN), 
        .isZero(isZero)
    );

    parameterized_adder_e4m3 u_add (
        .a(bf16_in),
        .b(prod_abc), 
        .sum(bf16_out)
    );

endmodule




