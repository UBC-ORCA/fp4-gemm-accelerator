`timescale 1ns/1ps

module fp4_multiplier (

    input  logic signed [4:0] a_i,
    input  logic signed [4:0] b_i,

    output logic signed [9:0] product_o

);

    assign product_o = a_i * b_i;

endmodule
