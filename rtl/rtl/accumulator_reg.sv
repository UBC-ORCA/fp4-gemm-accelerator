`timescale 1ns/1ps

module accumulator_reg (

    input  logic clk,
    input  logic rst_n,

    input  logic clear_i,
    input  logic we_i,

    input  logic signed [15:0] d_i,

    output logic signed [15:0] q_o

);

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n)
            q_o <= '0;

        else if (clear_i)
            q_o <= '0;

        else if (we_i)
            q_o <= d_i;

    end

endmodule
