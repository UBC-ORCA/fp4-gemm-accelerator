`timescale 1ns/1ps

module accumulator_reg (

    input  logic clk,

    input  logic clear_i,
    input  logic we_i,

    input  logic signed [15:0] d_i,

    output logic signed [15:0] q_o

);

    always_ff @(posedge clk) begin
        /* Reset is undefined */

        if (clear_i)
            q_o <= '0;

        else if (we_i)
            q_o <= d_i;

    end

endmodule
