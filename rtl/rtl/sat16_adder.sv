`timescale 1ns/1ps

module sat16_adder (

    input  logic signed [15:0] accum_i,
    input  logic signed [9:0]  product_i,

    output logic signed [15:0] accum_next_o

);

    logic signed [16:0] sum;

    always_comb begin

        sum = accum_i + product_i;

        if (sum > 17'sd32767)
            accum_next_o = 16'sd32767;

        else if (sum < -17'sd32768)
            accum_next_o = -16'sd32768;

        else
            accum_next_o = sum[15:0];

    end

endmodule
