`timescale 1ns/1ps

module sat16_adder (

    input  logic signed [15:0] accum_i,
    input  logic signed [7:0] prod_i,
    output logic signed [15:0] accum_next_o

);

    logic signed [16:0] sum;
    logic signed [16:0] accum_ext;

    assign accum_ext   = {accum_i[15], accum_i};

    always_comb begin

        sum = accum_ext + prod_i;
        /* Saturation logic */
        /* Negative overflow */
        if (sum[16] & ~sum[15]) begin
            accum_next_o = 16'h8000;       
        
        /* Positive Overflow */
        end else if (~sum[16] & sum[15]) begin
            accum_next_o = 16'h7fff;
        end else begin
            accum_next_o = sum[15:0];
        end
    end

endmodule
