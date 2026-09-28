`timescale 1ns/1ps

module sat16_adder (

    input  logic signed [15:0] accum_i,
    input  logic        product_sign_i,             
    input  logic [7:0]  product_mag_i,
    output logic signed [15:0] accum_next_o

);

    logic signed [16:0] sum;
    logic signed [16:0] accum_ext;
    logic [8:0] product_ext;

    assign accum_ext   = {accum_i[15], accum_i};
    assign product_ext = {1'b0, product_mag_i};

    always_comb begin

        sum = product_sign_i ? 
                accum_ext - product_ext 
                : accum_ext + product_ext;

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
