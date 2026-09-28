`timescale 1ns/1ps

module sat16_adder
#(
    parameter int ACC_WIDTH = 12
) (

    input  logic signed [15:0] accum_i,
    input  logic signed [7:0] prod_i,
    output logic signed [15:0] accum_next_o

);
    localparam logic signed [ACC_WIDTH-1:0] MAX_VAL =
    {1'b0, {(ACC_WIDTH-1){1'b1}}};

    localparam logic signed [ACC_WIDTH-1:0] MIN_VAL =
        {1'b1, {(ACC_WIDTH-1){1'b0}}};

    logic signed [ACC_WIDTH:0] sum;
    logic signed [ACC_WIDTH:0] accum_ext;

    assign accum_ext   = {accum_i[ACC_WIDTH-1], accum_i};

    always_comb begin

        sum = accum_ext + prod_i;
        /* Saturation logic */
        /* Negative overflow */
        if (sum[ACC_WIDTH] & ~sum[ACC_WIDTH-1]) begin
            accum_next_o = MIN_VAL;       
        
        /* Positive Overflow */
        end else if (~sum[ACC_WIDTH] & sum[ACC_WIDTH-1]) begin
            accum_next_o = MAX_VAL;
        end else begin
            accum_next_o = sum[ACC_WIDTH-1:0];
        end
    end

endmodule
