`timescale 1ns / 1ps

module int16_to_bf16 
import mx_pkg::*;
(
    input  logic signed [15:0] int_in,
    output bf16_t              bf16_out
);

    logic        sign;
    logic [15:0] abs_val;
    logic [3:0]  lzc; // Leading zero count
    
    assign sign    = int_in[15];
    assign abs_val = sign ? -int_in : int_in;

    // Priority Encoder / Leading Zero Count for the absolute magnitude
    always_comb begin
        if      (abs_val[15]) lzc = 4'd0;
        else if (abs_val[14]) lzc = 4'd1;
        else if (abs_val[13]) lzc = 4'd2;
        else if (abs_val[12]) lzc = 4'd3;
        else if (abs_val[11]) lzc = 4'd4;
        else if (abs_val[10]) lzc = 4'd5;
        else if (abs_val[9])  lzc = 4'd6;
        else if (abs_val[8])  lzc = 4'd7;
        else if (abs_val[7])  lzc = 4'd8;
        else if (abs_val[6])  lzc = 4'd9;
        else if (abs_val[5])  lzc = 4'd10;
        else if (abs_val[4])  lzc = 4'd11;
        else if (abs_val[3])  lzc = 4'd12;
        else if (abs_val[2])  lzc = 4'd13;
        else if (abs_val[1])  lzc = 4'd14;
        else                  lzc = 4'd15;
    end

    // Shift left to normalize (hidden bit aligns to MSB position)
    logic [15:0] norm_val;
    assign norm_val = abs_val << lzc;

    // Unrounded exponent calculation based on BF16 bias (127)
    // INT16 MSB (bit 15) represents 2^15 relative to 2^0
    // Note that there's a -2 bias because of the FP4 encoding.
    // Therefore the intial_exp = (127 + 15 - 2) - lzc
    logic [7:0] initial_exp;
    //assign initial_exp = 8'd140 - {4'b0, lzc};
assign initial_exp = 8'd142 - {4'b0, lzc};

    // Rounding logic: Extract explicit mantissa bits, guard, round, and sticky
    // Normalized string: [15] is hidden bit, [14:8] are target 7 mantissa bits
    logic [6:0] mant_base;
    logic       g, r, s;
    
    assign mant_base = norm_val[14:8];
    assign g         = norm_val[7];
    assign r         = norm_val[6];
    assign s         = |norm_val[5:0];

    logic round_up;
    assign round_up = g && (r || s || mant_base[0]); // Round to Nearest, Ties to Even

    // Final packing with rounding adjustment
    always_comb begin
        if (int_in == 16'd0) begin
            bf16_out = '0;
        end else begin
            bf16_out.sign = sign;
            if (round_up) begin
                if (mant_base == 7'h7F) begin
                    bf16_out.mant = '0;
                    bf16_out.exp  = initial_exp + 1'b1;
                end else begin
                    bf16_out.mant = mant_base + 1'b1;
                    bf16_out.exp  = initial_exp;
                end
            end else begin
                bf16_out.mant = mant_base;
                bf16_out.exp  = initial_exp;
            end
        end
    end

endmodule
