`timescale 1ns / 1ps

module int16_to_bf16_scaled
import mx_pkg::*;
(
    input  logic signed [15:0] int_in,
    input  logic [7:0]         scale_a,
    input  logic [7:0]         scale_b,
    output bf16_t              bf16_out
);

    //------------------------------------------------------------
    // Absolute value and sign extraction
    //------------------------------------------------------------

    logic        sign;
    logic [15:0] abs_val;
    logic [3:0]  lzc;

    assign sign    = int_in[15];
    assign abs_val = sign ? -int_in : int_in;


    //------------------------------------------------------------
    // Leading zero count
    //------------------------------------------------------------

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


    //------------------------------------------------------------
    // Normalize magnitude
    //------------------------------------------------------------

    logic [15:0] norm_val;

    assign norm_val = abs_val << lzc;


    //------------------------------------------------------------
    // Mantissa extraction and BF16 rounding
    //------------------------------------------------------------

    logic [6:0] mant_base;
    logic       g, r, s;
    logic       round_up;

    assign mant_base = norm_val[14:8];

    assign g = norm_val[7];
    assign r = norm_val[6];
    assign s = |norm_val[5:0];

    // Round-to-nearest-even
    assign round_up = g && (r || s || mant_base[0]);


    //------------------------------------------------------------
    // Scaled exponent computation
    //
    // initial_exp = 127 + 15 - lzc
    // scaled_exp  = initial_exp + scaleA + scaleB - 254
    //
    // Simplifies to:
    //
    // scaled_exp = scaleA + scaleB - lzc - 112
    //------------------------------------------------------------

    logic signed [9:0] scaled_exp;
    logic signed [9:0] rounded_scaled_exp;

    assign scaled_exp =
            $signed({2'b0,scale_a})
          + $signed({2'b0,scale_b})
          - $signed({6'b0,lzc})
          - 10'sd112;

    // Increment exponent if mantissa rounding overflows.
    assign rounded_scaled_exp =
            scaled_exp +
            ((round_up && (mant_base == 7'h7F))
                ? 10'sd1
                : 10'sd0);


    //------------------------------------------------------------
    // Final BF16 packing
    //------------------------------------------------------------

    always_comb begin

        //--------------------------------------------------------
        // Zero bypass
        //--------------------------------------------------------

        if (int_in == 16'd0) begin

            bf16_out = '0;

        end

        //--------------------------------------------------------
        // Overflow -> Infinity
        //--------------------------------------------------------

        else if (rounded_scaled_exp >= 10'sd255) begin

            bf16_out.sign = sign;
            bf16_out.exp  = 8'hFF;
            bf16_out.mant = 7'h00;

        end

        //--------------------------------------------------------
        // Underflow -> Flush-to-zero
        //--------------------------------------------------------

        else if (rounded_scaled_exp <= 10'sd0) begin

            bf16_out.exp  = 8'h00;
            bf16_out.mant = 7'h00;
            bf16_out.sign = 1'b0;

        end

        //--------------------------------------------------------
        // Normal BF16 value
        //--------------------------------------------------------

        else begin

            bf16_out.sign = sign;
            bf16_out.exp  = rounded_scaled_exp[7:0];

            if (round_up) begin

                if (mant_base == 7'h7F) begin
                    bf16_out.mant = 7'h00;
                end
                else begin
                    bf16_out.mant = mant_base + 1'b1;
                end

            end
            else begin
                bf16_out.mant = mant_base;
            end

        end

    end

endmodule
