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
    logic [4:0]  lzc;

    assign sign    = int_in[15];
    assign abs_val = sign ? -int_in : int_in;


    //------------------------------------------------------------
    // Leading zero count
    //------------------------------------------------------------
    always_comb begin
        casez (abs_val)
            16'b1???_????_????_????: lzc = 5'd0;
            16'b01??_????_????_????: lzc = 5'd1;
            16'b001?_????_????_????: lzc = 5'd2;
            16'b0001_????_????_????: lzc = 5'd3;
            16'b0000_1???_????_????: lzc = 5'd4;
            16'b0000_01??_????_????: lzc = 5'd5;
            16'b0000_001?_????_????: lzc = 5'd6;
            16'b0000_0001_????_????: lzc = 5'd7;
            16'b0000_0000_1???_????: lzc = 5'd8;
            16'b0000_0000_01??_????: lzc = 5'd9;
            16'b0000_0000_001?_????: lzc = 5'd10;
            16'b0000_0000_0001_????: lzc = 5'd11;
            16'b0000_0000_0000_1???: lzc = 5'd12;
            16'b0000_0000_0000_01??: lzc = 5'd13;
            16'b0000_0000_0000_001?: lzc = 5'd14;
            16'b0000_0000_0000_0001: lzc = 5'd15;             
            default:  lzc = 5'd16;
        endcase
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
    logic [7:0] rounded_mant;

    assign mant_base = norm_val[14:8];

    assign g = norm_val[7];
    assign r = norm_val[6];
    assign s = |norm_val[5:0];

    // Round-to-nearest-even
    assign round_up = g && (r || s || mant_base[0]);
    assign rounded_mant = {1'b0, mant_base} + {7'b0, round_up};

    //------------------------------------------------------------
    // Scaled exponent computation
    //
    // initial_exp = 127 + 15 - lzc - 2
    // scaled_exp  = initial_exp + scaleA + scaleB - 254
    //
    // Simplifies to:
    //
    // scaled_exp = scaleA + scaleB - lzc - 114
    //------------------------------------------------------------

    logic signed [9:0] rounded_scaled_exp;

    logic signed [9:0] sgn_scale_a;
    logic signed [9:0] sgn_scale_b;
    logic signed [9:0] sgn_bf16_exp;

    logic signed [9:0] round_up_exp_increase;       
    logic round_exp_ov;
    // Increment exponent if mantissa rounding overflows.
    assign round_up_exp_increase = {9'd0, rounded_mant[7]};

    assign sgn_scale_a = {2'b0, scale_a};
    assign sgn_scale_b = {2'b0, scale_b};
    assign sgn_bf16_exp = {5'b0, lzc};

    assign rounded_scaled_exp = sgn_scale_a + sgn_scale_b + 
        round_up_exp_increase - sgn_bf16_exp - 10'sd114;

    assign round_exp_ov = ~rounded_scaled_exp[9] & (rounded_scaled_exp[8]
                            | &rounded_scaled_exp[7:0]);

    //------------------------------------------------------------
    // Final BF16 packing
    //------------------------------------------------------------

    always_comb begin

        bf16_out.sign = sign;

        //--------------------------------------------------------
        // Zero bypass or Underflow Management
        //--------------------------------------------------------

        // Only handle case where negative exponent occurs
        if (int_in == 16'd0 || rounded_scaled_exp[9]) begin
            bf16_out.exp = '0;
            bf16_out.mant = '0;
        end

        //--------------------------------------------------------
        // Exponent Overflow behaviour, goes to NaN
        //--------------------------------------------------------

        else if (round_exp_ov) begin
            bf16_out.exp  = 8'hFF;
            bf16_out.mant = 7'h40;
        end
  
        //--------------------------------------------------------
        // Normal BF16 value
        //--------------------------------------------------------

        else begin
            bf16_out.exp  = rounded_scaled_exp[7:0];
            bf16_out.mant = rounded_mant[6:0];
        end

    end

endmodule
