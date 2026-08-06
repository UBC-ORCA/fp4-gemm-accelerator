`timescale 1ns / 1ps

module e8m0_scale 
import mx_pkg::*;
(
    input  bf16_t      bf16_in,
    input  logic [7:0] scale_a,
    input  logic [7:0] scale_b,
    output bf16_t      bf16_out
);

    logic signed [9:0] delta;
    logic signed [9:0] calculated_exp;

    // Compute delta = scaleA + scaleB - 254
    assign delta = $signed({2'b0, scale_a}) + $signed({2'b0, scale_b}) - 10'd254;
    assign calculated_exp = $signed({2'b0, bf16_in.exp}) + delta;

    always_comb begin
        // 1. Pass-through sign
        bf16_out.sign = bf16_in.sign;

        // 2. Zero-Bypass handling & special cases
        if (bf16_in.exp == 8'h0 && bf16_in.mant == 7'h0) begin
            bf16_out.exp  = '0;
            bf16_out.mant = '0;
        end
        // Input is NaN or Infinity (all exponent bits high)
        else if (bf16_in.exp == 8'hFF) begin
            bf16_out.exp  = 8'hFF;
            bf16_out.mant = bf16_in.mant;
        end
        // 3. Overflow handling
        else if (calculated_exp >= 10'sd255) begin
            bf16_out.exp  = 8'hFF;  // Infinity encoding
            bf16_out.mant = 7'h0;
        end
        // 4. Underflow handling (Flush-to-Zero)
        else if (calculated_exp <= 10'sd0) begin
            bf16_out.exp  = 8'h0;
            bf16_out.mant = 7'h0;
        end
        // 5. Normal inside range bounds
        else begin
            bf16_out.exp  = calculated_exp[7:0];
            bf16_out.mant = bf16_in.mant;
        end
    end

endmodule
