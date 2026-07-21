`timescale 1ns / 1ps

module parameterized_adder_e4m3 
import fp4_pkg::*;
#(
    parameter int EXP_WIDTH  = 8,
    parameter int MANT_WIDTH = 7
)(
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] sum 
);

    // Local aliases mapping back to standard structural layouts
    bf16_t op_a, op_b, op_res;
    assign op_a = bf16_t'(a);
    assign op_b = bf16_t'(b);
    assign sum  = logic_vector_pack(op_res);

    // Helper function to cast structural representations cleanly back to logic arrays
    function automatic logic [15:0] logic_vector_pack(bf16_t in_struct);
        return {in_struct.sign, in_struct.exp, in_struct.mant};
    endfunction

    // Structural signals for the datapath
    logic signed [8:0] exp_diff;
    bf16_t max_op, min_op;
    
    // Internal extended mantissas:
    // [Hidden Bit: 1] + [MANT_WIDTH: 7] + [Guard: 1] + [Round: 1] + [Sticky: 1] = 11 bits
    localparam int EXT_MANT_WIDTH = 1 + MANT_WIDTH + 3; 
    
    logic [EXT_MANT_WIDTH-1:0] max_mant;
    logic [EXT_MANT_WIDTH-1:0] min_mant;
    logic [EXT_MANT_WIDTH-1:0] min_mant_shifted;

    logic eff_sub;
    logic eff_sub_sign;
    logic [EXT_MANT_WIDTH:0] sum_mant_ext;
    logic signed [EXT_MANT_WIDTH:0] abs_sum_mant_ext; 
    logic [7:0]  final_exp;
    logic [EXT_MANT_WIDTH:0] norm_mant;
    logic [3:0]  lzc; 

    logic max_op_subnorm, min_op_subnorm;

    always_comb begin 

        /* 
            Default initializations
        */
        logic [EXT_MANT_WIDTH-1:0] lost_bits = 'b0;
        logic [6:0] r_mant = 'b0;
        logic  g, r, s, round_up;
        g = 1'b0;
        r = 1'b0;
        s = 'b0;
        round_up = 'b0;
        op_res = 'b0;
        lzc = 'b0;
        eff_sub_sign = 'b0;

        // ---------------------------------------------------------------------
        // 1. OPERAND SORTING & EXPONENT ALIGNMENT
        // ---------------------------------------------------------------------
        exp_diff = $signed({1'b0, op_a.exp}) - $signed({1'b0, op_b.exp});

        if (exp_diff >= 0) begin
            max_op   = op_a;
            min_op   = op_b;
        end else begin
            max_op   = op_b;
            min_op   = op_a;
            exp_diff = -exp_diff;
        end

        max_op_subnorm = max_op.exp == '0;
        min_op_subnorm = min_op.exp == '0;

        /* subnormal exponent accounting */
        exp_diff += max_op_subnorm - min_op_subnorm;

        // Extract mantissas and append hidden bits (handle zero operands)
        max_mant = {~max_op_subnorm , max_op.mant, 3'b000};
        min_mant = {~min_op_subnorm, min_op.mant, 3'b000};

        // Align smaller operand's mantissa with dynamic sticky-bit retention
        if (exp_diff >= EXT_MANT_WIDTH) begin
            min_mant_shifted = '0;
            if (min_mant != '0) min_mant_shifted[0] = 1'b1; 
        end else begin
            lost_bits = min_mant << (EXT_MANT_WIDTH - exp_diff);
            min_mant_shifted = min_mant >> exp_diff;
            if (lost_bits != '0) min_mant_shifted[0] = 1'b1;
        end

        // ---------------------------------------------------------------------
        // 2. EFFECTIVE OPERATION & ADDITION/SUBTRACTION
        // ---------------------------------------------------------------------
        eff_sub = max_op.sign ^ min_op.sign;

        if (eff_sub) begin
            sum_mant_ext = {1'b0, max_mant} - {1'b0, min_mant_shifted};
        end else begin
            sum_mant_ext = {1'b0, max_mant} + {1'b0, min_mant_shifted};
        end

        // ---------------------------------------------------------------------
        // 3. NORMALIZATION
        // ---------------------------------------------------------------------
        final_exp = max_op.exp;
        abs_sum_mant_ext = sum_mant_ext[EXT_MANT_WIDTH] ? -sum_mant_ext : sum_mant_ext;
        norm_mant = abs_sum_mant_ext;
        eff_sub_sign = sum_mant_ext[EXT_MANT_WIDTH] & eff_sub == 1'b1;

        if (sum_mant_ext == '0) begin
            op_res = '0; 
        end 
        else if (eff_sub == 1'b0 && sum_mant_ext[EXT_MANT_WIDTH]) begin
            // Overflow during addition: shift right and preserve sticky bit
            norm_mant = sum_mant_ext >> 1;
            norm_mant[0] = norm_mant[0] | sum_mant_ext[0]; 
            final_exp = final_exp + 1'b1;  
        
        // subnormal add, when the hidden bit is 1, promote exponent
        end else if ((eff_sub == 1'b0) && sum_mant_ext[EXT_MANT_WIDTH-1]
                        && max_op_subnorm && min_op_subnorm) begin
            final_exp = final_exp + 1'b1; 

        /* Mantissa hidden bit is 0 */
        end else if ((eff_sub == 1'b1) && !abs_sum_mant_ext[EXT_MANT_WIDTH-1]
                        && (~max_op_subnorm || ~min_op_subnorm)) begin
            // Cancellation during subtraction: shift left by LZC
            if      (abs_sum_mant_ext[10]) lzc = 4'd0;
            else if (abs_sum_mant_ext[9])  lzc = 4'd1;
            else if (abs_sum_mant_ext[8])  lzc = 4'd2;
            else if (abs_sum_mant_ext[7])  lzc = 4'd3;
            else if (abs_sum_mant_ext[6])  lzc = 4'd4;
            else if (abs_sum_mant_ext[5])  lzc = 4'd5;
            else if (abs_sum_mant_ext[4])  lzc = 4'd6;
            else if (abs_sum_mant_ext[3])  lzc = 4'd7;
            else if (abs_sum_mant_ext[2])  lzc = 4'd8;
            else if (abs_sum_mant_ext[1])  lzc = 4'd9;
            else                       lzc = 4'd10;

            if (final_exp > {4'b0, lzc}) begin
                norm_mant = abs_sum_mant_ext << lzc;
                final_exp = final_exp - {4'b0, lzc};
            end else begin
                norm_mant = abs_sum_mant_ext << (final_exp - 1);
                final_exp = 8'h0;
            end
        end 

        // ---------------------------------------------------------------------
        // 4. ROUNDING (Round to Nearest, Ties to Even) & PACKING
        // ---------------------------------------------------------------------
        if (sum_mant_ext != '0) begin
 
            // --- FIXED BIT INDEXING HERE ---
            r_mant = norm_mant[9:3]; // Extract ONLY the 7 fractional mantissa bits
            g      = norm_mant[2];   // Guard bit
            r      = norm_mant[1];   // Round bit
            s      = norm_mant[0];   // Sticky bit

            round_up = g && (r || s || r_mant[0]);

            // toggle if eff_sub is 1, and there's a sign flip.
            op_res.sign = max_op.sign ^ eff_sub_sign;

            if (round_up) begin
                if (r_mant == 7'h7F) begin
                    // Mantissa overflow on rounding up
                    op_res.mant = '0;
                    op_res.exp  = final_exp + 1'b1;
                end else begin
                    op_res.mant = r_mant + 1'b1;
                    op_res.exp  = final_exp;
                end
            end else begin
                op_res.mant = r_mant;
                op_res.exp  = final_exp;
            end

            // Infinity/Overflow Saturation Guard
            if (final_exp >= 8'hFF) begin
                op_res.exp  = 8'hFF;
                op_res.mant = '0;
            end
        end
    end


// --- [stev] ---
// ---------------------------------------------------------------------
    // DEBUG LOGGING BLOCK
    // ---------------------------------------------------------------------
    // This block triggers whenever the output changes, printing the full 
    // internal state of the unpack, shift, math, and rounding stages.
    // always @(sum) begin
    //     // Only print valid operations (skipping initial/X states in simulation)
    //     if (^a !== 1'bx && ^b !== 1'bx) begin
    //         $display("----------------------------------------------------------------");
    //         $display("[DEBUG ADDER] Inputs: A = 16'h%4h | B = 16'h%4h", a, b);
    //         $display("  Unpacked A: Sign=%b, Exp=8'h%2h, Mant=7'h%2h", op_a.sign, op_a.exp, op_a.mant);
    //         $display("  Unpacked B: Sign=%b, Exp=8'h%2h, Mant=7'h%2h", op_b.sign, op_b.exp, op_b.mant);
    //         $display("----------------------------------------------------------------");
    //         $display("  Alignment Stage:");
    //         $display("    Exp Diff      = %d", exp_diff);
    //         $display("    Max Mant (Int)= 11'b%11b", max_mant);
    //         $display("    Min Mant (Int)= 11'b%11b", min_mant);
    //         $display("    Shifted Min   = 11'b%11b", min_mant_shifted);
    //         $display("----------------------------------------------------------------");
    //         $display("  Arithmetic Stage:");
    //         $display("    Effective Sub = %b", eff_sub);
    //         $display("    Effective Sub Sign = %b", eff_sub_sign);
    //         $display("    Sum Mant Ext  = 12'b%12b", sum_mant_ext);
    //         $display("    Abs Sub Sign = 12'b%12b", abs_sum_mant_ext);
    //         $display("----------------------------------------------------------------");
    //         $display("  Normalization Stage:");
    //         $display("    Final Exp Pre = 8'h%2h", final_exp);
    //         $display("    Norm Mant     = 12'b%12b", norm_mant);
    //         $display("----------------------------------------------------------------");
    //         $display("  Rounding & Packing Stage (Targeting Bits 9:3):");
    //         $display("    Extracted Mantissa Bits [9:3] = 7'b%7b (Hex: 7'h%2h)", norm_mant[9:3], norm_mant[9:3]);
    //         $display("    Guard (Bit 2)                 = %b", norm_mant[2]);
    //         $display("    Round (Bit 1)                 = %b", norm_mant[1]);
    //         $display("    Sticky (Bit 0)                = %b", norm_mant[0]);
    //         $display("    Packed Result                 = 16'h%4h", sum);
    //         $display("----------------------------------------------------------------\n");
    //     end
    // end
// --- [end] ---

endmodule
