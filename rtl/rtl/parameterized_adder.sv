`timescale 1ns / 1ps

module parameterized_adder
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
    bf16_t op_a, op_b, op_b_norm, op_res;
    assign op_a = bf16_t'(a);
    assign op_b = bf16_t'(b);
    assign sum  = logic_vector_pack(op_res);

    // Helper function to cast structural representations cleanly back to logic arrays
    function automatic logic [15:0] logic_vector_pack(bf16_t in_struct);
        return {in_struct.sign, in_struct.exp, in_struct.mant};
    endfunction
    logic signed [8:0] exp_diff;

    // Structural signals for the datapath
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
    logic [8:0]  final_exp;
    logic [EXT_MANT_WIDTH:0] norm_mant;
    logic [3:0]  sum_lzc; 

    logic max_op_norm, min_op_norm;

    logic final_exp_overflow;
    assign final_exp_overflow = final_exp[8] | &final_exp[7:0];


    /* Leading Zero Counter */
    always_comb begin
        casez(abs_sum_mant_ext)
            11'b1??????????: sum_lzc = 4'd0;
            11'b01?????????: sum_lzc = 4'd1;
            11'b001????????: sum_lzc = 4'd2;
            11'b0001???????: sum_lzc = 4'd3;
            11'b00001??????: sum_lzc = 4'd4;
            11'b000001?????: sum_lzc = 4'd5;
            11'b0000001????: sum_lzc = 4'd6;
            11'b00000001???: sum_lzc = 4'd7;
            11'b000000001??: sum_lzc = 4'd8;
            11'b0000000001?: sum_lzc = 4'd9;
            default:        sum_lzc = 4'd10; // Covers the case of all zeros
        endcase
    end


    always_comb begin 

        /* 
            Default initializations
        */
        logic [EXT_MANT_WIDTH-1:0] lost_bits = 'b0;
        logic [7:0] r_mant = 'b0;
        logic  g, r, s, round_up;
        logic signed [EXP_WIDTH:0] op_a_exp_signed;
        logic signed [EXP_WIDTH:0] op_b_exp_signed;


        g = 1'b0;
        r = 1'b0;
        s = 'b0;
        round_up = 'b0;
        op_res = 'b0;
        eff_sub_sign = 'b0;

        op_a_exp_signed = {1'b0, op_a.exp};
        op_b_exp_signed = {1'b0, op_b.exp};

        // ---------------------------------------------------------------------
        // 1. OPERAND SORTING & EXPONENT ALIGNMENT
        // ---------------------------------------------------------------------
        exp_diff = op_a_exp_signed - op_b_exp_signed;

        {max_op, min_op, exp_diff} = exp_diff >= 0 ? {op_a, op_b, exp_diff}
                                                     : {op_b, op_a, -exp_diff};

        // Optimization for flush to 0: 
        // Check if max_op.exp != 0,
        // -> to flush to 0, need to AND the max_op.exp 
        // and replicated signal of max_op_exp != 0.
        // Therefore when not nromal, the mantissa goes to 0.

        max_op_norm = (|max_op.exp);
        min_op_norm = (|min_op.exp);

        // Extract mantissas and append hidden bits (handle zero operands)
        max_mant = {max_op_norm , max_op.mant & {7{max_op_norm}}, 3'b000};
        min_mant = {min_op_norm, min_op.mant & {7{min_op_norm}}, 3'b000};

        // Align smaller operand's mantissa with dynamic sticky-bit retention
        if (exp_diff >= EXT_MANT_WIDTH) begin
            lost_bits = min_mant;
            min_mant_shifted = '0;
        end else begin
            lost_bits = min_mant << (EXT_MANT_WIDTH - exp_diff);
            min_mant_shifted = min_mant >> exp_diff;
        end
        min_mant_shifted[0] = |lost_bits; // Sticky bit retention

        // ---------------------------------------------------------------------
        // 2. EFFECTIVE OPERATION & ADDITION/SUBTRACTION
        // ---------------------------------------------------------------------
        eff_sub = max_op.sign ^ min_op.sign;

        sum_mant_ext = eff_sub ? 
            {1'b0, max_mant} - {1'b0, min_mant_shifted} 
            : {1'b0, max_mant} + {1'b0, min_mant_shifted};

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
    
        /* Cancellation logic */
        end else if ((eff_sub == 1'b1) && !abs_sum_mant_ext[EXT_MANT_WIDTH-1]) begin
            if (final_exp > {4'b0, sum_lzc}) begin
                norm_mant = abs_sum_mant_ext << sum_lzc;
                final_exp = final_exp - {4'b0, sum_lzc};

            /* Result is subnormal */
            end else begin
                norm_mant = 7'h0;
                final_exp = 8'h0;
        // ---------------------------------------------------------------------
            end
        end 

        // 4. ROUNDING (Round to Nearest, Ties to Even) & PACKING
        // ---------------------------------------------------------------------
        if (sum_mant_ext != '0) begin
 
            // --- FIXED BIT INDEXING HERE ---
            g      = norm_mant[2];   // Guard bit
            r      = norm_mant[1];   // Round bit
            s      = norm_mant[0];   // Sticky bit

            round_up = g && (r || s || norm_mant[3]);
            r_mant = {1'b0, norm_mant[9:3]} + {7'b0, round_up}; 

            // toggle if eff_sub is 1, and there's a sign flip.
            op_res.sign = max_op.sign ^ eff_sub_sign;
            op_res.mant = r_mant[6:0];
            op_res.exp = final_exp + r_mant[7];

            // Infinity/Overflow Saturation Guard
            if (final_exp_overflow) begin
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
