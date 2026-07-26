`timescale 1ns / 1ps


/*
    Note that NVFP8 (E4M3) *does not* encode
    infinities.
*/
// module normalize_e4m3
// import fp4_pkg::*;
// #( parameter int OUTPUT_EXP_BITS = 8)
//  (
//     input fp4_scaler_e4m3_t e4m3,
//     output logic [2:0] normalized_mant,
    
//     /* Encodes *the actual* value (unbiased)*/
//     output logic signed [OUTPUT_EXP_BITS-1:0] normalized_exp, 
//     output logic is_zero,
//     output logic is_subnormal,
//     output logic is_nan
// );

//     localparam logic [OUTPUT_EXP_BITS-1:0] E4M3_BIAS = 'd7;
//     localparam logic [OUTPUT_EXP_BITS-1:0] OUT_BIAS = (1 << (OUTPUT_EXP_BITS - 1)) - 1;
//     logic [3:0] sub_shift_l;
//     // logic is_subnormal;

//     assign is_zero = e4m3[6:0] == 'b0;
//     assign is_subnormal = (e4m3.exp == 'b0) && (!is_zero);
//     assign is_nan = &e4m3[6:0];
//     always_comb begin
//         sub_shift_l = 4'd0;

//         if (is_subnormal) begin
//             /* Shift back mantissa to correct place */
//             unique casez (e4m3.mant)
//                 3'b1??: sub_shift_l = 4'd1; 
//                 3'b01?: sub_shift_l = 4'd2;
//                 3'b001: sub_shift_l = 4'd3;
//                 default: begin // 3'b000
//                     assert (is_zero)
//                     else $error("Mantissa is zero but reached this statement"); 
//                     sub_shift_l = 4'd0;
//                 end
//             endcase
 
//             normalized_mant = e4m3.mant << sub_shift_l;
//             normalized_exp = 1'b1 - E4M3_BIAS - sub_shift_l;
//         end else begin 
//             normalized_mant = e4m3.mant;
//             normalized_exp = e4m3.exp - E4M3_BIAS; 
//         end
//     end

//     always_comb begin
//         if (is_subnormal) begin
            
//         end
//     end

// endmodule


module e4m3_mul
import fp4_pkg::*; (
    input fp4_scaler_e4m3_t A8,
    input fp4_scaler_e4m3_t B8,
    input logic signed [15:0] q14_2_C_in,
    output bf16_t PABC, // E8M7 (intermediate)
    output logic isNaN,
    output logic isZero
);

    
    localparam logic [14:0] BF16_NAN = 15'h7fc0;
    localparam int E4M3_M = 3;
    localparam int E4M3_E = 4;
    localparam int Q14_2_FRAC = 0;
    //localparam int Q14_2_FRAC = 2;
    localparam int Q14_2_WIDTH = 16;

    localparam int BF16_E = 8;
    localparam int BF16_M = 7;

    localparam logic [BF16_E-1:0] E4M3_BIAS = (1 << (E4M3_E-1))-1;


    /* 2*(1+M3) + 16 int bits.
        This is because the encoding for |INT16.MIN_VAL|
            does have a representation in 16 bits
     */
    localparam logic [7:0] PABC_FIXED_PRODUCT_WIDTH = 
                                2 * (1 + E4M3_M) + Q14_2_WIDTH;                       
    
    /* Decimal places for the product A * B * C
        2 * 3 (3 from E4M3), then 2 from Q14.2
     */
    localparam logic [7:0] PABC_DECIMAL_PLACES = 2*E4M3_M + Q14_2_FRAC;;

    /* MSB and LSB locations for the {1,A} * {1,B} * C mantissa product */
    localparam SHIFTED_M_MSB = PABC_FIXED_PRODUCT_WIDTH-2;
    localparam SHIFTED_M_LSB = PABC_FIXED_PRODUCT_WIDTH-BF16_M-1;

    function automatic e4m3_is_zero(fp4_scaler_e4m3_t x);
        e4m3_is_zero = x[6:0] == 'b0;
    endfunction

    function automatic e4m3_is_nan(fp4_scaler_e4m3_t x);
        e4m3_is_nan = &x[6:0];
    endfunction

    function automatic e4m3_is_subnormal(fp4_scaler_e4m3_t x); 
        e4m3_is_subnormal = (x.exp == 'b0) && (!e4m3_is_zero(x));
    endfunction


    // Declare the variables
    logic A_is_zero;
    logic A_is_nan;
    logic A_is_subnormal;
    logic [3:0] A_mant_ext;
    // logic [2:0] A_mant_norm;
    logic signed [7:0] A_exp_norm;

    assign A_is_zero = e4m3_is_zero(A8);
    assign A_is_nan = e4m3_is_nan(A8);
    assign A_is_subnormal = e4m3_is_subnormal(A8);
    assign A_mant_ext = {~A_is_subnormal, A8.mant}; 
    assign A_exp_norm = A8.exp - E4M3_BIAS + A_is_subnormal;

    logic B_is_zero;
    logic B_is_nan;
    logic B_is_subnormal;
    logic [3:0] B_mant_ext;
    // logic [2:0] B_mant_norm;
    logic signed [7:0] B_exp_norm;

    assign B_is_zero = e4m3_is_zero(B8);
    assign B_is_nan = e4m3_is_nan(B8);
    assign B_is_subnormal = e4m3_is_subnormal(B8);
    assign B_mant_ext = {~B_is_subnormal, B8.mant};
    assign B_exp_norm = B8.exp - E4M3_BIAS + B_is_subnormal;

    logic C_is_zero;
    logic C_sign;

    /* 1 extra bit to account for the sign */
    logic signed [Q14_2_WIDTH:0] C_absval;
    logic S_PAB;

    


    // normalize_e4m3
    // #(.OUTPUT_EXP_BITS(8)) normA8 (
    //     .e4m3(A8), 
    //     .normalized_mant(A_mant_norm), 
    //     .normalized_exp(A_exp_norm), 
    //     .is_zero(A_is_zero), 
    //     .is_nan(A_is_nan), 
    //     .is_subnormal(A_is_subnormal)
    // );

    // normalize_e4m3
    // #(.OUTPUT_EXP_BITS(8)) normB8 (
    //     .e4m3(B8), 
    //     .normalized_mant(B_mant_norm), 
    //     .normalized_exp(B_exp_norm), 
    //     .is_zero(B_is_zero), 
    //     .is_nan(B_is_nan), 
    //     .is_subnormal(B_is_subnormal)
    // );

    // temp variables for multiplication
    logic [PABC_FIXED_PRODUCT_WIDTH-1:0] TEMP_M_PABC;
    logic [14:0] TEMP_PABC;

    /* Point Adjustment Signals */
    logic [7:0] lzc; // leading zeros count
    logic [PABC_FIXED_PRODUCT_WIDTH-1:0] shifted_m_pabc;
    logic [6:0] pre_round_mant;
    logic signed [7:0] exp_shift_amount;

    /* ROUNDING SIGNALS */
    /* Guard, Round and Sticky bits */
    logic g, r, s;
    logic round_up;

    /* Final Values */
    logic [6:0] actual_m_pabc;
    logic [BF16_E-1:0] actual_exp_pabc;        

    assign C_is_zero = q14_2_C_in == 'b0;
    assign C_sign = q14_2_C_in[Q14_2_WIDTH-1];
    assign C_absval = C_sign ? -q14_2_C_in : q14_2_C_in;

    /* Integer multiplication */
    assign TEMP_M_PABC = C_absval[Q14_2_WIDTH-1:0] * A_mant_ext * B_mant_ext; 

    assign S_PAB = A8.sign ^ B8.sign ^ C_sign;

    /* 
        Detect leading 0 from the product, 
        then round.
     */
    always_comb begin
        casez (TEMP_M_PABC)
            24'b1???????????????????????: lzc = 5'd0;
            24'b01??????????????????????: lzc = 5'd1;
            24'b001?????????????????????: lzc = 5'd2;
            24'b0001????????????????????: lzc = 5'd3;
            24'b00001???????????????????: lzc = 5'd4;
            24'b000001??????????????????: lzc = 5'd5;
            24'b0000001?????????????????: lzc = 5'd6;
            24'b00000001????????????????: lzc = 5'd7;
            24'b000000001???????????????: lzc = 5'd8;
            24'b0000000001??????????????: lzc = 5'd9;
            24'b00000000001?????????????: lzc = 5'd10;
            24'b000000000001????????????: lzc = 5'd11;
            24'b0000000000001???????????: lzc = 5'd12;
            24'b00000000000001??????????: lzc = 5'd13;
            24'b000000000000001?????????: lzc = 5'd14;
            24'b0000000000000001????????: lzc = 5'd15;
            24'b00000000000000001???????: lzc = 5'd16;
            24'b000000000000000001??????: lzc = 5'd17;
            24'b0000000000000000001?????: lzc = 5'd18;
            24'b00000000000000000001????: lzc = 5'd19;
            24'b000000000000000000001???: lzc = 5'd20;
            24'b0000000000000000000001??: lzc = 5'd21;
            24'b00000000000000000000001?: lzc = 5'd22;
            24'b000000000000000000000001: lzc = 5'd23;
            24'b000000000000000000000000: lzc = 5'd24;
            default:                     lzc = 5'd0; 
        endcase

        shifted_m_pabc = TEMP_M_PABC << lzc;

        /* Compute the delta between the location of the point 
            before shift adjustment and the desired position 
            relative from MSB.
         */
        exp_shift_amount = PABC_FIXED_PRODUCT_WIDTH 
                            - (PABC_DECIMAL_PLACES) // original point (3 + 3 + 2)
                            - (lzc + 1);

        /* 
            ============ ROUNDING LOGIC =============
        */

        pre_round_mant = shifted_m_pabc[SHIFTED_M_MSB : SHIFTED_M_LSB];
        g = shifted_m_pabc[SHIFTED_M_LSB-1];
        r = shifted_m_pabc[SHIFTED_M_LSB-2];
        s = |shifted_m_pabc[SHIFTED_M_LSB-3:0];

        round_up = g & (r | s | pre_round_mant[0]);

        if (round_up) begin
            if (pre_round_mant == 7'h7F) begin
                // mantissa becomes 1.000000
                // exponent +1
                actual_m_pabc = 7'h0;
                actual_exp_pabc = A_exp_norm + B_exp_norm + 8'h7F + 1'd1 
                                    + exp_shift_amount;
            end else begin
                actual_m_pabc = pre_round_mant + 7'd1;
                actual_exp_pabc = A_exp_norm + B_exp_norm 
                                    + exp_shift_amount + 8'h7F;
            end
        end else begin 
            actual_m_pabc = pre_round_mant;
            actual_exp_pabc = A_exp_norm + B_exp_norm 
                                + exp_shift_amount + 8'h7F;
        end
    end

    // finalize the output
    assign TEMP_PABC = {actual_exp_pabc, actual_m_pabc};    

    assign isNaN = A_is_nan | B_is_nan;
    assign isZero = (A_is_zero | B_is_zero | C_is_zero) & !isNaN;

    assign PABC =  {S_PAB, 
        isNaN ? BF16_NAN : (isZero ? 15'b0 : TEMP_PABC)
    };
    
`ifndef SYNTHESIS
always_comb begin
    $display("================ E4M3 MUL DEBUG ================");
    $display("A8        = %02h", A8);
    $display("B8        = %02h", B8);
    $display("C_in      = %d (0x%04h)",
             $signed(q14_2_C_in),
             q14_2_C_in);

    $display("-----------------------------------------------");
    $display("A sign    = %0d", A8.sign);
    $display("A exp     = %0d (raw %b)",
             A8.exp,
             A8.exp);
    $display("A mant    = %03b", A8.mant);
    $display("A subnorm = %0d", A_is_subnormal);
    $display("A exp norm= %0d",
             $signed(A_exp_norm));
    $display("A mant ext= %04b",
             A_mant_ext);


    $display("-----------------------------------------------");
    $display("B sign    = %0d", B8.sign);
    $display("B exp     = %0d (raw %b)",
             B8.exp,
             B8.exp);
    $display("B mant    = %03b", B8.mant);
    $display("B subnorm = %0d", B_is_subnormal);
    $display("B exp norm= %0d",
             $signed(B_exp_norm));
    $display("B mant ext= %04b",
             B_mant_ext);


    $display("-----------------------------------------------");
    $display("C abs     = %0d",
             $signed(C_absval));
    $display("C sign    = %0d",
             C_sign);

    $display("-----------------------------------------------");
    $display("TEMP_M_PABC       = %h",
             TEMP_M_PABC);

    $display("TEMP_M_PABC bin   = %b",
             TEMP_M_PABC);

    $display("lzc               = %0d",
             lzc);

    $display("shifted_m_pabc    = %h",
             shifted_m_pabc);

    $display("PABC_FIXED_WIDTH  = %0d",
             PABC_FIXED_PRODUCT_WIDTH);

    $display("PABC_DECIMAL      = %0d",
             PABC_DECIMAL_PLACES);

    $display("exp_shift_amount  = %0d",
             $signed(exp_shift_amount));


    $display("-----------------------------------------------");
    $display("pre_round_mant    = %b",
             pre_round_mant);

    $display("g=%0d r=%0d s=%0d",
             g,r,s);

    $display("round_up          = %0d",
             round_up);


    $display("-----------------------------------------------");
    $display("actual mant       = %b",
             actual_m_pabc);

    $display("actual exp        = %0d (0x%02h)",
             actual_exp_pabc,
             actual_exp_pabc);

    $display("TEMP_PABC         = %h",
             TEMP_PABC);

    $display("PABC output       = %h",
             PABC);

    $display("================================================");

end

`endif

endmodule
