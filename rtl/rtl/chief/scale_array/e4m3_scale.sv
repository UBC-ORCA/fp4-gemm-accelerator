`timescale 1ns / 1ps

/*
    E4M3-scaled int16 -> BF16 for N_SCALE_UNITS lanes sharing one weight scale.

        scaled_o[i] = tile_i[i] * 2^TILE_EXP_BIAS * as_i[i] * ws_i

    Multiply only (no accumulation). The tile is multiplied in its integer
    encoding, so the only rounding is the final one to BF16:

      S2  - E4M3 decode: M = {~subnormal, mant}, exp = exp + subnormal
          - fixup_i = h_a,i ? (Mb << 3) : 0  (registered, feeds the DSP D port)
          - broadcast multiply  ma_i * Mb     (one shared DSP, 4 lanes)
          - |tile_i|, exponent partial sum, sign, NaN/zero flags
      S3  - prod = (ma_i*Mb + fixup_i) * |tile_i|        (lane DSP: pre-adder
                 = Ma_i * Mb * |tile_i|                   + 8x16 multiply)
          - LZC normalize, RNE round to BF16, special-value override

    Latency LATENCY = 2 from inputs to scaled_o. Inputs should come straight
    from registers (the caller's capture stage).

    Sidebands are delayed by LATENCY so they come out aligned with scaled_o:
      ctrl_i/ctrl_o : synchronous reset to 0   (valid, end token, ...)
      data_i/data_o : not reset                (BRAM accumulator, addresses)

    Scales are assumed positive (UE4M3), but signs are XORed anyway.
    No BF16 overflow/underflow handling: with E4M3 scales and an int16 tile
    the result exponent cannot leave the BF16 normal range.
*/
module e4m3_scale
import chief_pkg::*;
#(
    parameter  int N_SCALE_UNITS  = 4,
    parameter  int TILE_EXP_BIAS  = -2,
    parameter  int TILE_INT_WIDTH = 16,
    parameter  int CTRL_W         = 1,
    parameter  int DATA_W         = 1,
    localparam int LATENCY        = 2
)(
    input  logic                             clk_i,
    input  logic                             rst_ni,

    input  chief_scaler_e4m3_t               ws_i,
    input  chief_scaler_e4m3_t               as_i   [0:N_SCALE_UNITS-1],
    input  logic signed [TILE_INT_WIDTH-1:0] tile_i [0:N_SCALE_UNITS-1],
    input  logic [CTRL_W-1:0]                ctrl_i,
    input  logic [DATA_W-1:0]                data_i,

    output bf16_t                            scaled_o [0:N_SCALE_UNITS-1],
    output logic [CTRL_W-1:0]                ctrl_o,
    output logic [DATA_W-1:0]                data_o
);

    function automatic int IEEE_EXP_BIAS(int EXP_WIDTH);
        return (1 << (EXP_WIDTH-1)) - 1;
    endfunction

    function automatic logic e4m3_is_zero(chief_scaler_e4m3_t x);
        return x[6:0] == '0;
    endfunction

    function automatic logic e4m3_is_nan(chief_scaler_e4m3_t x);
        return &x[6:0];
    endfunction

    function automatic logic e4m3_is_subnormal(chief_scaler_e4m3_t x);
        return (x.exp == '0) && !e4m3_is_zero(x);
    endfunction

    localparam logic [14:0] BF16_NAN = 15'h7fc0;
    localparam int BF16_E    = 8;
    localparam int BF16_M    = 7;
    localparam int E4M3_BIAS = IEEE_EXP_BIAS(E4M3_E);

    localparam int LANE_PROD_WIDTH = 2*E4M3_M + 1;          // ma * Mb   (7)
    localparam int MAB_WIDTH       = 2*(E4M3_M + 1);        // Ma * Mb   (8)

    /* |Mab * tile| < 2^MAB_WIDTH * 2^(TILE_INT_WIDTH-1): 23 bits.
       The DSP sees a 9-bit {0, Mab} times a 16-bit signed tile. */
    localparam int PROD_W      = MAB_WIDTH + TILE_INT_WIDTH - 1;
    localparam int CLOG_PROD_W = $clog2(PROD_W + 1);

    /* prod is an integer with 2*E4M3_M fraction bits from Mab and
       -TILE_EXP_BIAS from the tile. After the LZC shift the leading 1 sits
       at bit PROD_W-1-lzc, so

         exp_bf16 = (a_exp + w_exp) - lzc + round_carry + EXP_CONST            */
    localparam int EXP_CONST = (PROD_W - 1) - 2*E4M3_M + TILE_EXP_BIAS
                               - 2*E4M3_BIAS + IEEE_EXP_BIAS(BF16_E);

    // Holds a_exp + w_exp + EXP_CONST before the LZC correction
    localparam int EXP_PART_W = BF16_E + 2;

    //------------------------------------------------------------
    // Stage 2: decode, broadcast multiply
    //  1. Decode FP status signals
    //  2. Perform broadcast multiply, product split into 2 terms:
    //      As[3:0] * Ws = As[2:0] * Ws + As[3] * (Ws << 3) 
    //  3. Exponent calculation 
    //
    //------------------------------------------------------------

    // Weight scale (shared by every lane)
    logic                ws_is_zero, ws_is_nan, ws_is_sub;
    logic [E4M3_M:0]     ws_mant_ext;
    logic [E4M3_E:0]     ws_exp;

    assign ws_is_zero  = e4m3_is_zero(ws_i);
    assign ws_is_nan   = e4m3_is_nan(ws_i);
    assign ws_is_sub   = e4m3_is_subnormal(ws_i);
    assign ws_mant_ext = {~ws_is_sub, ws_i.mant};
    assign ws_exp      = (E4M3_E+1)'(ws_i.exp) + (E4M3_E+1)'(ws_is_sub);

    // Activation scales (per lane)
    logic [E4M3_M:0]       as_mant_ext [0:N_SCALE_UNITS-1];

    logic [LANE_PROD_WIDTH-1:0] bcast_prod   [0:N_SCALE_UNITS-1];
    logic                       bcast_hidden [0:N_SCALE_UNITS-1];
    logic [MAB_WIDTH-1:0]       fixup_d      [0:N_SCALE_UNITS-1];

    logic [TILE_INT_WIDTH-1:0] tile_abs_d [0:N_SCALE_UNITS-1];

    logic [EXP_PART_W-1:0] exp_part_d [0:N_SCALE_UNITS-1];
    logic                  sign_d     [0:N_SCALE_UNITS-1];
    logic                  nan_d      [0:N_SCALE_UNITS-1];
    logic                  zero_d     [0:N_SCALE_UNITS-1];

    for (genvar i = 0; i < N_SCALE_UNITS; i++) begin : gen_s2_lane
        logic            as_is_zero, as_is_nan, as_is_sub;
        logic [E4M3_E:0] as_exp;
        logic            tile_neg;

        assign as_is_zero     = e4m3_is_zero(as_i[i]);
        assign as_is_nan      = e4m3_is_nan(as_i[i]);
        assign as_is_sub      = e4m3_is_subnormal(as_i[i]);
        assign as_mant_ext[i] = {~as_is_sub, as_i[i].mant};
        assign as_exp         = (E4M3_E+1)'(as_i[i].exp) + (E4M3_E+1)'(as_is_sub);

        /* Ma*Mb = ma*Mb + h_a * (Mb << E4M3_M), where ma is Ma[2:0]. 
            The gate is resolved here so the lane DSP's 
            D input is a plain register (no mux before it). */
        assign fixup_d[i] = bcast_hidden[i]
                            ? MAB_WIDTH'({ws_mant_ext, {E4M3_M{1'b0}}})
                            : '0;

        always_comb begin
            int e;
            e = int'(as_exp) + int'(ws_exp) + EXP_CONST;
            exp_part_d[i] = e[EXP_PART_W-1:0];
        end

        // |tile| fits TILE_INT_WIDTH unsigned bits, including |-2^15|
        assign tile_neg      = tile_i[i][TILE_INT_WIDTH-1];
        assign tile_abs_d[i] = tile_neg ? TILE_INT_WIDTH'(-tile_i[i])
                                        : TILE_INT_WIDTH'(tile_i[i]);

        assign sign_d[i] = tile_neg;
        assign nan_d[i]  = as_is_nan | ws_is_nan;
        assign zero_d[i] = (as_is_zero | ws_is_zero | (tile_i[i] == '0)) & ~nan_d[i];
    end : gen_s2_lane

    // Shared DSP: packed {ma_3..ma_0} * Mb, one 7-bit slot per lane
    e4m3_broadcast_mul #(
        .N_SCALE_UNITS(N_SCALE_UNITS)
    ) u_bcast (
        .row_scale_mant_ext_i     (as_mant_ext),
        .col_scale_mant_ext_i     (ws_mant_ext),
        .row_scale_mant_hidden_d_o(bcast_hidden),
        .prod_d_o                 (bcast_prod)
    );

    // Stage 2 -> 3 registers
    logic [LANE_PROD_WIDTH-1:0]       bcast_prod_q   [0:N_SCALE_UNITS-1];
    logic [MAB_WIDTH-1:0]             fixup_q        [0:N_SCALE_UNITS-1];
    logic [TILE_INT_WIDTH-1:0]        tile_abs_q     [0:N_SCALE_UNITS-1];
    logic [EXP_PART_W-1:0]            exp_part_q     [0:N_SCALE_UNITS-1];
    logic                             sign_q         [0:N_SCALE_UNITS-1];
    logic                             nan_q          [0:N_SCALE_UNITS-1];
    logic                             zero_q         [0:N_SCALE_UNITS-1];

    always_ff @(posedge clk_i) begin : s2_regs
        for (int i = 0; i < N_SCALE_UNITS; i++) begin
            bcast_prod_q[i]   <= bcast_prod[i];
            fixup_q[i]        <= fixup_d[i];
            tile_abs_q[i]     <= tile_abs_d[i];
            exp_part_q[i]     <= exp_part_d[i];
            sign_q[i]         <= sign_d[i];
            nan_q[i]          <= nan_d[i];
            zero_q[i]         <= zero_d[i];
        end
    end : s2_regs

    //------------------------------------------------------------
    // Stage 3: lane multiply, normalize, round
    //------------------------------------------------------------
    bf16_t scaled_d [0:N_SCALE_UNITS-1];
    bf16_t scaled_q [0:N_SCALE_UNITS-1];

    for (genvar i = 0; i < N_SCALE_UNITS; i++) begin : gen_s3_lane
        (* use_dsp = "yes" *)
        logic [PROD_W-1:0]           prod_abs;
        logic [PROD_W-1:0]           prod_shifted;
        logic [CLOG_PROD_W-1:0]      lzc;
        logic [BF16_M-1:0]           rounded_mant;
        logic                        round_carry;
        logic [BF16_E-1:0]           total_exp;

        /* One expression so it maps to the DSP as (A + D) * B:
           A = bcast slot, D = fixup, B = |tile|. */
        assign prod_abs = PROD_W'(
            (MAB_WIDTH'(bcast_prod_q[i]) + fixup_q[i]) * tile_abs_q[i]);

        lzc_enc #(
            .N(PROD_W)
        ) u_lzc (
            .req(prod_abs),
            .idx(lzc)
        );

        assign prod_shifted = prod_abs << lzc;

        // Drop the hidden 1 at the MSB, round the fraction bits only
        round_rne #(
            .INPUT_WIDTH (PROD_W-1),
            .OUTPUT_WIDTH(BF16_M)
        ) u_round (
            .round_input_in(prod_shifted[PROD_W-2:0]),
            .round_output_o(rounded_mant),
            .round_carry_o (round_carry)
        );

        assign total_exp = BF16_E'(exp_part_q[i] - EXP_PART_W'(lzc)
                                   + round_carry);

        always_comb begin
            scaled_d[i].sign = sign_q[i];
            if (nan_q[i])       {scaled_d[i].exp, scaled_d[i].mant} = BF16_NAN;
            else if (zero_q[i]) {scaled_d[i].exp, scaled_d[i].mant} = '0;
            else                {scaled_d[i].exp, scaled_d[i].mant} = {total_exp, rounded_mant};
        end
    end : gen_s3_lane

    always_ff @(posedge clk_i) begin : s3_regs
        for (int i = 0; i < N_SCALE_UNITS; i++) scaled_q[i] <= scaled_d[i];
    end : s3_regs

    assign scaled_o = scaled_q;

    //------------------------------------------------------------
    // Sideband delay lines (LATENCY deep)
    //------------------------------------------------------------
    logic [CTRL_W-1:0] ctrl_pipe_q [0:LATENCY-1];
    logic [DATA_W-1:0] data_pipe_q [0:LATENCY-1];

    always_ff @(posedge clk_i) begin : ctrl_pipe
        if (!rst_ni) begin
            for (int s = 0; s < LATENCY; s++) ctrl_pipe_q[s] <= '0;
        end else begin
            ctrl_pipe_q[0] <= ctrl_i;
            for (int s = 1; s < LATENCY; s++) ctrl_pipe_q[s] <= ctrl_pipe_q[s-1];
        end
    end : ctrl_pipe

    always_ff @(posedge clk_i) begin : data_pipe
        data_pipe_q[0] <= data_i;
        for (int s = 1; s < LATENCY; s++) data_pipe_q[s] <= data_pipe_q[s-1];
    end : data_pipe

    assign ctrl_o = ctrl_pipe_q[LATENCY-1];
    assign data_o = data_pipe_q[LATENCY-1];

endmodule
