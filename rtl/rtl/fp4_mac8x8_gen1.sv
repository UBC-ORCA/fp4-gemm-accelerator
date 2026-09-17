`timescale 1ns/1ps
/******************************************************************************
 * fp4_mac8x8_gen1.sv
 *
 * GEN1 FP4 outer-product MAC tile for the UBC ORCA fp4-gemm-accelerator.
 *
 * ---------------------------------------------------------------------------
 * HIGH-LEVEL BEHAVIOR
 * ---------------------------------------------------------------------------
 * This module implements a single 8x8 accumulator tile T.
 *
 * One "MAC operation" takes:
 *   - a 32-bit packed operand A containing 8 FP4 values
 *   - a 32-bit packed operand B containing 8 FP4 values
 *
 * Each FP4 is stored in one nibble:
 *   [sign | mag2 | mag1 | mag0]
 *
 * The module:
 *   1) unpacks the 8 FP4 values from A and the 8 FP4 values from B
 *   2) converts each FP4 into a signed small integer "quanta" value
 *   3) computes the OUTER PRODUCT of the two 8-element vectors
 *   4) accumulates the 64 products into an 8x8 signed INT16 tile T
 *
 * In math form:
 *
 *   for i = 0..7:
 *     for j = 0..7:
 *       T[i][j] <- T[i][j] + A_q[i] * B_q[j]
 *
 * where A_q[i] and B_q[j] are the signed integer "quanta" decoded from FP4.
 *
 * ---------------------------------------------------------------------------
 * WHY INTEGER QUANTA?
 * ---------------------------------------------------------------------------
 * The README suggests converting FP4 inputs into small integers that count
 * "quanta" of 0.5.  The FP4 value set is:
 *
 *   +/- {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 *
 * which maps to integer quanta:
 *
 *   +/- {0, 1, 2, 3, 4, 6, 8, 12}
 *
 * This makes the MAC datapath simple: decode -> signed integer multiply ->
 * INT16 accumulation.
 *
 * ---------------------------------------------------------------------------
 * GEN1 SCOPE OF THIS MODULE
 * ---------------------------------------------------------------------------
 * This module only implements the core 8x8 tile MAC behavior:
 *
 *   - clear tile
 *   - one outer-product MAC per enable pulse
 *   - readback of the 8x8 INT16 tile as 32 x 32-bit words
 *
 * It does NOT implement:
 *   - scaling factors
 *   - BF16/Binary16 conversion
 *   - vector register plumbing
 *   - BRAM U post-processing
 *   - later GEN2/GEN3 multi-cycle behaviors
 *
 ******************************************************************************/

module fp4_mac8x8_gen1 (
    input  logic         clk,
    input  logic         rst_n,

    // clear_i:
    //   Synchronously zero the entire 8x8 tile T.
    input  logic         clear_i,

    // mac_en_i:
    //   When high for one cycle, perform one 8x8 outer-product MAC:
    //     T += outer_product(a_packed_i, b_packed_i)
    input  logic         mac_en_i, //ctrl signals
	input logic max_en_i,
	input logic        add_en_i,
	input logic [2:0]  add_row_i,

    // 32-bit packed operands, each containing 8 x FP4 values.
    // Nibble layout:
    //   fp4[0] = bits [ 3: 0]
    //   fp4[1] = bits [ 7: 4]
    //   ...
    //   fp4[7] = bits [31:28]
    input  logic [31:0]  a_packed_i, // [stev] - need to rename to rs1_rdata_i
    input  logic [31:0]  b_packed_i, // [stev] - need to rename to rs2_rdata_i


  // ------------------------------------------------------------
  // Move-from-tile ISA interface
  // ------------------------------------------------------------
  input  logic        mv_en_i,        // pulse for mve/mvo/mv2
  input  logic [1:0]  mv_op_i,        // 0=mve, 1=mvo, 2=mv2
  input  logic [4:0]  mv_row_i,       // instruction rs1 field
  input  logic [4:0]  mv_pair_i,      // instruction rs2 field
  output logic [31:0] mv_data_o,       // value to integer rd writeback
//[stev] - maybe reduce the amount of rd ports

    // Read interface for exposing the 8x8 INT16 tile as 32 x 32-bit words.
    //
    // Each read address returns TWO INT16 values packed into one 32-bit word:
    //
    //   rd_addr = 0  -> {T[0][1], T[0][0]}
    //   rd_addr = 1  -> {T[0][3], T[0][2]}
    //   ...
    //   rd_addr = 31 -> {T[7][7], T[7][6]}
    //
    // This matches the README concept that the FP4 tile contains:
    //   8 * 8 * 16b = 64 * 16b = 32 * 32b words.
    input  logic         rd_en_i,
    input  logic [4:0]   rd_addr_i,
    output logic [31:0]  rd_data_o,

  // ------------------------------------------------------------
  // Tile memory operations: ld2MAC64 / st2MAC64
  // rs2 field encoding:
  //   rs2[4:2] = row i
  //   rs2[1:0] = pair j  -> columns 2*j and 2*j+1
  // ------------------------------------------------------------
  input  logic        ld2_en_i,          // pulse for ld2MAC64 completion/write tile
  input  logic        st2_en_i,          // pulse for st2MAC64 commit/clear tile
  input  logic [4:0]  tile_mem_rs2_i,    // encoded tile selector from instruction rs2
  input  logic [31:0] ld2_data_i,        // loaded memory word to write into tile
  output logic [31:0] st2_data_o         // tile pair to send to memory
//[stev] - need to cleanup and reuse ports, create mux for internal logic
);

logic [31:0] mv_data;
assign mv_data_o = mv_data;
    /**************************************************************************
     * INTERNAL TILE STORAGE
     *
     * T[r][c] is the signed INT16 accumulation tile.
     *
     * r = row index   (0..7)
     * c = column index(0..7)
     *
     * One MAC operation updates all 64 entries:
     *   T[r][c] += A_q[r] * B_q[c]
     **************************************************************************/
    logic signed [15:0] T [0:7][0:7];

// --- [stev] ---
logic dump_next; //quick debug flag
// --- [end] ---

    /**************************************************************************
     * UNPACKED FP4 NIBBLES
     *
     * We unpack each 32-bit operand into 8 nibbles.
     *
     * Example:
     *   a_fp4[0] = a_packed_i[ 3: 0]
     *   a_fp4[1] = a_packed_i[ 7: 4]
     *   ...
     *   a_fp4[7] = a_packed_i[31:28]
     **************************************************************************/
    logic [3:0] a_fp4 [0:7];
    logic [3:0] b_fp4 [0:7];

    /**************************************************************************
     * DECODED SIGNED QUANTA VALUES
     *
     * After unpacking each FP4 nibble, we decode it into a signed integer
     * in the set:
     *
     *   {-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12}
     *
     * These are the "quanta counts" corresponding to FP4 values measured
     * in units of 0.5.
     *
     * We use 5 signed bits because the range -12..12 fits in signed 5-bit.
     **************************************************************************/
    logic signed [4:0] a_q [0:7];
    logic signed [4:0] b_q [0:7];

    /**************************************************************************
     * FP4 -> SIGNED QUANTA DECODE FUNCTION
     *
     * ASSUMED FP4 nibble format:
     *   fp4[3]   = sign bit
     *   fp4[2:0] = magnitude code
     *
     * Magnitude code mapping:
     *   3'b000 ->  0   (0.0)
     *   3'b001 ->  1   (0.5)
     *   3'b010 ->  2   (1.0)
     *   3'b011 ->  3   (1.5)
     *   3'b100 ->  4   (2.0)
     *   3'b101 ->  6   (3.0)
     *   3'b110 ->  8   (4.0)
     *   3'b111 -> 12   (6.0)
     *
     * If sign=1, the decoded magnitude is negated.
     *
     * IMPORTANT:
     * If your project uses a different 3-bit E2M1 encoding order, THIS is the
     * one place in the RTL to change.
     **************************************************************************/
    function automatic logic signed [4:0] fp4_to_quanta (
        input logic [3:0] fp4
    );
        logic       sign;
        logic [2:0] mag;
        logic signed [4:0] q_mag;
        begin
            sign = fp4[3];
            mag  = fp4[2:0];

            unique case (mag)
                3'b000: q_mag =  5'sd0;
                3'b001: q_mag =  5'sd1;
                3'b010: q_mag =  5'sd2;
                3'b011: q_mag =  5'sd3;
                3'b100: q_mag =  5'sd4;
                3'b101: q_mag =  5'sd6;
                3'b110: q_mag =  5'sd8;
                3'b111: q_mag =  5'sd12;
                default: q_mag = 5'sd0;
            endcase

            if (sign)
                fp4_to_quanta = -q_mag;
            else
                fp4_to_quanta = q_mag;
        end
    endfunction

    /**************************************************************************
     * COMBINATIONAL UNPACK LOGIC
     *
     * Break each 32-bit packed word into 8 nibbles.
     *
     * We use a generate block so the intent is very explicit and compact.
     **************************************************************************/
    genvar g;
    generate
        for (g = 0; g < 8; g++) begin : GEN_UNPACK
            assign a_fp4[g] = a_packed_i[g*4 +: 4];
            assign b_fp4[g] = b_packed_i[g*4 +: 4];
        end
    endgenerate

    /**************************************************************************
     * COMBINATIONAL DECODE LOGIC
     *
     * Convert each unpacked FP4 nibble into its signed integer quanta value.
     *
     * This happens combinationally from the input words, so when mac_en_i
     * is asserted on a clock edge, the tile update can directly use a_q/b_q.
     **************************************************************************/
    generate
        for (g = 0; g < 8; g++) begin : GEN_DECODE
            assign a_q[g] = fp4_to_quanta(a_fp4[g]);
            assign b_q[g] = fp4_to_quanta(b_fp4[g]);
        end
    endgenerate

    /**************************************************************************
     * Signals for maxMAC64 rs1
     **************************************************************************/
	logic signed [15:0] max_scalar;
	assign max_scalar = a_packed_i[15:0];

    /**************************************************************************
     * Signals and functions for addMAC64 rs1, rs2
     **************************************************************************/
function automatic logic signed [15:0] sat16_add (
    input logic signed [15:0] a,
    input logic signed [15:0] b
);
    logic signed [16:0] tmp;
    begin
        tmp = a + b;

        if (tmp > 32767)
            sat16_add = 16'sd32767; // signed int16 max = 32767, min = -32768
        else if (tmp < -32768)
            sat16_add = -16'sd32768;
        else
            sat16_add = tmp[15:0];
    end
endfunction

	logic signed [15:0] add_scalar;
	assign add_scalar = b_packed_i[15:0];


    /**************************************************************************
     * Signals and functions for:
     * 1) mvoMAC64 rd, rs1, rs2 
     * 2) mveMAC64 rd, rs1, rs2 
     * 3) mv2MAC64 rd, rs1, rs2 
     **************************************************************************/
	localparam logic [1:0] MV_EVEN = 2'd0;
	localparam logic [1:0] MV_ODD  = 2'd1;
	localparam logic [1:0] MV_PAIR = 2'd2;

	logic [2:0] mv_row_idx;
	logic [1:0] mv_pair_idx;
	logic [2:0] mv_even_col_idx;
	logic [2:0] mv_odd_col_idx;

	assign mv_row_idx      = mv_row_i[2:0];
	assign mv_pair_idx     = mv_pair_i[1:0];
	assign mv_even_col_idx = {mv_pair_idx, 1'b0};  // 2*rs2
	assign mv_odd_col_idx  = {mv_pair_idx, 1'b1};  // 2*rs2+1

//always_comb begin
//  mv_data_o = 32'h0;

//  unique case (mv_op_i)
//    MV_EVEN: begin
//      mv_data_o = {{16{T[mv_row_idx][mv_even_col_idx][15]}}, //[stev] - sign extension to 32 bits here, need to check
//                    T[mv_row_idx][mv_even_col_idx]};
//    end

//    MV_ODD: begin
//      mv_data_o = {{16{T[mv_row_idx][mv_odd_col_idx][15]}},
//                    T[mv_row_idx][mv_odd_col_idx]};
//    end

//    MV_PAIR: begin

 //     mv_data_o = {T[mv_row_idx][mv_odd_col_idx],
//                   T[mv_row_idx][mv_even_col_idx]};
//    end

//    default: begin
//      mv_data_o = 32'h0;
//    end
//  endcase
//end


    /**************************************************************************
     * Signals and functions for:
     * 1) ld2MAC64 rs2, IMM12(rs1) 
     * 2) st2MAC64 rs2, IMM12(rs1)
     **************************************************************************/
  logic [2:0] tm_row_idx;
  logic [1:0] tm_pair_idx;
  logic [2:0] tm_even_col_idx;
  logic [2:0] tm_odd_col_idx;

  assign tm_row_idx      = tile_mem_rs2_i[4:2];
  assign tm_pair_idx     = tile_mem_rs2_i[1:0];
  assign tm_even_col_idx = {tm_pair_idx, 1'b0};  // 2*j
  assign tm_odd_col_idx  = {tm_pair_idx, 1'b1};  // 2*j+1


  always_comb begin
    st2_data_o = {T[tm_row_idx][tm_odd_col_idx],
                  T[tm_row_idx][tm_even_col_idx]};
  end



/**************************************************************************
     * MAIN TILE UPDATE LOGIC
     **************************************************************************/
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Asynchronous active-low reset clears everything cleanly
            for (i = 0; i < 8; i++) begin
                for (j = 0; j < 8; j++) begin
                    T[i][j] <= '0;
                end
            end
            dump_next <= 1'b0;
            mv_data   <= 32'h0;
        end 
        else begin 
            // 1. Core Hardware Instructions Tree
            if (clear_i) begin
                $display("[fp4_mac8x8_gen1] zzMAC64 triggered ...");
                for (i = 0; i < 8; i++) begin
                    for (j = 0; j < 8; j++) begin
                        T[i][j] <= '0;
                    end
                end
                dump_next <= 1'b1;
            end 
            else if (max_en_i) begin
                $display("[fp4_mac8x8_gen1] maxMAC64 triggered ...");
                dump_next <= 1'b1;
                for (i = 0; i < 8; i++) begin
                    for (j = 0; j < 8; j++) begin
                        if ($signed(T[i][j]) < max_scalar)
                            T[i][j] <= max_scalar;
                    end
                end
            end 
            else if (mv_en_i) begin
                unique case (mv_op_i)
                    MV_EVEN: begin
                        mv_data <= {{16{T[mv_row_idx][mv_even_col_idx][15]}}, T[mv_row_idx][mv_even_col_idx]};
                        T[mv_row_idx][mv_even_col_idx] <= '0;
                    end
                    MV_ODD: begin
                        mv_data <= {{16{T[mv_row_idx][mv_odd_col_idx][15]}}, T[mv_row_idx][mv_odd_col_idx]};
                        T[mv_row_idx][mv_odd_col_idx] <= '0;
                    end
                    MV_PAIR: begin
                        $display("[fp4_mac8x8_gen1] mv2MAC64 triggered ...");
                        dump_next <= 1'b1;
                        mv_data   <= {T[mv_row_idx][mv_odd_col_idx], T[mv_row_idx][mv_even_col_idx]};
                        T[mv_row_idx][mv_even_col_idx] <= '0;
                        T[mv_row_idx][mv_odd_col_idx]  <= '0;
                    end
                    default: ;
                endcase
            end 
            else if (ld2_en_i) begin
                T[tm_row_idx][tm_even_col_idx] <= ld2_data_i[15:0];
                T[tm_row_idx][tm_odd_col_idx]  <= ld2_data_i[31:16];
            end 
            else if (st2_en_i) begin
                T[tm_row_idx][tm_even_col_idx] <= '0;
                T[tm_row_idx][tm_odd_col_idx]  <= '0;
            end 
            else if (add_en_i) begin
                $display("[fp4_mac8x8_gen1] addMAC64 triggered ...");
                dump_next <= 1'b1;
                for (j = 0; j < 8; j++) begin
                    T[add_row_i][j] <= sat16_add(T[add_row_i][j], add_scalar);
                end
            end 
            else if (mac_en_i) begin
                $display("[fp4_mac8x8_gen1] hwMAC64 triggered ...");
                dump_next <= 1'b1;
                for (i = 0; i < 8; i++) begin
                    for (j = 0; j < 8; j++) begin
                        T[i][j] <= T[i][j] + (a_q[i] * b_q[j]);
                    end
                end
            end

            // 2. Clear simulation debug flag on the cycle immediately following a dump
            if (dump_next) begin
                dump_next <= 1'b0;
                $display("[fp4_mac8x8_gen1] AFTER MAC INST");
                $display("========== [fp4_mac8x8_gen1] MV DEBUG ==========");
                $display("mv_op_i        = %0d", mv_op_i);
                $display("mv_row_i       = %0d", mv_row_i);
                $display("mv_pair_i      = %0d", mv_pair_i);
                $display("mv_data        = 0x%08h", {T[mv_row_idx][mv_odd_col_idx], T[mv_row_idx][mv_even_col_idx]});
                $display("========== [fp4_mac8x8_gen1] ====================");
            end
        end // Cleanly closes the "else" (non-reset) block
    end // Closes always_ff

    /**************************************************************************
     * READBACK PATH
     *
     * The 8x8 INT16 tile is exposed as 32 x 32-bit words.
     * Each read word packs TWO adjacent columns from the same row:
     *
     *   rd_addr 0  -> {T[0][1], T[0][0]}
     *   rd_addr 1  -> {T[0][3], T[0][2]}
     *   rd_addr 2  -> {T[0][5], T[0][4]}
     *   rd_addr 3  -> {T[0][7], T[0][6]}
     *
     *   rd_addr 4  -> {T[1][1], T[1][0]}
     *   ...
     *   rd_addr 31 -> {T[7][7], T[7][6]}
     *
     * Address mapping:
     *   row      = rd_addr_i[4:2]      // 0..7
     *   pair_idx = rd_addr_i[1:0]      // 0..3
     *   col0     = pair_idx*2
     *   col1     = pair_idx*2 + 1
     *
     * rd_en_i is optional gating here; if low we simply return zero.
     **************************************************************************/
    always_comb begin
        logic [2:0] row;
        logic [1:0] pair_idx;
        logic [2:0] col0, col1;

        rd_data_o = 32'h0000_0000;

        if (rd_en_i) begin
            row      = rd_addr_i[4:2];
            pair_idx = rd_addr_i[1:0];

            col0 = {pair_idx, 1'b0};      // 0,2,4,6
            col1 = {pair_idx, 1'b0} + 1;  // 1,3,5,7

            // Pack high halfword = later column, low halfword = earlier column.
            rd_data_o = {T[row][col1], T[row][col0]};
        end
    end



endmodule
