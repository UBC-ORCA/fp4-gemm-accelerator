`timescale 1ns/1ps

/**
    Scale + accumulate for N_SCALE_UNITS lanes (one row group, one column
    per cycle):

        bram_acc_o[i] = bram_acc_i[i] + tile_in_i[i] * as_i[i] * ws_i

    Pipeline (latency 4, one token per cycle):
      S1   input capture
      S2-3 e4m3_scale (decode + broadcast mul, lane mul + round)
      S4   BF16 add with the BRAM accumulator

    The BRAM accumulator and write address ride through e4m3_scale as its
    data sideband; valid/end token ride as its (reset) ctrl sideband.
**/
module scale_accum_array_e4m3
import chief_pkg::*;
#(
    parameter int N_SCALE_UNITS  = 4,
    parameter int TILE_INT_BIAS  = -2,
    parameter int TILE_INT_WIDTH = 16,
    parameter int COL_AW         = 3,
    parameter int ROW_AW         = 1
)
(
    input  logic clk_i,
    input  logic rst_ni,

    // Data channels
    input  chief_scaler_e4m3_t ws_i,
    input  chief_scaler_e4m3_t as_i [0:N_SCALE_UNITS-1],
    input  logic signed [TILE_INT_WIDTH-1:0]
                               tile_in_i [0:N_SCALE_UNITS-1],

    // BRAM read side (accumulator returned for the same cell)
    input  logic [COL_AW-1:0]  bram_rd_col_base_addr_i,
    input  logic [ROW_AW-1:0]  bram_rd_row_addr_i,
    input  bf16_t              bram_acc_i [0:N_SCALE_UNITS-1],
    input  logic               input_valid_i,

    // BRAM write side
    output logic               out_valid_o,
    output logic [COL_AW-1:0]  bram_wr_col_addr_o,
    output logic [ROW_AW-1:0]  bram_wr_row_addr_o,
    output bf16_t              bram_acc_o [0:N_SCALE_UNITS-1],

    // Final token of the stream
    input  logic               end_tok_i,
    output logic               end_tok_o
);

    // Sideband payloads carried through e4m3_scale
    typedef struct packed {
        logic valid;
        logic end_tok;
    } ctrl_t;

    typedef struct packed {
        bf16_t [N_SCALE_UNITS-1:0] acc;
        logic  [COL_AW-1:0]        col;
        logic  [ROW_AW-1:0]        row;
    } data_t;

    //------------------------------------------------------------
    // Stage 1: input capture
    //------------------------------------------------------------
    chief_scaler_e4m3_t               ws_s1;
    chief_scaler_e4m3_t               as_s1   [0:N_SCALE_UNITS-1];
    logic signed [TILE_INT_WIDTH-1:0] tile_s1 [0:N_SCALE_UNITS-1];
    ctrl_t                            ctrl_s1;
    data_t                            data_s1;

    always_ff @(posedge clk_i) begin : s1_ctrl
        if (!rst_ni) ctrl_s1 <= '0;
        else         ctrl_s1 <= '{valid: input_valid_i, end_tok: end_tok_i};
    end : s1_ctrl

    always_ff @(posedge clk_i) begin : s1_data
        ws_s1       <= ws_i;
        data_s1.col <= bram_rd_col_base_addr_i;
        data_s1.row <= bram_rd_row_addr_i;
        for (int i = 0; i < N_SCALE_UNITS; i++) begin
            as_s1[i]       <= as_i[i];
            tile_s1[i]     <= tile_in_i[i];
            data_s1.acc[i] <= bram_acc_i[i];
        end
    end : s1_data

    //------------------------------------------------------------
    // Stages 2-3: scaling (multiply only)
    //------------------------------------------------------------
    bf16_t scaled_s3 [0:N_SCALE_UNITS-1];
    ctrl_t ctrl_s3;
    data_t data_s3;

    e4m3_scale #(
        .N_SCALE_UNITS (N_SCALE_UNITS),
        .TILE_EXP_BIAS (TILE_INT_BIAS),
        .TILE_INT_WIDTH(TILE_INT_WIDTH),
        .CTRL_W        ($bits(ctrl_t)),
        .DATA_W        ($bits(data_t))
    ) u_scale (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .ws_i    (ws_s1),
        .as_i    (as_s1),
        .tile_i  (tile_s1),
        .ctrl_i  (ctrl_s1),
        .data_i  (data_s1),
        .scaled_o(scaled_s3),
        .ctrl_o  (ctrl_s3),
        .data_o  (data_s3)
    );

    //------------------------------------------------------------
    // Stage 4: BF16 accumulation
    //------------------------------------------------------------
    bf16_t acc_result_d [0:N_SCALE_UNITS-1];
    bf16_t acc_result_q [0:N_SCALE_UNITS-1];
    ctrl_t ctrl_s4;
    logic [COL_AW-1:0] col_s4;
    logic [ROW_AW-1:0] row_s4;

    for (genvar i = 0; i < N_SCALE_UNITS; i++) begin : gen_add
        parameterized_adder u_add (
            .a  (data_s3.acc[i]),
            .b  (scaled_s3[i]),
            .sum(acc_result_d[i])
        );
    end : gen_add

    always_ff @(posedge clk_i) begin : s4_ctrl
        if (!rst_ni) begin 
            ctrl_s4 <= '0; 
        end else begin 
            ctrl_s4 <= ctrl_s3;
        end
    end : s4_ctrl

    always_ff @(posedge clk_i) begin : s4_data
        col_s4 <= data_s3.col;
        row_s4 <= data_s3.row;
        for (int i = 0; i < N_SCALE_UNITS; i++) begin 
             acc_result_q[i] <= acc_result_d[i];
        end
    end : s4_data

    assign out_valid_o        = ctrl_s4.valid;
    assign end_tok_o          = ctrl_s4.end_tok;
    assign bram_wr_col_addr_o = col_s4;
    assign bram_wr_row_addr_o = row_s4;
    assign bram_acc_o         = acc_result_q;

endmodule
