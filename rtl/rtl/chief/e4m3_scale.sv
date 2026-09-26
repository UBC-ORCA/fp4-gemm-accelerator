`timescale 1ns / 1ps

module e4m3_scale 
import fp4_pkg::*;
#(
    // Tile exponent biasing
    // True value is tile_i * 2**-TILE_EXP_BIAS
    parameter int TILE_EXP_BIAS = -2
)
(
    input logic clk_i, 
    input logic rst_ni, 

    // Input data payload 
    input logic input_valid_i,
    input logic signed [15:0] tile_i, 
    input fp4_scaler_e4m3_t a_scale_i,
    input fp4_scaler_e4m3_t w_scale_i,
    input bf16_t bram_acc_i,

    input logic [2:0] bram_rd_col_addr_i,
    input logic [1:0] bram_rd_row_addr_i, 

    // Output payload
    output logic out_valid_o, 
    output bf16_t bram_acc_o,
    output logic [2:0] bram_wr_col_addr_o,
    output logic [1:0] bram_wr_row_addr_o,

    // Indicators for the final token to be streamed
    // to the BRAM  
    input logic end_tok_i, 
    output logic end_tok_o

);
    // Pipeline Shift Registers,
    // MSB contains the final stage, while LSB 
    // captures the input
    localparam N_PIPE_STAGES = 3;
    logic [N_PIPE_STAGES-1:0] valid_pipe_q;
    logic [2:0] addr_col_pipe_q [N_PIPE_STAGES-1:0];
    logic [1:0] addr_row_pipe_q [N_PIPE_STAGES-1:0];
    logic [N_PIPE_STAGES-1:0] end_tok_pipe_q; 

    assign bram_wr_col_addr_o = addr_col_pipe_q[N_PIPE_STAGES-1];
    assign bram_wr_row_addr_o = addr_row_pipe_q[N_PIPE_STAGES-1];
    assign end_tok_o = end_tok_pipe_q[N_PIPE_STAGES-1];

    // Pipeline registers for each stage 
    // Stage 1: Input Capture 
    logic signed [15:0] tile_s1; 
    fp4_scaler_e4m3_t   a_scale_s1;
    fp4_scaler_e4m3_t   w_scale_s1;
    bf16_t              bram_acc_s1;

    // Stage 2: Scaling operation 
    bf16_t bram_acc_s2; 
    bf16_t scaled_tile_s2_d; 
    bf16_t scaled_tile_s2_q;  
    logic  scaled_tile_zero_s2_d; 
    logic  scaled_tile_nan_s2_d;
    logic  scaled_tile_zero_s2_q; 
    logic  scaled_tile_nan_s2_q;

    // Stage 3: (Shared ?) Accumulation (with E8M0)
    bf16_t acc_result_d;    
    bf16_t acc_result_q;

    assign bram_acc_o = acc_result_q;

    // Control signal propagation
    always_ff @(posedge clk_i or negedge rst_ni) begin 
        if (~rst_ni) begin
            valid_pipe_q <= 4'b0;
        end else begin
            // For any i > 0, shift_regs[i] gets shift_regs[i-1] 
            valid_pipe_q <= 
                {valid_pipe_q[N_PIPE_STAGES-2:0], input_valid_i};
            
            end_tok_pipe_q <= 
                {end_tok_pipe_q[N_PIPE_STAGES-2:0], end_tok_i};
        end
    end

    // Address Propagation 
    always_ff @(posedge clk_i) begin 
        addr_row_pipe_q[0] <= bram_rd_row_addr_i;
        addr_col_pipe_q[0] <= bram_rd_col_addr_i;

        for (int i = 1; i < N_PIPE_STAGES; ++i) begin 
            addr_row_pipe_q[i] <= addr_row_pipe_q[i-1];
            addr_col_pipe_q[i] <= addr_col_pipe_q[i-1];
        end 
    end


    //------------------------------------------------------------ 
    // Pipeline Stage 1: Input Capture
    //------------------------------------------------------------ 
    always_ff @(posedge clk_i) begin : pipe_in_capture_stage
        tile_s1 <= tile_i; 
        a_scale_s1 <= a_scale_i;  
        w_scale_s1 <= w_scale_i;
        bram_acc_s1 <= bram_acc_i;  
    end : pipe_in_capture_stage

    //------------------------------------------------------------ 
    // Pipeline Stage 2: Tile Value conversion and scaling  
    //------------------------------------------------------------ 
    e4m3_mul mul (
        .A8(a_scale_s1), 
        .B8(w_scale_s1),
        .q14_2_C_in(tile_s1),
        .PABC(scaled_tile_s2_d),
        .isNaN(scaled_tile_nan_s2_d), 
        .isZero(scaled_tile_zero_s2_d)
    );
    
    always_ff @(posedge clk_i) begin : pipe_scale
        scaled_tile_s2_q <= scaled_tile_s2_d;    
        scaled_tile_zero_s2_q <= scaled_tile_zero_s2_d;
        scaled_tile_nan_s2_q <= scaled_tile_nan_s2_d;
        bram_acc_s2 <= bram_acc_s1;
    end : pipe_scale
    
    //------------------------------------------------------------ 
    // Pipeline Stage 3: Accumulation and save output
    //------------------------------------------------------------ 

    parameterized_adder_e4m3 u_add (
        .a(bram_acc_s2),
        .b(scaled_tile_s2_q), 
        .sum(acc_result_d)
    );
    
    always_ff @(posedge clk_i) begin : tile_acc 
        acc_result_q <= acc_result_d;  
    end : tile_acc 

endmodule




