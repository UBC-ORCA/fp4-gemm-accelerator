`timescale 1ns / 1ps

module e4m3_scale 
import fp4_pkg::*;
(
    input logic clk_i, 
    input logic rst_n_i, 

    // Input data payload 
    input logic payload_valid_in,
    input logic signed [15:0] acc_q14_2_in, 
    input fp4_scaler_e4m3_t a_scale_in,
    input fp4_scaler_e4m3_t w_scale_in,
    input bf16_t bf16_in,

    input logic [2:0] bram_rd_col_addr_in,
    input logic [1:0] bram_rd_row_addr_in, 

    // Output payload
    output logic out_valid_o, 
    output bf16_t bf16_out,
    output logic [2:0] bram_wr_col_addr_o,
    output logic [1:0] bram_wr_row_addr_o

);
    /* e7m6 prod */
    bf16_t prod_abc;
    logic isNaN, isZero;

    // Pipeline Shift Registers,
    // MSB contains the final stage, while LSB 
    // captures the input
    localparam N_PIPE_STAGES = 4;
    logic [N_PIPE_STAGES-1:0] valid_pipe_q;
    logic [2:0] addr_col_pipe_q [N_PIPE_STAGES-1:0];
    logic [1:0] addr_row_pipe_q [N_PIPE_STAGES-1:0];

    assign bram_wr_col_addr_o = addr_col_pipe_q[N_PIPE_STAGES-1];
    assign bram_wr_row_addr_o = addr_row_pipe_q[N_PIPE_STAGES-1];

    // Pipeline registers for each stage 
    // Stage 1: Input Capture 
    //





    // Stage 2: Scaling operation 

    

    // Stage 3: (Shared ?) Accumulation (with E8M0)

    always_ff @(posedge clk_i or negedge rst_n_i) begin 
        if (~rst_n_i) begin
            valid_pipe_q <= 4'b0;
        end else begin
            // For any i > 0, shift_regs[i] gets shift_regs[i-1] 
            valid_pipe_q <= 
                {valid_pipe_q[N_PIPE_STAGES-2:0], payload_valid_in};
            
        end
    end


    // Address Propagation 
    always_ff @(posedge clk_i) begin 
        addr_row_pipe_q[0] <= bram_rd_row_addr_in;
        addr_col_pipe_q[0] <= bram_rd_col_addr_in;

        for (int i = 1; i < N_PIPE_STAGES; ++i) begin 
            addr_row_pipe_q[i] <= addr_row_pipe_q[i-1];
            addr_col_pipe_q[i] <= addr_col_pipe_q[i-1];
        end 
    end

    e4m3_mul mul (
        .A8(a_scale_in), 
        .B8(w_scale_in),
        .q14_2_C_in(acc_q14_2_in),
        .PABC(prod_abc),
        .isNaN(isNaN), 
        .isZero(isZero)
    );

    parameterized_adder_e4m3 u_add (
        .a(bf16_in),
        .b(prod_abc), 
        .sum(bf16_out)
    );

endmodule




