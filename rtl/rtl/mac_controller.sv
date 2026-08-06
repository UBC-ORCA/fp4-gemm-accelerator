// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

module mac_controller #(
    parameter int VL = 32, 
    parameter int TT = 8
) (
    input  logic                        clk_i,
    input  logic                        rst_ni,

    // Request interface
    input  logic                        req_valid_i,
    output logic                        req_ready_o,
    input  cve2_pkg::mac_op_e           cf_req_op_i,
    input  logic [31:0]                 rs1_i,
    input  logic [31:0]                 rs2_i,

    // Status outputs
    output logic                        busy_o,
    output logic                        done_o,

    // Context / Control to VMAC engine
    input  logic [4:0]                  vs1_i,
    input  logic [4:0]                  weight_blk_i,
    input  logic [31:0]                 base_i,

    output logic                        vmac_start_o,
    input  logic                        vmac_busy_i,
    input  logic                        vmac_done_i,
    input  logic                        vmac_complete_i,

    // Memory / GPR scalar interface
    output logic                        scalar_we_o,
    output logic [4:0]                  scalar_waddr_o,
    input  logic [4:0]                  scalar_waddr_i,

    // Scale register interface
    output logic [31:0]                 act_scale_lo_o,
    output logic [31:0]                 act_scale_hi_o,
    output logic [31:0]                 weight_scale_lo_o,
    output logic [31:0]                 weight_scale_hi_o,

    output logic                        act_scale_ready_o,
    output logic                        weight_scale_ready_o,

    // Scale FSM handshake ports
    input  logic                        scale_busy_i,
    input  logic                        scale_done_i,

    //--------------------------------------------------
    // Accumulator BRAM Structured Interface
    //--------------------------------------------------
    output logic                        accum_rd_en_o,
    output logic [4:0]                  accum_rd_tile_o,
    output logic [2:0]                  accum_rd_row_o,
    output logic [2:0]                  accum_rd_col_o,
    input  logic [15:0]                 accum_rd_data_i,

    output logic                        accum_wr_en_o,
    output logic [4:0]                  accum_wr_tile_o,
    output logic [2:0]                  accum_wr_row_o,
    output logic [2:0]                  accum_wr_col_o,
    output logic [15:0]                 accum_wr_data_o
);

    logic [31:0] act_scale_lo_q;
    logic [31:0] act_scale_hi_q;
    logic [31:0] weight_scale_lo_q;
    logic [31:0] weight_scale_hi_q;

    logic        act_scale_pulse;
    logic        weight_scale_pulse;

    cve2_pkg::mac_op_e op_q;

    // ISA Bitfield Extraction for MAC_BIAS Controls
    logic [2:0]  bias_col;
    logic [2:0]  bias_row;
    logic [4:0]  bias_tile;
    logic [15:0] bias_value;

    assign bias_col   = rs1_i[2:0];
    assign bias_row   = rs1_i[5:3];
    assign bias_tile  = rs1_i[10:6];
    assign bias_value = rs2_i[15:0];

    // FSM State Definition
    typedef enum logic [1:0] {
        IDLE,
        CLEAR,
        EXEC,
        DONE
    } state_e;
    state_e state_q, state_d;

    logic [4:0] scalar_waddr_q;
    logic       brd_phase_q, brd_phase_d;   // BRAM read: 0=issue read, 1=capture+writeback

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            brd_phase_q        <= 1'b0;
            op_q               <= cve2_pkg::OP_ZZ;
            state_q            <= IDLE;
            scalar_waddr_q     <= '0;
            act_scale_lo_q     <= '0;
            act_scale_hi_q     <= '0;
            weight_scale_lo_q  <= '0;
            weight_scale_hi_q  <= '0;
            act_scale_pulse    <= 1'b1;
            weight_scale_pulse <= 1'b0;
        end else begin
            state_q            <= state_d;
            brd_phase_q        <= brd_phase_d;

            act_scale_pulse    <= 1'b0;
            weight_scale_pulse <= 1'b0;

	    // Fire weight-scale context only when the previous VMAC has completed.
	    if ((state_q == EXEC) &&
    		(op_q == cve2_pkg::OP_MAC_WS) &&
    		vmac_complete_i) begin
    		weight_scale_pulse <= 1'b1;
	    end

            if (req_valid_i && req_ready_o) begin
                op_q           <= cf_req_op_i;
                scalar_waddr_q <= scalar_waddr_i;

                unique case (cf_req_op_i)
                    cve2_pkg::OP_MAC_AS: begin
                        act_scale_lo_q  <= rs1_i;
                        act_scale_hi_q  <= rs2_i;
                        act_scale_pulse <= 1'b1;
                    end
                    cve2_pkg::OP_MAC_WS: begin
                        weight_scale_lo_q  <= rs1_i;
                        weight_scale_hi_q  <= rs2_i;
                        //weight_scale_pulse <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    // Next State Logic
    always_comb begin
        state_d     = state_q;
        brd_phase_d = brd_phase_q;

        case (state_q)
            IDLE: begin
                brd_phase_d = 1'b0;

                if (req_valid_i) begin
                    if (cf_req_op_i == cve2_pkg::OP_VMAC)
                        state_d = CLEAR;
                    else
                        state_d = EXEC;
                end
            end

            CLEAR: begin
                state_d = EXEC;
            end

            EXEC: begin
                if (op_q == cve2_pkg::OP_BRAM_RD) begin
                    if (!brd_phase_q) begin
                        brd_phase_d = 1'b1;   // read issued; capture/writeback next cycle
                    end else begin
                        brd_phase_d = 1'b0;
                        state_d     = DONE;
                    end
                end
                else if ((op_q == cve2_pkg::OP_MAC_AS) ||
                         (op_q == cve2_pkg::OP_MAC_BIAS) ||
                         (op_q == cve2_pkg::OP_ACC_BANK)) begin // Finishes execution in one cycle
                    state_d = DONE;
                end
		else if (op_q == cve2_pkg::OP_MAC_WS) begin
//[stev]
//`ifdef DEBUG_PRINT
//$display("WS EXEC complete=%0d", vmac_complete_i);
//`endif
//end

    			if (vmac_complete_i) begin
        			state_d = DONE;
			end
		end 
                else if (op_q == cve2_pkg::OP_VMAC) begin //[stev] - need to make this return imm
                   // if (vmac_done_i) begin
                        state_d = DONE;
                   // end
                end
            end

            DONE: begin
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    // Output Decode Logic
    always_comb begin
        req_ready_o    = 1'b0;
        busy_o         = 1'b1;
        done_o         = 1'b0;
        scalar_we_o    = 1'b0;
        scalar_waddr_o = scalar_waddr_q;

        vmac_start_o   = (state_q == CLEAR);

        act_scale_lo_o    = act_scale_lo_q;
        act_scale_hi_o    = act_scale_hi_q;
        weight_scale_lo_o = weight_scale_lo_q;
        weight_scale_hi_o = weight_scale_hi_q;

        act_scale_ready_o    = act_scale_pulse;
        weight_scale_ready_o = weight_scale_pulse;

        // Structured Accumulator Memory Defaults
        accum_rd_en_o   = 1'b0;
        accum_rd_tile_o = '0;
        accum_rd_row_o  = '0;
        accum_rd_col_o  = '0;

        accum_wr_en_o   = 1'b0;
        accum_wr_tile_o = '0;
        accum_wr_row_o  = '0;
        accum_wr_col_o  = '0;
        accum_wr_data_o = '0;

        case (state_q)
            IDLE: begin
                req_ready_o = 1'b1;
                busy_o      = 1'b0;
            end

            CLEAR: begin
                // Occupied clearing cycle; busy_o defaults to 1'b1, req_ready_o defaults to 1'b0
            end

            EXEC: begin
                unique case (op_q)
                    cve2_pkg::OP_MAC_BIAS: begin
                        accum_wr_en_o   = 1'b1;
                        accum_wr_tile_o = bias_tile;
                        accum_wr_row_o  = bias_row;
                        accum_wr_col_o  = bias_col;
                        accum_wr_data_o = bias_value;
                    end
                    cve2_pkg::OP_BRAM_RD: begin
                        if (!brd_phase_q) begin
                            // phase 0: issue the (synchronous) BRAM read
                            accum_rd_en_o   = 1'b1;
                            accum_rd_tile_o = bias_tile;    // rs1[10:6]
                            accum_rd_row_o  = bias_row;     // rs1[5:3]
                            accum_rd_col_o  = bias_col;     // rs1[2:0]
                        end else begin
                            // phase 1: data is valid, write it back to the GPR
                            scalar_we_o = 1'b1;
                        end
                    end
                    default: ;
                endcase
            end

            DONE: begin
                done_o = 1'b1;
            end

            default: ;
        endcase
    end

endmodule
