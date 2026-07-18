// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

module mac_controller #(
    parameter int VL = 32, 
    parameter int TT = 8
) (
    input  logic                       clk_i,
    input  logic                       rst_ni,

    // Request interface
    input  logic                       req_valid_i,
    output logic                       req_ready_o,
    input  cve2_pkg::mac_op_e          cf_req_op_i,
    input  logic [31:0]                rs1_i,
    input  logic [31:0]                rs2_i,

    // Status outputs
    output logic                       busy_o,
    output logic                       done_o,

    // Control to MAC array
    output logic                       mac_en_o,
    output logic                       clear_o,

    // Vector Register File Interface
    input  logic [4:0]                 vs1_i,
    input  logic [4:0]                 weight_blk_i,
    input  logic [31:0]                base_i,

    output logic [4:0]                 mac_vrf_raddr_o,
    output logic [4:0]                 mac_vrf_relem_o, 
    input  logic [31:0]                mac_vrf_rdata_i, 

    // Weight memory interface
    output logic                       data_req_o,
    input  logic                       data_gnt_i,

    output logic [31:0]                data_addr_o,
    output logic                       data_we_o,
    output logic [3:0]                 data_be_o,
    output logic [31:0]                data_wdata_o,

    input  logic                       data_rvalid_i,
    input  logic [31:0]                data_rdata_i,
    input  logic                       data_err_i,

    // MV Inst Interface
    output logic                       mv_en_o,
    output logic [1:0]                 mv_mode_o,
    output logic [2:0]                 mv_even_col_idx_o,
    output logic [2:0]                 mv_odd_col_idx_o,
    output logic [2:0]                 mv_row_idx_o,
    input  logic [4:0]                 mv_row_i,
    input  logic [4:0]                 mv_pair_i,

    output logic                       scalar_we_o,
    output logic [4:0]                 scalar_waddr_o,
    input  logic [4:0]                 scalar_waddr_i,

    // Optimized Vector Slices
    output logic [3:0]                 act_vector_o    [0:TT-1],
    output logic [3:0]                 weight_vector_o [0:TT-1],

    // Scale register interface
    output logic [31:0]                act_scale_lo_o,
    output logic [31:0]                act_scale_hi_o,
    output logic [31:0]                weight_scale_lo_o,
    output logic [31:0]                weight_scale_hi_o,

    output logic                       act_scale_ready_o,
    output logic                       weight_scale_ready_o,
    output logic                       mac_snapshot_valid_o,

    // Scale FSM handshake ports
    input  logic                       scale_busy_i,
    input  logic                       scale_done_i,

    //--------------------------------------------------
    // Accumulator BRAM Structured Interface (Patched)
    //--------------------------------------------------
    output logic                       accum_rd_en_o,
    output logic [4:0]                 accum_rd_tile_o,
    output logic [2:0]                 accum_rd_row_o,
    output logic [2:0]                 accum_rd_col_o,
    input  logic [15:0]                accum_rd_data_i,

    output logic                       accum_wr_en_o,
    output logic [4:0]                 accum_wr_tile_o,
    output logic [2:0]                 accum_wr_row_o,
    output logic [2:0]                 accum_wr_col_o,
    output logic [15:0]                accum_wr_data_o
);

    logic [31:0] act_scale_lo_q;
    logic [31:0] act_scale_hi_q;
    logic [31:0] weight_scale_lo_q;
    logic [31:0] weight_scale_hi_q;

    logic        snapshot_valid_q;
    logic        act_scale_pulse;
    logic        weight_scale_pulse;

    logic [4:0]  vs1_q;
    logic [4:0]  weight_blk_q;
    logic [31:0] base_q;

    logic        mem_req_sent_q;
    logic        mem_req_sent_d;

    cve2_pkg::mac_op_e op_q;

    localparam int CNT_W = $clog2(VL);
    logic [CNT_W-1:0] count_q;
    logic [CNT_W-1:0] count_d;

    // ISA Bitfield Extraction for MAC_BIAS Controls
    logic [2:0]  bias_col;
    logic [2:0]  bias_row;
    logic [4:0]  bias_tile;
    logic [15:0] bias_value;

    assign bias_col   = rs1_i[2:0];
    assign bias_row   = rs1_i[5:3];
    assign bias_tile  = rs1_i[10:6];
    assign bias_value = rs2_i[15:0];

    // VRF flattening logic
    logic [4:0] elem_idx;
    logic [4:0] mac_vrf_addr;

    assign elem_idx     = count_q;
    assign mac_vrf_addr = vs1_q;

    logic        vmac_last_q;

    typedef enum logic [1:0] { IDLE, EXEC, DONE } state_e;
    state_e state_q, state_d;

    logic [31:0] act_packed;
    logic [31:0] weight_packed;

    always_comb begin
        act_packed    = mac_vrf_rdata_i;
        weight_packed = data_rdata_i;
        if (op_q == cve2_pkg::OP_MAC) begin
            act_packed    = rs1_i;
            weight_packed = rs2_i;
        end
    end

    localparam logic [1:0] MV_EVEN = 2'd0;
    localparam logic [1:0] MV_ODD  = 2'd1;
    localparam logic [1:0] MV_PAIR = 2'd2;

    logic [1:0] mv_pair_idx;
    assign mv_row_idx_o      = mv_row_i[2:0];
    assign mv_pair_idx       = mv_pair_i[1:0];
    assign mv_even_col_idx_o = {mv_pair_idx, 1'b0};
    assign mv_odd_col_idx_o  = {mv_pair_idx, 1'b1};

    logic [4:0] scalar_waddr_q;
    logic       brd_phase_q, brd_phase_d;   // BRAM read: 0=issue read, 1=capture+writeback

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            brd_phase_q        <= 1'b0;
            op_q               <= cve2_pkg::OP_ZZ;
            vs1_q              <= '0;
            weight_blk_q       <= '0; 
            base_q             <= '0; 
            state_q            <= IDLE;
            count_q            <= '0;
            mem_req_sent_q     <= 1'b0;
            scalar_waddr_q     <= '0;
            act_scale_lo_q     <= '0;
            act_scale_hi_q     <= '0;
            weight_scale_lo_q  <= '0;
            weight_scale_hi_q  <= '0;
            snapshot_valid_q   <= 1'b0;
            act_scale_pulse    <= 1'b0;
            weight_scale_pulse <= 1'b0;
            vmac_last_q        <= 1'b0;
        end else begin
            state_q            <= state_d;
            count_q            <= count_d;
            mem_req_sent_q     <= mem_req_sent_d;
            brd_phase_q        <= brd_phase_d;

            snapshot_valid_q   <= 1'b0;
            act_scale_pulse    <= 1'b0;
            weight_scale_pulse <= 1'b0;

            if (req_valid_i && req_ready_o) begin
                op_q           <= cf_req_op_i;
                vs1_q          <= vs1_i;
                weight_blk_q   <= weight_blk_i;
                base_q         <= base_i;
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
                        weight_scale_pulse <= 1'b1;
                    end
                    default: ;
                endcase
            end

            vmac_last_q <= (op_q == cve2_pkg::OP_VMAC) && data_rvalid_i && (count_q == (VL-1));

            if (vmac_last_q) begin
                snapshot_valid_q <= 1'b1;
            end
        end
    end

    always_comb begin
        state_d        = state_q;
        count_d        = count_q;
        brd_phase_d    = brd_phase_q;
        mem_req_sent_d = mem_req_sent_q;

        case (state_q)
            IDLE: begin
                count_d        = '0;
                mem_req_sent_d = 1'b0;
                brd_phase_d    = 1'b0;
                if (req_valid_i) begin
                    state_d = EXEC;
                end
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
                else if ((op_q == cve2_pkg::OP_ZZ )    ||
                    (op_q == cve2_pkg::OP_MAC)    ||
                    (op_q == cve2_pkg::OP_MVE)    ||
                    (op_q == cve2_pkg::OP_MVO)    ||
                    (op_q == cve2_pkg::OP_MV2)    ||
                    (op_q == cve2_pkg::OP_MAC_AS) ||
                    (op_q == cve2_pkg::OP_MAC_WS) ||
                    (op_q == cve2_pkg::OP_MAC_BIAS) ||
                    (op_q == cve2_pkg::OP_ACC_BANK)) begin // Finishes execution in one cycle
                    state_d = DONE;
                end 
                else if (op_q == cve2_pkg::OP_VMAC) begin
                    if (!mem_req_sent_q) begin
                        if (data_gnt_i) begin
                            mem_req_sent_d = 1'b1;
                        end
                    end else begin
                        if (data_rvalid_i) begin
                            mem_req_sent_d = 1'b0;
                            if (count_q == (VL-1)) begin
                                state_d = DONE;
                                count_d = '0;
                            end else begin
                                count_d = count_q + 1'b1;
                            end
                        end
                    end
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

    always_comb begin
        req_ready_o    = 1'b0;
        busy_o         = 1'b1;
        done_o         = 1'b0;
        clear_o        = 1'b0;
        mv_en_o        = 1'b0;
        mv_mode_o      = MV_EVEN;
        scalar_we_o    = 1'b0;
        scalar_waddr_o = scalar_waddr_q;

        mac_en_o = ((state_q == EXEC) && (op_q == cve2_pkg::OP_VMAC) && data_rvalid_i) || 
                   ((state_q == EXEC) && (op_q == cve2_pkg::OP_MAC));
        clear_o = (state_q == EXEC) && (op_q == cve2_pkg::OP_ZZ);

        mac_vrf_raddr_o = '0;
        mac_vrf_relem_o = '0;
        data_req_o      = 1'b0;
        data_addr_o     = '0;
        data_we_o       = 1'b0;
        data_be_o       = 4'b1111; 
        data_wdata_o    = '0;

        act_scale_lo_o    = act_scale_lo_q;
        act_scale_hi_o    = act_scale_hi_q;
        weight_scale_lo_o = weight_scale_lo_q;
        weight_scale_hi_o = weight_scale_hi_q;

        act_scale_ready_o    = act_scale_pulse;
        weight_scale_ready_o = weight_scale_pulse;
        mac_snapshot_valid_o = snapshot_valid_q;

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
            EXEC: begin
                unique case (op_q)
                    cve2_pkg::OP_MVE: begin
                        mv_en_o     = 1'b1;
                        mv_mode_o   = MV_EVEN;
                        scalar_we_o = 1'b1;
                    end
                    cve2_pkg::OP_MVO: begin
                        mv_en_o     = 1'b1;
                        mv_mode_o   = MV_ODD;
                        scalar_we_o = 1'b1;
                    end
                    cve2_pkg::OP_MV2: begin
                        mv_en_o     = 1'b1;
                        mv_mode_o   = MV_PAIR;
                        scalar_we_o = 1'b1;
                    end
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
                            accum_rd_tile_o = bias_tile;   // rs1[10:6]
                            accum_rd_row_o  = bias_row;    // rs1[5:3] (even -> reads pair n, n+1)
                            accum_rd_col_o  = bias_col;    // rs1[2:0]
                        end else begin
                            // phase 1: data is valid, write it back to the GPR
                            scalar_we_o = 1'b1;
                        end
                    end
                    default: ;
                endcase

                if (op_q == cve2_pkg::OP_VMAC) begin
                    mac_vrf_raddr_o = mac_vrf_addr;
                    mac_vrf_relem_o = elem_idx;
                    if (!mem_req_sent_q) begin
                        data_req_o  = 1'b1;
                        data_addr_o = base_q + (count_q << 2); 
                    end
                end
            end
            DONE: begin
                done_o = 1'b1;
            end
            default: ;
        endcase
    end

    genvar k;
    generate
        for (k = 0; k < TT; k++) begin : GEN_UNPACK_NIBBLES
            assign act_vector_o[k]    = act_packed[4*k +: 4];
            assign weight_vector_o[k] = weight_packed[4*k +: 4];
        end
    endgenerate

    // simulation debugging hooks
    always_ff @(posedge clk_i) begin
        if (rst_ni && (op_q == cve2_pkg::OP_VMAC) && data_rvalid_i) begin
            $display(
                "[%0t] [VMAC] vreg=v%0d elem=%0d flat=%0d vrf=%08x mem_addr=%08x weight=%08x mem_req=%0b mem_gnt=%0b mem_rvalid=%0b mac_en=%0b",
                $time, mac_vrf_addr, elem_idx, count_q, mac_vrf_rdata_i,
                base_q + (count_q << 2), data_rdata_i, mem_req_sent_q, data_gnt_i, data_rvalid_i, mac_en_o
            );
        end
    end

endmodule
