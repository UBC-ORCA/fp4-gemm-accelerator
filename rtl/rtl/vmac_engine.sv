// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0

module vmac_engine #(
    parameter int VL = 32,
    parameter int TT = 8
) (
    input  logic clk_i,
    input  logic rst_ni,

    //----------------------------------------
    // Control
    //----------------------------------------
    input  logic start_i,
    output logic busy_o,
    output logic done_o,

    //----------------------------------------
    // Context
    //----------------------------------------
    input  logic [4:0]  vs1_i,
    input  logic [4:0]  weight_blk_i,
    input  logic [31:0] base_i,

    //----------------------------------------
    // VRF
    //----------------------------------------
    output logic [4:0]  mac_vrf_raddr_o,
    output logic [4:0]  mac_vrf_relem_o,
    input  logic [31:0] mac_vrf_rdata_i,

    //----------------------------------------
    // LSU
    //----------------------------------------
    output logic        data_req_o,
    input  logic        data_gnt_i,

    output logic [31:0] data_addr_o,

    input  logic        data_rvalid_i,
    input  logic [31:0] data_rdata_i,

    //----------------------------------------
    // MAC array
    //----------------------------------------
    output logic        clear_o,
    output logic        mac_en_o,

    output logic [3:0]  act_vector_o    [0:TT-1],
    output logic [3:0]  weight_vector_o [0:TT-1],

    //----------------------------------------
    // Snapshot
    //----------------------------------------
    output logic        snapshot_valid_o
);

    typedef enum logic [1:0] {
        IDLE,
        CLEAR,
        EXEC,
        DONE
    } state_e;
    state_e state_q, state_d;

    logic [4:0]  vs1_q;
    logic [4:0]  weight_blk_q;
    logic [31:0] base_q;

    logic mem_req_sent_q;
    logic mem_req_sent_d;

    localparam int CNT_W = $clog2(VL);
    logic [CNT_W-1:0] count_q;
    logic [CNT_W-1:0] count_d;

    logic vmac_last_q;

    logic [31:0] act_packed;
    logic [31:0] weight_packed;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q          <= IDLE;
            vs1_q            <= '0;
            weight_blk_q     <= '0;
            base_q           <= '0;
            count_q          <= '0;
            mem_req_sent_q   <= 1'b0;
            vmac_last_q      <= 1'b0;
            snapshot_valid_o <= 1'b0;
        end else begin
            state_q        <= state_d;
            count_q        <= count_d;
            mem_req_sent_q <= mem_req_sent_d;

            if (start_i) begin
                vs1_q        <= vs1_i;
                weight_blk_q <= weight_blk_i;
                base_q       <= base_i;
            end

            vmac_last_q      <= data_rvalid_i && (count_q == (VL - 1));
            snapshot_valid_o <= vmac_last_q;
        end
    end
always_ff @(posedge clk_i) begin
    if (start_i)
        $display("%0t VMAC START", $time);

    if (done_o)
        $display("%0t VMAC DONE", $time);

    if (snapshot_valid_o)
        $display("%0t SNAPSHOT_VALID", $time);
end
    always_comb begin
        state_d        = state_q;
        count_d        = count_q;
        mem_req_sent_d = mem_req_sent_q;

        case (state_q)
            IDLE: begin
                count_d        = '0;
                mem_req_sent_d = 1'b0;
                if (start_i) begin
                    state_d = CLEAR;
                end
            end

            CLEAR: begin
                state_d = EXEC;
            end

            EXEC: begin
                if (!mem_req_sent_q) begin
                    if (data_gnt_i) begin
                        mem_req_sent_d = 1'b1;
                    end
                end else begin
                    if (data_rvalid_i) begin
                        mem_req_sent_d = 1'b0;
                        if (count_q == (VL - 1)) begin
                            state_d = DONE;
                            count_d = '0;
                        end else begin
                            count_d = count_q + 1'b1;
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

    // Control/Status Outputs
    assign busy_o = (state_q != IDLE);
    assign done_o = (state_q == DONE);

    assign clear_o  = (state_q == CLEAR);
    assign mac_en_o = (state_q == EXEC) && data_rvalid_i;

    // VRF Interface
    assign mac_vrf_raddr_o = vs1_q;
    assign mac_vrf_relem_o = count_q;

    // Memory Interface
    assign data_req_o  = (state_q == EXEC) && !mem_req_sent_q;
    assign data_addr_o = base_q + (count_q << 2);

    // Unpack Nibbles
    assign act_packed    = mac_vrf_rdata_i;
    assign weight_packed = data_rdata_i;

    genvar k;
    generate
        for (k = 0; k < TT; k++) begin : GEN_UNPACK_NIBBLES
            assign act_vector_o[k]    = act_packed[4*k +: 4];
            assign weight_vector_o[k] = weight_packed[4*k +: 4];
        end
    endgenerate

`ifdef MAC_DEBUG
    always_ff @(posedge clk_i) begin
        if (rst_ni && (state_q == EXEC) && data_rvalid_i) begin
            $display(
                "[%0t] [VMAC] vreg=v%0d elem=%0d vrf=%08x mem_addr=%08x weight=%08x mem_req=%0b mem_gnt=%0b mem_rvalid=%0b mac_en=%0b",
                $time, vs1_q, count_q, mac_vrf_rdata_i,
                base_q + (count_q << 2), data_rdata_i, mem_req_sent_q, data_gnt_i, data_rvalid_i, mac_en_o
            );
        end
    end
`endif

endmodule
