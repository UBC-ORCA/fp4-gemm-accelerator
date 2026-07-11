`timescale 1ns/1ps

module cve2_cf_unit (

    input  logic                     clk_i,
    input  logic                     rst_ni,

    input  logic                     req_valid_i,
    input  cve2_pkg::mac_op_e        cf_req_op_i,
    input  logic [31:0]              req_instr_i,
    input  logic [31:0]              req_rs1_i,
    input  logic [31:0]              req_rs2_i,

    output logic                     req_ready_o,
    output logic                     busy_o,
    output logic                     done_o,

    output logic                     scalar_we_o,
    output logic [4:0]               scalar_waddr_o,
    output logic [31:0]              scalar_wdata_o,

    output logic                     data_req_o,
    input  logic                     data_gnt_i,
    output logic [31:0]              data_addr_o,
    output logic                     data_we_o,
    output logic [3:0]               data_be_o,
    output logic [31:0]              data_wdata_o,

    input  logic [31:0]              data_rdata_i,
    input  logic                     data_rvalid_i,
    input  logic                     data_err_i
);

    // ============================================================
    // TILE CONTROL SIGNALS
    // ============================================================

    logic clear_i;
    logic mac_en_i, max_en_i, add_en_i;
    logic mv_en_i, ld2_en_i, st2_en_i;
    logic [1:0] mv_op_i;
    logic [2:0] add_row_i;

    assign add_row_i = req_instr_i[19:15][2:0];

    logic [4:0] mv_row  = req_instr_i[19:15];
    logic [4:0] mv_pair = req_instr_i[24:20];

    logic [31:0] mv_data;
    logic [31:0] st2_data;

    // ============================================================
    // FSM
    // ============================================================

    typedef enum logic [1:0] {
        S_IDLE,
        S_EXEC,
        S_DONE
    } state_e;

    state_e state, state_n;

    logic [31:0] instr_q;
    logic [31:0] rs1_q, rs2_q;
    cve2_pkg::mac_op_e op_q;

    // latch instruction
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state   <= S_IDLE;
            instr_q <= '0;
            rs1_q   <= '0;
            rs2_q   <= '0;
            op_q    <= cve2_pkg::OP_ZZ;
        end else begin
            state <= state_n;

            if (req_valid_i && req_ready_o) begin
                instr_q <= req_instr_i;
                rs1_q   <= req_rs1_i;
                rs2_q   <= req_rs2_i;
                op_q    <= cf_req_op_i;

// --- [stev] ---
`ifdef CPU_DEBUG
            $display("========== [CF] LATCH ==========");
            $display("req_instr_i = %08h", req_instr_i);
            $display("req_rs1_i   = %08h", req_rs1_i);
            $display("req_rs2_i   = %08h", req_rs2_i);
            $display("cf_req_op_i = %0d", cf_req_op_i);
            $display("rd          = %0d", req_instr_i[11:7]);
            $display("row(rs1)    = %0d", req_instr_i[19:15]);
            $display("pair(rs2)   = %0d", req_instr_i[24:20]);
        $display("mv_data        = 0x%08h", mv_data);
            $display("===============================");
`endif

// --- [end] ---


            end
        end
    end

    // ============================================================
    // STATE TRANSITION
    // ============================================================

    always_comb begin
        state_n = state;

        unique case (state)

            S_IDLE: if (req_valid_i) state_n = S_EXEC;

            S_EXEC: state_n = S_DONE;

            S_DONE: state_n = S_IDLE;

        endcase
    end

    // ============================================================
    // DEFAULTS
    // ============================================================

    always_comb begin

        clear_i  = 0;
        mac_en_i = 0;
        max_en_i = 0;
        add_en_i = 0;
        mv_en_i  = 0;
        mv_op_i  = 0;
        ld2_en_i = 0;
        st2_en_i = 0;

        data_req_o  = 0;
        data_we_o   = 0;
        data_addr_o = rs1_q;
        data_wdata_o= st2_data;
        data_be_o   = 4'b1111;

        //scalar_we_o   = 0;
        scalar_waddr_o= instr_q[11:7];
        scalar_wdata_o= mv_data;

// --- [stev] ---
`ifdef CPU_DEBUG
    if (state == S_EXEC &&
       ((op_q == cve2_pkg::OP_MVE) ||
        (op_q == cve2_pkg::OP_MVO) ||
        (op_q == cve2_pkg::OP_MV2))) begin

        $display("========== [cve2_cf_unit] CF EXEC ==========");
        $display("state          = %0d", state);
        $display("op_q           = %0d", op_q);
        $display("mv_en_i        = %0b", mv_en_i);
        $display("mv_op_i        = %0d", mv_op_i);
        $display("mv_row         = %0d", mv_row);
        $display("mv_pair        = %0d", mv_pair);
        $display("mv_data        = 0x%08h", mv_data);
        $display("scalar_wdata_o = 0x%08h", scalar_wdata_o);
        $display("scalar_we_o    = %0b", scalar_we_o);
        $display("rd             = %0d", scalar_waddr_o);
        $display("========== [cve2_cf_unit] ===================");
    end
`endif

// --- [end] ---

        req_ready_o = (state == S_IDLE);
        busy_o      = (state != S_IDLE);
        done_o      = (state == S_DONE);

        // ========================================================
        // EXEC PULSE (1-cycle true hardware execution)
        // ========================================================

        if (state == S_EXEC) begin

            unique case (op_q)

                cve2_pkg::OP_ZZ: clear_i = 1;

                cve2_pkg::OP_MAC: mac_en_i = 1;
                cve2_pkg::OP_MAX: max_en_i = 1;
                cve2_pkg::OP_ADD: add_en_i = 1;

                cve2_pkg::OP_MVE: begin mv_en_i = 1; mv_op_i = 0; end
                cve2_pkg::OP_MVO: begin mv_en_i = 1; mv_op_i = 1; end
                cve2_pkg::OP_MV2: begin mv_en_i = 1; mv_op_i = 2; end

                cve2_pkg::OP_LD2: begin
                    data_req_o = 1;
                    data_we_o  = 0;
                end

                cve2_pkg::OP_ST2: begin
                    data_req_o = 1;
                    data_we_o  = 1;
                end

                default: ;

            endcase
        end
    end

// --- [stev] ---
`ifdef CPU_DEBUG
always_comb begin
    if (mv_en_i) begin
        $display("\n========== [CF -> MAC] ==========");
        $display("clear_i     = %0b", clear_i);
        $display("mac_en_i    = %0b", mac_en_i);
        $display("mv_en_i     = %0b", mv_en_i);
        $display("mv_op_i     = %0d", mv_op_i);

        $display("mv_row      = %0d", mv_row);
        $display("mv_pair     = %0d", mv_pair);

        $display("instr_q     = %08h", instr_q);
        $display("req_instr_i = %08h", req_instr_i);

        $display("rs1_q       = %08h", rs1_q);
        $display("rs2_q       = %08h", rs2_q);
        $display("mv_data        = 0x%08h", mv_data);

        $display("=================================");
    end
end

always_comb begin
    if (state == S_EXEC) begin
        $display("\n========== [CF OUTPUTS] ==========");
        $display("mv_data        = %08h", mv_data);
        $display("scalar_wdata_o = %08h", scalar_wdata_o);
        $display("scalar_we_o    = %0b", scalar_we_o);
        $display("scalar_waddr_o = %0d", scalar_waddr_o);
        $display("done_o         = %0b", done_o);
        $display("busy_o         = %0b", busy_o);
        $display("==================================");
    end
end
`endif



// --- [end] ---


    // ============================================================
    // TILE INSTANTIATION (UNCHANGED)
    // ============================================================

    fp4_mac8x8_gen1 u_mac (
        .clk(clk_i),
        .rst_n(rst_ni),

        .clear_i(clear_i),
        .mac_en_i(mac_en_i),
        .max_en_i(max_en_i),
        .add_en_i(add_en_i),
        .add_row_i(add_row_i),

        .a_packed_i(rs1_q),
        .b_packed_i(rs2_q),

        .mv_en_i(mv_en_i),
        .mv_op_i(mv_op_i),
        .mv_row_i(mv_row),
        .mv_pair_i(mv_pair),
        .mv_data_o(mv_data),

        .rd_en_i(1'b0),
        .rd_addr_i('0),
        .rd_data_o(),

        .ld2_en_i(ld2_en_i),
        .st2_en_i(st2_en_i),
        .tile_mem_rs2_i(instr_q[24:20]),
        .ld2_data_i(data_rdata_i),
        .st2_data_o(st2_data)
    );

    // ============================================================
    // WRITEBACK
    // ============================================================

    assign scalar_we_o =
        (op_q == cve2_pkg::OP_MVE) ||
        (op_q == cve2_pkg::OP_MVO) ||
        (op_q == cve2_pkg::OP_MV2);


// --- [stev] ---
`ifdef CPU_DEBUG
always_ff @(posedge clk_i) begin
    if (mv_en_i) begin
        $display("========== [cve2_cf_unit] CF <- MAC ==========");
        $display("mv_en_i        = %0b", mv_en_i);
        $display("mv_op_i        = %0d", mv_op_i);
        $display("mv_data        = 0x%08h", mv_data);
        $display("scalar_wdata_o = 0x%08h", scalar_wdata_o);
        $display("scalar_we_o    = %0b", scalar_we_o);
        $display("========== [cve2_cf_unit] =====================");
    end
end

always_ff @(posedge clk_i) begin
    $display("\n========== [CF] CYCLE ==========");
    $display("state      = %0d -> %0d", state, state_n);
    $display("req_valid  = %0b", req_valid_i);
    $display("req_ready  = %0b", req_ready_o);

    $display("LIVE:");
    $display("  req_instr = %08h", req_instr_i);
    $display("  req_rs1   = %08h", req_rs1_i);
    $display("  req_rs2   = %08h", req_rs2_i);
        $display("mv_data        = 0x%08h", mv_data);

    $display("LATCHED:");
    $display("  instr_q   = %08h", instr_q);
    $display("  rs1_q     = %08h", rs1_q);
    $display("  rs2_q     = %08h", rs2_q);
    $display("  op_q      = %0d", op_q);

    $display("================================");
end
`endif

// --- [end] ---

endmodule
