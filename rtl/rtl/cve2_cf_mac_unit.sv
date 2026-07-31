`timescale 1ns/1ps

module cve2_cf_mac_unit
(
    output logic                        scalar_we_o,
    output logic [4:0]                  scalar_waddr_o,
    output logic [31:0]                 scalar_wdata_o,

    output logic                        data_req_o,
    input  logic                        data_gnt_i,
    output logic [31:0]                 data_addr_o,
    output logic                        data_we_o,
    output logic [3:0]                  data_be_o,
    output logic [31:0]                 data_wdata_o,

    input  logic [31:0]                 data_rdata_i,
    input  logic                        data_rvalid_i,
    input  logic                        data_err_i,

    input  logic                        clk_i,
    input  logic                        rst_ni,

    input  logic                        req_valid_i,
    input  cve2_pkg::mac_op_e           cf_req_op_i,
    input  logic [31:0]                 req_instr_i,
    input  logic [31:0]                 req_rs1_i,
    input  logic [31:0]                 req_rs2_i,

    output logic                        req_ready_o,
    output logic                        busy_o,
    output logic                        done_o,

    output logic [4:0]                  mac_vrf_raddr_o,
    output logic [4:0]                  mac_vrf_relem_o,
    input  logic [31:0]                 mac_vrf_rdata_i
);

    localparam int TT = 8;

    logic [4:0]  vs1;
    logic unsigned [11:0] imm12;
    logic [31:0] weight_base;
    logic [31:0] weight_addr;

    assign vs1         = req_instr_i[11:7];
    assign imm12       = req_instr_i[31:20];
    assign weight_base = req_rs1_i;
    assign weight_addr = weight_base + imm12;

    // BRAM_RD returns the accumulator read pair; MV ops return the raw tile.
    assign scalar_wdata_o = bram_rd_data;

    logic [4:0]  scalar_waddr;
    assign scalar_waddr = req_instr_i[11:7];


    logic signed [15:0] scale_tile_value [0:1]; 

    logic [31:0] act_scale_lo, act_scale_hi;
    logic [31:0] weight_scale_lo, weight_scale_hi;
    logic        act_scale_ready, weight_scale_ready;
    logic        snapshot_valid;

    logic signed [15:0] tile_snapshot [0:TT-1][0:TT-1];

    logic                snapshot_valid_q;
    logic                act_scale_valid_q;
    logic                weight_scale_valid_q;

    // Inbound staged registers (Pending context holding)
    logic [31:0]        ctx_act_scale_lo;
    logic [31:0]        ctx_act_scale_hi;
    logic [31:0]        ctx_weight_scale_lo;
    logic [31:0]        ctx_weight_scale_hi;
    logic signed [15:0] ctx_tile_snapshot [0:TT-1][0:TT-1];

    // Execution isolated registers (Active scale datapath context)

    logic                context_ready;
    logic                context_accept;
    logic                scale_busy;
    logic                scale_write;
    logic                scale_done;

    assign context_ready = snapshot_valid_q && act_scale_valid_q && weight_scale_valid_q;

    // Staging Context Logic
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            snapshot_valid_q     <= 1'b0;
            act_scale_valid_q    <= 1'b0;
            weight_scale_valid_q <= 1'b0;
            ctx_act_scale_lo     <= '0;
            ctx_act_scale_hi     <= '0;
            ctx_weight_scale_lo  <= '0;
            ctx_weight_scale_hi  <= '0;
            ctx_tile_snapshot    <= '{default: '{default: '0}};
        end else begin
            if (snapshot_valid && !snapshot_valid_q) begin
                snapshot_valid_q  <= 1'b1;
                ctx_tile_snapshot <= tile_snapshot;
            end
            if (act_scale_ready && !act_scale_valid_q) begin
                act_scale_valid_q <= 1'b1;
                ctx_act_scale_lo  <= act_scale_lo;
                ctx_act_scale_hi  <= act_scale_hi;
            end
            if (weight_scale_ready && !weight_scale_valid_q) begin
                weight_scale_valid_q <= 1'b1;
                ctx_weight_scale_lo  <= weight_scale_lo;
                ctx_weight_scale_hi  <= weight_scale_hi;
            end
            if (context_accept) begin
                snapshot_valid_q     <= 1'b0;
                act_scale_valid_q    <= 1'b0;
                weight_scale_valid_q <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------
    // Dynamic Tile Selection Logic
    // Current accumulator tile/bank selected by accBank(T).
    // Persists until the next accBank.
    //------------------------------------------------------------
    logic [4:0] current_tile_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)
            current_tile_q <= 5'b0;
        else if (req_valid_i && req_ready_o && (cf_req_op_i == cve2_pkg::OP_ACC_BANK))
            current_tile_q <= req_rs1_i[4:0];
    end

    // Frozen target bank for the in-flight fold (latched at context_accept)
    logic [4:0] scale_tile_q;

    // Scaler Private Isolated Context Latching
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            scale_tile_q          <= 5'b0;
        end else if (context_accept) begin
            scale_tile_q          <= current_tile_q;   // Freeze bank choice for this fold
        end
    end

    //------------------------------------------------------------
    // Accumulator Block RAM Interconnect Signals
    //------------------------------------------------------------
    logic        bram_rd_en;
    logic [4:0]  bram_rd_tile;
    logic [2:0]  bram_rd_row;
    logic [2:0]  bram_rd_col;
    logic [31:0] bram_rd_data; 

    logic        bram_wr_en;
    logic [4:0]  bram_wr_tile;
    logic [2:0]  bram_wr_row;
    logic [2:0]  bram_wr_col;
    logic [31:0] bram_wr_data;
    logic        bram_wr_pair;   // 1 = paired (scale) write, 0 = single-cell (bias) write

    logic        ctrl_accum_rd_en;
    logic [4:0]  ctrl_accum_rd_tile;
    logic [2:0]  ctrl_accum_rd_row;
    logic [2:0]  ctrl_accum_rd_col;

    logic        ctrl_accum_wr_en;
    logic [4:0]  ctrl_accum_wr_tile;
    logic [2:0]  ctrl_accum_wr_row;
    logic [2:0]  ctrl_accum_wr_col;
    logic [15:0] ctrl_accum_wr_data;

    logic [15:0] scale_accum_in  [0:1];
    logic [15:0] scale_accum_out [0:1];

//PATCH
// Wire declarations for pipeline scaling
    logic        scale_rd_en;
    logic [2:0]  scale_rd_col;
    logic [1:0]  scale_rd_row_group;
    logic [2:0]  scale_wr_col;
    logic [1:0]  scale_wr_row_group;
//PATCH_end

    always_comb begin
        if (scale_busy) begin
            //bram_rd_en   = 1'b1;
		bram_rd_en   = scale_rd_en;
            bram_rd_tile = scale_tile_q;
            bram_rd_row  = {scale_rd_row_group, 1'b0};
            bram_rd_col  = scale_rd_col;

            bram_wr_en   = scale_write;
            bram_wr_tile = scale_tile_q;
            bram_wr_row  = {scale_wr_row_group, 1'b0};
            bram_wr_col  = scale_wr_col;
            bram_wr_data = {scale_accum_out[1], scale_accum_out[0]};
            bram_wr_pair = 1'b1;   // Scale fold writes row pairs
        end else begin
            bram_rd_en   = ctrl_accum_rd_en;
            bram_rd_tile = ctrl_accum_rd_tile;
            bram_rd_row  = ctrl_accum_rd_row;
            bram_rd_col  = ctrl_accum_rd_col;

            bram_wr_en   = ctrl_accum_wr_en;
            bram_wr_tile = ctrl_accum_wr_tile;
            bram_wr_row  = ctrl_accum_wr_row;
            bram_wr_col  = ctrl_accum_wr_col;
            bram_wr_data = {16'b0, ctrl_accum_wr_data};
            bram_wr_pair = 1'b0;   // Single-cell write
        end
    end

    // Direct unpack from fast single-cycle BRAM read channel
    always_comb begin
        scale_accum_in[0] = bram_rd_data[15:0];  
        scale_accum_in[1] = bram_rd_data[31:16]; 
    end

    //------------------------------------------------------------
    // Submodule Instantiations
    //------------------------------------------------------------
    logic        mac_en;
    logic        clear;
    logic [3:0]  act_vector [0:TT-1];
    logic [3:0]  weight_vector [0:TT-1];
    
    logic        mem_req;
    logic [31:0] mem_addr;
    logic        mem_we;
    logic [3:0]  mem_be;
    logic [31:0] mem_wdata;

    assign data_req_o   = mem_req;
    assign data_addr_o  = mem_addr;
    assign data_we_o    = mem_we;
    assign data_be_o    = mem_be;
    assign data_wdata_o = mem_wdata;

    mac_controller #(
        .VL(32),
        .TT(TT)
    ) u_ctrl (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .req_valid_i          (req_valid_i),
        .cf_req_op_i          (cf_req_op_i),
        .rs1_i                (req_rs1_i),
        .rs2_i                (req_rs2_i),
        .mac_en_o             (mac_en),
        .clear_o              (clear),
        .vs1_i                (vs1),
        .weight_blk_i         (5'b0),
        .base_i               (weight_addr),
        .mac_vrf_raddr_o      (mac_vrf_raddr_o),
        .mac_vrf_relem_o      (mac_vrf_relem_o),
        .data_req_o           (mem_req),
        .data_gnt_i           (data_gnt_i),
        .data_addr_o          (mem_addr),
        .data_we_o            (mem_we),
        .data_be_o            (mem_be),
        .data_wdata_o         (mem_wdata),
        .data_rvalid_i        (data_rvalid_i),
        .data_rdata_i         (data_rdata_i),
        .data_err_i           (data_err_i),
        .act_vector_o         (act_vector),
        .weight_vector_o      (weight_vector),
        .mac_vrf_rdata_i      (mac_vrf_rdata_i),
        .scalar_waddr_i       (scalar_waddr),
        .scalar_waddr_o       (scalar_waddr_o),
        .scalar_we_o          (scalar_we_o),
        .act_scale_lo_o       (act_scale_lo),
        .act_scale_hi_o       (act_scale_hi),
        .weight_scale_lo_o    (weight_scale_lo),
        .weight_scale_hi_o    (weight_scale_hi),
        .act_scale_ready_o    (act_scale_ready),
        .weight_scale_ready_o (weight_scale_ready),
        .mac_snapshot_valid_o (snapshot_valid),
        .scale_busy_i         (scale_busy),
        .scale_done_i         (scale_done),
        .req_ready_o          (req_ready_o),
        //.busy_o               (busy_o),
        //.done_o               (done_o),
	.busy_o               (busy_main),
        .done_o               (done_main),
        .accum_rd_en_o        (ctrl_accum_rd_en),
        .accum_rd_tile_o      (ctrl_accum_rd_tile),
        .accum_rd_row_o       (ctrl_accum_rd_row),
        .accum_rd_col_o       (ctrl_accum_rd_col),
        .accum_rd_data_i      (bram_rd_data[15:0]), 
        .accum_wr_en_o        (ctrl_accum_wr_en),
        .accum_wr_tile_o      (ctrl_accum_wr_tile),
        .accum_wr_row_o       (ctrl_accum_wr_row),
        .accum_wr_col_o       (ctrl_accum_wr_col),
        .accum_wr_data_o      (ctrl_accum_wr_data)
    );

//[stev] - handshake

logic busy_main;
assign busy_o = busy_main || scale_busy;
logic done_main;
assign done_o = done_main || scale_done;

    mac_array #(
        .TT(TT)
    ) u_array (
        .clk                  (clk_i),
        .rst_n                (rst_ni),
        .mac_en_i             (mac_en),
        .clear_i              (clear),
        .act_i                (act_vector),
        .wt_i                 (weight_vector),
        .accum_o              (tile_snapshot)
    );

    mac_accum_bram #(
        .NTILES(32),
        .NROWS(8),
        .NCOLS(8)
    ) u_accum_bram (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .rd_en_i              (bram_rd_en),
        .rd_tile_i            (bram_rd_tile),
        .rd_row_i             (bram_rd_row),
        .rd_col_i             (bram_rd_col),
        .rd_data_o            (bram_rd_data),
        .wr_en_i              (bram_wr_en),
        .wr_tile_i            (bram_wr_tile),
        .wr_row_i             (bram_wr_row),
        .wr_col_i             (bram_wr_col),
        .wr_data_i            (bram_wr_data),
        .wr_pair_i            (bram_wr_pair)
    );

    mac_scale_fsm #(
        .NUM_GROUPS(32)
    ) u_scale_fsm (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .context_ready_i      (context_ready),
        .context_accept_o     (context_accept),
        .scale_busy_o         (scale_busy),
.scale_rd_en_o        (scale_rd_en),
        .scale_write_o        (scale_write),
        .scale_done_o         (scale_done),
        .scale_rd_col_o       (scale_rd_col),
        .scale_rd_row_group_o (scale_rd_row_group),
        .scale_wr_col_o       (scale_wr_col),
        .scale_wr_row_group_o (scale_wr_row_group)
    );

    always_comb begin
        // use write col: BRAM read is 1 cycle late, so it lines up with the accumulator
        scale_tile_value[0] = ctx_tile_snapshot[{scale_wr_row_group,1'b0}][scale_wr_col];
        scale_tile_value[1] = ctx_tile_snapshot[{scale_wr_row_group,1'b0}+1][scale_wr_col];
    end

    logic [7:0] scaleA [0:1];
    logic [7:0] scaleB [0:1];

    mac_scale_accum u_scale_accum0 (
        .tile_value(scale_tile_value[0]),
        .scaleA(scaleA[0]),
        .scaleB(scaleB[0]),
        .accumulator(scale_accum_in[0]),
        .accumulator_out(scale_accum_out[0])
    );

    mac_scale_accum u_scale_accum1 (
        .tile_value(scale_tile_value[1]),
        .scaleA(scaleA[1]),
        .scaleB(scaleB[1]),
        .accumulator(scale_accum_in[1]),
        .accumulator_out(scale_accum_out[1])
    );

    // Dynamic Scale Muxing Logic
    always_comb begin
        // scaleA by write row group (matches the late read)
        case(scale_wr_row_group)
            2'd0: begin
                scaleA[0] = ctx_act_scale_lo[7:0];
                scaleA[1] = ctx_act_scale_lo[15:8];
            end
            2'd1: begin
                scaleA[0] = ctx_act_scale_lo[23:16];
                scaleA[1] = ctx_act_scale_lo[31:24];
            end
            2'd2: begin
                scaleA[0] = ctx_act_scale_hi[7:0];
                scaleA[1] = ctx_act_scale_hi[15:8];
            end
            2'd3: begin
                scaleA[0] = ctx_act_scale_hi[23:16];
                scaleA[1] = ctx_act_scale_hi[31:24];
            end
        endcase

        // scaleB by write col (matches the late read)
        if (scale_wr_col < 4) begin
            scaleB[0] = ctx_weight_scale_lo[scale_wr_col*8 +: 8];
            scaleB[1] = ctx_weight_scale_lo[scale_wr_col*8 +: 8];
        end else begin
            scaleB[0] = ctx_weight_scale_hi[(scale_wr_col-4)*8 +: 8];
            scaleB[1] = ctx_weight_scale_hi[(scale_wr_col-4)*8 +: 8];
        end
    end

`ifdef BRAM_DEBUG
    // debug: log write col vs product col per fold write (should match after the fix)
    always_ff @(posedge clk_i) begin
        if (rst_ni && scale_busy && scale_write && scale_tile_q == 5'd0) begin
            $display("[CF_SCALE] wr_col=%0d prod_col(rd)=%0d | tile0=%4h scaleB0=%2h | accIn0=%4h -> accOut0=%4h",
                     scale_wr_col, scale_rd_col,
                     scale_tile_value[0], scaleB[0], scale_accum_in[0], scale_accum_out[0]);
        end
    end
`endif

endmodule
