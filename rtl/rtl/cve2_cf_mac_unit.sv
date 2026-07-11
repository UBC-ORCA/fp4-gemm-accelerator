`timescale 1ns/1ps
/******************************************************************************
 *
 * cve2_cf_mac_unit.sv
 *
 * Wrapper between the CVE2 custom-function interface and the MAC array.
 *
 * Responsibilities
 *   - Parse packed FP4 operands from rs1/rs2
 *   - Instantiate MAC controller
 *   - Instantiate MAC array
 *   - Connect controller outputs to MAC array inputs
 *
 ******************************************************************************/

module cve2_cf_mac_unit
(

// --- [stev] --- unused signals
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
    input  logic                     data_err_i,

// --- [end] ---



    input  logic                     clk_i,
    input  logic                     rst_ni,

    //------------------------------------------------------------
    // CVE2 request interface
    //------------------------------------------------------------

    input  logic                     req_valid_i,
    input  cve2_pkg::mac_op_e        cf_req_op_i,

    input  logic [31:0]              req_instr_i,
    input  logic [31:0]              req_rs1_i,
    input  logic [31:0]              req_rs2_i,

    //------------------------------------------------------------
    // Status
    //------------------------------------------------------------

    output logic                     req_ready_o,
    output logic                     busy_o,
    output logic                     done_o,

// --- [stev] ---
output logic [4:0] mac_vrf_raddr_o,
output  logic [2:0]   mac_vrf_relem_o,
input logic [31:0]   mac_vrf_rdata_i

// Weight memory interface
//output logic [31:0] weight_addr_o


// --- [end] ---
);


// --- [stev] ---
    //------------------------------------------------------------
    // Decoded VMAC instruction fields
    //------------------------------------------------------------

    // Temporary VMAC encoding:
    // [11:7]   = vs1 (vector source register)
    // [24:20]  = weight block index
    // req_rs1_i = base pointer
//TO BE RM
    logic [4:0] vs1;
logic [11:0] imm12;
    logic [31:0] weight_base;
logic [31:0] weight_addr;

    assign vs1        = req_instr_i[11:7];
assign imm12       = req_instr_i[31:20];
    assign weight_addr = weight_base + {{20{imm12[11]}}, imm12};
    assign weight_base = req_rs1_i;

// mv
    logic [4:0] mv_row  = req_instr_i[19:15];
    logic [4:0] mv_pair = req_instr_i[24:20];

	logic        mv_en;
	logic [1:0]  mv_mode;   // 0=even 1=odd 2=pair
	logic [2:0] mv_even_col_idx;
	logic [2:0] mv_odd_col_idx;
	logic [2:0] mv_row_idx;
logic [31:0] mv_data;
assign scalar_wdata_o = mv_data;

    logic [4:0] scalar_waddr;

        assign scalar_waddr = req_instr_i[11:7];


// --- [end] ---

// --- [stev] ---
//------------------------------------------------------------
// Controller -> Memory interface
//------------------------------------------------------------

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
// --- [end] ---


    localparam int TT = 8;

    //------------------------------------------------------------
    // Controller -> MAC array
    //------------------------------------------------------------

    logic mac_en;
    logic clear;

    //------------------------------------------------------------
    // Unpacked FP4 vectors
    //------------------------------------------------------------

    logic [3:0] act_vector [0:TT-1];
    logic [3:0] weight_vector [0:TT-1];

    //------------------------------------------------------------
    // MAC accumulator tile
    //------------------------------------------------------------

    logic signed [15:0] tile_accum [0:TT-1][0:TT-1];


    //------------------------------------------------------------
    // MAC controller
    //------------------------------------------------------------

    mac_controller u_ctrl
    (
        .clk_i(clk_i),
        .rst_ni(rst_ni),

        .req_valid_i(req_valid_i),
        .cf_req_op_i(cf_req_op_i),

        .rs1_i(req_rs1_i),
        .rs2_i(req_rs2_i),

        .mac_en_o(mac_en),
        .clear_o(clear),

// --- [stev] ---
//TEMP 
// New decoded VMAC fields
        .vs1_i        (vs1),
        //.weight_blk_i (weight_blk),
        .base_i       (weight_addr),

.mac_vrf_raddr_o(mac_vrf_raddr_o),
.mac_vrf_relem_o(mac_vrf_relem_o),

// [stev] - load weight
.data_req_o   (mem_req),
.data_gnt_i      (data_gnt_i),
.data_addr_o  (mem_addr),
.data_we_o    (mem_we),
.data_be_o    (mem_be),
.data_wdata_o (mem_wdata),

// Weight memory response
.data_rvalid_i   (data_rvalid_i),
.data_rdata_i    (data_rdata_i), //[stev] - may not need to pass into controller
.data_err_i      (data_err_i),

// post processed rsp data
.act_vector_o(act_vector),
.weight_vector_o(weight_vector),
.mac_vrf_rdata_i(mac_vrf_rdata_i),

//mv

      .mv_en_o(mv_en),
	 .mv_mode_o(mv_mode),   // 0=even 1=odd 2=pair
	.mv_even_col_idx_o(mv_even_col_idx),
	.mv_odd_col_idx_o(mv_odd_col_idx),
	 .mv_row_idx_o(mv_row_idx),
  	  .mv_row_i(mv_row),       // instruction rs1 field
    .mv_pair_i(mv_pair),      // instruction rs2 field

.scalar_waddr_i(scalar_waddr),
.scalar_waddr_o(scalar_waddr_o),
.scalar_we_o(scalar_we_o),

// --- [end] ---

        .req_ready_o(req_ready_o),
        .busy_o(busy_o),
        .done_o(done_o)
    );

    //------------------------------------------------------------
    // MAC array
    //------------------------------------------------------------

    mac_array
    #(
        .TT(TT)
    )
    u_array
    (
        .clk(clk_i),
        .rst_n(rst_ni),

        .mac_en_i(mac_en),
        .clear_i(clear),

        .act_i(act_vector),
        .wt_i(weight_vector),

        .accum_o(tile_accum),

//mv
      .mv_en_i(mv_en),
	 .mv_mode_i(mv_mode),   // 0=even 1=odd 2=pair
	.mv_even_col_idx_i(mv_even_col_idx),
	.mv_odd_col_idx_i(mv_odd_col_idx),
	 .mv_row_idx_i(mv_row_idx),
 .mv_data_o(mv_data)

    );

// --- [stev] ---
//------------------------------------------------------------
// Debug: MAC VRF read port
//------------------------------------------------------------
`ifdef VEC_DEBUG
always_ff @(posedge clk_i) begin
    if (rst_ni) begin
        $display("[MAC_VRF] addr=v%0d elem=%0d data=%08x",
                 mac_vrf_raddr_o,
                 mac_vrf_relem_o,
                 mac_vrf_rdata_i);
    end
end
//`endif

//`ifdef VEC_DEBUG
always_ff @(posedge clk_i) begin
    if (rst_ni) begin
        $display("[%0t] [MAC_VRF] raddr=v%0d relem=%0d rdata=%08x mac_en=%0b busy=%0b done=%0b",
                 $time,
                 mac_vrf_raddr_o,
                 mac_vrf_relem_o,
                 mac_vrf_rdata_i,
                 mac_en,
                 busy_o,
                 done_o);
    end
end
//`endif

always_ff @(posedge clk_i) begin
    if (rst_ni) begin
        $display("[%0t] [MAC_MEM] req=%0b gnt=%0b addr=%08x we=%0b be=%0h wdata=%08x rvalid=%0b rdata=%08x err=%0b busy=%0b done=%0b mac_en=%0b",
                 $time,
                 data_req_o,
                 data_gnt_i,
                 data_addr_o,
                 data_we_o,
                 data_be_o,
                 data_wdata_o,
                 data_rvalid_i,
                 data_rdata_i,
                 data_err_i,
                 busy_o,
                 done_o,
                 mac_en);
    end
end
`endif


//------------------------------------------------------------
// Debug: MAC Move / Scalar Writeback
//------------------------------------------------------------
`ifdef MV_DEBUG
always_ff @(posedge clk_i) begin
    if (rst_ni) begin
        $display("[%0t] [MAC_MV] op=%0d mv_en=%0b mode=%0d row=%0d even_col=%0d odd_col=%0d",
                 $time,
                 cf_req_op_i,
                 mv_en,
                 mv_mode,
                 mv_row_idx,
                 mv_even_col_idx,
                 mv_odd_col_idx);

        if (mv_en) begin
            $display("[%0t] [MAC_MV] DATA_OUT=%08x scalar_we=%0b scalar_waddr=x%0d",
                     $time,
                     mv_data,
                     scalar_we_o,
                     scalar_waddr_o);

            $display("[%0t] [MAC_MV] TILE[%0d][%0d]=%0d TILE[%0d][%0d]=%0d",
                     $time,
                     mv_row_idx,
                     mv_even_col_idx,
                     tile_accum[mv_row_idx][mv_even_col_idx],
                     mv_row_idx,
                     mv_odd_col_idx,
                     tile_accum[mv_row_idx][mv_odd_col_idx]);
        end
    end
end
`endif

// --- [end] ---

endmodule
