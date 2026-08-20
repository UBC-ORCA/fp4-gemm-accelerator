// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Minimal vector register file for RVV-Lite A.1 on CVE2.
// Separate from scalar RF.
//
// 2R1W BRAM-oriented version:
// - 32b, synchronous reads, dual read ports
// - 32b, synchronous writes
// - tiny separate v0 mask shadow with two bit-read taps so masked execution
//   does not consume either data-read port
// - element-level (32-bit) interface to avoid carrying 256-bit buses

module cve2_vec_regfile #(
  parameter int unsigned VLEN     = 256,
  parameter int unsigned SEW      = 32,
  parameter int unsigned NUM_REGS = 32
) (
  input  logic                         clk_i,
  input  logic                         rst_ni,

  // Two vector data read ports.
  input  logic [$clog2(NUM_REGS)-1:0]  raddr1_i,
  input  logic [$clog2(VLEN/SEW)-1:0]  relem1_i,
  output logic [SEW-1:0]               rdata1_o,

  // Write port (synchronous), one 32-bit element at a time.
  input  logic                         we_i,
  input  logic [$clog2(NUM_REGS)-1:0]  waddr_i,
  input  logic [$clog2(VLEN/SEW)-1:0]  welem_i,
  input  logic [SEW-1:0]               wdata_i
);

  localparam int unsigned LANES   = VLEN / SEW;
  localparam int unsigned REG_AW  = (NUM_REGS > 1) ? $clog2(NUM_REGS) : 1;
  localparam int unsigned ELEM_AW = (LANES    > 1) ? $clog2(LANES)    : 1;
  localparam int unsigned DEPTH   = NUM_REGS * LANES;
  localparam int unsigned ADDR_W  = REG_AW + ELEM_AW;

  // Vector Register File memory.
  // BRAM hint for Xilinx inference. This is a strong suggestion, not a guarantee.
  (* ram_style = "block" *)
  logic [SEW-1:0] vrf_mem [0:DEPTH-1];

  logic [ADDR_W-1:0] raddr1_flat;
  logic [ADDR_W-1:0] waddr_flat;

  assign raddr1_flat = {raddr1_i, relem1_i};
  assign waddr_flat  = {waddr_i,  welem_i};

  // Synchronous data read (2 ports)
  // Don't bother to forward data on read-under-write case (stall instead)
  always_ff @(posedge clk_i) begin
    rdata1_o <= vrf_mem[raddr1_flat];
  end

  // Synchronous element write (1 port)
  always_ff @(posedge clk_i) begin
    if (we_i) begin
      vrf_mem[waddr_flat] <= wdata_i;
    end
  end


`ifdef VEC_DEBUG
  always_ff @(posedge clk_i) begin
    if (rst_ni && we_i) begin
      $display("[VRF] w v%0d[%0d] <= %08x", waddr_i, welem_i, wdata_i);
    end
  end
`endif

endmodule
