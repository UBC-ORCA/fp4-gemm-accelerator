// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Minimal RVV-Lite A.1 vector unit for CVE2
//
// Patched for element-level synchronous-read VRF:
// - separate vector register file
// - SEW fixed to 32, LMUL fixed to 1
// - VLEN fixed (default 256b => 8 elements)
// - one-time VRF prime, then steady-state overlap
// - no artificial per-element VRF bubbles
// - dedicated v0 mask shadow read from VRF
// - hardware loop counter + post-increment address counter
//
// Supported subset:
// - vsetvli / vsetivli / vsetvl
// - vle32.v / vse32.v (unit stride only)

module cve2_vec_unit #(
  parameter int unsigned VLEN     = 1024,
  parameter int unsigned SEW      = 32,
  parameter int unsigned NUM_REGS = 32
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  // Request from ID stage
  input  logic        req_valid_i,
  input  logic [31:0] req_instr_i,
  input  logic [31:0] req_rs1_i,
  input  logic [31:0] req_rs2_i,
  output logic        req_ready_o,

  // Completion back to ID stage
  output logic        busy_o,
  output logic        done_o,
  output logic        scalar_we_o,
  output logic [4:0]  scalar_waddr_o,
  output logic [31:0] scalar_wdata_o,

  // Memory interface
  output logic        data_req_o,
  input  logic        data_gnt_i,
  output logic [31:0] data_addr_o,
  output logic        data_we_o,
  output logic [3:0]  data_be_o,
  output logic [31:0] data_wdata_o,
  input  logic [31:0] data_rdata_i,
  input  logic        data_rvalid_i,
  input  logic        data_err_i,

  // Reuse scalar EX hardware (Unused outputs grounded)
  output logic        ex_req_o,
  output logic        ex_is_mul_o,
  output logic [1:0]  ex_alu_op_o,
  output logic [31:0] ex_operand_a_o,
  output logic [31:0] ex_operand_b_o,
  input  logic [31:0] ex_result_i,
  input  logic        ex_valid_i,

  // MAC read interface
  input  logic [4:0]  mac_vrf_raddr_i,
  input  logic [4:0]  mac_vrf_relem_i,
  input  logic        mac_vrf_en_i,
  output logic [31:0] mac_vrf_rdata_o
);

  localparam int unsigned LANES   = VLEN / SEW;
  localparam int unsigned REG_AW  = (NUM_REGS > 1) ? $clog2(NUM_REGS) : 1;
  localparam int unsigned ELEM_AW = (LANES    > 1) ? $clog2(LANES)    : 1;

  // Ground unused EX pipeline interface outputs
  assign ex_req_o       = 1'b0;
  assign ex_is_mul_o    = 1'b0;
  assign ex_alu_op_o    = 2'd0;
  assign ex_operand_a_o = 32'd0;
  assign ex_operand_b_o = 32'd0;

  // ----------------------
  // Latched request
  // ----------------------
  logic        req_valid_q;
  logic [31:0] instr_q;
  logic [31:0] rs1_q, rs2_q;

  // Instruction fields
  wire [4:0] rd     = instr_q[11:7];
  wire [2:0] funct3 = instr_q[14:12];
  wire [4:0] rs1    = instr_q[19:15];
  wire [4:0] rs2    = instr_q[24:20];
  wire       vm     = instr_q[25];

  // vmem fields
  wire [1:0] mop = instr_q[27:26];
  wire       mew = instr_q[28];
  wire [2:0] nf  = instr_q[31:29];

  // Vector reg fields
  wire [4:0] vd  = rd;
  wire [4:0] vs3 = rd; // store data is encoded in rd field for STORE-FP

  function automatic logic vreg_idx_valid(input logic [4:0] idx);
    begin
      vreg_idx_valid = (idx < NUM_REGS);
    end
  endfunction

  // ----------------------
  // VRF interface
  // ----------------------
  logic [REG_AW-1:0]  raddr1, raddr2;
  logic [ELEM_AW-1:0] relem1, relem2;
  logic [SEW-1:0]     v_r1, v_r2;

  logic [ELEM_AW-1:0] mask_relem_a, mask_relem_b;
  logic               mask_bit_a, mask_bit_b_unused;

  logic               v_we;
  logic [REG_AW-1:0]  v_waddr;
  logic [ELEM_AW-1:0] v_welem;
  logic [SEW-1:0]     v_wdata;

  cve2_vec_regfile #(
    .VLEN(VLEN),
    .SEW(SEW),
    .NUM_REGS(NUM_REGS)
  ) i_vrf (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),

    .relem0_i     (mask_relem_a),
    .mask_bit_o   (mask_bit_a),
    .relem0b_i    (mask_relem_b),
    .mask_bit_b_o (mask_bit_b_unused),

    .raddr1_i     (raddr1),
    .relem1_i     (relem1),
    .rdata1_o     (v_r1),

    .raddr2_i     (raddr2),
    .relem2_i     (relem2),
    .rdata2_o     (v_r2),

    .we_i         (v_we),
    .waddr_i      (v_waddr),
    .welem_i      (v_welem),
    .wdata_i      (v_wdata)
  );

  // ----------------------
  // Decode
  // ----------------------
  typedef enum logic [1:0] {
    VOP_NONE,
    VOP_VSET,
    VOP_VLE32,
    VOP_VSE32
  } vop_e;

  vop_e vop_q, vop_d;

  function automatic logic instr_vregs_valid(input vop_e op, input logic [31:0] instr);
    logic [4:0] rd_i;
    begin
      rd_i = instr[11:7];

      unique case (op)
        VOP_VLE32,
        VOP_VSE32: instr_vregs_valid = vreg_idx_valid(rd_i);
        default:   instr_vregs_valid = 1'b1;
      endcase
    end
  endfunction

  localparam logic [6:0] OPC_OPV     = 7'h57;
  localparam logic [6:0] OPC_LOADFP  = 7'h07;
  localparam logic [6:0] OPC_STOREFP = 7'h27;

  localparam logic [2:0] F3_VSET = 3'b111;
  localparam logic [2:0] F3_W32  = 3'b110;

  function automatic vop_e decode_vop(input logic [31:0] instr);
    logic [6:0] op;
    logic [2:0] f3;
    begin
      op = instr[6:0];
      f3 = instr[14:12];

      if (op == OPC_OPV     && f3 == F3_VSET) return VOP_VSET;
      if (op == OPC_LOADFP  && f3 == F3_W32)  return VOP_VLE32;
      if (op == OPC_STOREFP && f3 == F3_W32)  return VOP_VSE32;

      return VOP_NONE;
    end
  endfunction

  function automatic logic vtype_supported(input logic [10:0] vtypei);
    logic [2:0] vlmul;
    logic [2:0] vsew;
    logic       vta;
    begin
      vlmul = vtypei[2:0];
      vsew  = vtypei[5:3];
      vta   = vtypei[6];
      vtype_supported = (vlmul == 3'b000) && (vsew == 3'b010) && (vta == 1'b1);
    end
  endfunction

  function automatic logic [$clog2(LANES+1)-1:0] compute_vl(input logic [31:0] avl);
    logic [$clog2(LANES+1)-1:0] tmp;
    begin
      if (avl > LANES) tmp = LANES[$clog2(LANES+1)-1:0];
      else             tmp = avl[$clog2(LANES+1)-1:0];
      return tmp;
    end
  endfunction

  function automatic logic is_unit_stride;
    is_unit_stride = (mop == 2'b00) && (mew == 1'b0) && (nf == 3'b000);
  endfunction

  // ----------------------
  // State
  // ----------------------
  typedef enum logic [2:0] {
    S_IDLE,
    S_VRF_READ,
    S_ALU,
    S_MEM_REQ,
    S_MEM_WAIT
  } state_e;

  state_e state_q, state_d;

  logic [$clog2(LANES+1)-1:0] vl_q, vl_d;
  logic [ELEM_AW-1:0]         idx_q, idx_d;
  logic [31:0]                mem_addr_q, mem_addr_d;
  logic                       done_d;

  logic        do_elem;
  logic [31:0] vset_avl;
  logic [10:0] vset_vtypei;

  logic               last_elem;
  logic [ELEM_AW-1:0] vrf_elem_idx;

  assign req_ready_o = ~req_valid_q;
  assign busy_o      = req_valid_q;
  assign done_o      = done_d;

  assign last_elem   = (idx_q == (vl_q[$bits(idx_q)-1:0] - 1'b1));

  // Decide element index for synchronous VRF read
  always_comb begin
    vrf_elem_idx = idx_q;

    unique case (state_q)
      // Overlap next store-data fetch when current store completes
      S_MEM_WAIT: begin
        if ((vop_q == VOP_VSE32) && data_rvalid_i && !data_err_i && !last_elem) begin
          vrf_elem_idx = idx_q + 1'b1;
        end
      end

      // Overlap next store-data fetch when masked store is skipped
      S_MEM_REQ: begin
        if ((vop_q == VOP_VSE32) && !do_elem && !last_elem) begin
          vrf_elem_idx = idx_q + 1'b1;
        end
      end

      default: begin
      end
    endcase
  end

  // VRF address assignment
  always_comb begin
    raddr1       = '0;
    raddr2       = '0;
    relem1       = vrf_elem_idx;
    relem2       = vrf_elem_idx;
    mask_relem_a = idx_q;
    mask_relem_b = '0;

    if (vop_q == VOP_VSE32) begin
      raddr1 = vs3[REG_AW-1:0];
    end

    // MAC has priority on VRF read port 1
    if (mac_vrf_en_i) begin
      raddr1 = mac_vrf_raddr_i[REG_AW-1:0];
      relem1 = mac_vrf_relem_i;
    end
  end

  assign mac_vrf_rdata_o = v_r1; 

  // ----------------------
  // Sequential
  // ----------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      req_valid_q <= 1'b0;
      instr_q     <= 32'd0;
      rs1_q       <= 32'd0;
      rs2_q       <= 32'd0;
      vop_q       <= VOP_NONE;

      state_q     <= S_IDLE;
      vl_q        <= LANES[$clog2(LANES+1)-1:0];
      idx_q       <= '0;
      mem_addr_q  <= 32'd0;
    end else begin
      state_q     <= state_d;
      vl_q        <= vl_d;
      idx_q       <= idx_d;
      mem_addr_q  <= mem_addr_d;
      vop_q       <= vop_d;

      if (req_valid_i && req_ready_o) begin
        req_valid_q <= 1'b1;
        instr_q     <= req_instr_i;
        rs1_q       <= req_rs1_i;
        rs2_q       <= req_rs2_i;
      end

      if (done_d) begin
        req_valid_q <= 1'b0;
      end
    end
  end

  // ----------------------
  // Main combinational control
  // ----------------------
  always_comb begin
    scalar_we_o    = 1'b0;
    scalar_waddr_o = 5'd0;
    scalar_wdata_o = 32'd0;

    data_req_o     = 1'b0;
    data_addr_o    = 32'd0;
    data_we_o      = 1'b0;
    data_be_o      = 4'b1111;
    data_wdata_o   = 32'd0;

    v_we           = 1'b0;
    v_waddr        = '0;
    v_welem        = '0;
    v_wdata        = '0;

    state_d        = state_q;
    vl_d           = vl_q;
    idx_d          = idx_q;
    mem_addr_d     = mem_addr_q;
    vop_d          = vop_q;
    done_d         = 1'b0;

    do_elem        = vm ? 1'b1 : mask_bit_a;
    vset_avl       = 32'd0;
    vset_vtypei    = 11'd0;

    if (req_valid_i && req_ready_o) begin
      vop_d      = decode_vop(req_instr_i);
      idx_d      = '0;
      mem_addr_d = req_rs1_i;

      if (!instr_vregs_valid(decode_vop(req_instr_i), req_instr_i)) begin
        state_d = S_ALU;
        vop_d   = VOP_NONE;
      end else begin
        unique case (decode_vop(req_instr_i))
          VOP_VLE32: state_d = S_MEM_REQ;
          VOP_VSE32: state_d = S_VRF_READ;

          VOP_VSET,
          VOP_NONE: state_d = S_ALU;

          default: state_d = S_ALU;
        endcase
      end
    end

    unique case (state_q)
      S_IDLE: begin
      end

      // One-time prime cycle for synchronous VRF
      S_VRF_READ: begin
        if (vl_q == '0) begin
          done_d  = 1'b1;
          state_d = S_IDLE;
        end else begin
          unique case (vop_q)
            VOP_VSE32: state_d = S_MEM_REQ;
            default:   state_d = S_ALU;
          endcase
        end
      end

      S_ALU: begin
        if ((vop_q != VOP_VSET) && (vl_q == '0)) begin
          done_d  = 1'b1;
          state_d = S_IDLE;
        end else begin
          unique case (vop_q)
            VOP_VSET: begin
              vset_avl    = rs1_q;
              vset_vtypei = instr_q[30:20];

              if (instr_q[31]) begin
                vset_avl    = {27'd0, instr_q[19:15]};
                vset_vtypei = {1'b0, instr_q[29:20]};
              end else if (instr_q[25] && (instr_q[31:26] == 6'b000000)) begin
                vset_avl    = rs2_q;
                vset_vtypei = 11'h000;
              end

              if (!instr_q[31] && !(instr_q[25] && (instr_q[31:26] == 6'b000000))) begin
                if (!vtype_supported(vset_vtypei)) vl_d = vl_q;
                else                               vl_d = compute_vl(vset_avl);
              end else begin
                vl_d = compute_vl(vset_avl);
              end

              scalar_we_o    = (rd != 5'd0);
              scalar_waddr_o = rd;
              scalar_wdata_o = {{(32-$clog2(LANES+1)){1'b0}}, vl_d};
              done_d         = 1'b1;
              state_d        = S_IDLE;
            end

            VOP_VLE32,
            VOP_VSE32: begin
              state_d = S_MEM_REQ;
            end

            default: begin
              done_d  = 1'b1;
              state_d = S_IDLE;
            end
          endcase
        end
      end

      S_MEM_REQ: begin
        if (vl_q == '0) begin
          done_d  = 1'b1;
          state_d = S_IDLE;
        end else if (!is_unit_stride()) begin
          done_d  = 1'b1;
          state_d = S_IDLE;
        end else if (!do_elem) begin
          mem_addr_d = mem_addr_q + 32'd4;

          if (last_elem) begin
            done_d  = 1'b1;
            state_d = S_IDLE;
          end else begin
            idx_d   = idx_q + 1'b1;
            state_d = S_MEM_REQ;
          end
        end else begin
          data_req_o  = 1'b1;
          data_addr_o = mem_addr_q;
          data_be_o   = 4'b1111;

          if (vop_q == VOP_VLE32) begin
            data_we_o    = 1'b0;
            data_wdata_o = 32'd0;
          end else begin
            data_we_o    = 1'b1;
            data_wdata_o = v_r1;
          end

          if (data_gnt_i) begin
            state_d = S_MEM_WAIT;
          end
        end
      end

      S_MEM_WAIT: begin
        if (vl_q == '0) begin
          done_d  = 1'b1;
          state_d = S_IDLE;
        end else if (data_rvalid_i) begin
          if (data_err_i) begin
            done_d  = 1'b1;
            state_d = S_IDLE;
          end else begin
            if (vop_q == VOP_VLE32) begin
              v_we    = 1'b1;
              v_waddr = vd[REG_AW-1:0];
              v_welem = idx_q;
              v_wdata = data_rdata_i;
            end

            mem_addr_d = mem_addr_q + 32'd4;

            if (last_elem) begin
              done_d  = 1'b1;
              state_d = S_IDLE;
            end else begin
              idx_d   = idx_q + 1'b1;
              state_d = S_MEM_REQ;
            end
          end
        end
      end

      default: begin
        state_d = S_IDLE;
      end
    endcase
  end

endmodule
