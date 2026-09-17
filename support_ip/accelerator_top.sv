module accelerator_top import cve2_pkg::*; #(
  parameter int unsigned MHPMCounterNum   = 10,
  parameter int unsigned MHPMCounterWidth = 40,
  parameter bit          RV32E            = 1'b0,
  parameter rv32m_e      RV32M            = RV32MSingleCycle,
  parameter bit          XInterface       = 1'b0,
  parameter BOOT_ADDR = 32'h00000000,
  parameter HART_ID = 32'h00000000
) ();

  // PS Interfaces
  logic pl_clk0;
  logic proc_reset_n;
  logic ps_reset_n;
  
  logic [39:0]s_axi_araddr;
  logic [2:0]s_axi_arprot;
  logic s_axi_arready;
  logic s_axi_arvalid;

  logic [31:0]s_axi_rdata;
  logic s_axi_rready;
  logic [1:0]s_axi_rresp;
  logic s_axi_rvalid;
  
  logic [39:0]s_axi_awaddr;
  logic [2:0]s_axi_awprot;
  logic s_axi_awready;
  logic s_axi_awvalid;
  
  logic [31:0]s_axi_wdata;
  logic s_axi_wready;
  logic [3:0]s_axi_wstrb;
  logic s_axi_wvalid;

  logic s_axi_bready;
  logic [1:0]s_axi_bresp;
  logic s_axi_bvalid;

  // Instruction memory interface
  logic                         instr_req_o;
  logic                         instr_gnt_i;
  logic                         instr_rvalid_i;
  logic [31:0]                  instr_addr_o;
  logic [31:0]                  instr_rdata_i;
  logic                         instr_err_i;

  // Data memory interface
  logic                         data_req_o;
  logic                         data_gnt_i;
  logic                         data_rvalid_i;
  logic                         data_we_o;
  logic [3:0]                   data_be_o;
  logic [31:0]                  data_addr_o;
  logic [31:0]                  data_wdata_o;
  logic [31:0]                  data_rdata_i;
  logic                         data_err_i;

  // X Interface (unused)
  logic                         x_issue_valid_o;
  logic                         x_issue_ready_i;
  x_issue_req_t                 x_issue_req_o;
  x_issue_resp_t                x_issue_resp_i;
  x_register_t                  x_register_o;
  logic                         x_commit_valid_o;
  x_commit_t                    x_commit_o;
  logic                         x_result_valid_i;
  logic                         x_result_ready_o;
  x_result_t                    x_result_i;

  // Interrupt inputs (unused)
  logic                         irq_software_i = '0;
  logic                         irq_timer_i= '0;
  logic                         irq_external_i = '0;
  logic [15:0]                  irq_fast_i = '0;
  logic                         irq_nm_i = '0;

  // Debug Interface (unused)
  logic                         debug_req_i;
  logic                         debug_halted_o;
  logic [31:0]                  dm_halt_addr_i;
  logic [31:0]                  dm_exception_addr_i;
  crash_dump_t                  crash_dump_o;

  ps ps_i(
    .pl_clk0(pl_clk0),
    .proc_reset_n(proc_reset_n),
    .ps_reset_n(ps_reset_n),
    .s_axi_araddr(s_axi_araddr),
    .s_axi_arprot(s_axi_arprot),
    .s_axi_arready(s_axi_arready),
    .s_axi_arvalid(s_axi_arvalid),
    .s_axi_awaddr(s_axi_awaddr),
    .s_axi_awprot(s_axi_awprot),
    .s_axi_awready(s_axi_awready),
    .s_axi_awvalid(s_axi_awvalid),
    .s_axi_bready(s_axi_bready),
    .s_axi_bresp(s_axi_bresp),
    .s_axi_bvalid(s_axi_bvalid),
    .s_axi_rdata(s_axi_rdata),
    .s_axi_rready(s_axi_rready),
    .s_axi_rresp(s_axi_rresp),
    .s_axi_rvalid(s_axi_rvalid),
    .s_axi_wdata(s_axi_wdata),
    .s_axi_wready(s_axi_wready),
    .s_axi_wstrb(s_axi_wstrb),
    .s_axi_wvalid(s_axi_wvalid)
  );

  cve2_core #(
    .PMPEnable        (1'b0),
    .PMPGranularity   (0),
    .PMPNumRegions    (4),
    .MHPMCounterNum   (MHPMCounterNum),
    .MHPMCounterWidth (MHPMCounterWidth),
    .RV32E            (RV32E),
    .RV32M            (RV32M),
    .RV32B            (RV32BNone),
    .DbgTriggerEn     (1'b1),
    .DbgHwBreakNum    (1),
    .XInterface       (XInterface)
  ) u_cve2_core (
    .clk_i(pl_clk0),
    .rst_ni(proc_reset_n),
    .test_en_i(0),

    .hart_id_i(HART_ID),
    .boot_addr_i(BOOT_ADDR),

    .instr_req_o,
    .instr_gnt_i,
    .instr_rvalid_i,
    .instr_addr_o,
    .instr_rdata_i,
    .instr_err_i,

    .data_req_o,
    .data_gnt_i,
    .data_rvalid_i,
    .data_we_o,
    .data_be_o,
    .data_addr_o,
    .data_wdata_o,
    .data_rdata_i,
    .data_err_i,

    // Core-V Extension Interface (CV-X-IF)
    .x_issue_valid_o,
    .x_issue_ready_i,
    .x_issue_req_o,
    .x_issue_resp_i,
    .x_register_o,
    .x_commit_valid_o,
    .x_commit_o,
    .x_result_valid_i,
    .x_result_ready_o,
    .x_result_i,

    .irq_software_i,
    .irq_timer_i,
    .irq_external_i,
    .irq_fast_i,
    .irq_nm_i,
    //.irq_pending_o(irq_pending),

    .debug_req_i,
    .debug_halted_o,
    .dm_halt_addr_i,
    .dm_exception_addr_i,
    .crash_dump_o,

    .fetch_enable_i(proc_reset_n)
    //.core_busy_o    (core_busy_d)
  );

  memory_system memory_system_i(
    .instr_req_o,
    .instr_gnt_i,
    .instr_rvalid_i,
    .instr_addr_o,
    .instr_rdata_i,
    .instr_err_i,

    .data_req_o,
    .data_gnt_i,
    .data_rvalid_i,
    .data_we_o,
    .data_be_o,
    .data_addr_o,
    .data_err_i,
    .data_wdata_o,
    .data_rdata_i,

    .s_axi_araddr,
    .s_axi_arprot,
    .s_axi_arready,
    .s_axi_arvalid,

    .s_axi_rdata,
    .s_axi_rresp,
    .s_axi_rready,
    .s_axi_rvalid,

    .s_axi_awaddr,
    .s_axi_awprot,
    .s_axi_awready,
    .s_axi_awvalid,

    .s_axi_wdata,
    .s_axi_wready,
    .s_axi_wstrb,
    .s_axi_wvalid,

    .s_axi_bresp,
    .s_axi_bready,
    .s_axi_bvalid,

    .clk(pl_clk0),
    .cve2_resetn(proc_reset_n),
    .axi_resetn(ps_reset_n)
  );                            

endmodule