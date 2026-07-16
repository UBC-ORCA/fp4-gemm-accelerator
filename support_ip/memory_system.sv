// Memory system that should be initialized as two seperate two port bram instances: main memory, mmio memory
// cve2 instruction bus perspective
// * [0x00000000 + PROGRAM_SIZE] main program memory
// cve2 data bus perspective
// * [0x00000000 + PROGRAM_SIZE] main program memory
// * [0x10000000 + MMIO_SIZE] mmio memory
// axi lite perspective
// * [0x00000000 + MMIO_SIZE] mmio memory

module memory_system#(
  parameter PROGRAM_SIZE = 32'h00080000,
  parameter MMIO_SIZE =    32'h00001000,
  parameter MMIO_START =   32'h10000000
)(
  // CVE2 Instruction memory interface
  input  logic                         instr_req_o,
  output logic                         instr_gnt_i,
  output logic                         instr_rvalid_i,
  input  logic [31:0]                  instr_addr_o,
  output logic [31:0]                  instr_rdata_i,
  output logic                         instr_err_i,

  // CVE2 Data memory interface
  input  logic                         data_req_o,
  output logic                         data_gnt_i,
  output logic                         data_rvalid_i,
  input  logic                         data_we_o,
  input  logic [3:0]                   data_be_o,
  input  logic [31:0]                  data_addr_o,
  input  logic [31:0]                  data_wdata_o,
  output logic [31:0]                  data_rdata_i,
  output logic                         data_err_i,

  // AXI LITE read
  input logic [39:0]s_axi_araddr,
  input logic [2:0]s_axi_arprot,
  output logic s_axi_arready,
  input logic s_axi_arvalid,

  output logic [31:0]s_axi_rdata,
  output logic [1:0]s_axi_rresp,
  input logic s_axi_rready,
  output logic s_axi_rvalid,
  
  // AXI LITE write
  input logic [39:0]s_axi_awaddr,
  input logic [2:0]s_axi_awprot,
  output logic s_axi_awready,
  input logic s_axi_awvalid,

  input logic [31:0]s_axi_wdata,
  output logic s_axi_wready,
  input logic [3:0]s_axi_wstrb,
  input logic s_axi_wvalid,

  output logic [1:0]s_axi_bresp,
  input logic s_axi_bready,
  output logic s_axi_bvalid,

  // misc
  input logic clk,
  input logic cve2_resetn,
  input logic axi_resetn
);

(* ram_decomp="power" *) logic [31:0] program_ram[PROGRAM_SIZE/4];
(* ram_decomp="power" *) logic [31:0] mmio_ram[MMIO_SIZE/4];

logic        program_a_en;
logic [31:0] program_a_addr;
logic        program_a_resetn;
logic        program_a_rvalid;
logic [31:0] program_a_rdata;
logic        program_a_error;

logic        program_b_en;
logic        program_b_wen;
logic [31:0] program_b_addr;
logic [31:0] program_b_wdata;
logic [3:0]  program_b_ben;
logic        program_b_resetn;
logic        program_b_rvalid;
logic [31:0] program_b_rdata;
logic        program_b_error;

logic        mmio_a_en;
logic [31:0] mmio_a_addr;
logic        mmio_a_resetn;
logic        mmio_a_rvalid;
logic [31:0] mmio_a_rdata;
logic        mmio_a_error;

logic        mmio_b_en;
logic        mmio_b_wen;
logic [31:0] mmio_b_addr;
logic [31:0] mmio_b_wdata;
logic [3:0]  mmio_b_ben;
logic        mmio_b_resetn;
logic        mmio_b_rvalid;
logic [31:0] mmio_b_rdata;
logic        mmio_b_error;

integer ittvar;

// Memory initializers
integer i;
initial begin
  $readmemh("program.mem", program_ram);
end
initial begin
  for(i = 0; i < MMIO_SIZE/4; i++) begin
    mmio_ram[i] = '0;
  end
end

// CVE2 instruction bus
assign program_a_en = instr_req_o;
assign program_a_addr = instr_addr_o;
assign program_a_resetn = cve2_resetn; 

assign instr_gnt_i = '1;
assign instr_rvalid_i = program_a_rvalid;
assign instr_rdata_i = program_a_rdata;
assign instr_err_i = program_a_error;

// CVE2 data bus
assign program_b_en = data_req_o & (data_addr_o < MMIO_START);
assign program_b_wen = data_we_o;
assign program_b_addr = data_addr_o;
assign program_b_wdata = data_wdata_o;
assign program_b_ben = data_be_o;
assign program_b_resetn = cve2_resetn;

assign mmio_b_en = data_req_o && ((data_addr_o >= MMIO_START) & (data_addr_o < MMIO_START + MMIO_SIZE));
assign mmio_b_wen = data_we_o;
assign mmio_b_addr = data_addr_o - MMIO_START;
assign mmio_b_wdata = data_wdata_o;
assign mmio_b_ben = data_be_o;
assign mmio_b_resetn = cve2_resetn;

assign data_gnt_i = '1;
assign data_rvalid_i = program_b_rvalid | mmio_b_rvalid;
assign data_rdata_i = mmio_b_rvalid ? mmio_b_rdata : program_b_rdata;
assign data_err_i = mmio_b_rvalid ? mmio_b_error : program_b_error;

// AXI Lite
logic [31:0] axi_buffer_data;
logic axi_buffer_error;
logic axi_buffer_full;

always @(posedge clk) begin
  if(~axi_resetn) begin
    axi_buffer_full = '0;
  end else begin
    if(~axi_buffer_full) begin
      axi_buffer_full <= mmio_a_rvalid & ~(s_axi_rready & s_axi_rvalid);
    end else begin
      axi_buffer_full <= mmio_a_rvalid | ~(s_axi_rready & s_axi_rvalid);
    end
  end
  if(mmio_a_rvalid) begin
    axi_buffer_data <= mmio_a_rdata;
    axi_buffer_error <= mmio_a_error;
  end
end

assign mmio_a_en = s_axi_arready & s_axi_arvalid;
assign mmio_a_addr = s_axi_araddr & (MMIO_SIZE-1) & ~32'b11;
assign mmio_a_resetn = axi_resetn;

assign s_axi_arready = ~axi_buffer_full | (s_axi_rvalid & s_axi_rready);

assign s_axi_rdata = axi_buffer_full ? axi_buffer_data : mmio_a_rdata;
assign s_axi_rresp = (axi_buffer_full ? axi_buffer_error : mmio_a_error) ? 2'b10 : 2'b00;
assign s_axi_rvalid = axi_buffer_full | mmio_a_rvalid;

// program BRAM Behavior
logic program_a_error_comb;
logic program_b_error_comb;
always @(posedge clk) begin
  program_a_rvalid <= program_a_en;
  if(~program_a_resetn) begin
    program_a_rvalid <= '0;
  end else if(program_a_en) begin
    program_a_rdata <= program_ram[program_a_addr/4];
    program_a_error <= program_a_error_comb;
  end 

  program_b_rvalid <= program_b_en;
  if(~program_b_resetn) begin
    program_b_rvalid <= '0;
  end else if(program_b_en) begin
    program_b_rdata <= program_ram[program_b_addr/4];
    program_b_error <= program_b_error_comb;
    for(ittvar = 0; ittvar < 4; ittvar++) begin
      if(program_b_wen && program_b_ben[ittvar]) program_ram[program_b_addr/4][8*ittvar+:8] <= program_b_wdata[8*ittvar+:8];
    end 
  end 
end
assign program_a_error_comb = (program_a_addr % 4) | (program_a_addr >= PROGRAM_SIZE);
assign program_b_error_comb = (program_b_addr % 4) | (program_b_addr >= PROGRAM_SIZE);

// mmio BRAM Behavior
logic mmio_a_error_comb;
logic mmio_b_error_comb;
always @(posedge clk) begin
  mmio_a_rvalid <= mmio_a_en;
  if(~mmio_a_resetn) begin
    mmio_a_rvalid <= '0;
  end else if(mmio_a_en) begin
    mmio_a_rdata <= mmio_ram[mmio_a_addr/4];
    mmio_a_error <= mmio_a_error_comb;
  end 

  mmio_b_rvalid <= mmio_b_en;
  if(~mmio_b_resetn) begin
    mmio_b_rvalid <= '0;
  end else if(mmio_b_en) begin
    mmio_b_rdata <= mmio_ram[mmio_b_addr/4];
    mmio_b_error <= mmio_b_error_comb;
    for(ittvar = 0; ittvar < 4; ittvar++) begin
      if(mmio_b_wen && mmio_b_ben[ittvar]) mmio_ram[mmio_b_addr/4][8*ittvar+:8] <= mmio_b_wdata[8*ittvar+:8];
    end 
  end 
end
assign mmio_a_error_comb = (mmio_a_addr % 4) | (mmio_a_addr >= MMIO_SIZE);
assign mmio_b_error_comb = (mmio_b_addr % 4) | (mmio_b_addr >= MMIO_SIZE);

// AXI write logic (this should never be used)
logic write_addr_req;
logic write_data_req;
assign s_axi_awready = ~write_addr_req;
assign s_axi_wready = ~write_data_req;
assign s_axi_bresp = 2'b10; // SLVERR
assign s_axi_bvalid = write_addr_req & write_data_req;

always @(posedge clk) begin
  if(~axi_resetn) begin
    write_addr_req <= '0;
  end else begin
    if(!write_addr_req) begin
      write_addr_req <= s_axi_awready & s_axi_awvalid;  
    end else begin
      write_addr_req <= ~(s_axi_bready & s_axi_bready);
    end
  end
end
always @(posedge clk) begin
  if(~axi_resetn) begin
    write_data_req <= '0;
  end else begin
    if(!write_data_req) begin
      write_data_req <= s_axi_awready & s_axi_awvalid;  
    end else begin
      write_data_req <= ~(s_axi_bready & s_axi_bready);
    end
  end
end

endmodule