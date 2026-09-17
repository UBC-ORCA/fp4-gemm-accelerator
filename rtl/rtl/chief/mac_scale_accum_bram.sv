`timescale 1ns/1ps

module mac_scale_accum_bram #(
    parameter int BANKS  = 4,
    parameter int DEPTH  = 16,
    parameter int DATA_W = 16
)(
    input  logic clk_i,
    input  logic rst_ni,

    //------------------------------------------------
    // Read interface
    //------------------------------------------------
    input  logic [3:0]        rd_addr_i [0:BANKS-1],
    input  logic              rd_en_i   [0:BANKS-1],
    output logic [DATA_W-1:0] rd_data_o [0:BANKS-1],

    //------------------------------------------------
    // Write interface
    //------------------------------------------------
    input  logic [3:0]        wr_addr_i [0:BANKS-1],
    input  logic              wr_en_i   [0:BANKS-1],
    input  logic [DATA_W-1:0] wr_data_i [0:BANKS-1]
);

    //------------------------------------------------
    // Four BRAM banks
    //------------------------------------------------
    (* ram_style = "block" *)
    logic [DATA_W-1:0] bank_mem [0:BANKS-1][0:DEPTH-1];

    //------------------------------------------------
    // BRAM behavior
    //------------------------------------------------
    genvar b;
    generate
        for (b=0; b<BANKS; b++) begin : GEN_BANK
            always_ff @(posedge clk_i) begin
                if (!rst_ni) begin
                    rd_data_o[b] <= '0;
                end
                else begin
                    // Synchronous read
                    if (rd_en_i[b]) begin
                        rd_data_o[b] <= bank_mem[b][rd_addr_i[b]];
                    end
                    // Synchronous write
                    if (wr_en_i[b]) begin
                        bank_mem[b][wr_addr_i[b]] <= wr_data_i[b];
                    end
                end
            end
        end
    endgenerate

endmodule
