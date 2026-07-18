// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Structured coordinate-addressed Accumulator Tile BRAM module.
// Fixed to completely eliminate asynchronous lookups, forcing true BRAM inference.

module mac_accum_bram (
    input  logic        clk_i,
    input  logic        rst_ni,

    //----------------------------------
    // Read Port (Tile Structural Coordinates)
    //----------------------------------
    input  logic        rd_en_i,
    input  logic [4:0]  rd_tile_i,
    input  logic [2:0]  rd_row_i,    
    input  logic [2:0]  rd_col_i,
    output logic [31:0] rd_data_o, 

    //----------------------------------
    // Write Port (Tile Structural Coordinates)
    //----------------------------------
    input  logic        wr_en_i,
    input  logic [4:0]  wr_tile_i,
    input  logic [2:0]  wr_row_i,    
    input  logic [2:0]  wr_col_i,
    input  logic [31:0] wr_data_i,   
    input  logic        wr_pair_i    // 1 = paired write (scale), 0 = single-cell (bias)
);

    // Architectural Dimension Parameters
    localparam int unsigned NTILES = 32;
    localparam int unsigned TT     = 8;
    localparam int unsigned DEPTH  = NTILES * (TT / 2) * TT; // 32 * 4 * 8 = 1024 physical entries
    localparam int unsigned ADDR_W = 10;

    // Accumulator Tile Memory Array (Maps natively onto 1 RAMB36)
    (* ram_style = "block" *)
    logic [31:0] accum_mem [0:DEPTH-1];

    //----------------------------------
    // Address Transformation Logic
    //----------------------------------
    logic [ADDR_W-1:0] rd_addr_flat;
    logic [ADDR_W-1:0] wr_addr_flat;

    assign rd_addr_flat = (ADDR_W'(rd_tile_i) << 5) + (ADDR_W'(rd_row_i[2:1]) << 3) + ADDR_W'(rd_col_i);
    assign wr_addr_flat = (ADDR_W'(wr_tile_i) << 5) + (ADDR_W'(wr_row_i[2:1]) << 3) + ADDR_W'(wr_col_i);

    //----------------------------------
    // Read Path (Fully Synchronous)
    //----------------------------------
    logic [ADDR_W-1:0] rd_addr_q;
    logic              rd_en_q;

    always_ff @(posedge clk_i) begin
        if (rd_en_i) begin
            assert(rd_row_i[0] == 1'b0) else
                $error("[BRAM_ACCUM_ERROR] Read architectural row must be even for paired layout, got %0d", rd_row_i);
            
            rd_addr_q <= rd_addr_flat;
        end
    end

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            rd_data_o <= '0;
        end else if (rd_en_q) begin
            rd_data_o <= accum_mem[rd_addr_q];
        end
    end

    // Tracking registers for legacy diagnostic debug blocks
    logic [4:0] rd_tile_q;
    logic [2:0] rd_row_q;
    logic [2:0] rd_col_q;

    always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
            rd_en_q   <= 1'b0;
            rd_tile_q <= '0;
            rd_row_q  <= '0;
            rd_col_q  <= '0;
        end else begin
            rd_en_q   <= rd_en_i;
            rd_tile_q <= rd_tile_i;
            rd_row_q  <= rd_row_i;
            rd_col_q  <= rd_col_i;
        end
    end

    //----------------------------------
    // Write Path (Masked Byte/Half-word Write Enables)
    //----------------------------------
    logic [1:0]  wr_mask;
    logic [15:0] data_lower;
    logic [15:0] data_upper;

    always_comb begin
        if (wr_pair_i) begin
            // Paired Mode: update both halves simultaneously
            wr_mask    = 2'b11;
            data_lower = wr_data_i[15:0];
            data_upper = wr_data_i[31:16];
        end else begin
            // Single Cell (Bias) Mode: Mask based on target row parity
            if (wr_row_i[0] == 1'b0) begin
                wr_mask    = 2'b01; // Write even row (lower 16 bits), block upper
                data_lower = wr_data_i[15:0];
                data_upper = 16'h0000; // Don't care
            end else begin
                wr_mask    = 2'b10; // Write odd row (upper 16 bits), block lower
                data_lower = 16'h0000; // Don't care
                data_upper = wr_data_i[15:0]; // Single-cell entry payload is in [15:0]
            end
        end
    end

    // Standard Xilinx Byte-Wide Write Enable Inference Block
    always_ff @(posedge clk_i) begin
        if (wr_en_i) begin
            if (wr_pair_i) begin
                assert(wr_row_i[0] == 1'b0) else
                    $error("[BRAM_ACCUM_ERROR] Paired write row must be even, got %0d", wr_row_i);
            end
            
            if (wr_mask[0]) accum_mem[wr_addr_flat][15:0]  <= data_lower;
            if (wr_mask[1]) accum_mem[wr_addr_flat][31:16] <= data_upper;
        end
    end

    //----------------------------------
    // Simulation Debug Dumps
    //----------------------------------
`ifdef BRAM_DEBUG
    always_ff @(posedge clk_i) begin
        if (rst_ni) begin
            if (rd_en_q) begin
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY READ COMPLETE:", $time);
                $display("[BRAM_ACCUM_DEBUG]   Coordinates -> Tile=%2d | Rows=%1d,%1d | Col=%1d", rd_tile_q, rd_row_q, rd_row_q+1, rd_col_q);
                $display("[BRAM_ACCUM_DEBUG]   Payload     -> Out Data=32'h%h", rd_data_o);
            end

            if (wr_en_i) begin
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY WRITE TRANSACTION COMMITTED:", $time);
                $display("[BRAM_ACCUM_DEBUG]   Coordinates -> Tile=%2d | Targeted Row=%1d | Col=%1d", wr_tile_i, wr_row_i, wr_col_i);
                $display("[BRAM_ACCUM_DEBUG]   Write Mode  -> Paired=%0b | Mask=2'b%b", wr_pair_i, wr_mask);
            end
        end
    end
`endif

endmodule
