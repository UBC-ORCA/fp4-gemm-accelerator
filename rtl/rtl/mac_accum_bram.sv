// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Structured coordinate-addressed Accumulator Tile BRAM module.
// True Xilinx Vivado dual-bank RAMB18 inference layout using split 
// row-parity physical mapping to guarantee artifact-free synthesis.

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

    // Architectural Dimension Parameters (1024 words per memory bank)
    localparam int unsigned NTILES = 32;
    localparam int unsigned DEPTH  = 32 * 4 * 8; // 1024 words
    localparam int unsigned ADDR_W = 10;

    // Split Memory Banks (Maps flawlessly onto Xilinx Block RAM Primitives)
    (* ram_style = "block" *) logic [15:0] bram_low  [0:DEPTH-1];
    (* ram_style = "block" *) logic [15:0] bram_high [0:DEPTH-1];

    //----------------------------------
    // Physical Address Calculations (10 bits: 5-tile, 2-pair, 3-col)
    //----------------------------------
    logic [ADDR_W-1:0] rd_addr;
    logic [ADDR_W-1:0] wr_addr_flat;

    assign rd_addr      = (ADDR_W'(rd_tile_i) << 5) | (ADDR_W'(rd_row_i[2:1]) << 3) | ADDR_W'(rd_col_i);
    assign wr_addr_flat = (ADDR_W'(wr_tile_i) << 5) | (ADDR_W'(wr_row_i[2:1]) << 3) | ADDR_W'(wr_col_i);

    //----------------------------------
    // Synchronous Read Process (Dual Parallel Memory Ports)
    //----------------------------------
    logic [15:0] low_q;
    logic [15:0] high_q;

    always_ff @(posedge clk_i) begin
        if (rd_en_i) begin
            // Safety enforcement: ensure read alignments hit even group row boundaries
            assert(rd_row_i[0] == 1'b0) else
                $error("[BRAM_ACCUM_ERROR] Read row must be even for paired layout, got %0d", rd_row_i);

            low_q  <= bram_low[rd_addr];
            high_q <= bram_high[rd_addr];
        end
    end

    // Latency Option B: Direct structural assign combination 
    // Cycle N: rd_en_i + address presented
    // Cycle N+1: low_q / high_q updated -> rd_data_o output matches perfectly
    assign rd_data_o = {high_q, low_q};

    // Tracking registers to properly align historical debug statements
    logic        rd_en_q;
    logic [4:0]  rd_tile_q;
    logic [2:0]  rd_row_q;
    logic [2:0]  rd_col_q;

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
    // Synchronous Write Process
    //----------------------------------
    always_ff @(posedge clk_i) begin
        if (wr_en_i) begin
            if (wr_pair_i) begin
                // Paired write (Scale update): Even row maps to low bank, Odd to high bank
                assert(wr_row_i[0] == 1'b0) else
                    $error("[BRAM_ACCUM_ERROR] Paired write row must be even, got %0d", wr_row_i);

                bram_low[wr_addr_flat]  <= wr_data_i[15:0];
                bram_high[wr_addr_flat] <= wr_data_i[31:16];
            end else begin
                // Single-cell write (Bias update): Drive sub-word based exclusively on row parity
                if (wr_row_i[0] == 1'b0) begin
                    bram_low[wr_addr_flat]  <= wr_data_i[15:0];
                end else begin
                    bram_high[wr_addr_flat] <= wr_data_i[15:0];
                end
            end
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
                $display("[BRAM_ACCUM_DEBUG]    Coordinates -> Tile=%2d | Rows=%1d,%1d | Col=%1d", rd_tile_q, rd_row_q, rd_row_q+1, rd_col_q);
                $display("[BRAM_ACCUM_DEBUG]    Payload     -> Out Data=32'h%h", rd_data_o);
                $display("[BRAM_ACCUM_DEBUG]                 row0=%4h row1=%4h", rd_data_o[15:0], rd_data_o[31:16]);
            end

            if (wr_en_i) begin
                $display("[BRAM_ACCUM_DEBUG] [%0t ns] MEMORY WRITE TRANSACTION COMMITTED:", $time);
                $display("[BRAM_ACCUM_DEBUG]    Coordinates -> Tile=%2d | Row=%1d | Col=%1d", wr_tile_i, wr_row_i, wr_col_i);
                $display("[BRAM_ACCUM_DEBUG]    Payload     -> In Data=32'h%h | Pair Write=%0b", wr_data_i, wr_pair_i);
            end
        end
    end

    // Visual matrix snapshot generation logic
    integer r, c, addr;
    always_ff @(posedge clk_i) begin
        //if (rst_ni && wr_en_i) begin
            // We evaluate one cycle late to avoid delta-cycle evaluation race hazards
            //#0.1;
            
            //--------------------------------------------------
            // Tile 0 Visual Matrix Monitor Dump
            //--------------------------------------------------
            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE 0 @ time %0t", $time);
            for (r = 0; r < 8; r++) begin
                $write("Row %0d :", r);
                for (c = 0; c < 8; c++) begin
                    addr = (0 << 5) | ((r >> 1) << 3) | c;
                    $write(" %6h", (r[0] == 1'b0) ? bram_low[addr] : bram_high[addr]);
                end
                $write("\n");
            end
            $display("======================================================");

            //--------------------------------------------------
            // Tile 31 Visual Matrix Monitor Dump
            //--------------------------------------------------
            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE 31 @ time %0t", $time);
            for (r = 0; r < 8; r++) begin
                $write("Row %0d :", r);
                for (c = 0; c < 8; c++) begin
                    addr = (31 << 5) | ((r >> 1) << 3) | c;
                    $write(" %6h", (r[0] == 1'b0) ? bram_low[addr] : bram_high[addr]);
                end
                $write("\n");
            end
            $display("======================================================\n");
        //end
    end
`endif

endmodule
