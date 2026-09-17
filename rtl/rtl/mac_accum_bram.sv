// Copyright (c) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Optimized Single-Cycle Fast Accumulator Tile Block RAM Module.
// Features flexible dynamic indexing functions and forward bypassing
// to support zero-stall read-modify-write accumulation loops.

`timescale 1ns/1ps

module mac_accum_bram #(
    parameter int unsigned NTILES = 32,
    parameter int unsigned NROWS  = 8,
    parameter int unsigned NCOLS  = 8,
    parameter int unsigned DEPTH  = NTILES * (NROWS / 2) * NCOLS // 1024 words
) (
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

    localparam int unsigned ADDR_W = $clog2(DEPTH); // 10 bits

    // Split Memory Banks (Maps directly to Xilinx Block RAM Primitives)
    (* ram_style = "block" *) logic [15:0] bram_low  [0:DEPTH-1];
    (* ram_style = "block" *) logic [15:0] bram_high [0:DEPTH-1];

    //--------------------------------------------------------------------------
    // Dynamic / Changeable Indexing Rules
    // Edit this function to modify physical coordinate mapping dynamically.
    //--------------------------------------------------------------------------
    function automatic logic [ADDR_W-1:0] calc_addr(
        input logic [4:0] tile,
        input logic [2:0] row,
        input logic [2:0] col
    );
        // Default Rule: [Tile: 5 bits] | [Row Group (Pair): 2 bits] | [Col: 3 bits]
        return (ADDR_W'(tile) << 5) | (ADDR_W'(row[2:1]) << 3) | ADDR_W'(col);
    endfunction

    // Calculated addresses
    logic [ADDR_W-1:0] rd_addr;
    logic [ADDR_W-1:0] wr_addr;

    assign rd_addr = calc_addr(rd_tile_i, rd_row_i, rd_col_i);
    assign wr_addr = calc_addr(wr_tile_i, wr_row_i, wr_col_i);

    // Internal RAM output registers
    logic [15:0] ram_low_q;
    logic [15:0] ram_high_q;

    // Direct write-forwarding registers to eliminate write-to-read latency stalls
    logic [ADDR_W-1:0] wr_addr_q;
    logic [31:0]       wr_data_q;
    logic              wr_en_q;
    logic              wr_pair_q;
    logic [2:0]        wr_row_q;

    // Tracking read address for forwarding hazard checks
    logic [ADDR_W-1:0] rd_addr_q;

    //--------------------------------------------------------------------------
    // Low Bank Read / Write Process
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i) begin
        if (wr_en_i) begin
            if (wr_pair_i || (wr_row_i[0] == 1'b0)) begin
                bram_low[wr_addr] <= wr_data_i[15:0];
            end
        end
        if (rd_en_i) begin
            ram_low_q <= bram_low[rd_addr];
        end
    end

    //--------------------------------------------------------------------------
    // High Bank Read / Write Process
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i) begin
        if (wr_en_i) begin
            if (wr_pair_i) begin
                bram_high[wr_addr] <= wr_data_i[31:16];
            end else if (wr_row_i[0] == 1'b1) begin
                bram_high[wr_addr] <= wr_data_i[15:0];
            end
        end
        if (rd_en_i) begin
            ram_high_q <= bram_high[rd_addr];
        end
    end

    //--------------------------------------------------------------------------
    // Pipeline Forwarding Tracking Logic (For Single-Cycle Accumulation Loops)
    //--------------------------------------------------------------------------
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            wr_addr_q <= '0;
            wr_data_q <= '0;
            wr_en_q   <= 1'b0;
            wr_pair_q <= 1'b0;
            wr_row_q  <= '0;
            rd_addr_q <= '0;
        end else begin
            wr_addr_q <= wr_addr;
            wr_data_q <= wr_data_i;
            wr_en_q   <= wr_en_i;
            wr_pair_q <= wr_pair_i;
            wr_row_q  <= wr_row_i;
            if (rd_en_i) begin
                rd_addr_q <= rd_addr;
            end
        end
    end

    //--------------------------------------------------------------------------
    // Fast Output Muxing with Zero-Latency Forwarding Bypass
    //--------------------------------------------------------------------------
    logic [15:0] low_final;
    logic [15:0] high_final;

    always_comb begin
        // Hazard Detection: If reading the exact memory slot written on the immediate previous cycle
        if (wr_en_q && (rd_addr_q == wr_addr_q)) begin
            // Low sub-word bypass
            if (wr_pair_q || (wr_row_q[0] == 1'b0)) begin
                low_final = wr_data_q[15:0];
            end else begin
                low_final = ram_low_q;
            end

            // High sub-word bypass
            if (wr_pair_q) begin
                high_final = wr_data_q[31:16];
            end else if (wr_row_q[0] == 1'b1) begin
                high_final = wr_data_q[15:0];
            end else begin
                high_final = ram_high_q;
            end
        end else begin
            low_final  = ram_low_q;
            high_final = ram_high_q;
        end
    end

    assign rd_data_o = {high_final, low_final};

    //--------------------------------------------------------------------------
    // Assertions
    //--------------------------------------------------------------------------
`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rd_en_i) begin
            assert(rd_row_i[0] == 1'b0) else
                $error("[BRAM_ACCUM_ERROR] Read row must be even for paired layout, got %0d", rd_row_i);
        end
        if (wr_en_i && wr_pair_i) begin
            assert(wr_row_i[0] == 1'b0) else
                $error("[BRAM_ACCUM_ERROR] Paired write row must be even, got %0d", wr_row_i);
        end
    end
`endif

    //--------------------------------------------------------------------------
    // Simulation Visual Debug Display
    //--------------------------------------------------------------------------
`ifdef BRAM_DEBUG
    integer r, c, dbg_addr;
    always_ff @(posedge clk_i) begin
        if (rst_ni && wr_en_i) begin
            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE %0d SNAPSHOT @ time %0t", wr_tile_i, $time);
            for (r = 0; r < 8; r++) begin
                $write("Row %0d :", r);
                for (c = 0; c < 8; c++) begin
                    dbg_addr = calc_addr(wr_tile_i, r[2:0], c[2:0]);
                    $write(" %6h", (r[0] == 1'b0) ? bram_low[dbg_addr] : bram_high[dbg_addr]);
                end
                $write("\n");
            end
            $display("======================================================\n");
        end
    end
//`endif

//--------------------------------------------------------------------------
// Simulation Visual Debug Display
//--------------------------------------------------------------------------
//`ifdef BRAM_DEBUG
    // Register read address for logging debug output 1 cycle later
    logic [2:0] rd_row_q, rd_col_q;
    logic [4:0] rd_tile_q;
    logic       rd_en_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
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

    // Transaction Logger
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

    // Visual Matrix Snapshot Generator for Tiles 0, 1, and 31
    //integer r, c, dbg_addr;
    always_ff @(posedge clk_i) begin
       // if (rst_ni && wr_en_i) begin
            // Evaluate on write commit cycle
            
            //--------------------------------------------------
            // Tile 0 Visual Matrix Dump
            //--------------------------------------------------
            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE 0 SNAPSHOT @ time %0t", $time);
            for (r = 0; r < 8; r++) begin
                $write("Row %0d :", r);
                for (c = 0; c < 8; c++) begin
                    dbg_addr = calc_addr(5'd0, r[2:0], c[2:0]);
                    $write(" %6h", (r[0] == 1'b0) ? bram_low[dbg_addr] : bram_high[dbg_addr]);
                end
                $write("\n");
            end
            $display("======================================================");

            //--------------------------------------------------
            // Tile 1 Visual Matrix Dump
            //--------------------------------------------------
            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE 1 SNAPSHOT @ time %0t", $time);
            for (r = 0; r < 8; r++) begin
                $write("Row %0d :", r);
                for (c = 0; c < 8; c++) begin
                    dbg_addr = calc_addr(5'd1, r[2:0], c[2:0]);
                    $write(" %6h", (r[0] == 1'b0) ? bram_low[dbg_addr] : bram_high[dbg_addr]);
                end
                $write("\n");
            end
            $display("======================================================");

            //--------------------------------------------------
            // Tile 31 Visual Matrix Dump
            //--------------------------------------------------
            $display("\n======================================================");
            $display("ACCUMULATOR BRAM TILE 31 SNAPSHOT @ time %0t", $time);
            for (r = 0; r < 8; r++) begin
                $write("Row %0d :", r);
                for (c = 0; c < 8; c++) begin
                    dbg_addr = calc_addr(5'd31, r[2:0], c[2:0]);
                    $write(" %6h", (r[0] == 1'b0) ? bram_low[dbg_addr] : bram_high[dbg_addr]);
                end
                $write("\n");
            end
            $display("======================================================\n");
       // end
    end
`endif
endmodule
